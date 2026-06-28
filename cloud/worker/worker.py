"""
worker.py — Cloud worker entry point for distributed optimisation.

Runs on EC2 Spot Instances. Polls SQS for batches of parameter combos,
downloads historical data from S3, runs backtests via worker_backtest,
uploads results to S3, and self-terminates when the queue is empty.

Environment variables (set by bootstrap.sh):
    SQS_QUEUE_URL  — URL of the job queue
    S3_BUCKET      — Bucket name for data and results
    JOB_ID         — Unique job identifier
    AWS_REGION     — AWS region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import boto3
import pandas as pd

# Add project root to path so strategies are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cloud.worker.worker_backtest import run_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("cloud-worker")

# ── Configuration ─────────────────────────────────────────────────────

SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
S3_BUCKET = os.environ["S3_BUCKET"]
JOB_ID = os.environ["JOB_ID"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

EMPTY_POLL_TIMEOUT = 60  # seconds of empty queue before self-termination
VISIBILITY_TIMEOUT_EXTEND = 300  # extend visibility every 5 min
MAX_BATCH_RETRIES = 2

# ── AWS Clients ───────────────────────────────────────────────────────

sqs = boto3.client("sqs", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

# ── Spot Interruption Handling ────────────────────────────────────────

_shutting_down = False


def _handle_shutdown(signum, frame):
    global _shutting_down
    logger.warning("Received signal %s — finishing current batch then exiting.", signum)
    _shutting_down = True


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


def _check_spot_interruption() -> bool:
    """Check EC2 metadata for spot interruption notice (2-min warning)."""
    try:
        import urllib.request
        # IMDSv2: get token first
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT",
        )
        token = urllib.request.urlopen(token_req, timeout=2).read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            headers={"X-aws-ec2-metadata-token": token},
        )
        urllib.request.urlopen(req, timeout=2)
        return True  # if endpoint exists, interruption is imminent
    except Exception:
        return False


# ── Data Loading ──────────────────────────────────────────────────────

_cached_df: pd.DataFrame | None = None
_cached_meta: dict | None = None


def _load_job_data() -> tuple[pd.DataFrame, dict]:
    """Download historical data and metadata from S3 (cached after first call)."""
    global _cached_df, _cached_meta

    if _cached_df is not None and _cached_meta is not None:
        return _cached_df, _cached_meta

    logger.info("Downloading job data from s3://%s/jobs/%s/ ...", S3_BUCKET, JOB_ID)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        parquet_path = f.name
    s3.download_file(S3_BUCKET, f"jobs/{JOB_ID}/data.parquet", parquet_path)
    df = pd.read_parquet(parquet_path)
    os.unlink(parquet_path)

    meta_obj = s3.get_object(Bucket=S3_BUCKET, Key=f"jobs/{JOB_ID}/meta.json")
    meta = json.loads(meta_obj["Body"].read().decode())

    # Ensure time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True)

    _cached_df = df
    _cached_meta = meta
    logger.info("Loaded %d bars, symbol=%s", len(df), meta.get("symbol"))
    return df, meta


# ── Batch Processing ─────────────────────────────────────────────────

def _process_batch(message: dict) -> None:
    """Process a single SQS message containing a batch of parameter combos."""
    body = json.loads(message["Body"])
    batch_id = body["batch_id"]
    strategy_name = body["strategy_name"]
    combos = body["combos"]
    bt_config = body["backtest_config"]
    receipt_handle = message["ReceiptHandle"]

    logger.info("Processing batch %s: %d combos for strategy '%s'",
                batch_id, len(combos), strategy_name)

    df, meta = _load_job_data()

    results = []
    last_heartbeat = time.time()

    for idx, params in enumerate(combos):
        if _shutting_down or _check_spot_interruption():
            logger.warning("Interruption during batch %s at combo %d/%d — uploading partial results.",
                           batch_id, idx, len(combos))
            break

        # Extend visibility timeout periodically
        if time.time() - last_heartbeat > VISIBILITY_TIMEOUT_EXTEND:
            try:
                sqs.change_message_visibility(
                    QueueUrl=SQS_QUEUE_URL,
                    ReceiptHandle=receipt_handle,
                    VisibilityTimeout=600,
                )
                last_heartbeat = time.time()
            except Exception as exc:
                logger.warning("Failed to extend visibility: %s", exc)

        try:
            metrics = run_single(
                strategy_name=strategy_name,
                params=params,
                df=df,
                symbol=meta["symbol"],
                point=meta["point"],
                digits=meta["digits"],
                pip_value_per_lot=meta["pip_value_per_lot"],
                lot_size=bt_config["lot_size"],
                initial_balance=bt_config["initial_balance"],
                default_sl=bt_config["default_sl"],
                default_tp=bt_config["default_tp"],
                max_open_trades=bt_config.get("max_open_trades", 5),
            )
            if metrics["total_trades"] >= bt_config.get("min_trades", 3):
                metrics["params"] = params
                results.append(metrics)
        except Exception as exc:
            logger.debug("Combo %d in batch %s failed: %s", idx, batch_id, exc)

    # Upload results to S3
    result_key = f"jobs/{JOB_ID}/results/{batch_id}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=result_key,
        Body=json.dumps(results).encode(),
        ContentType="application/json",
    )
    logger.info("Uploaded %d results for batch %s → s3://%s/%s",
                len(results), batch_id, S3_BUCKET, result_key)

    # Delete the message from the queue
    sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)


# ── Main Loop ─────────────────────────────────────────────────────────

def main():
    logger.info("Worker starting: job=%s queue=%s bucket=%s", JOB_ID, SQS_QUEUE_URL, S3_BUCKET)

    empty_since: float | None = None

    while not _shutting_down:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,  # long poll
                VisibilityTimeout=600,  # 10 minutes
            )
        except Exception as exc:
            logger.error("SQS receive failed: %s — retrying in 5s", exc)
            time.sleep(5)
            continue

        messages = response.get("Messages", [])

        if not messages:
            if empty_since is None:
                empty_since = time.time()
                logger.info("Queue empty — will self-terminate after %ds of inactivity.",
                            EMPTY_POLL_TIMEOUT)
            elif time.time() - empty_since >= EMPTY_POLL_TIMEOUT:
                logger.info("Queue empty for %ds — self-terminating.", EMPTY_POLL_TIMEOUT)
                break
            continue

        empty_since = None

        for message in messages:
            try:
                _process_batch(message)
            except Exception as exc:
                logger.error("Failed to process batch: %s", exc, exc_info=True)

    logger.info("Worker shutting down.")
    _self_terminate()


def _self_terminate():
    """Attempt to shut down the EC2 instance."""
    logger.info("Initiating self-termination...")
    try:
        subprocess.run(["sudo", "shutdown", "-h", "now"], timeout=10)
    except Exception:
        logger.info("Could not self-terminate (not running on EC2). Exiting process.")
        sys.exit(0)


if __name__ == "__main__":
    main()
