"""
setup.py — One-time AWS infrastructure setup for cloud optimisation.

Creates persistent resources (S3 bucket, IAM role, security group) and
stores their identifiers in data/cloud_config.json for use by the
orchestrator.

Can be run standalone:
    python -m cloud.setup

Or triggered via the dashboard:
    POST /api/optimize/cloud/setup
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cloud.aws_infra import AWSInfra

logger = logging.getLogger("cloud.setup")

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "cloud_config.json"


def load_cloud_config() -> dict | None:
    """Load saved cloud config, or None if not set up yet."""
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_cloud_config(config: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, indent=2))
    logger.info("Saved cloud config to %s", _CONFIG_PATH)


def run_setup(
    region: str = "us-east-1",
    bucket_name: str | None = None,
) -> dict:
    """Run one-time AWS setup. Returns the cloud config dict.

    Idempotent — safe to run multiple times.
    """
    infra = AWSInfra(region=region)

    # Generate bucket name if not provided
    if not bucket_name:
        account_id = infra.account_id
        bucket_name = f"ri-cloud-{account_id}-{region}"

    logger.info("Setting up cloud infrastructure in %s ...", region)

    # 1. S3 bucket
    infra.ensure_bucket(bucket_name)

    # 2. IAM role + instance profile
    role_arn = infra.ensure_worker_role(bucket_name)

    # 3. Security group
    sg_id = infra.get_or_create_security_group()

    config = {
        "region": region,
        "bucket": bucket_name,
        "worker_role_arn": role_arn,
        "security_group_id": sg_id,
        "setup_complete": True,
    }

    save_cloud_config(config)
    logger.info("Cloud setup complete: %s", config)
    return config


def verify_setup() -> dict:
    """Check if cloud setup is complete and resources still exist.

    Returns status dict with 'ready' boolean.
    """
    config = load_cloud_config()
    if not config or not config.get("setup_complete"):
        return {"ready": False, "reason": "Cloud setup has not been run."}

    try:
        infra = AWSInfra(region=config["region"])

        # Verify bucket exists
        infra.s3.head_bucket(Bucket=config["bucket"])

        # Verify IAM role exists
        infra.iam.get_role(RoleName="ri-cloud-worker-role")

        return {"ready": True, "config": config}
    except Exception as exc:
        return {"ready": False, "reason": f"Resource check failed: {exc}"}


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket = os.environ.get("AWS_CLOUD_S3_BUCKET", "")
    result = run_setup(region=region, bucket_name=bucket or None)
    print(json.dumps(result, indent=2))
