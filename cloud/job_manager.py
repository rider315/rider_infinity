"""
job_manager.py — Job serialization, chunking, and enqueueing.

Handles packing historical data + parameter combos into S3/SQS-ready
payloads for distribution to cloud workers.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("cloud.job_manager")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 50


def serialize_job_data(
    df: pd.DataFrame,
    symbol: str,
    point: float,
    digits: int,
    pip_value_per_lot: float,
) -> tuple[bytes, bytes]:
    """Serialize DataFrame and metadata for upload to S3.

    Returns (parquet_bytes, meta_json_bytes).
    """
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", index=False)
    parquet_bytes = buf.getvalue()

    meta = {
        "symbol": symbol,
        "point": point,
        "digits": digits,
        "pip_value_per_lot": pip_value_per_lot,
    }
    meta_bytes = json.dumps(meta).encode()

    logger.info("Serialized %d bars → %d KB parquet, meta for %s",
                len(df), len(parquet_bytes) // 1024, symbol)
    return parquet_bytes, meta_bytes


def chunk_combos(
    combos: list[dict],
    batch_size: int = BATCH_SIZE,
) -> list[list[dict]]:
    """Split parameter combos into batches for SQS messages."""
    chunks = []
    for i in range(0, len(combos), batch_size):
        chunks.append(combos[i:i + batch_size])
    logger.info("Split %d combos into %d batches of ≤%d",
                len(combos), len(chunks), batch_size)
    return chunks


def build_sqs_messages(
    job_id: str,
    strategy_name: str,
    chunks: list[list[dict]],
    backtest_config: dict,
) -> list[dict]:
    """Build SQS message payloads from combo chunks."""
    messages = []
    for idx, chunk in enumerate(chunks):
        messages.append({
            "job_id": job_id,
            "batch_id": f"batch-{idx:05d}",
            "strategy_name": strategy_name,
            "combos": chunk,
            "backtest_config": backtest_config,
        })
    return messages


def create_worker_package() -> bytes:
    """Create a tar.gz of the worker code + strategies for deployment.

    Includes:
      - cloud/ package (worker code)
      - strategies/ package (all strategies)
      - instruments.py (symbol families + pip arithmetic)
      - backtest_core.py (the shared simulation kernel)
      - utils.py (logging / ATR helpers)

    ``instruments.py`` and ``backtest_core.py`` are what make the worker
    agree with the local engine. The worker used to carry its own
    hard-coded pip arithmetic, which was wrong by 1000x on gold.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add strategies directory
        strat_dir = _PROJECT_ROOT / "strategies"
        for py_file in strat_dir.glob("*.py"):
            arcname = f"strategies/{py_file.name}"
            tar.add(str(py_file), arcname=arcname)

        # Add cloud/worker directory
        worker_dir = _PROJECT_ROOT / "cloud" / "worker"
        for py_file in worker_dir.glob("*.py"):
            arcname = f"cloud/worker/{py_file.name}"
            tar.add(str(py_file), arcname=arcname)

        # Add cloud/__init__.py
        cloud_init = _PROJECT_ROOT / "cloud" / "__init__.py"
        if cloud_init.exists():
            tar.add(str(cloud_init), arcname="cloud/__init__.py")

        # Add the root-level modules the worker imports.
        for module in ("instruments.py", "backtest_core.py", "utils.py",
                       "session_manager.py", "evaluation_matrix.py"):
            module_file = _PROJECT_ROOT / module
            if module_file.exists():
                tar.add(str(module_file), arcname=module)

    package_bytes = buf.getvalue()
    logger.info("Created worker package: %d KB", len(package_bytes) // 1024)
    return package_bytes
