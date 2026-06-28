"""
result_collector.py — Polls S3 for completed batch results and merges them.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("cloud.result_collector")


class ResultCollector:
    """Collects and merges results from cloud workers via S3."""

    def __init__(self, aws_infra, bucket: str, job_id: str,
                 total_batches: int, target_metric: str = "profit_factor"):
        self.infra = aws_infra
        self.bucket = bucket
        self.job_id = job_id
        self.total_batches = total_batches
        self.target_metric = target_metric

        self._collected_batches: set[str] = set()
        self._all_results: list[dict] = []

    @property
    def completed_batches(self) -> int:
        return len(self._collected_batches)

    @property
    def progress(self) -> float:
        if self.total_batches == 0:
            return 1.0
        return self.completed_batches / self.total_batches

    @property
    def is_complete(self) -> bool:
        return self.completed_batches >= self.total_batches

    def poll(self) -> int:
        """Check S3 for new result files. Returns count of new batches found."""
        prefix = f"jobs/{self.job_id}/results/"
        keys = self.infra.list_objects(self.bucket, prefix)

        new_count = 0
        for key in keys:
            batch_id = key.rsplit("/", 1)[-1].replace(".json", "")
            if batch_id in self._collected_batches:
                continue

            try:
                results = self.infra.download_json(self.bucket, key)
                if isinstance(results, list):
                    self._all_results.extend(results)
                    self._collected_batches.add(batch_id)
                    new_count += 1
            except Exception as exc:
                logger.warning("Failed to download result %s: %s", key, exc)

        if new_count:
            logger.info("Collected %d new batches (%d/%d total, %d results)",
                        new_count, self.completed_batches,
                        self.total_batches, len(self._all_results))

        return new_count

    def get_ranked_results(self, top_n: int = 100) -> list[dict]:
        """Get top results sorted by target metric."""
        metric_key = self.target_metric
        reverse = metric_key not in ("max_drawdown", "max_drawdown_pct", "avg_loss_pips")

        sorted_results = sorted(
            self._all_results,
            key=lambda r: r.get(metric_key, 0) or 0,
            reverse=reverse,
        )
        return sorted_results[:top_n]

    def get_best(self) -> tuple[dict, float]:
        """Get best params and metric value."""
        ranked = self.get_ranked_results(1)
        if not ranked:
            return {}, 0.0
        best = ranked[0]
        return best.get("params", {}), best.get(self.target_metric, 0)

    @property
    def total_results(self) -> int:
        return len(self._all_results)
