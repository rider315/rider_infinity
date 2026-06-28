"""
cost_tracker.py — Real-time cost estimation and budget enforcement.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger("cloud.cost_tracker")


class CostTracker:
    """Tracks estimated cloud cost for a running optimisation job."""

    def __init__(self, spot_price_per_hour: float, max_cost: float = 10.0):
        self.spot_price_per_hour = spot_price_per_hour
        self.max_cost = max_cost
        self._instance_launch_times: dict[str, float] = {}
        self._terminated_instance_hours: float = 0.0

    def register_instances(self, instance_ids: list[str]) -> None:
        now = time.time()
        for iid in instance_ids:
            self._instance_launch_times[iid] = now

    def mark_terminated(self, instance_ids: list[str]) -> None:
        now = time.time()
        for iid in instance_ids:
            launch = self._instance_launch_times.pop(iid, None)
            if launch:
                hours = (now - launch) / 3600
                self._terminated_instance_hours += hours

    @property
    def running_instance_hours(self) -> float:
        now = time.time()
        hours = 0.0
        for launch in self._instance_launch_times.values():
            hours += (now - launch) / 3600
        return hours

    @property
    def total_instance_hours(self) -> float:
        return self._terminated_instance_hours + self.running_instance_hours

    @property
    def estimated_cost(self) -> float:
        # EC2 spot + ~$0.01 for SQS/S3 overhead
        return round(self.total_instance_hours * self.spot_price_per_hour + 0.01, 4)

    @property
    def budget_exceeded(self) -> bool:
        return self.estimated_cost >= self.max_cost

    @property
    def active_instances(self) -> int:
        return len(self._instance_launch_times)

    def to_dict(self) -> dict:
        return {
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "max_cost_usd": self.max_cost,
            "budget_remaining_usd": round(max(0, self.max_cost - self.estimated_cost), 4),
            "total_instance_hours": round(self.total_instance_hours, 3),
            "active_instances": self.active_instances,
            "spot_price_per_hour": self.spot_price_per_hour,
        }


def estimate_job_cost(
    total_combos: int,
    spot_price_per_hour: float,
    combos_per_second: float = 2.0,
    num_workers: int = 1,
) -> dict:
    """Estimate cost and time for a cloud optimisation job.

    Args:
        total_combos: Total parameter combinations to test.
        spot_price_per_hour: Current spot price per instance-hour.
        combos_per_second: Estimated backtests per second per worker.
        num_workers: Number of worker instances.

    Returns:
        Dict with estimated cost, time, and breakdown.
    """
    total_seconds = total_combos / (combos_per_second * num_workers)
    total_hours = total_seconds / 3600
    # Each instance runs for the full duration (plus ~2 min setup)
    setup_hours = 2.0 / 60  # 2 minutes
    instance_hours = (total_hours + setup_hours) * num_workers

    ec2_cost = instance_hours * spot_price_per_hour
    sqs_cost = 0.0000004 * total_combos  # ~$0.40 per million requests
    s3_cost = 0.005  # minimal PUT/GET costs

    total_cost = ec2_cost + sqs_cost + s3_cost

    return {
        "total_combos": total_combos,
        "num_workers": num_workers,
        "estimated_time_seconds": round(total_seconds + 120, 0),  # +2min setup
        "estimated_time_minutes": round((total_seconds + 120) / 60, 1),
        "estimated_cost_usd": round(total_cost, 4),
        "breakdown": {
            "ec2_spot": round(ec2_cost, 4),
            "sqs": round(sqs_cost, 6),
            "s3": round(s3_cost, 4),
        },
        "spot_price_per_hour": spot_price_per_hour,
        "instance_hours": round(instance_hours, 3),
    }
