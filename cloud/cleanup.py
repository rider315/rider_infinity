"""
cleanup.py — Stale resource cleanup for cloud optimisation.

Finds and terminates orphaned EC2 instances that have exceeded their
max runtime, and cleans up leftover S3/SQS resources from failed jobs.

Can be run standalone:
    python -m cloud.cleanup

Or called programmatically on dashboard startup.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

from cloud.aws_infra import AWSInfra
from cloud.setup import load_cloud_config

logger = logging.getLogger("cloud.cleanup")


def cleanup_stale_instances(region: str | None = None) -> list[str]:
    """Find and terminate EC2 instances that have exceeded max_runtime.

    Returns list of terminated instance IDs.
    """
    config = load_cloud_config()
    if not config:
        return []

    region = region or config.get("region", "us-east-1")
    infra = AWSInfra(region=region)

    instances = infra.get_all_ri_instances()
    now = datetime.now(timezone.utc)
    terminated = []

    for inst in instances:
        launch_time = inst["launch_time"]
        if isinstance(launch_time, str):
            launch_time = datetime.fromisoformat(launch_time)
        if launch_time.tzinfo is None:
            launch_time = launch_time.replace(tzinfo=timezone.utc)

        max_runtime = inst.get("max_runtime_mins", 120)
        deadline = launch_time + timedelta(minutes=max_runtime)

        if now > deadline:
            logger.warning(
                "Instance %s exceeded max runtime (%d min, launched %s). Terminating.",
                inst["instance_id"], max_runtime, launch_time.isoformat(),
            )
            infra.terminate_instances([inst["instance_id"]])
            terminated.append(inst["instance_id"])

    if terminated:
        logger.info("Terminated %d stale instances.", len(terminated))
    else:
        logger.info("No stale instances found.")

    return terminated


def cleanup_stale_queues(region: str | None = None) -> int:
    """Delete orphaned SQS queues from old jobs.

    Returns number of queues deleted.
    """
    config = load_cloud_config()
    if not config:
        return 0

    region = region or config.get("region", "us-east-1")
    infra = AWSInfra(region=region)
    sqs = infra.sqs

    deleted = 0
    try:
        resp = sqs.list_queues(QueueNamePrefix="ri-cloud-")
        for queue_url in resp.get("QueueUrls", []):
            try:
                attrs = sqs.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=["ApproximateNumberOfMessages",
                                    "ApproximateNumberOfMessagesNotVisible",
                                    "CreatedTimestamp"],
                )["Attributes"]

                total_msgs = (int(attrs.get("ApproximateNumberOfMessages", 0))
                              + int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)))
                created = datetime.fromtimestamp(
                    int(attrs.get("CreatedTimestamp", "0")), tz=timezone.utc,
                )
                age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600

                # Delete empty queues older than 4 hours
                if total_msgs == 0 and age_hours > 4:
                    sqs.delete_queue(QueueUrl=queue_url)
                    deleted += 1
                    logger.info("Deleted stale queue: %s (age: %.1fh)", queue_url, age_hours)
            except Exception as exc:
                logger.debug("Could not check queue %s: %s", queue_url, exc)
    except Exception as exc:
        logger.warning("Could not list SQS queues: %s", exc)

    return deleted


def run_full_cleanup() -> dict:
    """Run all cleanup tasks. Returns summary."""
    config = load_cloud_config()
    if not config:
        return {"skipped": True, "reason": "Cloud not set up."}

    region = config.get("region", "us-east-1")
    terminated = cleanup_stale_instances(region)
    deleted_queues = cleanup_stale_queues(region)

    return {
        "terminated_instances": terminated,
        "deleted_queues": deleted_queues,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_full_cleanup()
    print(f"Cleanup result: {result}")
