"""
aws_infra.py — AWS resource management for cloud optimisation.

Handles creation and management of S3 buckets, SQS queues, IAM roles,
EC2 launch templates, and Spot Instance fleets. All resources are
tagged with 'rider-infinity' for easy identification and cleanup.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger("cloud.aws_infra")

TAG_KEY = "Project"
TAG_VALUE = "rider-infinity"
RESOURCE_PREFIX = "ri-cloud"


class AWSInfra:
    """Manages AWS resources for distributed optimisation."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
        self.sqs = boto3.client("sqs", region_name=region)
        self.ec2 = boto3.client("ec2", region_name=region)
        self.iam = boto3.client("iam", region_name=region)
        self.sts = boto3.client("sts", region_name=region)
        self._account_id: str | None = None

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            self._account_id = self.sts.get_caller_identity()["Account"]
        return self._account_id

    # ── S3 ────────────────────────────────────────────────────────────

    def ensure_bucket(self, bucket_name: str) -> str:
        """Create S3 bucket if it doesn't exist. Returns bucket name."""
        try:
            self.s3.head_bucket(Bucket=bucket_name)
            logger.info("S3 bucket '%s' already exists.", bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                create_args: dict[str, Any] = {"Bucket": bucket_name}
                if self.region != "us-east-1":
                    create_args["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.region
                    }
                self.s3.create_bucket(**create_args)
                logger.info("Created S3 bucket '%s'.", bucket_name)
            else:
                raise

        # Set lifecycle rule to auto-delete job data after 7 days
        try:
            self.s3.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration={
                    "Rules": [{
                        "ID": "ri-auto-cleanup",
                        "Status": "Enabled",
                        "Filter": {"Prefix": "jobs/"},
                        "Expiration": {"Days": 7},
                    }]
                },
            )
        except ClientError:
            logger.warning("Could not set lifecycle policy on bucket.")

        # Tag the bucket
        try:
            self.s3.put_bucket_tagging(
                Bucket=bucket_name,
                Tagging={"TagSet": [{"Key": TAG_KEY, "Value": TAG_VALUE}]},
            )
        except ClientError:
            pass

        return bucket_name

    def upload_file(self, bucket: str, key: str, file_path: str) -> None:
        self.s3.upload_file(file_path, bucket, key)

    def upload_bytes(self, bucket: str, key: str, data: bytes,
                     content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=bucket, Key=key, Body=data,
                           ContentType=content_type)

    def list_objects(self, bucket: str, prefix: str) -> list[str]:
        """List object keys under a prefix."""
        keys = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def download_json(self, bucket: str, key: str) -> Any:
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode())

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects under a prefix. Returns count deleted."""
        keys = self.list_objects(bucket, prefix)
        if not keys:
            return 0
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            self.s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in batch]},
            )
        logger.info("Deleted %d objects under s3://%s/%s", len(keys), bucket, prefix)
        return len(keys)

    # ── SQS ───────────────────────────────────────────────────────────

    def create_job_queue(self, job_id: str) -> tuple[str, str]:
        """Create SQS queue + dead-letter queue for a job.

        Returns (queue_url, dlq_url).
        """
        dlq_name = f"{RESOURCE_PREFIX}-dlq-{job_id}"
        queue_name = f"{RESOURCE_PREFIX}-{job_id}"

        # Create DLQ
        dlq_resp = self.sqs.create_queue(
            QueueName=dlq_name,
            Attributes={
                "MessageRetentionPeriod": str(86400 * 4),  # 4 days
                "VisibilityTimeout": "600",
            },
            tags={TAG_KEY: TAG_VALUE, "JobId": job_id},
        )
        dlq_url = dlq_resp["QueueUrl"]
        dlq_arn = self.sqs.get_queue_attributes(
            QueueUrl=dlq_url, AttributeNames=["QueueArn"],
        )["Attributes"]["QueueArn"]

        # Create main queue with DLQ redrive policy
        queue_resp = self.sqs.create_queue(
            QueueName=queue_name,
            Attributes={
                "MessageRetentionPeriod": str(86400 * 4),
                "VisibilityTimeout": "600",
                "RedrivePolicy": json.dumps({
                    "deadLetterTargetArn": dlq_arn,
                    "maxReceiveCount": "3",
                }),
            },
            tags={TAG_KEY: TAG_VALUE, "JobId": job_id},
        )
        queue_url = queue_resp["QueueUrl"]

        logger.info("Created SQS queue: %s (DLQ: %s)", queue_name, dlq_name)
        return queue_url, dlq_url

    def send_messages(self, queue_url: str, messages: list[dict]) -> None:
        """Send messages to SQS in batches of 10."""
        for i in range(0, len(messages), 10):
            batch = messages[i:i + 10]
            entries = [
                {
                    "Id": str(idx),
                    "MessageBody": json.dumps(msg),
                }
                for idx, msg in enumerate(batch)
            ]
            self.sqs.send_message_batch(QueueUrl=queue_url, Entries=entries)
        logger.info("Sent %d messages to SQS.", len(messages))

    def get_queue_depth(self, queue_url: str) -> int:
        """Get approximate number of messages in queue."""
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"],
        )["Attributes"]
        return (int(attrs.get("ApproximateNumberOfMessages", 0))
                + int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0)))

    def purge_queue(self, queue_url: str) -> None:
        """Purge all messages from a queue."""
        try:
            self.sqs.purge_queue(QueueUrl=queue_url)
        except ClientError as e:
            if "PurgeQueueInProgress" not in str(e):
                raise

    def delete_queue(self, queue_url: str) -> None:
        try:
            self.sqs.delete_queue(QueueUrl=queue_url)
        except ClientError:
            logger.warning("Could not delete queue %s", queue_url)

    # ── IAM ───────────────────────────────────────────────────────────

    def ensure_worker_role(self, bucket_name: str) -> str:
        """Create or update IAM role for workers. Returns role ARN."""
        role_name = f"{RESOURCE_PREFIX}-worker-role"
        profile_name = f"{RESOURCE_PREFIX}-worker-profile"

        assume_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }],
        })

        try:
            resp = self.iam.get_role(RoleName=role_name)
            role_arn = resp["Role"]["Arn"]
            logger.info("Worker role already exists: %s", role_arn)
        except ClientError:
            resp = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=assume_policy,
                Description="Rider Infinity cloud worker role",
                Tags=[{"Key": TAG_KEY, "Value": TAG_VALUE}],
            )
            role_arn = resp["Role"]["Arn"]
            logger.info("Created worker role: %s", role_arn)

        # Attach inline policy scoped to the S3 bucket and SQS queues
        worker_policy = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject", "s3:PutObject", "s3:ListBucket",
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sqs:ReceiveMessage", "sqs:DeleteMessage",
                        "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility",
                    ],
                    "Resource": f"arn:aws:sqs:{self.region}:{self.account_id}:{RESOURCE_PREFIX}-*",
                },
            ],
        })

        self.iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{RESOURCE_PREFIX}-worker-policy",
            PolicyDocument=worker_policy,
        )

        # Instance profile
        try:
            self.iam.get_instance_profile(InstanceProfileName=profile_name)
        except ClientError:
            self.iam.create_instance_profile(InstanceProfileName=profile_name)
            self.iam.add_role_to_instance_profile(
                InstanceProfileName=profile_name, RoleName=role_name,
            )
            # IAM propagation delay
            logger.info("Waiting for IAM profile propagation...")
            time.sleep(10)

        return role_arn

    # ── EC2 ───────────────────────────────────────────────────────────

    def get_spot_price(self, instance_type: str = "c6a.large") -> float:
        """Get current spot price for the instance type in this region."""
        try:
            resp = self.ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=5,
            )
            prices = resp.get("SpotPriceHistory", [])
            if prices:
                return float(prices[0]["SpotPrice"])
        except ClientError:
            pass
        # Fallback estimate
        return 0.03

    def get_latest_amazon_linux_ami(self) -> str:
        """Get the latest Amazon Linux 2023 AMI ID."""
        resp = self.ec2.describe_images(
            Owners=["amazon"],
            Filters=[
                {"Name": "name", "Values": ["al2023-ami-2023*-x86_64"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        images = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
        if not images:
            raise RuntimeError("No Amazon Linux 2023 AMI found in region " + self.region)
        return images[0]["ImageId"]

    def get_default_vpc_subnet(self) -> str:
        """Get a subnet ID from the default VPC."""
        resp = self.ec2.describe_subnets(
            Filters=[{"Name": "default-for-az", "Values": ["true"]}],
        )
        subnets = resp.get("Subnets", [])
        if not subnets:
            raise RuntimeError("No default VPC subnet found. Please specify a subnet.")
        return subnets[0]["SubnetId"]

    def get_or_create_security_group(self) -> str:
        """Get or create a security group for workers (egress-only)."""
        sg_name = f"{RESOURCE_PREFIX}-worker-sg"
        try:
            resp = self.ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": [sg_name]}],
            )
            groups = resp.get("SecurityGroups", [])
            if groups:
                return groups[0]["GroupId"]
        except ClientError:
            pass

        # Create with egress-only (no inbound)
        resp = self.ec2.create_security_group(
            GroupName=sg_name,
            Description="Rider Infinity cloud worker — egress only",
        )
        sg_id = resp["GroupId"]

        # Revoke default egress and re-add (default allows all egress, which is fine)
        self.ec2.create_tags(
            Resources=[sg_id],
            Tags=[{"Key": TAG_KEY, "Value": TAG_VALUE}],
        )
        logger.info("Created security group: %s (%s)", sg_name, sg_id)
        return sg_id

    def launch_spot_fleet(
        self,
        num_instances: int,
        instance_type: str,
        user_data: str,
        job_id: str,
        max_runtime_mins: int = 120,
    ) -> list[str]:
        """Launch EC2 Spot Instances for a job. Returns list of instance IDs."""
        ami_id = self.get_latest_amazon_linux_ami()
        subnet_id = self.get_default_vpc_subnet()
        sg_id = self.get_or_create_security_group()
        profile_name = f"{RESOURCE_PREFIX}-worker-profile"

        user_data_b64 = base64.b64encode(user_data.encode()).decode()

        resp = self.ec2.run_instances(
            ImageId=ami_id,
            InstanceType=instance_type,
            MinCount=num_instances,
            MaxCount=num_instances,
            SubnetId=subnet_id,
            SecurityGroupIds=[sg_id],
            IamInstanceProfile={"Name": profile_name},
            UserData=user_data_b64,
            InstanceMarketOptions={
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            TagSpecifications=[{
                "ResourceType": "instance",
                "Tags": [
                    {"Key": TAG_KEY, "Value": TAG_VALUE},
                    {"Key": "JobId", "Value": job_id},
                    {"Key": "MaxRuntimeMins", "Value": str(max_runtime_mins)},
                    {"Key": "Name", "Value": f"{RESOURCE_PREFIX}-worker-{job_id}"},
                ],
            }],
            InstanceInitiatedShutdownBehavior="terminate",
        )

        instance_ids = [inst["InstanceId"] for inst in resp["Instances"]]
        logger.info("Launched %d spot instances for job %s: %s",
                     len(instance_ids), job_id, instance_ids)
        return instance_ids

    def terminate_instances(self, instance_ids: list[str]) -> None:
        """Terminate EC2 instances."""
        if not instance_ids:
            return
        try:
            self.ec2.terminate_instances(InstanceIds=instance_ids)
            logger.info("Terminated instances: %s", instance_ids)
        except ClientError as e:
            logger.warning("Failed to terminate some instances: %s", e)

    def get_job_instances(self, job_id: str) -> list[dict]:
        """Get all running/pending instances for a job."""
        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
                    {"Name": "tag:JobId", "Values": [job_id]},
                    {"Name": "instance-state-name",
                     "Values": ["pending", "running"]},
                ],
            )
        except ClientError:
            return []

        instances = []
        for reservation in resp.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                instances.append({
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "launch_time": inst["LaunchTime"].isoformat(),
                    "instance_type": inst["InstanceType"],
                })
        return instances

    def get_all_ri_instances(self) -> list[dict]:
        """Get all Rider Infinity instances (for cleanup)."""
        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": f"tag:{TAG_KEY}", "Values": [TAG_VALUE]},
                    {"Name": "instance-state-name",
                     "Values": ["pending", "running"]},
                ],
            )
        except ClientError:
            return []

        instances = []
        for reservation in resp.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                instances.append({
                    "instance_id": inst["InstanceId"],
                    "state": inst["State"]["Name"],
                    "launch_time": inst["LaunchTime"],
                    "job_id": tags.get("JobId", "unknown"),
                    "max_runtime_mins": int(tags.get("MaxRuntimeMins", "120")),
                })
        return instances

    # ── Cleanup ───────────────────────────────────────────────────────

    def cleanup_job(self, job_id: str, bucket: str,
                    queue_url: str | None = None,
                    dlq_url: str | None = None) -> None:
        """Clean up all resources for a completed job."""
        # Terminate any remaining instances
        instances = self.get_job_instances(job_id)
        if instances:
            ids = [i["instance_id"] for i in instances]
            self.terminate_instances(ids)

        # Delete SQS queues
        if queue_url:
            self.delete_queue(queue_url)
        if dlq_url:
            self.delete_queue(dlq_url)

        # Delete S3 job data (results may be kept for review)
        self.delete_prefix(bucket, f"jobs/{job_id}/")

        logger.info("Cleaned up all resources for job %s.", job_id)
