#!/bin/bash
# bootstrap.sh — EC2 user-data script for Rider Infinity cloud workers.
#
# This script runs on first boot of each EC2 Spot Instance. It:
#   1. Installs Python and dependencies
#   2. Downloads the worker package from S3
#   3. Starts the worker process
#
# Environment variables are passed via instance tags or launch template
# and injected below by the orchestrator before uploading to S3.

set -euo pipefail

exec > /var/log/ri-worker.log 2>&1
echo "=== Rider Infinity Cloud Worker Bootstrap ==="
echo "Timestamp: $(date -u)"

# ── Variables (injected by orchestrator) ──────────────────────────────
export SQS_QUEUE_URL="{{SQS_QUEUE_URL}}"
export S3_BUCKET="{{S3_BUCKET}}"
export JOB_ID="{{JOB_ID}}"
export AWS_REGION="{{AWS_REGION}}"

# ── Install dependencies ─────────────────────────────────────────────
echo "Installing Python and dependencies..."
yum update -y -q
yum install -y -q python3.11 python3.11-pip

pip3.11 install --quiet \
    pandas>=2.0 \
    numpy>=1.24 \
    pyarrow>=12.0 \
    boto3>=1.28.0 \
    pandas-ta>=0.3.14

# ── Download worker package ──────────────────────────────────────────
echo "Downloading worker package from S3..."
WORK_DIR="/opt/ri-worker"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

aws s3 cp "s3://${S3_BUCKET}/jobs/${JOB_ID}/worker-package.tar.gz" worker-package.tar.gz
tar xzf worker-package.tar.gz
rm worker-package.tar.gz

echo "Worker package contents:"
find . -type f -name "*.py" | head -20

# ── Start worker ─────────────────────────────────────────────────────
echo "Starting worker process..."
cd "$WORK_DIR"
python3.11 -m cloud.worker.worker
