"""
cloud — Cloud-Based Distributed Optimisation for Rider Infinity.

Provides AWS-backed parallel execution of strategy optimisation jobs
using EC2 Spot Instances, SQS for job distribution, and S3 for data
transfer.

CloudOptimizer is imported lazily to avoid requiring boto3 when only
the worker_backtest functions are needed (e.g. by the local optimizer).
"""


def __getattr__(name: str):
    if name == "CloudOptimizer":
        from .cloud_optimizer import CloudOptimizer
        return CloudOptimizer
    raise AttributeError(f"module 'cloud' has no attribute {name!r}")


__all__ = ["CloudOptimizer"]
