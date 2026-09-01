# 作用：导出 M-bucket workload 生成函数。
from .generate_m_buckets import build_records, bucket, manifest

__all__ = ["build_records", "bucket", "manifest"]
