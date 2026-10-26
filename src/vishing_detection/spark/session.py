"""Création de la SparkSession depuis configs/spark.yaml."""
from __future__ import annotations

from vishing_detection.config import get_spark_config


def get_spark(overrides: dict | None = None):
    from pyspark.sql import SparkSession

    cfg = {**get_spark_config().get("spark", {}), **(overrides or {})}
    builder = (
        SparkSession.builder
        .appName(cfg.get("app_name", "vishing-detection"))
        .master(cfg.get("master", "local[*]"))
        .config("spark.driver.memory", cfg.get("driver_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(cfg.get("shuffle_partitions", 16)))
        .config("spark.sql.parquet.compression.codec", cfg.get("parquet_compression", "snappy"))
        .config("spark.ui.showConsoleProgress", "false")
    )
    return builder.getOrCreate()
