"""Point d'entrée d'une étape : python -m vishing_detection.pipeline.run_stage <étape>."""
from __future__ import annotations

import argparse
from importlib import import_module

SPARK_STAGES = {
    "validate": "vishing_detection.spark.validate",
    "prepare": "vishing_detection.spark.prepare",
    "features_handcrafted": "vishing_detection.spark.extract_features",
    "ingest_robocalls": "vishing_detection.spark.ingest_robocalls",
}
PLAIN_STAGES = {
    "baselines": "vishing_detection.models.baselines",
    "embeddings": "vishing_detection.models.embeddings",
    "train": "vishing_detection.models.train",
    "evaluate": "vishing_detection.models.evaluate",
    "analyse_duree": "vishing_detection.models.analyse_duree",
    "embeddings_external": "vishing_detection.models.embeddings_external",
    "external_eval": "vishing_detection.models.external_eval",
    "bootstrap_duree": "vishing_detection.models.bootstrap_duree",
    "external_campaigns": "vishing_detection.models.external_campaigns",
    "embeddings_channel": "vishing_detection.models.embeddings_channel",
    "channel_eval": "vishing_detection.models.channel_eval",
    
}
STAGES = sorted(SPARK_STAGES | PLAIN_STAGES)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    stage = parser.parse_args(argv).stage

    if stage in SPARK_STAGES:
        from vishing_detection.spark.session import get_spark

        module = import_module(SPARK_STAGES[stage])
        spark = get_spark()
        try:
            module.run(spark)
        finally:
            spark.stop()
    else:
        import_module(PLAIN_STAGES[stage]).run()


if __name__ == "__main__":
    main()