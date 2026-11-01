"""Point d'entrée d'une étape : python -m vishing_detection.pipeline.run_stage <étape>."""
from __future__ import annotations

import argparse

SPARK_STAGES = {"validate", "prepare", "features_handcrafted"}
PLAIN_STAGES = {"baselines", "embeddings", "train", "evaluate", "analyse_duree"}
STAGES = sorted(SPARK_STAGES | PLAIN_STAGES)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    stage = parser.parse_args(argv).stage

    if stage in SPARK_STAGES:
        from vishing_detection.spark import extract_features, prepare, validate
        from vishing_detection.spark.session import get_spark

        spark = get_spark()
        try:
            {"validate": validate, "prepare": prepare,
             "features_handcrafted": extract_features}[stage].run(spark)
        finally:
            spark.stop()
    else:
        from vishing_detection.models import analyse_duree, baselines, embeddings, evaluate, train

        {"baselines": baselines, "embeddings": embeddings,
         "train": train, "evaluate": evaluate,
         "analyse_duree": analyse_duree}[stage].run()


if __name__ == "__main__":
    main()
