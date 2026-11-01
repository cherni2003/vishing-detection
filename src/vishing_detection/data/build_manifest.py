"""python -m vishing_detection.data.build_manifest vishguard --root data/raw/vishguard"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vishing_detection.config import get_config, get_params, resolve
from vishing_detection.data.ingestion import ingest_vishguard
from vishing_detection.data.schema import validate_manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=["vishguard"])
    ap.add_argument("--root", default="data/raw/vishguard")
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)

    cfg, params = get_config(), get_params()
    root = resolve(args.root)
    found = sorted(root.rglob("annotations_with_audio.json"))
    if not found:
        raise SystemExit(f"annotations_with_audio.json introuvable sous {root}")
    prefix = Path(args.root, found[0].parent.relative_to(root)).as_posix().rstrip("/") + "/"

    df = ingest_vishguard(found[0], prefix, seed=params["seed"],
                          test_size=cfg["training"]["test_size"],
                          validation_size=cfg["training"]["validation_size"],
                          languages=args.languages)
    errors = validate_manifest(df)
    missing = [p for p in df["audio_path"] if not resolve(p).exists()]
    if missing:
        errors.append(f"{len(missing)} fichiers audio introuvables (ex. {missing[0]})")
    if errors:
        raise SystemExit("manifest invalide : " + " ; ".join(errors))

    out = resolve(cfg["data"]["manifest_path"])
    if args.append and out.exists() and out.stat().st_size > 0:
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} appels -> {out}")
    print(df.groupby(["split", "language", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()