import pathlib
import zipfile

import pandas as pd

seg = pd.read_parquet("data/processed/external/segments.parquet")
print(f"segments.parquet : {len(seg)} segments, {seg.call_id.nunique()} appels")

out = pathlib.Path("vishing_code.zip")
out.unlink(missing_ok=True)

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in pathlib.Path("src").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
            z.write(p, p.as_posix())                      # chemins avec "/"
    for f in ("params.yaml", "pyproject.toml", "setup.py", "setup.cfg"):
        if pathlib.Path(f).exists():
            z.write(f, f)
    z.write("data/processed/external/segments.parquet", "segments.parquet")

names = zipfile.ZipFile(out).namelist()
print(len(names), "fichiers ;", names[:3])