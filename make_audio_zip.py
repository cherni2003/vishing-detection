import pathlib
import re
import zipfile

src = pathlib.Path("data/raw/ftc_robocalls/audio-wav-16khz")
out = pathlib.Path("robocall_audio.zip")
out.unlink(missing_ok=True)

seen, n_renamed = {}, 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:     # wav : pas de gain à compresser
    for p in sorted(src.glob("*.wav")):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", p.name)
        if safe in seen:
            raise SystemExit(f"Collision de noms : {p.name} et {seen[safe]} -> {safe}")
        seen[safe] = p.name
        n_renamed += safe != p.name
        z.write(p, f"audio-wav-16khz/{safe}")

print(f"{len(seen)} fichiers, {n_renamed} renommés")