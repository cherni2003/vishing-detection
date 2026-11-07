import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from vishing_detection.config import get_config, resolve

m = pd.read_parquet(resolve(get_config()["outputs"]["manifest"]))
d = json.load(open("data/raw/vishguard_dataset/vishguard_data.json", encoding="utf-8"))
j = pd.DataFrame([{"sample_id": x["id"], "text": x["script"], "type": x["type"]} for x in d])

t = m[["sample_id", "label", "split", "language", "duration_sec"]].merge(j, on="sample_id", how="left")
print("sans texte :", int(t.text.isna().sum()))
t = t.dropna(subset=["text"]).copy()

# contrôle : l'étiquette du manifest doit correspondre au type du JSON
print("étiquettes incohérentes :", int(((t.type == "fraudulent").astype(int) != t.label).sum()))
t["n_words"] = t.text.str.split().str.len()
t = t.drop(columns="type")
t.to_parquet("transcripts.parquet", index=False)

print(t.shape, t.split.value_counts().to_dict(), t.language.value_counts().to_dict())
for s in ("train", "validation", "test"):
    g = t[t.split == s]
    print(s, "AUC nb de mots :", round(roc_auc_score(g.label, g.n_words), 3),
          "| AUC durée :", round(roc_auc_score(g.label, g.duration_sec), 3))