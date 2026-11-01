import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feat = pd.read_parquet("data/features/handcrafted.parquet")
seg = pd.read_parquet("data/processed/segments.parquet")[["segment_id", "label", "split"]]
df = feat.merge(seg, on="segment_id")
cols = [c for c in feat.columns if c not in ("segment_id", "sample_id")]
df[cols] = df[cols].fillna(0)

tr, te = df[df.split == "train"], df[df.split == "test"]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
model.fit(tr[cols], tr["label"])
te = te.assign(p=model.predict_proba(te[cols])[:, 1])

call = te.groupby("sample_id").agg(p=("p", "mean"), label=("label", "first"))
res = {
    "segment_accuracy": accuracy_score(te.label, te.p > 0.5),
    "call_accuracy": accuracy_score(call.label, call.p > 0.5),
    "call_f1": f1_score(call.label, call.p > 0.5),
    "call_roc_auc": roc_auc_score(call.label, call.p),
    "baseline_duree_seule": 0.956,
}
print(json.dumps(res, indent=2))
json.dump(res, open("reports/mfcc_baseline.json", "w"), indent=2)