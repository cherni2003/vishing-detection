import pandas as pd
f = pd.read_csv("data/raw/ftc_robocalls/metadata.csv")
f = f[f.language == "en"].drop_duplicates("file_name").dropna(subset=["transcript"])
for w in ["please", "thank", "urgent", "immediately", "must", "press"]:
    print(w, round(f.transcript.str.contains(rf"\b{w}\b", case=False).mean(), 3))