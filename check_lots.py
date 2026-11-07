import os, glob
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

path = "transcripts.parquet" if os.path.exists("transcripts.parquet") else glob.glob("/kaggle/input/**/transcripts.parquet", recursive=True)[0]
t = pd.read_parquet(path)
t["lot"] = t.sample_id.str.split("_").str[0]

v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(t[t.split == "train"].text)
sim = (v.transform(t[t.split == "test"].text) @ v.transform(t[t.split == "train"].text).T).max(axis=1).toarray().ravel()
print("similarité test→train : médiane", round(float(np.median(sim)), 2), "| >0.9 :", round(float((sim > 0.9).mean()), 3))

aucs = []
for a, b in GroupKFold(5).split(t, groups=t.lot):
    vv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = LogisticRegression(max_iter=2000, C=10).fit(vv.fit_transform(t.text.iloc[a]), t.label.iloc[a])
    aucs.append(roc_auc_score(t.label.iloc[b], c.predict_proba(vv.transform(t.text.iloc[b]))[:, 1]))
print(t.lot.nunique(), "lots | AUC par lot exclu :", np.round(aucs, 3))


# 1. Quels mots séparent les classes ? (ajoutez à la fin de check_lots.py)
vv = TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True)
c = LogisticRegression(max_iter=2000, C=10).fit(vv.fit_transform(t.text), t.label)
w = pd.Series(c.coef_[0], index=vv.get_feature_names_out()).sort_values()
print("LÉGITIME :", list(w.index[:15]))
print("FRAUDE   :", list(w.index[-15:]))

# 2. AUC du TF-IDF uniquement sur les textes de test peu similaires au train
te = t[t.split == "test"].reset_index(drop=True)
vt = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
ct = LogisticRegression(max_iter=2000, C=10).fit(vt.fit_transform(t[t.split == "train"].text), t[t.split == "train"].label)
ok = sim < 0.5
print(int(ok.sum()), "textes de test avec similarité < 0,5 | AUC :",
      round(roc_auc_score(te.label[ok], ct.predict_proba(vt.transform(te.text[ok]))[:, 1]), 3))