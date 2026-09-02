# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn>=1.5", "sentence-transformers>=3.0", "numpy>=1.26"]
# ///
"""Embed a subset of the 20 Newsgroups corpus with all-MiniLM-L6-v2.

Run once with `uv run data-raw/newsgroups.py data-raw`, then
`Rscript data-raw/newsgroups.R` to reduce and package the result. The model
weights and the corpus are downloaded on first use.
"""
import csv
import re
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

CATEGORIES = [
    "rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics",
    "rec.autos", "sci.med", "misc.forsale", "soc.religion.christian",
]
PER_GROUP = 300

out = sys.argv[1]
ng = fetch_20newsgroups(
    subset="train", categories=CATEGORIES,
    remove=("headers", "footers", "quotes"), random_state=1,
)
rng = np.random.default_rng(1)
texts, labels = [], []
for c in CATEGORIES:
    idx = [i for i, (t, y) in enumerate(zip(ng.data, ng.target))
           if ng.target_names[y] == c and len(t.strip()) > 200]
    for i in rng.choice(idx, size=min(PER_GROUP, len(idx)), replace=False):
        texts.append(ng.data[i])
        labels.append(c)

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

np.savetxt(f"{out}/newsgroups_embedding.csv", emb, delimiter=",", fmt="%.6g")
with open(f"{out}/newsgroups_group.txt", "w") as f:
    f.write("\n".join(labels) + "\n")
with open(f"{out}/newsgroups_snippet.csv", "w", newline="") as f:
    w = csv.writer(f)
    for t in texts:
        w.writerow([re.sub(r"\s+", " ", t.strip())[:120]])
print(f"embedded {len(texts)} documents, shape {emb.shape}")
