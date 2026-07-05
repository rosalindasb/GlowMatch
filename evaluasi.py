"""
Evaluasi sistem rekomendasi Glowmatch (Content-Based Filtering)

METRIK EVALUASI (Per-Product Evaluation, dalam kategori yang sama —
konsisten dengan hard filter kategori yang dipakai di recommender.py & app.py):
- Precision@5 : proporsi item relevan dalam Top-5 rekomendasi
- NDCG@5      : kualitas ranking berdasarkan posisi rekomendasi

DEFINISI RELEVANSI:
Sebuah produk dianggap relevan jika memiliki minimal 1 skin_type yang sama
dengan produk input. Produk dengan label "all skin types" juga dianggap relevan.

CATATAN METODOLOGI PENTING:
Rekomendasi SELALU dibatasi ke kategori yang sama dengan produk input — ini
SENGAJA dibuat konsisten dengan recommender.py (hard filter kategori aktif
by default) dan app.py (get_recs() juga filter kategori).

CATATAN: Recall@K sengaja TIDAK digunakan pada evaluasi ini. Recall@K
membutuhkan populasi "item relevan" sebagai pembagi, dan pada dataset ini
populasi tersebut bisa mencapai ratusan produk per kategori (banyak yang
berlabel "all skin types"), sementara sistem hanya menampilkan Top-5. Ini
membuat Recall@5 bernilai kecil secara struktural (bukan indikator kualitas
sistem), sehingga tidak dipakai sebagai metrik utama pada penelitian ini.
"""

import pickle
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR      = "model"
TFIDF_PATH     = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH    = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "products_df.pkl")

K = 5  # Top-K
MIN_CANDIDATES_PER_CATEGORY = 5  # sama seperti di recommender.py


# ==============================================================
# HELPER — print & formatting (gaya konsisten dengan file lain)
# ==============================================================
def section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"{title}")
    print(f"{'=' * 65}")


def subsection(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ==============================================================
# LOAD MODEL
# ==============================================================
section("LOAD MODEL")
print("[LOAD] Memuat model...")

with open(TFIDF_PATH, "rb") as f:
    vectorizer = pickle.load(f)
with open(MATRIX_PATH, "rb") as f:
    tfidf_matrix = pickle.load(f)
with open(DATAFRAME_PATH, "rb") as f:
    df = pickle.load(f)

print(f"       {len(df)} produk loaded")


# ==============================================================
# RELEVANCE FUNCTION — identik dengan recommender.py
# ==============================================================
def skin_type_overlap(input_skin: str, rec_skin: str) -> bool:
    input_set = set(x.strip().lower() for x in str(input_skin).split(","))
    rec_set   = set(x.strip().lower() for x in str(rec_skin).split(","))
    if "all skin types" in rec_set:
        return True
    return len(input_set & rec_set) > 0


# ==============================================================
# NDCG FUNCTION
# ==============================================================
def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances):
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(relevances) / ideal_dcg if ideal_dcg > 0 else 0.0


# ==============================================================
# EVALUATION — precision & ndcg, dalam kategori yang sama, similarity
# dihitung SEKALI untuk seluruh matrix (konsisten dengan optimasi di
# recommender.py)
# ==============================================================
section(f"EVALUASI PER-PRODUK — PRECISION / NDCG @{K}")
start = time.time()
total = len(df)

print(f"Menghitung cosine similarity matrix penuh ({total} x {total}), sekali saja...")
full_sim = cosine_similarity(tfidf_matrix)
print(f"Selesai. Similarity matrix shape: {full_sim.shape}")

categories = df["category"].values
skin_types = df["skin_type"].values

precision_list = []
ndcg_list      = []
cat_metrics    = {}  # category -> {"precision": [...], "ndcg": [...]}

skipped = 0
skipped_by_cat = {}

for pos in range(total):
    if (pos + 1) % 500 == 0 or (pos + 1) == total:
        print(f"       Progress: {pos + 1}/{total} produk diproses...")

    row_cat  = categories[pos]
    row_skin = skin_types[pos]
    sims     = full_sim[pos]

    # Kandidat: kategori sama, exclude diri sendiri — SAMA PERSIS dengan
    # hard filter kategori di recommender.py & app.py
    same_cat_mask = (categories == row_cat)
    same_cat_mask[pos] = False
    cat_positions = np.where(same_cat_mask)[0]

    if len(cat_positions) < MIN_CANDIDATES_PER_CATEGORY:
        skipped += 1
        skipped_by_cat[row_cat] = skipped_by_cat.get(row_cat, 0) + 1
        continue

    # Top-K dari kandidat SEKATEGORI (bukan dari seluruh dataset)
    order = np.argsort(-sims[cat_positions])
    topk_pos = cat_positions[order[:K]]

    # Relevance vector buat Top-K (dipakai Precision & NDCG)
    relevances = [
        1 if skin_type_overlap(row_skin, skin_types[p]) else 0
        for p in topk_pos
    ]

    # ---- PRECISION@K ----
    precision = sum(relevances) / K
    precision_list.append(precision)

    # ---- NDCG@K ----
    ndcg = ndcg_at_k(relevances)
    ndcg_list.append(ndcg)

    bucket = cat_metrics.setdefault(row_cat, {"precision": [], "ndcg": []})
    bucket["precision"].append(precision)
    bucket["ndcg"].append(ndcg)

elapsed = time.time() - start

# ==============================================================
# RESULT
# ==============================================================
subsection(f"Produk yang dilewati (kandidat sekategori < {MIN_CANDIDATES_PER_CATEGORY})")
print(f"Total produk dilewati : {skipped}  ({pct(skipped / total)})")
if skipped_by_cat:
    for cat, n in sorted(skipped_by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25}: {n} produk")

n_eval = len(precision_list)

section(f"HASIL EVALUASI PER-PRODUK @ {K}")
print(f"Produk dievaluasi : {n_eval} dari {total}  ({pct(n_eval / total)})")
print(f"Produk dilewati   : {skipped}")

print(f"\nPrecision@{K} : {pct(np.mean(precision_list))}")
print(f"NDCG@{K}      : {np.mean(ndcg_list):.4f}")

subsection(f"Distribusi metrik per kategori")
for cat, m in sorted(cat_metrics.items(), key=lambda x: -np.mean(x[1]["precision"])):
    n = len(m["precision"])
    print(
        f"  {cat:<25}: "
        f"Precision={np.mean(m['precision']):.4f}  "
        f"NDCG={np.mean(m['ndcg']):.4f}  "
        f"(n={n})"
    )

print(f"\nWaktu eksekusi evaluasi : {elapsed:.2f} detik")
print(f"\n[DONE] Evaluasi selesai.")