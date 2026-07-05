"""
Eksplorasi kandidat produk demo untuk skripsi — SECARA TRANSPARAN.

PENTING (baca dulu sebelum pakai):
Script ini SENGAJA TIDAK mengurutkan produk berdasarkan similarity
tertinggi. Kalau kriteria pemilihan demo adalah "similarity setinggi
mungkin", itu cherry-picking dan bisa jadi celah pertanyaan pas sidang
("kok hasilnya mulus semua, ini dipilih gimana?").

Yang dilakukan script ini cuma MENYARING produk yang layak dijadikan
demo secara metodologis (bukan yang paling "bagus" angkanya):
  1. Skip produk dengan skin_type input == "all skin types"
     → supaya filter skin type di funnel kerasa jalan, bukan no-op.
  2. Skip kategori dengan kandidat sekategori < MIN_CATEGORY_SIZE
     → supaya top-5 emang hasil "milih dari banyak", bukan menang WO.
  3. Hitung brand diversity di top-5 (berapa brand unik dari 5 rekomendasi)
     → cuma info tambahan, BUKAN dipakai buat ranking/filter.

Output: tabel kandidat per kategori (1 wakil per kategori), dengan
angka apa adanya (precision skin type, similarity range, brand
diversity). Kamu yang menilai mana yang paling representatif untuk
diceritakan di Bab 4 — bukan script yang milih "pemenang".

Cara pakai:
    python suggest_demo.py
    python suggest_demo.py --min-category-size 20
    python suggest_demo.py --per-category 3
"""

from __future__ import annotations

import argparse
import pickle
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR      = "model"
TFIDF_PATH     = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH    = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "products_df.pkl")

TOP_N = 5


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")


def subsection(title: str) -> None:
    print(f"\n{'-' * 65}")
    print(f"{title}")
    print(f"{'-' * 65}")


def skin_type_overlap(input_skins: set[str], rec_skin_str) -> bool:
    rec_skins = set(str(rec_skin_str).lower().split(","))
    return len(input_skins & rec_skins) > 0 or "all skin types" in rec_skins


def load_model():
    with open(TFIDF_PATH,     "rb") as f: vectorizer   = pickle.load(f)
    with open(MATRIX_PATH,    "rb") as f: tfidf_matrix = pickle.load(f)
    with open(DATAFRAME_PATH, "rb") as f: df           = pickle.load(f)
    return vectorizer, tfidf_matrix, df


def evaluate_single_product(pos: int, df: pd.DataFrame, full_sim: np.ndarray) -> dict | None:
    """
    Hitung metrik apa adanya untuk satu produk sebagai kandidat demo.
    Return None kalau produk ini tidak memenuhi syarat kelayakan dasar
    (skin type generik / kandidat sekategori kurang).
    """
    row = df.iloc[pos]
    input_skin_raw = str(row["skin_type"]).strip().lower()

    # syarat 1: skin type input harus spesifik, bukan "all skin types"
    if input_skin_raw == "all skin types":
        return None

    category = row["category"]
    same_cat_mask = (df["category"].values == category)
    same_cat_mask[pos] = False
    cat_positions = np.where(same_cat_mask)[0]

    return {
        "pos": pos,
        "category": category,
        "n_candidates_in_category": len(cat_positions),
        "row": row,
        "cat_positions": cat_positions,
    }


def build_demo_candidate(info: dict, df: pd.DataFrame, full_sim: np.ndarray) -> dict:
    pos = info["pos"]
    row = info["row"]
    cat_positions = info["cat_positions"]
    sims = full_sim[pos]

    order = np.argsort(-sims[cat_positions])
    top_pos = cat_positions[order[:TOP_N]]

    input_skins = set(str(row["skin_type"]).lower().split(","))
    skin_hits = [skin_type_overlap(input_skins, df.iloc[p]["skin_type"]) for p in top_pos]
    precision_skin = sum(skin_hits) / TOP_N

    top_brands = [df.iloc[p]["brand"] for p in top_pos]
    brand_diversity = len(set(b.lower() for b in top_brands))

    top_sims = sims[top_pos]

    return {
        "product_name": row["product_name"],
        "brand": row["brand"],
        "category": row["category"],
        "skin_type": row["skin_type"],
        "n_candidates_in_category": info["n_candidates_in_category"],
        "precision_skin_at_5": precision_skin,
        "brand_diversity_top5": brand_diversity,
        "sim_min": top_sims.min(),
        "sim_max": top_sims.max(),
        "sim_mean": top_sims.mean(),
        "top5_names": [df.iloc[p]["product_name"] for p in top_pos],
        "top5_brands": top_brands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksplorasi kandidat demo produk (transparan, bukan cherry-pick)")
    parser.add_argument("--min-category-size", type=int, default=15,
                         help="Minimal jumlah kandidat sekategori agar produk layak jadi demo (default: 15)")
    parser.add_argument("--per-category", type=int, default=2,
                         help="Berapa kandidat ditampilkan per kategori (default: 2, dipilih acak/random_state tetap)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed untuk sampling kandidat per kategori")
    args = parser.parse_args()

    vectorizer, tfidf_matrix, df = load_model()
    total = len(df)
    print(f"[LOAD] {total} produk dimuat dari model tersimpan")

    section("STEP 1 — HITUNG COSINE SIMILARITY MATRIX (SEKALI, DIPAKAI ULANG)")
    full_sim = cosine_similarity(tfidf_matrix)
    print(f"Matrix similarity shape: {full_sim.shape}")

    section("STEP 2 — SARING PRODUK YANG LAYAK JADI KANDIDAT DEMO")
    print("Syarat kelayakan (BUKAN berdasarkan similarity tinggi):")
    print(f"  1. skin_type input spesifik (bukan 'all skin types')")
    print(f"  2. kandidat sekategori >= {args.min_category_size} produk")

    eligible = []
    skipped_generic_skin = 0
    skipped_small_category = 0

    for pos in range(total):
        info = evaluate_single_product(pos, df, full_sim)
        if info is None:
            skipped_generic_skin += 1
            continue
        if info["n_candidates_in_category"] < args.min_category_size:
            skipped_small_category += 1
            continue
        eligible.append(info)

    print(f"\nTotal produk                                  : {total}")
    print(f"Di-skip (skin_type = 'all skin types')        : {skipped_generic_skin}")
    print(f"Di-skip (kandidat sekategori < {args.min_category_size})            : {skipped_small_category}")
    print(f"Sisa produk layak jadi kandidat demo          : {len(eligible)}")

    section("STEP 3 — SAMPLE KANDIDAT PER KATEGORI (RANDOM, BUKAN DIURUTKAN SIMILARITY)")
    print("Catatan: pengambilan sampel di bawah ini RANDOM per kategori (pakai seed tetap")
    print("supaya reproducible), BUKAN diurutkan dari similarity tertinggi ke terendah.")
    print("Tujuannya cuma kasih kamu beberapa opsi wajar per kategori untuk dinilai manual.\n")

    eligible_df = pd.DataFrame(eligible)
    all_candidates = []

    for cat, group in eligible_df.groupby("category"):
        n_sample = min(args.per_category, len(group))
        sampled = group.sample(n_sample, random_state=args.seed)
        subsection(f"Kategori: {cat}  ({len(group)} produk layak, menampilkan {n_sample})")
        for _, info_row in sampled.iterrows():
            candidate = build_demo_candidate(info_row.to_dict(), df, full_sim)
            all_candidates.append(candidate)

            print(f"\n  Produk        : {candidate['product_name']} ({candidate['brand']})")
            print(f"  Skin type     : {candidate['skin_type']}")
            print(f"  Kandidat sekategori     : {candidate['n_candidates_in_category']}")
            print(f"  Precision@5 skin type   : {candidate['precision_skin_at_5']:.2f}  "
                  f"({int(candidate['precision_skin_at_5']*TOP_N)}/{TOP_N} match)")
            print(f"  Brand diversity top-5   : {candidate['brand_diversity_top5']}/{TOP_N} brand berbeda")
            print(f"  Similarity (min/mean/max): {candidate['sim_min']:.4f} / "
                  f"{candidate['sim_mean']:.4f} / {candidate['sim_max']:.4f}")
            print(f"  Top-5:")
            for name, brand in zip(candidate["top5_names"], candidate["top5_brands"]):
                print(f"      - {name} ({brand})")

    section("RINGKASAN — SILAKAN PILIH MANUAL SESUAI CERITA YANG MAU DIBANGUN")
    print("Saran cara milih (bukan berdasarkan angka tertinggi):")
    print("  - 1 contoh dengan brand_diversity tinggi (4-5/5)")
    print("    → nunjukkin sistem nangkep kemiripan KONSEP, bukan cuma sesama brand.")
    print("  - 1 contoh dengan precision skin type < 1.0 (misal 0.6 atau 0.8)")
    print("    → jujur nunjukkin batas sistem, enak buat bahasan 'keterbatasan' di Bab 4/5.")
    print("  - 1 contoh dari kategori dengan kandidat banyak (misal Moisturizer/Toner)")
    print("    → nunjukkin sistem tetap presisi walau pilihannya banyak.")
    print("\nJangan pilih berdasarkan similarity tertinggi doang — itu bukan indikator")
    print("kualitas rekomendasi, cuma indikator seberapa mirip TEKS dua produk.")

    print(f"\nTotal kandidat ditampilkan: {len(all_candidates)}")


if __name__ == "__main__":
    main()