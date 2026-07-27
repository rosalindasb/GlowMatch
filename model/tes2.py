"""
Script bukti terminal untuk Tabel 4.9 (contoh rekomendasi + alasan)
pada subbab 4.7 Rekomendasi Produk.

PENTING: Script ini TIDAK menulis ulang logika rekomendasi. Fungsi
build_model() diimport dari recommender.py, sedangkan logika get_recs()
dan generate_reasons() direplikasi PERSIS sesuai rekomendasi.py (termasuk
whitelist SKINCARE_VOCAB), sehingga hasil yang ditampilkan konsisten
dengan sistem yang sesungguhnya berjalan di aplikasi web.

Cara pakai:
    python generate_bukti_tabel_4_9.py "Nama Produk" "Brand"

Contoh:
    python generate_bukti_tabel_4_9.py "Cica Dark Spot" "NPure"

Jalankan file ini di folder yang sama dengan recommender.py dan folder
data/, lalu screenshot output yang muncul.
"""

import sys
import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender import DATA_PATH, build_model

# Whitelist kosakata skincare (identik dengan SKINCARE_VOCAB di rekomendasi.py)
SKINCARE_VOCAB = {
    "niacinamide", "retinol", "vitamin", "hyaluronic", "hyaluronate",
    "ceramide", "ceramides", "salicylic", "glycolic", "lactic", "mandelic",
    "centella", "cica", "snail", "rice", "beras", "green", "tea", "licorice",
    "aloe", "vera", "collagen", "kolagen", "squalane", "panthenol",
    "peptide", "peptida", "antioxidant", "antioksidan", "spf", "sunscreen",
    "zinc", "sulfur", "charcoal", "arang", "kojic", "azelaic", "ferulic",
    "bakuchiol", "probiotic", "prebiotic", "postbiotic", "madecassoside",
    "allantoin", "glutathione", "arbutin", "mugwort", "propolis", "honey",
    "madu", "oat", "oatmeal", "avocado", "alpukat", "cucumber", "timun",
    "chamomile", "argan", "jojoba", "shea", "ginseng", "turmeric", "kunyit",
    "clay", "tanah", "liat", "spirulina", "algae", "rumput",
    "laut", "caffeine", "kafein", "tranexamic", "adenosine", "biome",
    "brightening", "mencerahkan", "cerah", "whitening", "hydrating",
    "melembapkan", "lembap", "moisturizing", "moisturizer", "soothing",
    "menenangkan", "calming", "exfoliating", "eksfoliasi", "cleansing",
    "membersihkan", "purifying", "balancing", "menyeimbangkan", "repairing",
    "memperbaiki", "repair", "protecting", "melindungi", "proteksi",
    "nourishing", "menutrisi", "nutrisi", "smoothing", "menghaluskan",
    "firming", "mengencangkan", "aging", "pore", "pori", "acne", "jerawat",
    "blemish", "noda", "spot", "flek", "kusam", "dull", "brighten",
    "oil", "minyak", "redness", "sensitive", "sensitif",
    "barrier", "gentle", "lembut", "refreshing", "menyegarkan", "glow",
    "glowing", "radiant", "bercahaya", "blackhead", "komedo", "whitehead",
    "pigmentation", "pigmentasi", "wrinkle", "keriput", "elastisitas",
    "elasticity", "pores", "scar", "bekas", "luka", "hydrate", "hydration",
    "exfoliate", "clarifying", "detox", "renewal", "regenerasi",
    "antiaging", "uv", "meratakan", "bright", "blemishes",
    "dark", "spots", "protect", "strengthen", "nourish",
    "irritated", "soothe", "calm", "restore",
    "menghidrasi", "menyamarkan", "wash", "exfo", "melembutkan",
}


def top_contributing_terms(tfidf_matrix, feature_names, idx_a, idx_b, top_k=3):
    vec_a = tfidf_matrix[idx_a].toarray().flatten()
    vec_b = tfidf_matrix[idx_b].toarray().flatten()
    contrib = vec_a * vec_b
    order = contrib.argsort()[::-1]
    terms = []
    for i in order:
        if contrib[i] <= 0:
            break
        term = feature_names[i]
        if term not in SKINCARE_VOCAB:
            continue
        terms.append(term)
        if len(terms) >= top_k:
            break
    return terms


def generate_reasons(df, tfidf_matrix, feature_names, idx_input, idx_rec, sim_score):
    input_row = df.loc[idx_input]
    rec_row = df.loc[idx_rec]
    reasons = [f"Kategori: {rec_row['category']}"]

    input_skins = set(x.strip().lower() for x in str(input_row["skin_type"]).split(","))
    rec_skins = set(x.strip().lower() for x in str(rec_row["skin_type"]).split(","))
    shared = input_skins & rec_skins
    if "all skin types" in rec_skins:
        reasons.append("Skin type: cocok untuk semua jenis kulit")
    elif shared:
        reasons.append("Skin type cocok: " + ", ".join(s.title() for s in sorted(shared)[:4]))

    if str(input_row["brand"]).strip().lower() == str(rec_row["brand"]).strip().lower():
        reasons.append(f"Brand sama: {rec_row['brand']}")

    terms = top_contributing_terms(tfidf_matrix, feature_names, idx_input, idx_rec)
    if terms:
        reasons.append("Kata kunci: " + ", ".join(terms))

    return reasons


def cetak_tabel_4_9(df, tfidf_matrix, feature_names, nama_produk: str, brand: str) -> None:
    print("\n" + "=" * 70)
    print(f"TABEL CONTOH REKOMENDASI: {nama_produk} ({brand})")
    print("=" * 70)

    match = df[
        (df["product_name"].str.lower() == nama_produk.lower())
        & (df["brand"].str.lower() == brand.lower())
    ]
    if match.empty:
        print(f"[ERROR] Produk '{nama_produk}' dari brand '{brand}' tidak ditemukan.")
        return

    idx = match.index[0]
    row_cat = df.loc[idx, "category"]
    print(f"Produk input : {df.loc[idx,'product_name']} ({df.loc[idx,'brand']})")
    print(f"Kategori     : {row_cat}")
    print(f"Skin type    : {df.loc[idx,'skin_type']}\n")

    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    same_cat = (df["category"] == row_cat).values.copy()
    same_cat[idx] = False
    cat_pos = np.where(same_cat)[0]
    order = np.argsort(-sims[cat_pos])
    top5 = cat_pos[order[:5]]

    for rank, p in enumerate(top5, start=1):
        rec = df.loc[p]
        reasons = generate_reasons(df, tfidf_matrix, feature_names, idx, p, sims[p])
        print(f"#{rank}  {rec['product_name']:<45} {rec['brand']:<18} sim={sims[p]*100:5.2f}%")
        print(f"      skin_type : {rec['skin_type']}")
        print(f"      alasan    : {'; '.join(reasons)}\n")


def main() -> None:
    if len(sys.argv) >= 3:
        nama_produk, brand = sys.argv[1], sys.argv[2]
    else:
        nama_produk, brand = "Cica Dark Spot", "NPure"

    df = pd.read_csv(DATA_PATH)
    with contextlib.redirect_stdout(io.StringIO()):
        vectorizer, tfidf_matrix = build_model(df)
    feature_names = list(vectorizer.get_feature_names_out())

    cetak_tabel_4_9(df, tfidf_matrix, feature_names, nama_produk, brand)


if __name__ == "__main__":
    main()