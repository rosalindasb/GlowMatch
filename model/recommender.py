"""
Sistem Rekomendasi Skincare - Content-Based Filtering
menggunakan TF-IDF + Cosine Similarity.

Perubahan:
- [FIX] Hard filter category by default (rekomendasi selalu dalam kategori sama)
- [FIX] Demo produk diganti ke nama yang valid di dataset
- [FIX] Evaluasi sekarang ke seluruh produk (bukan sample 50)
- [TUNE] Category weight dinaikkan 6x di build_content_features.py
- [TUNE] ngram_range diubah (1,2) → (1,1) agar kata kunci nama produk
         lebih mudah match tanpa tergantung urutan bigram
- [PERF] evaluate_model() dioptimasi: cosine_similarity dihitung SEKALI untuk
         seluruh matrix (bukan 1866 + 1866 kali secara terpisah), lalu dipakai
         ulang untuk precision keseluruhan maupun breakdown per kategori.

Cara pakai:
    python recommender.py
    python recommender.py --product "Acne Patch Day" --top 5
    python recommender.py --product "Advanced Snail 92 All in One Cream" --skin-type oily,acne
    python recommender.py --product "Rice Toner" --no-category-filter --top 5
    python recommender.py --evaluate
    python recommender.py --suggest-demo
"""

from __future__ import annotations

import argparse
import pickle
import os
import time

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH      = "data/sociolla_skincare_featured.csv"
MODEL_DIR      = "model"
TFIDF_PATH     = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH    = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "products_df.pkl")

MIN_CANDIDATES_PER_CATEGORY = 5  # produk dengan kandidat sekategori < ini di-skip saat evaluasi


# ==============================================================
# HELPER — print & formatting
# ==============================================================
def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(part / total) * 100:.2f}%"


def section(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"{title}")
    print(f"{'=' * 65}")


def subsection(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"{title}")
    print(f"{'-' * 60}")


def skin_type_overlap(input_skins: set[str], rec_skin_str) -> bool:
    """True kalau ada irisan skin type, atau produk rekomendasi 'all skin types'."""
    rec_skins = set(str(rec_skin_str).lower().split(","))
    return len(input_skins & rec_skins) > 0 or "all skin types" in rec_skins


def validate_alignment(df: pd.DataFrame, tfidf_matrix) -> bool:
    """
    Validasi bahwa index dataframe SEJAJAR dengan baris tfidf_matrix
    (0..n-1 berurutan). Ini krusial karena kode mengakses tfidf_matrix[idx]
    memakai index dataframe secara langsung — kalau index tidak sejajar,
    similarity yang diambil bisa salah produk tanpa error apapun.
    """
    n = len(df)
    is_aligned = (
        tfidf_matrix.shape[0] == n
        and list(df.index) == list(range(n))
    )
    status = "OK — index dataframe sejajar dengan baris tfidf_matrix" if is_aligned \
        else "PERINGATAN — index TIDAK sejajar, similarity bisa salah produk!"
    print(f"[VALIDASI] Alignment df.index vs tfidf_matrix: {status}")
    return is_aligned


# ==============================================================
# BUILD & SAVE MODEL
# ==============================================================

def build_model(df: pd.DataFrame):
    """Fit TF-IDF pada kolom content, simpan model ke disk."""
    section("BUILD MODEL — FIT TF-IDF VECTORIZER")
    start = time.time()

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 1),     # [TUNE] unigram only — lebih baik untuk matching nama produk
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(df["content"])
    elapsed = time.time() - start

    n_docs, n_terms = tfidf_matrix.shape
    nnz = tfidf_matrix.nnz
    sparsity = 1 - (nnz / (n_docs * n_terms)) if n_docs * n_terms else 0

    print(f"Waktu fitting        : {elapsed:.2f} detik")
    print(f"Matrix shape         : {tfidf_matrix.shape}  ({n_docs} dokumen x {n_terms} term)")
    print(f"Jumlah vocabulary    : {len(vectorizer.vocabulary_)}")
    print(f"Non-zero entries     : {nnz}")
    print(f"Sparsity             : {sparsity*100:.2f}% dari matrix bernilai 0")

    subsection("Top 15 term dengan akumulasi bobot TF-IDF tertinggi (seluruh dokumen)")
    term_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    top_idx = term_scores.argsort()[::-1][:15]
    for i in top_idx:
        print(f"  {feature_names[i]:<20}: {term_scores[i]:.2f}")

    validate_alignment(df, tfidf_matrix)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TFIDF_PATH,     "wb") as f: pickle.dump(vectorizer,   f)
    with open(MATRIX_PATH,    "wb") as f: pickle.dump(tfidf_matrix, f)
    with open(DATAFRAME_PATH, "wb") as f: pickle.dump(df,           f)

    print(f"\nModel disimpan ke folder '{MODEL_DIR}/'")
    return vectorizer, tfidf_matrix


def load_model():
    """Load model dari disk."""
    with open(TFIDF_PATH,     "rb") as f: vectorizer   = pickle.load(f)
    with open(MATRIX_PATH,    "rb") as f: tfidf_matrix = pickle.load(f)
    with open(DATAFRAME_PATH, "rb") as f: df           = pickle.load(f)
    return vectorizer, tfidf_matrix, df


# ==============================================================
# REKOMENDASI
# ==============================================================

def get_recommendations(
    product_name: str,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 5,
    filter_skin_type: str | None = None,
    filter_category: bool = True,
    custom_category: str | None = None,
    exclude_same_brand: bool = False,
) -> pd.DataFrame:
    """
    Return top_n rekomendasi produk berdasarkan cosine similarity.

    Parameters
    ----------
    product_name      : nama produk input (case-insensitive, partial match)
    df                : dataframe produk
    tfidf_matrix      : TF-IDF matrix
    top_n             : jumlah rekomendasi
    filter_skin_type  : filter skin type (contoh: "oily" atau "oily,acne")
    filter_category   : kalau True, rekomendasi hanya dari kategori yang sama
    custom_category   : override kategori filter (opsional)
    exclude_same_brand: kalau True, exclude produk dari brand yang sama
    """
    # Cari produk input
    mask = df["product_name"].str.lower().str.contains(
        product_name.lower(), regex=False
    )
    matches = df[mask]

    if matches.empty:
        print(f"[ERROR] Produk '{product_name}' tidak ditemukan.")
        print("[INFO]  Coba cek ejaan atau gunakan nama parsial.")
        return pd.DataFrame()

    if len(matches) > 1:
        print(f"[INFO] Ditemukan {len(matches)} produk yang cocok, menggunakan yang pertama:")
        for _, r in matches.iterrows():
            print(f"       - {r['product_name']} ({r['brand']})")

    idx = matches.index[0]
    input_product = df.loc[idx]

    print(f"\n[INPUT] {input_product['product_name']} — {input_product['brand']}")
    print(f"        Kategori  : {input_product['category']}")
    print(f"        Skin type : {input_product['skin_type']}")
    print(f"        Harga     : Rp{input_product['price']:,.0f}")
    print(f"        Rating    : {input_product['rating']:.2f}")

    # Hitung cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    sim_df = df.copy()
    sim_df["similarity"] = cos_sim
    sim_df = sim_df[sim_df.index != idx]

    print(f"\n        [FUNNEL FILTER]")
    print(f"        Kandidat awal (semua produk lain)     : {len(sim_df)}")

    # Hard filter kategori — default aktif
    if filter_category:
        cat = custom_category if custom_category else input_product["category"]
        sim_df = sim_df[sim_df["category"] == cat]
        print(f"        Setelah filter kategori '{cat}'        : {len(sim_df)}")

    # Filter skin type opsional
    if filter_skin_type:
        skin_types = [s.strip().lower() for s in filter_skin_type.split(",")]
        def has_skin_type(st_str):
            st_list = [s.strip().lower() for s in str(st_str).split(",")]
            return any(s in st_list for s in skin_types) or "all skin types" in st_list
        before = len(sim_df)
        sim_df = sim_df[sim_df["skin_type"].apply(has_skin_type)]
        print(f"        Setelah filter skin type '{filter_skin_type}' : {len(sim_df)}  (dari {before})")

    if exclude_same_brand:
        before = len(sim_df)
        sim_df = sim_df[sim_df["brand"].str.lower() != input_product["brand"].lower()]
        print(f"        Setelah exclude brand sama             : {len(sim_df)}  (dari {before})")

    if sim_df.empty:
        print("[INFO] Tidak ada produk yang cocok dengan filter yang diberikan.")
        return pd.DataFrame()

    result = (
        sim_df.sort_values("similarity", ascending=False)
        .head(top_n)[["product_name", "brand", "category", "skin_type", "price", "rating", "similarity"]]
        .reset_index(drop=True)
    )
    result.index += 1

    print(f"        Similarity kandidat (min/mean/max)     : "
          f"{sim_df['similarity'].min():.4f} / {sim_df['similarity'].mean():.4f} / {sim_df['similarity'].max():.4f}")

    return result


# ==============================================================
# DISPLAY
# ==============================================================

def display_recommendations(result: pd.DataFrame) -> None:
    if result.empty:
        return

    print(f"\n{'='*65}")
    print(f"{'TOP REKOMENDASI':^65}")
    print(f"{'='*65}")

    for i, row in result.iterrows():
        print(f"\n#{i}  {row['product_name']}")
        print(f"    Brand      : {row['brand']}")
        print(f"    Kategori   : {row['category']}")
        print(f"    Skin type  : {row['skin_type']}")
        print(f"    Harga      : Rp{row['price']:,.0f}")
        print(f"    Rating     : {row['rating']:.2f}")
        print(f"    Similarity : {row['similarity']:.4f}")


# ==============================================================
# EVALUASI — seluruh produk, dioptimasi (1x hitung similarity matrix)
# ==============================================================

def evaluate_model(df: pd.DataFrame, tfidf_matrix) -> None:
    """
    Evaluasi Precision@5 terhadap SELURUH produk.

    Metrik:
    - Precision@5 kategori   : proporsi top-5 yang kategorinya sama
                               (SELALU 1.0 karena hard filter kategori aktif
                               sejak awal — ini sanity check, bukan metrik
                               evaluasi yang independen)
    - Precision@5 skin type  : proporsi top-5 yang skin type-nya overlap
                               (metrik utama yang bermakna)

    Optimasi: cosine_similarity dihitung SEKALI untuk seluruh matrix
    (n x n), lalu barisnya dipakai ulang untuk precision keseluruhan
    maupun breakdown per kategori — bukan dihitung berulang per produk.
    """
    section("EVALUASI MODEL — PRECISION@5 (SELURUH PRODUK)")
    start = time.time()
    total = len(df)

    if not validate_alignment(df, tfidf_matrix):
        print("[EVAL] Dibatalkan: index dataframe tidak sejajar dengan tfidf_matrix.")
        return

    print(f"Menghitung cosine similarity matrix penuh ({total} x {total}), sekali saja...")
    full_sim = cosine_similarity(tfidf_matrix)
    print(f"Selesai. Similarity matrix shape: {full_sim.shape}")

    categories = df["category"].values
    skin_types = df["skin_type"].values

    precision_cat: list[float] = []
    precision_skin: list[float] = []
    cat_precisions: dict[str, list[float]] = {}
    skipped = 0
    skipped_by_cat: dict[str, int] = {}

    for pos in range(total):
        if (pos + 1) % 500 == 0 or (pos + 1) == total:
            print(f"       Progress: {pos + 1}/{total} produk diproses...")

        row_cat = categories[pos]
        sims = full_sim[pos]

        same_cat_mask = (categories == row_cat)
        same_cat_mask[pos] = False
        cat_positions = np.where(same_cat_mask)[0]

        if len(cat_positions) < MIN_CANDIDATES_PER_CATEGORY:
            skipped += 1
            skipped_by_cat[row_cat] = skipped_by_cat.get(row_cat, 0) + 1
            continue

        order = np.argsort(-sims[cat_positions])
        top5_pos = cat_positions[order[:MIN_CANDIDATES_PER_CATEGORY]]

        # Precision@5 kategori — dijamin 1.0 karena cat_positions sudah difilter sekategori
        precision_cat.append(1.0)

        # Precision@5 skin type overlap
        input_skins = set(str(skin_types[pos]).lower().split(","))
        skin_match = sum(
            skin_type_overlap(input_skins, skin_types[p]) for p in top5_pos
        )
        prec_skin = skin_match / MIN_CANDIDATES_PER_CATEGORY
        precision_skin.append(prec_skin)
        cat_precisions.setdefault(row_cat, []).append(prec_skin)

    elapsed = time.time() - start

    subsection(f"Produk yang dilewati (kandidat sekategori < {MIN_CANDIDATES_PER_CATEGORY})")
    print(f"Total produk dilewati : {skipped}  ({pct(skipped, total)})")
    if skipped_by_cat:
        for cat, n in sorted(skipped_by_cat.items(), key=lambda x: -x[1]):
            print(f"  {cat:<25}: {n} produk")

    n_eval = len(precision_skin)
    avg_cat  = float(np.mean(precision_cat))  if precision_cat  else float("nan")
    avg_skin = float(np.mean(precision_skin)) if precision_skin else float("nan")
    std_skin = float(np.std(precision_skin))  if precision_skin else float("nan")

    section("HASIL EVALUASI")
    print(f"Total produk dievaluasi              : {n_eval} dari {total}  ({pct(n_eval, total)})")
    print(f"\nPrecision@5 kategori (rata-rata)     : {avg_cat:.4f} ({avg_cat*100:.2f}%)")
    print("  Catatan: metrik ini otomatis 1.0 (100%) karena rekomendasi sudah")
    print("  di-hard-filter ke kategori yang sama sejak awal. Angka ini hanya")
    print("  sanity check bahwa filter kategori bekerja, BUKAN metrik evaluasi")
    print("  yang independen — metrik utama yang bermakna adalah skin type.")
    print(f"\nPrecision@5 skin type (rata-rata)    : {avg_skin:.4f} ({avg_skin*100:.2f}%)")
    print(f"Precision@5 skin type (std deviasi)  : {std_skin:.4f}")
    if precision_skin:
        print(f"Precision@5 skin type (min / max)    : {min(precision_skin):.4f} / {max(precision_skin):.4f}")

    subsection("Distribusi Precision@5 skin type per kategori")
    for cat, precs in sorted(cat_precisions.items(), key=lambda x: -np.mean(x[1])):
        arr = np.array(precs)
        print(f"  {cat:<25}: mean={arr.mean():.4f} ({arr.mean()*100:.2f}%)  "
              f"n={len(arr):<5} std={arr.std():.4f}")

    print(f"\nWaktu eksekusi evaluasi : {elapsed:.2f} detik")


# ==============================================================
# CHARACTERIZE DEMO SCENARIOS — bantu pilih contoh, TANPA cherry-pick
# ==============================================================

def characterize_demo_scenarios(
    df: pd.DataFrame,
    tfidf_matrix,
    top_n_show: int = 5,
    min_candidates: int = 15,
) -> None:
    """
    PENTING: fungsi ini TIDAK meranking produk berdasarkan seberapa "bagus"
    similarity-nya. Meranking berdasarkan similarity tinggi & seragam adalah
    bentuk cherry-picking (pilih hasil yang paling menguntungkan narasi),
    yang riskan dipertanyakan saat sidang skripsi.

    Alih-alih, fungsi ini mengelompokkan produk ke beberapa skenario NATURAL
    yang muncul apa adanya dari data, supaya kamu bisa pilih contoh yang
    representatif untuk storytelling di Bab 4 — bukan yang paling "mulus":

    A. Skin type spesifik + kategori populasinya besar (Moisturizer/Toner/dst)
       → skenario "khas", paling representatif buat contoh utama
    B. Similarity tertinggi secara alami (kemungkinan near-duplicate/varian
       lini produk yang sama) → menarik untuk dijelaskan sebagai temuan,
       BUKAN bukti akurasi sistem
    C. Precision@5 skin type TIDAK sempurna (< 100%) → contoh realistis untuk
       bahan diskusi limitasi di Bab Pembahasan, supaya laporan tidak
       terkesan "semua kasus mulus"

    Urutan dalam tiap kelompok BUKAN ranking kualitas — cuma diambil sample
    untuk ditinjau manual.
    """
    section("KARAKTERISASI SKENARIO DEMO (BUKAN RANKING 'PALING BAGUS')")
    print("Catatan: tool ini membantu MELIHAT skenario yang ada di data secara")
    print("apa adanya. Pemilihan contoh akhir tetap keputusan kamu — dan kalau")
    print("ditanya penguji, kamu bisa jelaskan dasar pemilihannya secara jujur")
    print("(representatif, bukan yang paling tinggi similarity-nya).\n")

    total = len(df)
    full_sim = cosine_similarity(tfidf_matrix)
    categories = df["category"].values
    skin_types = df["skin_type"].values
    brands = df["brand"].values
    names = df["product_name"].values

    rows = []
    for pos in range(total):
        row_skin_raw = str(skin_types[pos]).strip().lower()
        row_cat = categories[pos]

        same_cat_mask = (categories == row_cat)
        same_cat_mask[pos] = False
        cat_positions = np.where(same_cat_mask)[0]
        if len(cat_positions) < min_candidates:
            continue

        sims = full_sim[pos]
        order = np.argsort(-sims[cat_positions])
        top5_pos = cat_positions[order[:5]]
        if len(top5_pos) < 5:
            continue

        top5_sims = sims[top5_pos]
        input_skins = set(row_skin_raw.split(","))
        top5_skin_match = sum(skin_type_overlap(input_skins, skin_types[p]) for p in top5_pos)

        rows.append({
            "product_name": names[pos],
            "brand": brands[pos],
            "category": row_cat,
            "skin_type": skin_types[pos],
            "is_all_skin_types": row_skin_raw == "all skin types",
            "n_kandidat_kategori": len(cat_positions),
            "max_sim": float(top5_sims.max()),
            "avg_sim": float(top5_sims.mean()),
            "skin_precision5": top5_skin_match / 5,
        })

    if not rows:
        print("[INFO] Tidak ada produk yang memenuhi syarat minimum kandidat kategori.")
        return

    result_df = pd.DataFrame(rows)

    # ---- Skenario A: skin type spesifik + kategori populasinya besar ----
    subsection("A. Skenario 'khas' — skin type spesifik, kategori populasinya besar")
    print("(diurutkan berdasarkan jumlah kandidat sekategori, BUKAN similarity)")
    cat_pop = result_df["category"].value_counts()
    top_cats = cat_pop.head(3).index.tolist()
    scenario_a = result_df[
        (~result_df["is_all_skin_types"]) & (result_df["category"].isin(top_cats))
    ].sample(min(top_n_show, len(result_df)), random_state=42)
    for _, r in scenario_a.iterrows():
        print(f"  - {r['product_name']} ({r['brand']}) | {r['category']} | skin_type={r['skin_type']} "
              f"| kandidat_kategori={r['n_kandidat_kategori']} | skin_precision5={r['skin_precision5']:.2f}")

    # ---- Skenario B: similarity tertinggi secara alami ----
    subsection("B. Similarity tertinggi secara alami (kemungkinan near-duplicate/varian produk)")
    print("(untuk dijelaskan sebagai TEMUAN, bukan diklaim sebagai bukti akurasi)")
    scenario_b = result_df.nlargest(top_n_show, "max_sim")
    for _, r in scenario_b.iterrows():
        print(f"  - {r['product_name']} ({r['brand']}) | {r['category']} | max_sim={r['max_sim']:.4f}")

    # ---- Skenario C: precision skin type tidak sempurna ----
    subsection("C. Precision@5 skin type TIDAK sempurna — bahan diskusi limitasi")
    imperfect = result_df[(result_df["skin_precision5"] < 1.0) & (result_df["skin_precision5"] > 0)]
    if imperfect.empty:
        print("  Tidak ditemukan kasus dengan precision sebagian pada sample ini.")
    else:
        scenario_c = imperfect.sample(min(top_n_show, len(imperfect)), random_state=42)
        for _, r in scenario_c.iterrows():
            print(f"  - {r['product_name']} ({r['brand']}) | {r['category']} | skin_type={r['skin_type']} "
                  f"| skin_precision5={r['skin_precision5']:.2f}")

    print(f"\n[INGAT] Metrik akurasi utama tetap Precision@5 agregat (skin type & kategori")
    print(f"        dari evaluate_model), BUKAN similarity dari satu contoh demo manapun.")
    print(f"        Contoh demo di atas cuma untuk ilustrasi/interpretability di Bab 4.")


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Skincare Recommender System")
    parser.add_argument("--product",            type=str,  default=None,  help="Nama produk input")
    parser.add_argument("--top",                type=int,  default=5,     help="Jumlah rekomendasi (default: 5)")
    parser.add_argument("--skin-type",          type=str,  default=None,  help="Filter skin type (contoh: oily,acne)")
    parser.add_argument("--no-category-filter", action="store_true",      help="Matikan hard filter kategori")
    parser.add_argument("--rebuild",            action="store_true",      help="Rebuild model dari awal")
    parser.add_argument("--evaluate",           action="store_true",      help="Jalankan evaluasi seluruh produk")
    parser.add_argument("--suggest-demo",       action="store_true",      help="Tampilkan skenario demo (bukan ranking 'terbaik') untuk bantu pilih contoh laporan")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    print(f"[LOAD] {len(df)} produk dari {DATA_PATH}")

    model_exists = all(os.path.exists(p) for p in [TFIDF_PATH, MATRIX_PATH, DATAFRAME_PATH])

    if args.rebuild or not model_exists:
        vectorizer, tfidf_matrix = build_model(df)
    else:
        print("[LOAD] Memuat model dari disk...")
        vectorizer, tfidf_matrix, df = load_model()
        print(f"       Matrix shape: {tfidf_matrix.shape}")
        validate_alignment(df, tfidf_matrix)

    if args.evaluate:
        evaluate_model(df, tfidf_matrix)

    if args.suggest_demo:
        characterize_demo_scenarios(df, tfidf_matrix)

    if args.product:
        result = get_recommendations(
            product_name     = args.product,
            df               = df,
            tfidf_matrix     = tfidf_matrix,
            top_n            = args.top,
            filter_skin_type = args.skin_type,
            filter_category  = not args.no_category_filter,
        )
        display_recommendations(result)
    else:
        demo_products = [
            "Acne Patch Day",
            "Advanced Snail 92 All in One Cream",
            "Rice Toner",
        ]
        print("\n[DEMO] Menjalankan demo rekomendasi...")
        print("[DEMO] Setiap produk otomatis difilter dengan skin_type-nya sendiri,")
        print("[DEMO] supaya rekomendasi konsisten relevan dari sisi kategori maupun skin type.\n")
        for prod in demo_products:
            mask = df["product_name"].str.lower().str.contains(prod.lower(), regex=False)
            matches = df[mask]
            demo_skin_filter = None
            if not matches.empty:
                input_skin = str(matches.iloc[0]["skin_type"]).strip().lower()
                if input_skin and input_skin != "all skin types":
                    demo_skin_filter = input_skin

            result = get_recommendations(
                product_name     = prod,
                df               = df,
                tfidf_matrix     = tfidf_matrix,
                top_n            = 5,
                filter_category  = True,
                filter_skin_type = demo_skin_filter,
            )
            display_recommendations(result)
            print()


if __name__ == "__main__":
    main()