"""
Gabungkan fitur teks untuk TF-IDF content-based filtering.

Strategi bobot (pengulangan teks):
- category    : 6x (fitur paling diskriminatif)
- skin_type   : 5x
- product_name: 4x (kata kunci nama produk berpengaruh)
- brand       : 1x (info tambahan)
- description : 1x (konteks lengkap)

Output: data/sociolla_skincare_featured.csv

Catatan tambahan (versi laporan):
- Sebelum build_content dijalankan, dilakukan pengecekan missing value pada
  kolom-kolom sumber (category, skin_type, brand, description) agar tidak
  ada komponen content yang kosong tanpa disadari.
- Setelah content dibuat, dilakukan verifikasi bobot (memastikan tiap
  komponen benar-benar terulang sesuai jumlah yang diklaim), pengecekan
  konten duplikat antar produk berbeda, dan estimasi ukuran vocabulary
  sebagai pembanding kasar terhadap vocabulary TF-IDF yang sebenarnya.
"""

from __future__ import annotations

import re
import time
import pandas as pd


INPUT_PATH  = "data/sociolla_skincare_labeled.csv"
OUTPUT_PATH = "data/sociolla_skincare_featured.csv"

# bobot pengulangan per kolom, dipakai juga untuk verifikasi di laporan
WEIGHTS = {
    "category": 6,
    "skin_type": 5,
    "product_name": 4,
    "brand": 1,
    "description": 1,
}


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


def show_examples(items, label: str, max_items: int = 5) -> None:
    items = list(items)
    if not items:
        return
    print(f"  Contoh {label} (menampilkan {min(len(items), max_items)} dari {len(items)}):")
    for item in items[:max_items]:
        print(f"    - {item}")
    if len(items) > max_items:
        print(f"    ... dan {len(items) - max_items} lainnya")


# ==============================================================
# CORE LOGIC
# ==============================================================
def clean_text(text: str) -> str:
    """Lowercase, buang karakter non-alfanumerik kecuali spasi, rapikan whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_content(row: pd.Series) -> str:
    """
    Gabungkan fitur teks dengan bobot via pengulangan.
    skin_type dipisah koma → dijadikan spasi biar TF-IDF bisa baca tiap token.
    """
    category     = clean_text(row["category"])
    skin_type    = clean_text(str(row["skin_type"]).replace(",", " "))
    product_name = clean_text(row["product_name"])
    brand        = clean_text(row["brand"])
    description  = clean_text(row["description"] if pd.notna(row["description"]) else "")

    parts = (
        f"{category} " * WEIGHTS["category"] +
        f"{skin_type} " * WEIGHTS["skin_type"] +
        f"{product_name} " * WEIGHTS["product_name"] +
        f"{brand} " * WEIGHTS["brand"] +
        f"{description} " * WEIGHTS["description"]
    )

    return re.sub(r"\s+", " ", parts).strip()


# ==============================================================
# MAIN
# ==============================================================
def main() -> None:
    import os

    start_time = time.time()

    df = pd.read_csv(INPUT_PATH)
    section("STEP 1 — LOAD DATA")
    print(f"[LOAD] {len(df)} produk dari '{INPUT_PATH}'")
    print(f"Strategi bobot content : category {WEIGHTS['category']}x, "
          f"skin_type {WEIGHTS['skin_type']}x, product_name {WEIGHTS['product_name']}x, "
          f"brand {WEIGHTS['brand']}x, description {WEIGHTS['description']}x")

    # ---- validasi kolom sumber sebelum build ----
    section("STEP 2 — VALIDASI KOLOM SUMBER SEBELUM BUILD CONTENT")
    total = len(df)
    source_cols = ["category", "skin_type", "product_name", "brand", "description"]
    for col in source_cols:
        if col not in df.columns:
            print(f"  [WARNING] Kolom '{col}' tidak ditemukan di dataset!")
            continue
        n_missing = df[col].isna().sum()
        n_empty_str = (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col:<14}: missing={n_missing:>4} ({pct(n_missing, total)})  "
              f"empty-string={n_empty_str:>4} ({pct(n_empty_str, total)})")

    # ---- build content ----
    section("STEP 3 — BUILD KOLOM CONTENT")
    df["content"] = df.apply(build_content, axis=1)

    empty_content = (df["content"].str.strip() == "").sum()
    print(f"Kolom 'content' berhasil dibuat untuk {total} produk")
    print(f"Content kosong total       : {empty_content}  ({pct(empty_content, total)})")
    print(f"Panjang rata-rata (char)   : {df['content'].str.len().mean():.0f}")
    print(f"Panjang median (char)      : {df['content'].str.len().median():.0f}")
    print(f"Panjang min / max (char)   : {df['content'].str.len().min()} / {df['content'].str.len().max()}")

    # ---- contoh before/after clean_text ----
    subsection("Contoh clean_text (before → after) — pembuktian pembersihan karakter spesial")
    raw_names = df["product_name"].astype(str)
    cleaned_names = raw_names.apply(clean_text)
    changed_mask = raw_names != cleaned_names
    examples = [f"'{b}'  →  '{a}'" for b, a in zip(raw_names[changed_mask], cleaned_names[changed_mask])]
    show_examples(examples, "product_name sebelum → sesudah clean_text", max_items=5)

    # ---- verifikasi bobot ----
    section("STEP 4 — VERIFIKASI PEMBOBOTAN CONTENT")
    print("Memverifikasi apakah tiap komponen benar-benar terulang sesuai bobot yang diklaim,")
    print("dengan menghitung jumlah kemunculan teks komponen di dalam kolom 'content'.")
    print("Catatan: hitungan bisa LEBIH BESAR dari target (bukan berarti salah) kalau kata")
    print("yang sama kebetulan juga muncul di komponen lain, misal kategori 'toner' ikut")
    print("kehitung karena muncul juga di dalam product_name.\n")

    sample_check = df.sample(min(3, total), random_state=42)
    for _, row in sample_check.iterrows():
        content = row["content"]
        print(f"  Produk: {str(row['product_name'])[:55]}")
        for col in ["category", "skin_type", "product_name", "brand"]:
            raw_val = str(row[col]).replace(",", " ") if col == "skin_type" else row[col]
            cleaned_val = clean_text(raw_val)
            if not cleaned_val:
                print(f"    {col:<14}: (kosong, dilewati)")
                continue
            actual_count = content.count(cleaned_val)
            expected = WEIGHTS[col]
            status = "OK" if actual_count >= expected else "CEK ULANG"
            print(f"    {col:<14}: muncul {actual_count}x di content (target {expected}x)  [{status}]")
        print()

    # ---- cek konten duplikat antar produk berbeda ----
    section("STEP 5 — CEK KONTEN DUPLIKAT (RISIKO COSINE SIMILARITY = 1.0)")
    dup_content_groups = df[df.duplicated(subset=["content"], keep=False)].groupby("content")
    n_dup_groups = dup_content_groups.ngroups
    n_dup_products = sum(len(g) for _, g in dup_content_groups)
    print(f"Jumlah grup content yang identik persis antar produk berbeda : {n_dup_groups} grup")
    print(f"Total produk yang terlibat                                  : {n_dup_products}  ({pct(n_dup_products, total)})")
    if n_dup_groups > 0:
        examples = []
        for _, group in list(dup_content_groups)[:3]:
            names = group["product_name"].tolist()
            examples.append(f"{len(names)} produk sama persis: {', '.join(names[:3])}"
                             + (f", ... (+{len(names)-3} lainnya)" if len(names) > 3 else ""))
        show_examples(examples, "grup produk dengan content identik", max_items=3)
    else:
        print("  Tidak ditemukan content yang identik persis antar produk berbeda. Baik.")

    # ---- estimasi ukuran vocabulary (perkiraan kasar, bukan TF-IDF asli) ----
    section("STEP 6 — ESTIMASI UKURAN VOCABULARY (PERKIRAAN KASAR)")
    all_tokens = set()
    for c in df["content"]:
        all_tokens.update(c.split())
    print(f"Jumlah token unik di seluruh kolom 'content' : {len(all_tokens)}")
    print("(Ini estimasi kasar dari split spasi biasa — vocabulary TF-IDF sebenarnya")
    print(" akan lebih kecil karena ada filter min_df, max_df, dan tokenisasi sklearn.)")

    # ---- contoh sample content lengkap ----
    section("STEP 7 — CONTOH HASIL AKHIR (RANDOM SAMPLE)")
    for _, row in df[["product_name", "category", "skin_type", "content"]].sample(3, random_state=42).iterrows():
        print(f"\n  Produk   : {row['product_name']}")
        print(f"  Kategori : {row['category']}")
        print(f"  Skin type: {row['skin_type']}")
        print(f"  Content  : {row['content'][:200]}...")

    elapsed = time.time() - start_time
    section("SELESAI")
    print(f"Waktu eksekusi build_content_features : {elapsed:.2f} detik")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Disimpan ke     : {OUTPUT_PATH}")
    print(f"Kolom final     : {list(df.columns)}")
    print(f"Total produk    : {len(df)}")


if __name__ == "__main__":
    main()