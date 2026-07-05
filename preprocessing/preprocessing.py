"""
Preprocessing data skincare Sociolla.

Langkah-langkah:
1. Drop produk noise (Product Testing, bundle Buy 2 Get 1, dll)
2. Drop baris tanpa rating (NaN)
3. Bersihkan deskripsi yang invalid ('-', '--', NaN)
4. Normalisasi teks: product_name, brand
5. Drop varian produk (#2, #3, dst) — keep yang #1 atau tanpa nomor
6. Drop duplikat & reset index, simpan

Catatan:
- Kolom ingredients sudah tidak ada sejak scraping (dihapus dari scraper)
- Kolom image_url dipertahankan untuk ditampilkan di website
- Kolom skin_type dibiarkan kosong → akan diisi di step assign_skin_type.py
- Rating tidak di-fill median, langsung drop yang NaN

Catatan tambahan (versi laporan):
- Semua fungsi mencetak ringkasan before/after + contoh data yang terdampak,
  supaya output terminal bisa langsung dipakai sebagai bukti/screenshot di
  laporan skripsi (Bab Hasil & Pembahasan).
- Tabel funnel (ringkasan seluruh tahapan) dicetak di akhir proses.
"""

from __future__ import annotations

import re
import time
import pandas as pd


INPUT_PATH  = "data/sociolla_skincare_raw.csv"
OUTPUT_PATH = "data/sociolla_skincare_clean.csv"


# ==============================================================
# HELPER
# ==============================================================
def pct(part: int, total: int) -> str:
    """Format persentase 'part dari total', aman dari pembagian nol."""
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
    """Cetak maksimal `max_items` contoh, dengan info total jika lebih banyak."""
    items = list(items)
    if not items:
        return
    print(f"  Contoh {label} (menampilkan {min(len(items), max_items)} dari {len(items)}):")
    for item in items[:max_items]:
        print(f"    - {item}")
    if len(items) > max_items:
        print(f"    ... dan {len(items) - max_items} lainnya")


# ==============================================================
# STEP 0: Ringkasan data mentah (SEBELUM preprocessing)
# ==============================================================
def initial_summary(df: pd.DataFrame) -> None:
    section("RINGKASAN DATA MENTAH (SEBELUM PREPROCESSING)")

    total = len(df)
    print(f"Total baris (produk)       : {total}")
    print(f"Total kolom                : {len(df.columns)}")
    print(f"Kolom                      : {list(df.columns)}")

    if "brand" in df.columns:
        print(f"Jumlah brand unik          : {df['brand'].nunique()}")
    if "category" in df.columns:
        print(f"Jumlah kategori unik       : {df['category'].nunique()}")

    subsection("Missing values per kolom")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("  Tidak ada missing value.")
    else:
        for col, n in missing.items():
            print(f"  {col:<20}: {n:>6} baris  ({pct(n, total)})")

    subsection("Duplikat")
    dup_full = df.duplicated().sum()
    print(f"  Duplikat baris identik (semua kolom)      : {dup_full}")
    if {"product_name", "brand"}.issubset(df.columns):
        dup_pn_brand = df.duplicated(subset=["product_name", "brand"]).sum()
        print(f"  Duplikat berdasarkan (product_name, brand): {dup_pn_brand}")

    if "category" in df.columns:
        subsection("Distribusi kategori (data mentah)")
        print(df["category"].value_counts().to_string())

    if "price" in df.columns:
        subsection("Statistik harga (Rp) — data mentah")
        print(df["price"].describe().round(0).to_string())

    if "rating" in df.columns:
        subsection("Statistik rating — data mentah (termasuk NaN diabaikan otomatis oleh describe)")
        print(df["rating"].describe().round(4).to_string())


# ==============================================================
# STEP 1: Load data
# ==============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    section("STEP 1 — LOAD DATA")
    print(f"[LOAD] {len(df)} baris, {len(df.columns)} kolom dari '{path}'")
    return df


# ==============================================================
# STEP 2: Drop baris noise
# ==============================================================
NOISE_KEYWORDS = [
    "not for sale",
    "product testing",
    "sample",
    "tester",
    "buy 2 get",
    "buy 3 get",
    "b2g1",
    "b3g1",
]

def drop_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 2 — DROP PRODUK NOISE")
    before = len(df)
    mask_noise = df["product_name"].str.lower().str.contains(
        "|".join(NOISE_KEYWORDS), na=False
    )
    dropped_names = df.loc[mask_noise, "product_name"].tolist()
    df = df[~mask_noise].copy()
    after = len(df)

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus : {before - after} baris  ({pct(before - after, before)})")
    show_examples(dropped_names, "produk noise yang dihapus")
    return df


# ==============================================================
# STEP 3: Drop baris tanpa rating
# ==============================================================
def drop_missing_rating(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 3 — DROP BARIS TANPA RATING")
    before = len(df)
    missing_mask = df["rating"].isna()
    dropped_names = df.loc[missing_mask, "product_name"].tolist()
    df = df.dropna(subset=["rating"]).copy()
    after = len(df)

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus : {before - after} baris  ({pct(before - after, before)}) tanpa rating")
    show_examples(dropped_names, "produk tanpa rating yang dihapus")
    return df


# ==============================================================
# STEP 4: Bersihkan kolom description
# ==============================================================
INVALID_DESC = {"-", "--", "—", "–", "n/a", "na"}

def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 4 — BERSIHKAN KOLOM DESCRIPTION")

    original = df["description"].copy()

    def _clean(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s.lower() in INVALID_DESC or s == "":
            return None
        s = re.sub(r"\s+", " ", s)
        return s

    df["description"] = df["description"].apply(_clean)

    total = len(df)
    missing_before = original.isna().sum()
    invalid_only = original.apply(
        lambda x: isinstance(x, str) and x.strip().lower() in INVALID_DESC
    ).sum()
    missing_after = df["description"].isna().sum()

    print(f"Total baris                              : {total}")
    print(f"Deskripsi NaN sejak awal                 : {missing_before}  ({pct(missing_before, total)})")
    print(f"Deskripsi berisi placeholder tidak valid  : {invalid_only}  ({pct(invalid_only, total)})  contoh: '-', '--', 'N/A'")
    print(f"Total deskripsi kosong/invalid → NaN      : {missing_after}  ({pct(missing_after, total)})")

    # contoh perubahan whitespace berlebih (before -> after)
    changed_mask = (
        original.notna()
        & df["description"].notna()
        & (original.astype(str) != df["description"].astype(str))
    )
    examples = [
        f"'{b}'  →  '{a}'"
        for b, a in zip(original[changed_mask], df.loc[changed_mask, "description"])
    ]
    show_examples(examples, "deskripsi yang dirapikan whitespace-nya (before → after)")

    # contoh placeholder yang diubah jadi NaN
    placeholder_examples = original[
        original.apply(lambda x: isinstance(x, str) and x.strip().lower() in INVALID_DESC)
    ].tolist()
    show_examples([f"'{v}'  →  NaN" for v in placeholder_examples], "placeholder invalid yang di-NaN-kan")

    return df


# ==============================================================
# STEP 5: Normalisasi product_name & brand
# ==============================================================
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 5 — NORMALISASI product_name & brand")

    original_name = df["product_name"].copy()
    original_brand = df["brand"].copy()

    df["product_name"] = df["product_name"].str.strip().str.replace(r"\s+", " ", regex=True)
    df["brand"]        = df["brand"].str.strip().str.replace(r"\s+", " ", regex=True)

    changed_name_mask = original_name.astype(str) != df["product_name"].astype(str)
    changed_brand_mask = original_brand.astype(str) != df["brand"].astype(str)

    print(f"product_name yang berubah (spasi berlebih/leading-trailing) : {changed_name_mask.sum()}")
    print(f"brand yang berubah                                          : {changed_brand_mask.sum()}")
    print(f"Jumlah brand unik setelah normalisasi                       : {df['brand'].nunique()}")
    if "category" in df.columns:
        print(f"Jumlah kategori unik                                        : {df['category'].nunique()}")

    name_examples = [
        f"'{b}'  →  '{a}'"
        for b, a in zip(original_name[changed_name_mask], df.loc[changed_name_mask, "product_name"])
    ]
    show_examples(name_examples, "product_name sebelum → sesudah normalisasi")

    brand_examples = [
        f"'{b}'  →  '{a}'"
        for b, a in zip(original_brand[changed_brand_mask], df.loc[changed_brand_mask, "brand"])
    ]
    show_examples(brand_examples, "brand sebelum → sesudah normalisasi")

    return df


# ==============================================================
# STEP 6: Drop varian produk (#2, #3, dst)
# ==============================================================
def drop_variant_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus produk varian (#2, #3, dst) — keep yang #1 atau tanpa nomor varian.
    Contoh yang dihapus      : 'Plum Blossom Skin Renewal Serum #2'
    Contoh yang dipertahankan: 'Plum Blossom Skin Renewal Serum #1'
    """
    section("STEP 6 — DROP VARIAN PRODUK (#2, #3, dst)")
    before  = len(df)
    mask    = df["product_name"].str.contains(r"\s#[2-9]\d*$", regex=True, na=False)
    dropped = df[mask][["product_name", "brand"]].copy()
    df      = df[~mask].copy()
    after   = len(df)

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus : {before - after} baris  ({pct(before - after, before)})")

    dropped_list = [f"{row['product_name']} ({row['brand']})" for _, row in dropped.iterrows()]
    show_examples(dropped_list, "varian produk yang dihapus")
    return df


# ==============================================================
# STEP 7: Final check, dedup & simpan
# ==============================================================
def final_check(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 7 — DEDUPLIKASI & RINGKASAN AKHIR")

    before = len(df)
    dup_mask = df.duplicated(subset=["product_name", "brand"])
    dup_examples = [
        f"{row['product_name']} ({row['brand']})"
        for _, row in df[dup_mask][["product_name", "brand"]].iterrows()
    ]
    df = df.drop_duplicates(subset=["product_name", "brand"]).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[DEDUP] {before - after} duplikat dibuang  ({pct(before - after, before)})")
        show_examples(dup_examples, "duplikat yang dibuang")
    else:
        print("[DEDUP] Tidak ada duplikat (product_name, brand) yang tersisa.")

    subsection("Missing values (data bersih)")
    missing = df.isnull().sum()
    total = len(df)
    if missing.sum() == 0:
        print("  Tidak ada missing value.")
    else:
        for col, n in missing[missing > 0].sort_values(ascending=False).items():
            print(f"  {col:<20}: {n:>6} baris  ({pct(n, total)})")

    subsection("Distribusi kategori (data bersih)")
    print(df["category"].value_counts().to_string())

    subsection("Statistik harga (Rp) — data bersih")
    print(df["price"].describe().round(0).to_string())

    subsection("Statistik rating — data bersih")
    print(df["rating"].describe().round(4).to_string())

    print(f"\nJumlah brand unik (data bersih)    : {df['brand'].nunique()}")
    print(f"Jumlah kategori unik (data bersih) : {df['category'].nunique()}")
    print(f"Produk dengan image_url            : {(df['image_url'].astype(str).str.strip() != '').sum()}")

    return df


# ==============================================================
# FUNNEL SUMMARY (ringkasan seluruh tahapan)
# ==============================================================
def print_funnel(funnel: list[tuple[str, int, int]], raw_total: int) -> None:
    section("TABEL RINGKASAN TAHAPAN PREPROCESSING (DATA FUNNEL)")

    header = f"{'Tahapan':<32}{'Sebelum':>10}{'Sesudah':>10}{'Dihapus':>10}{'% Tersisa dari Raw':>22}"
    print(header)
    print("-" * len(header))
    for step_name, before, after in funnel:
        removed = before - after
        retained_pct = pct(after, raw_total)
        print(f"{step_name:<32}{before:>10}{after:>10}{removed:>10}{retained_pct:>22}")

    final_after = funnel[-1][2]
    print("-" * len(header))
    print(f"\nTotal data mentah   : {raw_total} produk")
    print(f"Total data bersih   : {final_after} produk")
    print(f"Retensi data akhir  : {pct(final_after, raw_total)} dari data mentah")
    print(f"Total data dibuang  : {raw_total - final_after} produk  ({pct(raw_total - final_after, raw_total)})")


# ==============================================================
# MAIN
# ==============================================================
def main() -> None:
    import os

    start_time = time.time()

    df = load_data(INPUT_PATH)
    raw_total = len(df)
    initial_summary(df)

    funnel: list[tuple[str, int, int]] = []

    n0 = len(df)
    df = drop_noise_rows(df)
    funnel.append(("Drop produk noise", n0, len(df)))

    n1 = len(df)
    df = drop_missing_rating(df)
    funnel.append(("Drop baris tanpa rating", n1, len(df)))

    n2 = len(df)
    df = clean_description(df)
    funnel.append(("Bersihkan description (tanpa drop baris)", n2, len(df)))

    n3 = len(df)
    df = normalize_text_columns(df)
    funnel.append(("Normalisasi teks (tanpa drop baris)", n3, len(df)))

    n4 = len(df)
    df = drop_variant_numbers(df)
    funnel.append(("Drop varian produk (#2, #3, dst)", n4, len(df)))

    n5 = len(df)
    df = final_check(df)
    funnel.append(("Deduplikasi (product_name, brand)", n5, len(df)))

    print_funnel(funnel, raw_total)

    elapsed = time.time() - start_time
    section("SELESAI")
    print(f"Waktu eksekusi preprocessing : {elapsed:.2f} detik")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Disimpan ke                  : {OUTPUT_PATH}")
    print(f"Total produk bersih final    : {len(df)}")


if __name__ == "__main__":
    main()