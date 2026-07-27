"""
Preprocessing data skincare Sociolla — Glowmatch (CBF: TF-IDF + Cosine Similarity)

VERSI REVISI 3
============================================================================
Perubahan dari versi sebelumnya:
- Scraper (scrape_sociolla.py) sudah direvisi agar TIDAK melakukan filtering
  atau dedup kualitas data (exclude bundle, exclude nama tidak valid, dedup
  lintas kategori). Seluruh keputusan tersebut dipindahkan ke sini, sesuai
  prinsip metodologi: filtering & dedup adalah bagian dari Data Preparation,
  bukan Data Understanding.
- Ditambahkan 2 step baru: (1) exclude bundle/pack/kit/set, (2) exclude
  size/packaging variant (mini/miniature/travel size/trial size).

FIX DI VERSI 2 (hasil audit manual terhadap clean CSV run sebelumnya):
- BUG 1 (volume suffix): strip_size_markers() sebelumnya cuma membuang kata
  penanda ("Miniature", "Mini", dst) tapi TIDAK membuang angka volume yang
  menempel di belakangnya (mis. "15ml", "30ml", "120gr"). Akibatnya nama
  hasil strip jadi "...Mask 15ml" bukan "...Mask", sehingga gagal cocok
  dengan base yang sebenarnya ADA di dataset.
  Fix: tambah _VOLUME_TRAILING_PATTERN yang membuang token volume/berat di
  ujung nama SETELAH kata penanda ukuran dibuang.
- BUG 2 (marker belum lengkap, versi 2): "refill pack" dan "twin pack"
  ditambahkan ke SIZE_MARKER_PATTERNS dengan kriteria base-harus-ada.

FIX DI VERSI 3 (revisi konsep, hasil audit lanjutan):
- TEMUAN: produk seperti "Senka Perfect Whip White 100gr Twin Pack" masih
  lolos ke katalog akhir. Ini BUKAN bug pencocokan base — memang tidak ada
  base single-unit "Senka Perfect Whip White" di dataset, jadi sesuai
  kriteria versi 2 (base-harus-ada), produk itu SENGAJA dipertahankan.
- REVISI KONSEP: setelah dipikir ulang, "twin pack" dan "refill pack"
  DIPINDAHKAN dari kategori size/packaging variant (yang butuh base) KE
  kategori bundle (exclusion tanpa syarat base). Alasannya:
    * Kata "mini"/"miniature"/"travel size"/"trial size" menandakan produk
      yang SAMA persis, cuma kuantitas/volumenya lebih kecil — 1 unit fisik,
      cuma lebih sedikit isinya. Kalau base-nya nggak ada, produk itu ADALAH
      satu-satunya bentuk produk itu dijual → sah sebagai satu unit analisis.
    * "Twin pack"/"refill pack" itu beda secara konsep: SECARA FISIK ada
      lebih dari satu unit produk dalam satu listing (2 tube sekaligus,
      1 tube + 1 kantong refill sekaligus). Ini melanggar unit analisis
      "satu produk individual" (Bab 1.2) TERLEPAS dari base-nya ada atau
      tidak — sama seperti alasan bundle pada umumnya, bukan alasan
      duplikasi konten seperti size variant.
  Karena itu twin pack/refill pack sekarang dihapus TANPA SYARAT (seperti
  bundle/kit/set), bukan cuma kalau base-nya ketemu.
  TRADE-OFF yang perlu disadari: kalau ada produk yang HANYA pernah dijual
  dalam bentuk twin pack/refill pack (tidak ada versi single unit-nya sama
  sekali di dataset), produk itu akan HILANG SELURUHNYA dari katalog akhir
  — bukan cuma dianggap "duplikat lalu di-drop salah satu", tapi memang
  tidak direkomendasikan sama sekali. Ini keputusan sadar (konsisten dengan
  definisi unit analisis di Bab 1.2), tapi perlu ditulis eksplisit di Bab
  3.3.1 supaya bisa dijawab kalau ditanya penguji kenapa jumlah produk
  Cleanser/dst berkurang beberapa item dibanding sebelumnya.
- Regex "twin pack" diperketat jadi eksplisit menerima 0 atau 1 spasi di
  antara "twin" dan "pack" (mencakup "Twin Pack" maupun "Twinpack" satu
  kata) — ini sudah benar sejak versi 2, dipertahankan di versi 3.

URUTAN STEP & ALASAN PENEMPATAN
----------------------------------------------------------------------------
1.  Load data
2.  Exclude bundle/pack/kit/set/twin pack/refill pack   <- diperluas v3
3.  Exclude size/packaging variant (mini/travel/trial)
4.  Drop produk noise (sisa)
5.  Drop baris tanpa rating
6.  Bersihkan kolom description
7.  Normalisasi product_name & brand
8.  Drop varian produk (#2, #3, dst)
9.  Deduplikasi & ringkasan akhir

Step 2 dan 3 sengaja ditempatkan PALING AWAL (sebelum drop noise/rating/
description), karena keduanya menjawab pertanyaan yang sama secara
konseptual dengan step "drop noise" dan "drop varian nomor": yaitu
"apakah baris ini layak dianggap SATU produk individual yang valid untuk
dibandingkan secara konten (TF-IDF)?" — bukan soal kualitas rating,
deskripsi, atau kerapian teks. Step 3 (size variant) ditempatkan SETELAH
step 2 (bundle, termasuk twin/refill pack) karena kriteria "base produk
harus ada di dataset" pada step 3 seharusnya dicek terhadap populasi yang
SUDAH bersih dari bundle — supaya base yang dijadikan acuan bukan bundle
yang toh akan dibuang juga.

Catatan gaya kode (dipertahankan dari versi lama):
- Semua fungsi mencetak ringkasan before/after + contoh data yang
  terdampak dalam "kotak" terpisah (section/subsection), supaya output
  terminal bisa langsung di-screenshot satu-satu tanpa kepotong tahap lain.
- Tabel funnel (ringkasan seluruh tahapan) dicetak di akhir proses.
"""

from __future__ import annotations

import re
import time
import pandas as pd


INPUT_PATH  = "data/sociolla_skincare_raw.csv"
OUTPUT_PATH = "data/sociolla_skincare_clean.csv"


# ==============================================================
# HELPER (dipertahankan dari versi lama)
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
# DETEKSI BUNDLE/PACK/KIT/SET/TWIN PACK/REFILL PACK
# ==============================================================
# PENTING (perbaikan dari versi lama):
# - TIDAK memakai pola kurung siku generik `\[.{1,30}\]` karena itu ikut
#   menangkap "[NEW]", "[Glutathione]" (nama bahan aktif), "[ECOCERT
#   Certified]", "[Limited Edition]" — semua ITU BUKAN bundle.
#   Solusinya: tidak ada aturan "kurung siku apa saja", yang ada hanya
#   pencocokan kata kunci spesifik dengan word-boundary. Karena kata kunci
#   dicek di seluruh nama (termasuk di dalam kurung siku), kurung siku yang
#   isinya bukan kata kunci bundle otomatis tidak pernah match.
# - "free" TIDAK dipakai berdiri sendiri (itu menangkap "Cruelty-Free",
#   "Oil-Free" yang notabene klaim produk/tekstur, bukan bundle). "free"
#   hanya dipakai dalam pola frasa "buy X get Y free" / "free gift" / dst.
# - "kit" dipertahankan sebagai kata kunci utuh (contoh valid: "Ceramide
#   Ato Travel Kit" — isinya toner+cleanser+lotion+cream+gel sekaligus).
# - "pack" SENGAJA tidak dijadikan kata kunci tunggal, karena dalam
#   penamaan skincare Korea "pack" sering berarti JENIS PRODUK (mis.
#   "Sleeping Pack", "Eye Pack" = sejenis masker, BUKAN kemasan/paket
#   berisi banyak produk). "pack" hanya dianggap indikasi bundle jika
#   muncul dalam konteks eksplisit (mis. "3-pack", "value pack",
#   "gift pack", "bonus pack").
# - Pola "Full+Mini Size" (ukuran penuh + mini dijual sepaket) dimasukkan
#   sebagai indikasi bundle, BUKAN size variant biasa.
# - REVISI v3: "twin pack"/"twinpack" & "refill pack" DIPINDAHKAN ke sini
#   (bundle, exclusion TANPA SYARAT base) — sebelumnya sempat diperlakukan
#   sebagai size variant yang butuh base ADA dulu. Alasan pemindahan: kedua
#   istilah ini menandakan LEBIH DARI SATU unit fisik produk dalam satu
#   listing (2 tube sekaligus / tube + refill sekaligus), yang melanggar
#   definisi unit analisis "satu produk individual" (Bab 1.2) terlepas dari
#   ada tidaknya versi single-unit-nya di dataset — beda dengan
#   mini/miniature/travel/trial size yang tetap 1 unit fisik, cuma lebih
#   kecil volumenya. Lihat catatan TRADE-OFF di header file: produk yang
#   HANYA pernah dijual dalam bentuk twin/refill pack akan hilang total
#   dari katalog akhir, ini keputusan sadar bukan bug.

BUNDLE_STANDALONE_WORDS = ["bundle", "kit", "duo", "trio", "gwp", "paket", "set"]

BUNDLE_PHRASE_PATTERNS = [
    r"\bvalue\s+pack\b",
    r"\bgift\s+pack\b",
    r"\bbonus\s+pack\b",
    r"\d+\s*-?\s*pack\b",                  # "3-pack", "2 pack"
    r"\bfull\s*\+\s*mini\b",               # "Full+Mini Size"
    r"\bbuy\s*\d*\s*get\s*\d*\s*free\b",   # "buy 1 get 1 free", "buy get free"
    r"\bget\s+\d+\s+free\b",
    r"\bfree\s+gift\b",
    r"\bgift\s+with\s+purchase\b",
    r"\bb\s*\d\s*g\s*\d\b",                # "b2g1", "b1g1", "b3g1"
    r"\bbuy\s*\d+\s*get\s*\d+\b",          # "buy 2 get 1" (tanpa kata 'free')
    r"\btwin\s*pack\b",                    # "Twin Pack" & "Twinpack" (v3: dipindah dari size variant)
    r"\brefill\s+pack\b",                  # "Refill Pack" (v3: dipindah dari size variant)
    r"\(\s*\d+\s*pcs\s*\)",                # "(3 pcs)", "(2 Pcs)" -- mis. "... Box (3 pcs)"
    r"\bisi\s+\d+\s*(?:pcs|pieces|buah)?\b",  # "isi 3", "isi 3 pcs" (penamaan Indonesia)
]

_BUNDLE_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in BUNDLE_STANDALONE_WORDS) + r")\b",
    flags=re.IGNORECASE,
)
_BUNDLE_PHRASE_PATTERN = re.compile("|".join(BUNDLE_PHRASE_PATTERNS), flags=re.IGNORECASE)


def is_bundle_product(name) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    if _BUNDLE_WORD_PATTERN.search(name):
        return True
    if _BUNDLE_PHRASE_PATTERN.search(name):
        return True
    return False


def get_bundle_mask(df: pd.DataFrame) -> pd.Series:
    return df["product_name"].apply(is_bundle_product)


# ==============================================================
# DETEKSI SIZE / PACKAGING VARIANT (mini, miniature, travel size, trial size)
# ==============================================================
# KRITERIA WAJIB: TIDAK blanket-delete semua yang mengandung kata penanda.
# Baris hanya dihapus jika ADA produk "base" yang identik (nama produk
# setelah dibuang kata penanda ukuran + angka volume di belakangnya, brand
# yang sama) yang SUDAH ADA di dataset. Contoh:
#   - "Rice Mask Miniature" (I'm From) & base "Rice Mask" (I'm From) ADA
#     -> HAPUS yang miniature, keep base-nya
#   - "Lacoco Aloe Vera Soothing Mist Mini" & base "Lacoco Aloe Vera
#     Soothing Mist" TIDAK ADA -> JANGAN dihapus (mungkin memang hanya
#     dijual dalam ukuran itu)
# Semua pencocokan kata penanda pakai word-boundary supaya "mini" yang jadi
# bagian nama produk asli (mis. brand/nama yang kebetulan diawali "Mini")
# tidak salah kena, karena \bmini\b butuh batas kata utuh di kedua sisi.
#
# CATATAN v3: "twin pack" & "refill pack" TIDAK lagi ada di sini — sudah
# dipindah ke deteksi bundle (exclusion tanpa syarat base), lihat komentar
# di bagian bundle di atas.
#
# FIX v2 (dipertahankan): angka volume/berat yang menempel di belakang kata
# penanda ("Miniature 15ml", "Mini 30ml") ikut dibuang lewat
# _VOLUME_TRAILING_PATTERN SETELAH kata penanda ukuran dibuang, supaya
# base-matching tidak gagal gara-gara sisa angka volume.

SIZE_MARKER_PATTERNS = [
    r"\btravel\s+size\b",
    r"\btrial\s+size\b",
    r"\bminiature\b",
    r"\bmini\b",
]
_SIZE_MARKER_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in SIZE_MARKER_PATTERNS]

# Token volume/berat di ujung nama, mis. "15ml", "30 ml", "120gr", "1.5oz".
# Sengaja hanya dibuang kalau ada di UJUNG string (setelah kata penanda
# ukuran dibuang), bukan di tengah, supaya tidak salah membuang angka yang
# memang bagian nama produk (mis. varian bernomor, konsentrasi bahan aktif).
_VOLUME_TRAILING_PATTERN = re.compile(
    r"\s*\d+(?:[.,]\d+)?\s*(?:ml|l|g|gr|gram|kg|oz)\.?\s*$",
    flags=re.IGNORECASE,
)


def has_size_marker(name) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    return any(p.search(name) for p in _SIZE_MARKER_REGEX)


def strip_size_markers(name: str) -> str:
    s = name
    for p in _SIZE_MARKER_REGEX:
        s = p.sub("", s)
    s = _VOLUME_TRAILING_PATTERN.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -")
    return s


def _norm_key(name: str, brand: str) -> tuple:
    n = re.sub(r"\s+", " ", str(name)).strip().lower()
    b = re.sub(r"\s+", " ", str(brand)).strip().lower()
    return (n, b)


def get_size_variant_mask(df: pd.DataFrame) -> pd.Series:
    """
    Mengembalikan boolean mask baris yang akan dihapus karena size variant
    dari base produk yang SUDAH ADA di df (brand sama). Baris yang
    mengandung kata penanda tapi base-nya tidak ditemukan TIDAK ditandai
    (dipertahankan).
    """
    has_marker = df["product_name"].apply(has_size_marker)

    # Pool base: baris TANPA kata penanda -> kandidat "produk asli"
    base_pool = df.loc[~has_marker]
    base_keys = set(
        _norm_key(n, b) for n, b in zip(base_pool["product_name"], base_pool["brand"])
    )

    mask = pd.Series(False, index=df.index)
    for idx in df.index[has_marker]:
        row = df.loc[idx]
        stripped = strip_size_markers(str(row["product_name"]))
        key = _norm_key(stripped, row["brand"])
        if key in base_keys:
            mask.loc[idx] = True
    return mask


# ==============================================================
# STEP 0: Ringkasan data mentah (SEBELUM preprocessing apa pun)
# ==============================================================
NOISE_KEYWORDS = [
    "not for sale",
    "product testing",
    "sample",
    "tester",
]


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

    subsection("Deteksi produk bundle/pack/kit/set/twin pack/refill pack (belum dihapus)")
    bundle_mask = get_bundle_mask(df)
    n_bundle = int(bundle_mask.sum())
    print(f"  Terdeteksi sebagai bundle/pack/kit/set/twin/refill : {n_bundle}  ({pct(n_bundle, total)})")
    show_examples(df.loc[bundle_mask, "product_name"].tolist(), "produk terindikasi bundle")

    subsection("Deteksi size/packaging variant: mini/miniature/travel/trial (belum dihapus)")
    size_mask = get_size_variant_mask(df)
    n_size = int(size_mask.sum())
    print(f"  Terdeteksi sebagai size variant (base ditemukan) : {n_size}  ({pct(n_size, total)})")
    print("  Catatan: dihitung dari populasi mentah (masih termasuk bundle) untuk keperluan")
    print("  pelaporan Bab 4.1.3. Angka final yang beneran dihapus ada di STEP 3 (setelah")
    print("  bundle dibuang), lihat funnel di akhir output.")
    show_examples(df.loc[size_mask, "product_name"].tolist(), "produk terindikasi size variant")

    subsection("Deteksi produk noise (not for sale/testing/sample/tester)")
    noise_mask = df["product_name"].str.lower().str.contains("|".join(NOISE_KEYWORDS), na=False)
    n_noise = int(noise_mask.sum())
    print(f"  Terdeteksi sebagai noise : {n_noise}  ({pct(n_noise, total)})")
    show_examples(df.loc[noise_mask, "product_name"].tolist(), "produk terindikasi noise")

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
        subsection("Statistik rating — data mentah (NaN diabaikan otomatis oleh describe)")
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
# STEP 2: Exclude bundle/pack/kit/set/twin pack/refill pack
# ==============================================================
def exclude_bundle_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sistem CBF membandingkan kemiripan konten antar PRODUK INDIVIDUAL.
    Bundle (termasuk twin pack & refill pack sejak v3) adalah lebih dari
    satu unit fisik produk dalam satu listing, sehingga: (a) kolom content
    jadi tidak koheren / harga & rating tidak sebanding dengan produk
    tunggal, (b) di luar unit analisis yang didefinisikan di Ruang Lingkup
    (Bab 1.2). Lihat Bab 3.3.1 untuk penjelasan lengkap & trade-off twin
    pack/refill pack di header file.
    """
    section("STEP 2 — EXCLUDE BUNDLE/PACK/KIT/SET/TWIN PACK/REFILL PACK")
    before = len(df)
    mask = get_bundle_mask(df)
    dropped = df.loc[mask, ["product_name", "brand"]]
    df = df[~mask].copy()
    after = len(df)

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus : {before - after} baris  ({pct(before - after, before)})")
    dropped_list = [f"{r['product_name']} ({r['brand']})" for _, r in dropped.iterrows()]
    show_examples(dropped_list, "produk bundle/pack/kit/set/twin/refill yang dihapus")
    return df


# ==============================================================
# STEP 3: Exclude size/packaging variant (mini/travel/trial)
# ==============================================================
def exclude_size_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produk miniature/travel size/trial size dari produk yang sama
    menghasilkan konten TF-IDF yang nyaris identik dengan produk basenya
    (contoh temuan: Rice Mask vs Rice Mask Miniature, cosine similarity
    96%+), sehingga tidak memberi rekomendasi yang benar-benar baru. Hanya
    dihapus jika base produknya (nama tanpa kata penanda + tanpa angka
    volume, brand sama) SUDAH ADA di dataset — bukan blanket-delete kata
    "mini"/dst. (twin pack/refill pack sudah ditangani di STEP 2, bukan di
    sini — lihat catatan v3 di header file.)
    """
    section("STEP 3 — EXCLUDE SIZE/PACKAGING VARIANT (mini/travel/trial)")
    before = len(df)

    has_marker = df["product_name"].apply(has_size_marker)
    mask = get_size_variant_mask(df)

    dropped = df.loc[mask, ["product_name", "brand"]].copy()
    dropped["base_ditemukan"] = dropped["product_name"].apply(strip_size_markers)

    kept_no_base_mask = has_marker & ~mask
    kept_no_base = df.loc[kept_no_base_mask, ["product_name", "brand"]]

    df = df[~mask].copy()
    after = len(df)

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus (size variant, base ditemukan) : {before - after}  ({pct(before - after, before)})")
    print(f"Mengandung kata penanda ukuran tapi base TIDAK ditemukan (dipertahankan) : {len(kept_no_base)}")

    dropped_list = [
        f"{r['product_name']} ({r['brand']})  ->  base: '{r['base_ditemukan']}'"
        for _, r in dropped.iterrows()
    ]
    show_examples(dropped_list, "size variant yang dihapus (base ditemukan)")

    kept_list = [f"{r['product_name']} ({r['brand']})" for _, r in kept_no_base.iterrows()]
    show_examples(kept_list, "mengandung kata ukuran tapi DIPERTAHANKAN (base tidak ditemukan)")
    return df


# ==============================================================
# STEP 4: Drop produk noise (sisa: not for sale/testing/sample/tester)
# ==============================================================
def drop_noise_rows(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 4 — DROP PRODUK NOISE (not for sale/testing/sample/tester)")
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
# STEP 5: Drop baris tanpa rating
# ==============================================================
def drop_missing_rating(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 5 — DROP BARIS TANPA RATING")
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
# STEP 6: Bersihkan kolom description
# ==============================================================
INVALID_DESC = {"-", "--", "—", "–", "n/a", "na"}

def clean_description(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 6 — BERSIHKAN KOLOM DESCRIPTION")

    before = len(df)
    original = df["description"].copy()

    invalid_mask = (
        original.isna() |
        original.apply(
            lambda x: isinstance(x, str) and (
                x.strip() == "" or
                x.strip().lower() in INVALID_DESC
            )
        )
    )

    dropped_names = df.loc[invalid_mask, "product_name"].tolist()
    df = df.loc[~invalid_mask].copy()

    df["description"] = (
        df["description"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    after = len(df)

    missing_before = original.isna().sum()
    invalid_only = original.apply(
        lambda x: isinstance(x, str) and (
            x.strip() == "" or
            x.strip().lower() in INVALID_DESC
        )
    ).sum()

    print(f"Sebelum : {before} baris")
    print(f"Sesudah : {after} baris")
    print(f"Dihapus : {before-after} baris ({pct(before-after, before)})")

    print(f"\nRincian data yang dihapus:")
    print(f"- Deskripsi NaN sejak awal                : {missing_before}")
    print(f"- Placeholder tidak valid ('-', '--', 'N/A', dll): {invalid_only}")

    show_examples(
        dropped_names,
        "produk yang dihapus karena description kosong/tidak valid"
    )

    return df


# ==============================================================
# STEP 7: Normalisasi product_name & brand
# ==============================================================
def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 7 — NORMALISASI product_name & brand")

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
        f"'{b}'  ->  '{a}'"
        for b, a in zip(original_name[changed_name_mask], df.loc[changed_name_mask, "product_name"])
    ]
    show_examples(name_examples, "product_name sebelum -> sesudah normalisasi")

    brand_examples = [
        f"'{b}'  ->  '{a}'"
        for b, a in zip(original_brand[changed_brand_mask], df.loc[changed_brand_mask, "brand"])
    ]
    show_examples(brand_examples, "brand sebelum -> sesudah normalisasi")

    return df


# ==============================================================
# STEP 8: Drop varian produk (#2, #3, dst)
# ==============================================================
def drop_variant_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus produk varian (#2, #3, dst) — keep yang #1 atau tanpa nomor varian.
    """
    section("STEP 8 — DROP VARIAN PRODUK (#2, #3, dst)")
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
# STEP 9: Final check, dedup & simpan
# ==============================================================
def final_check(df: pd.DataFrame) -> pd.DataFrame:
    section("STEP 9 — DEDUPLIKASI & RINGKASAN AKHIR")

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

    header = f"{'Tahapan':<44}{'Sebelum':>10}{'Sesudah':>10}{'Dihapus':>10}{'% Tersisa dari Raw':>22}"
    print(header)
    print("-" * len(header))
    for step_name, before, after in funnel:
        removed = before - after
        retained_pct = pct(after, raw_total)
        print(f"{step_name:<44}{before:>10}{after:>10}{removed:>10}{retained_pct:>22}")

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
    df = exclude_bundle_products(df)
    funnel.append(("Exclude bundle/pack/kit/set/twin/refill", n0, len(df)))

    n1 = len(df)
    df = exclude_size_variants(df)
    funnel.append(("Exclude size/packaging variant", n1, len(df)))

    n2 = len(df)
    df = drop_noise_rows(df)
    funnel.append(("Drop produk noise", n2, len(df)))

    n3 = len(df)
    df = drop_missing_rating(df)
    funnel.append(("Drop baris tanpa rating", n3, len(df)))

    n4 = len(df)
    df = clean_description(df)
    funnel.append(("Bersihkan description (drop baris invalid)", n4, len(df)))

    n5 = len(df)
    df = normalize_text_columns(df)
    funnel.append(("Normalisasi teks (tanpa drop baris)", n5, len(df)))

    n6 = len(df)
    df = drop_variant_numbers(df)
    funnel.append(("Drop varian produk (#2, #3, dst)", n6, len(df)))

    n7 = len(df)
    df = final_check(df)
    funnel.append(("Deduplikasi (product_name, brand)", n7, len(df)))

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