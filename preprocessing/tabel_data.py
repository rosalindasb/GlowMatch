"""
Script bukti terminal untuk:
- Ringkasan kualitas data mentah (missing value, duplikat, noise)
- Statistik deskriptif kolom price & rating
- Pemeriksaan missing value pada seluruh kolom
- Jumlah brand unik pada data mentah
- Tabel 4.2 (Distribusi Kategori Data Mentah)
- Tabel 4.3 (Ringkasan Tahapan Preprocessing / Data Funnel)
- Tabel 4.4 (Perbandingan Kategori Sebelum & Sesudah Preprocessing)
- Visualisasi distribusi price & rating (histogram + bar chart)

PENTING: Script ini TIDAK menulis ulang logika preprocessing. Semua fungsi
di-import langsung dari preprocessing.py, sehingga angka yang dihasilkan
dijamin identik dengan hasil pipeline utama (single source of truth).

Cara pakai:
    python generate_bukti_tabel.py

Jalankan file ini di folder yang sama dengan preprocessing.py, lalu
screenshot tiap bagian output yang muncul untuk dilampirkan sebagai
bukti di laporan. File gambar distribusi_price_rating.png akan otomatis
tersimpan di folder yang sama dan bisa langsung dipakai sebagai Gambar
di laporan (bukan screenshot terminal).

Dependency tambahan yang perlu diinstall:
    pip install matplotlib
"""

import contextlib
import io

import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import (
    INPUT_PATH,
    NOISE_KEYWORDS,
    load_data,
    drop_noise_rows,
    drop_missing_rating,
    clean_description,
    normalize_text_columns,
    drop_variant_numbers,
    final_check,
)


def cetak_ringkasan_kualitas_data(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RINGKASAN KUALITAS DATA MENTAH")
    print("=" * 60)

    total = len(df)
    print(f"Total baris (produk) : {total}")
    print(f"Total kolom          : {len(df.columns)}")
    print(f"Kolom                : {list(df.columns)}")

    print("\n-- Missing value per kolom (hanya yang > 0) --")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, n in missing.items():
        print(f"  {col:<15}: {n:>6} baris  ({n/total*100:.2f}%)")

    print("\n-- Duplikat --")
    print(f"  Duplikat baris identik (semua kolom)       : {df.duplicated().sum()}")
    print(f"  Duplikat berdasarkan (product_name, brand) : {df.duplicated(subset=['product_name','brand']).sum()}")

    print("\n-- Produk noise (kata kunci pada nama produk) --")
    mask_noise = df["product_name"].str.lower().str.contains("|".join(NOISE_KEYWORDS), na=False)
    print(f"  Jumlah produk mengandung kata kunci noise  : {mask_noise.sum()}")
    contoh = df.loc[mask_noise, "product_name"].head(3).tolist()
    for c in contoh:
        print(f"    - {c}")


def cetak_statistik_numerik(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STATISTIK DESKRIPTIF KOLOM NUMERIK")
    print("=" * 60)

    # --- Price ---
    price = df["price"].dropna()
    print("\n-- Kolom price (Rupiah) --")
    print(f"  Jumlah data     : {len(price)}")
    print(f"  Minimum         : Rp{price.min():,.0f}")
    print(f"  Maksimum        : Rp{price.max():,.0f}")
    print(f"  Rata-rata       : Rp{price.mean():,.0f}")
    print(f"  Median          : Rp{price.median():,.0f}")

    # --- Rating ---
    rating = df["rating"].dropna()
    print("\n-- Kolom rating (skala 0-5) --")
    print(f"  Jumlah data (memiliki rating) : {len(rating)}")
    print(f"  Minimum                       : {rating.min():.2f}")
    print(f"  Maksimum                      : {rating.max():.2f}")
    print(f"  Rata-rata                     : {rating.mean():.2f}")
    print(f"  Median                        : {rating.median():.2f}")


def cetak_missing_value_lengkap(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PEMERIKSAAN MISSING VALUE - SELURUH KOLOM")
    print("=" * 60)

    total = len(df)
    missing = df.isnull().sum()

    print(f"{'Kolom':<18}{'Missing':>10}{'Persentase':>14}")
    print("-" * 42)
    for col in df.columns:
        n = missing[col]
        pct = n / total * 100
        print(f"{col:<18}{n:>10}{pct:>13.2f}%")



def cetak_tabel_4_2(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("TABEL DISTRIBUSI KATEGORI DATA MENTAH")
    print("=" * 55)

    total = len(df)
    vc = df["category"].value_counts()

    print(f"{'No':<4}{'Kategori':<25}{'Jumlah Produk':>15}")
    print("-" * 44)
    for i, (cat, n) in enumerate(vc.items(), start=1):
        print(f"{i:<4}{cat:<25}{n:>15}")
    print("-" * 44)
    print(f"{'':<4}{'Total':<25}{total:>15}")


def cetak_jumlah_brand(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("JUMLAH BRAND UNIK - DATA MENTAH")
    print("=" * 60)
    print(f"  Jumlah brand unik : {df['brand'].nunique()}")

def cetak_tabel_4_3(funnel: list[tuple[str, int, int]], raw_total: int) -> None:
    print("\n" + "=" * 70)
    print("TABEL RINGKASAN TAHAPAN PREPROCESSING")
    print("=" * 70)

    print(f"{'Langkah':<8}{'Proses':<38}{'Dihapus':>10}{'Sisa':>10}")
    print("-" * 66)
    print(f"{'Awal':<8}{'Dataset mentah':<38}{'-':>10}{raw_total:>10}")

    total_dihapus = 0
    for i, (nama, before, after) in enumerate(funnel, start=1):
        dihapus = before - after
        total_dihapus += dihapus
        print(f"{i:<8}{nama:<38}{dihapus:>10}{after:>10}")

    print("-" * 66)
    final_after = funnel[-1][2]
    print(f"{'Akhir':<8}{'':<38}{total_dihapus:>10}{final_after:>10}")


def cetak_tabel_4_4(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("TABEL PERBANDINGAN KATEGORI SEBELUM & SESUDAH")
    print("=" * 60)

    vc_raw = df_raw["category"].value_counts()
    vc_clean = df_clean["category"].value_counts()

    print(f"{'No':<4}{'Kategori':<22}{'Sebelum':>10}{'Sesudah':>10}{'Dihapus':>10}")
    print("-" * 56)

    total_raw = total_clean = total_dihapus = 0
    for i, (cat, sebelum) in enumerate(vc_raw.items(), start=1):
        sesudah = int(vc_clean.get(cat, 0))
        dihapus = sebelum - sesudah
        total_raw += sebelum
        total_clean += sesudah
        total_dihapus += dihapus
        print(f"{i:<4}{cat:<22}{sebelum:>10}{sesudah:>10}{dihapus:>10}")

    print("-" * 56)
    print(f"{'':<4}{'Total':<22}{total_raw:>10}{total_clean:>10}{total_dihapus:>10}")


def _format_rupiah_singkat(nilai: float, _pos=None) -> str:
    """Format angka jadi Rupiah singkat yang mudah dibaca orang awam,
    misal 500000 -> 'Rp500rb', 1500000 -> 'Rp1,5jt'."""
    if nilai >= 1_000_000:
        juta = round(nilai / 1_000_000, 1)
        if juta == int(juta):
            return f"Rp{int(juta)}jt"
        return f"Rp{juta:.1f}jt".replace(".", ",")
    if nilai >= 1_000:
        ribu = nilai / 1_000
        if ribu == int(ribu):
            return f"Rp{int(ribu)}rb"
        return f"Rp{ribu:.0f}rb"
    return f"Rp{int(nilai)}"


def buat_visualisasi_price(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Simpan histogram distribusi price sebagai file PNG tersendiri,
    dengan sumbu X berformat Rupiah singkat (Rp500rb, Rp1jt, dst) agar
    mudah dibaca orang awam, bukan notasi ilmiah (1e6)."""

    price = df["price"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(price, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title("Distribusi Harga Produk (Price)")
    ax.set_xlabel("Harga")
    ax.set_ylabel("Jumlah Produk")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_rupiah_singkat))
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    out_path = f"{output_dir}/distribusi_price.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[INFO] Visualisasi price disimpan ke: {out_path}")
    plt.show()


def buat_visualisasi_rating(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Simpan bar chart distribusi rating sebagai file PNG tersendiri.
    Rating menggunakan bar chart (bukan histogram) karena nilainya
    diskrit/terbatas pada rentang 0-5 dengan interval yang cenderung tetap,
    sehingga bar chart per nilai lebih representatif dibanding histogram bin.
    """

    rating = df["rating"].dropna()

    # dibulatkan ke 1 desimal agar tidak terlalu banyak bar
    rating_rounded = rating.round(1)
    rating_counts = rating_rounded.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color="#DD8452",
        edgecolor="white",
    )
    ax.set_title("Distribusi Rating Produk")
    ax.set_xlabel("Rating (0-5)")
    ax.set_ylabel("Jumlah Produk")
    ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    out_path = f"{output_dir}/distribusi_rating.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n[INFO] Visualisasi rating disimpan ke: {out_path}")
    plt.show()


def main() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        df = load_data(INPUT_PATH)
    raw_total = len(df)
    df_raw_copy = df.copy()  # simpan salinan data mentah untuk Tabel 4.4

    # jalankan seluruh pipeline persis seperti main() di preprocessing.py,
    # tapi TANPA print verbose per-langkah (cetak_tabel_4_x dipanggil
    # terpisah di bawah, setelah pipeline selesai)
    cetak_ringkasan_kualitas_data(df)
    cetak_statistik_numerik(df)
    cetak_missing_value_lengkap(df)
    cetak_jumlah_brand(df)
    cetak_tabel_4_2(df)
    buat_visualisasi_price(df)
    buat_visualisasi_rating(df)

    funnel: list[tuple[str, int, int]] = []

    with contextlib.redirect_stdout(io.StringIO()):
        n0 = len(df)
        df = drop_noise_rows(df)
        funnel.append(("Penghapusan baris noise", n0, len(df)))

        n1 = len(df)
        df = drop_missing_rating(df)
        funnel.append(("Penghapusan produk tanpa rating", n1, len(df)))

        n2 = len(df)
        df = clean_description(df)
        funnel.append(("Pembersihan deskripsi tidak valid", n2, len(df)))

        n3 = len(df)
        df = normalize_text_columns(df)
        funnel.append(("Normalisasi teks", n3, len(df)))

        n4 = len(df)
        df = drop_variant_numbers(df)
        funnel.append(("Penghapusan varian bernomor", n4, len(df)))

        n5 = len(df)
        df = final_check(df)
        funnel.append(("Penghapusan duplikat", n5, len(df)))

    cetak_tabel_4_3(funnel, raw_total)
    cetak_tabel_4_4(df_raw_copy, df)


if __name__ == "__main__":
    main()