"""
Script bukti terminal untuk statistik kolom `content` dan contoh
pembobotan fitur konten (subbab 4.4 Pembangunan Fitur Konten).

PENTING: Script ini TIDAK menulis ulang logika pembangunan fitur konten.
Fungsi build_content() dan clean_text() di-import langsung dari
build_content_features.py, sehingga hasil yang ditampilkan dijamin
identik dengan hasil pipeline utama (single source of truth).

Cara pakai:
    python generate_bukti_tabel_4_6.py

Jalankan file ini di folder yang sama dengan build_content_features.py
dan folder data/, lalu screenshot bagian output yang muncul.
"""

import pandas as pd

from build_content_features import (
    INPUT_PATH,
    WEIGHTS,
    build_content,
    clean_text,
)


def cetak_statistik_content(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("STATISTIK KOLOM CONTENT")
    print("=" * 55)

    total = len(df)
    lengths = df["content"].str.len()

    print(f"Total produk               : {total}")
    print(f"Content kosong             : {(df['content'].str.strip() == '').sum()}")
    print(f"Panjang rata-rata (char)   : {lengths.mean():.0f}")
    print(f"Panjang median (char)      : {lengths.median():.0f}")
    print(f"Panjang minimum (char)     : {lengths.min()}")
    print(f"Panjang maksimum (char)    : {lengths.max()}")

    dup = df[df.duplicated(subset=["content"], keep=False)]
    print(f"\nGrup content identik antar produk berbeda : {dup.groupby('content').ngroups if len(dup) else 0}")
    print(f"Total produk terlibat dalam duplikat      : {len(dup)}")

    all_tokens = set()
    for c in df["content"]:
        all_tokens.update(c.split())
    print(f"\nEstimasi vocabulary (split spasi kasar)   : {len(all_tokens)} token unik")
    print("(vocabulary TF-IDF sebenarnya lebih kecil karena filter min_df/max_df & tokenizer sklearn)")


def cetak_contoh_pembobotan(df: pd.DataFrame, nama_produk: str) -> None:
    print("\n" + "=" * 60)
    print(f"CONTOH PEMBOBOTAN CONTENT — {nama_produk}")
    print("=" * 60)

    row = df[df["product_name"] == nama_produk].iloc[0]
    print(f"Kategori  : {row['category']}")
    print(f"Skin type : {row['skin_type']}")
    print(f"Brand     : {row['brand']}")

    print("\n-- Verifikasi jumlah pengulangan tiap atribut di content --")
    content = row["content"]
    for col in ["category", "skin_type", "product_name", "brand"]:
        raw_val = str(row[col]).replace(",", " ") if col == "skin_type" else row[col]
        cleaned_val = clean_text(raw_val)
        actual_count = content.count(cleaned_val) if cleaned_val else 0
        expected = WEIGHTS[col]
        status = "OK" if actual_count >= expected else "CEK ULANG"
        print(f"  {col:<14}: muncul {actual_count}x di content (target {expected}x)  [{status}]")

    print(f"\nPotongan content (400 karakter pertama):")
    print(f'  "{content[:400]}..."')


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["content"] = df.apply(build_content, axis=1)

    cetak_statistik_content(df)
    cetak_contoh_pembobotan(df, "Licorice Advanced Peeling Gel")


if __name__ == "__main__":
    main()