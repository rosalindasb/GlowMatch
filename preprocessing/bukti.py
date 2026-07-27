"""
Script bukti terminal untuk Tabel 4.5 (Distribusi Hasil Ekstraksi Skin Type)
dan breakdown mekanisme fallback vs konsolidasi.

PENTING: Script ini TIDAK menulis ulang logika ekstraksi skin type. Fungsi
assign_skin_type_debug() di-import langsung dari assign_skin_type.py,
sehingga angka yang dihasilkan dijamin identik dengan hasil pipeline utama
(single source of truth).

Cara pakai:
    python generate_bukti_tabel_4_5.py

Jalankan file ini di folder yang sama dengan assign_skin_type.py dan
folder data/, lalu screenshot output yang muncul sebagai bukti Tabel 4.5.
"""

import contextlib
import io

import pandas as pd

from assign_skin_type import (
    INPUT_PATH,
    ORDER,
    assign_skin_type_debug,
)


def cetak_tabel_4_5(df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("TABEL DISTRIBUSI HASIL EKSTRAKSI SKIN TYPE")
    print("=" * 55)

    total = len(df)
    all_types: list[str] = []
    for st in df["skin_type"]:
        all_types.extend(x.strip() for x in str(st).split(","))

    counts = pd.Series(all_types).value_counts()

    print(f"{'No':<4}{'Skin Type':<18}{'Jumlah Produk':>15}{'Persentase':>14}")
    print("-" * 51)
    for i, (skin, n) in enumerate(counts.items(), start=1):
        print(f"{i:<4}{skin.title():<18}{n:>15}{n/total*100:>13.1f}%")

    print("\nCatatan: satu produk dapat memiliki lebih dari satu label,")
    print("sehingga total persentase melebihi 100%.")


def cetak_breakdown_all_skin_types(df: pd.DataFrame, debug_results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("BREAKDOWN 'ALL SKIN TYPES' — FALLBACK vs KONSOLIDASI")
    print("=" * 60)

    total = len(df)
    n_fallback = sum(
        1 for r in debug_results
        if not r["matched_types"] and not r["consolidated"]
    )
    n_consolidated = sum(1 for r in debug_results if r["consolidated"])
    n_total_all_skin = (df["skin_type"].str.contains("all skin types")).sum()

    print(f"Fallback murni (tidak ada keyword match sama sekali) : {n_fallback}")
    print(f"Hasil konsolidasi (match >= 4 dari 5 tipe murni)      : {n_consolidated}")
    print(f"Total mengandung label 'all skin types'               : {n_total_all_skin}")

    print("\nContoh kasus konsolidasi:")
    contoh = 0
    for (_, row), r in zip(df.iterrows(), debug_results):
        if r["consolidated"] and contoh < 3:
            print(f"  {row['product_name']}")
            print(f"    sebelum : {r['label_before_consolidation']}")
            print(f"    sesudah : {r['label']}")
            contoh += 1


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    debug_results = [
        assign_skin_type_debug(
            row["product_name"],
            row["description"] if pd.notna(row["description"]) else ""
        )
        for _, row in df.iterrows()
    ]
    df["skin_type"] = [r["label"] for r in debug_results]

    cetak_tabel_4_5(df)
    cetak_breakdown_all_skin_types(df, debug_results)


if __name__ == "__main__":
    main()