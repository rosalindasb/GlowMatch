"""
Script bukti terminal untuk statistik matriks TF-IDF (shape, sparsity)
dan Tabel 4.7 (top term dengan akumulasi bobot tertinggi).

PENTING: Script ini TIDAK menulis ulang logika pembangunan model. Fungsi
build_model() di-import langsung dari recommender.py, sehingga vectorizer
dan matriks yang dihasilkan dijamin identik dengan model yang benar-benar
dipakai sistem (single source of truth).

Cara pakai:
    python generate_bukti_tabel_4_7.py

Jalankan file ini di folder yang sama dengan recommender.py dan folder
data/, lalu screenshot output yang muncul.
"""

import contextlib
import io

import numpy as np
import pandas as pd

from recommender import DATA_PATH, build_model


def cetak_statistik_matriks(tfidf_matrix, vectorizer) -> None:
    print("\n" + "=" * 55)
    print("STATISTIK MATRIKS TF-IDF")
    print("=" * 55)

    n_docs, n_terms = tfidf_matrix.shape
    nnz = tfidf_matrix.nnz
    sparsity = 1 - (nnz / (n_docs * n_terms))

    print(f"Matrix shape       : {tfidf_matrix.shape}  ({n_docs} dokumen x {n_terms} term)")
    print(f"Jumlah vocabulary  : {len(vectorizer.vocabulary_)}")
    print(f"Non-zero entries   : {nnz}")
    print(f"Sparsity           : {sparsity*100:.2f}%")


def cetak_tabel_4_7(tfidf_matrix, vectorizer, top_n: int = 10) -> None:
    print("\n" + "=" * 55)
    print(f"TABEL TOP {top_n} TERM AKUMULASI BOBOT TF-IDF TERTINGGI")
    print("=" * 55)

    term_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    feature_names = vectorizer.get_feature_names_out()
    top_idx = term_scores.argsort()[::-1][:top_n]

    print(f"{'No':<4}{'Term':<20}{'Akumulasi Bobot':>18}")
    print("-" * 42)
    for i, idx in enumerate(top_idx, start=1):
        print(f"{i:<4}{feature_names[idx]:<20}{term_scores[idx]:>18.2f}")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    with contextlib.redirect_stdout(io.StringIO()):
        vectorizer, tfidf_matrix = build_model(df)

    cetak_statistik_matriks(tfidf_matrix, vectorizer)
    cetak_tabel_4_7(tfidf_matrix, vectorizer)


if __name__ == "__main__":
    main()