import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("model/tfidf_matrix.pkl", "rb") as f: mat = pickle.load(f)
with open("model/products_df.pkl",  "rb") as f: df  = pickle.load(f)

# Hitung semua similarity sekaligus
sim_matrix = (mat @ mat.T).toarray()
np.fill_diagonal(sim_matrix, 0)  # exclude self-similarity

# Ambil top 10 pasangan tertinggi
idx = np.unravel_index(np.argsort(sim_matrix, axis=None)[-10:], sim_matrix.shape)
pairs = list(zip(idx[0], idx[1]))[::-1]

print("Top 10 pasangan produk dengan similarity tertinggi:\n")
for i, (a, b) in enumerate(pairs, 1):
    if a < b:  # hindari duplikat
        print(f"#{i}")
        print(f"  Produk A : {df.loc[a, 'product_name']} ({df.loc[a, 'brand']})")
        print(f"  Produk B : {df.loc[b, 'product_name']} ({df.loc[b, 'brand']})")
        print(f"  Similarity: {sim_matrix[a,b]*100:.2f}%")
        print(f"  Kategori : {df.loc[a, 'category']} | Skin type: {df.loc[a, 'skin_type']}")
        print()