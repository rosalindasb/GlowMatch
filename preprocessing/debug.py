import pandas as pd
df_raw = pd.read_csv("data/sociolla_skincare_raw.csv")

# cek total produk tanpa rating SEBELUM noise removal
tanpa_rating_awal = df_raw[df_raw['rating'].isna()]
print(len(tanpa_rating_awal))  # harusnya 263

# cek 31 produk noise yang kehapus di step 1
noise_keywords = ["not for sale", "product testing", "sample", "tester", "buy 2 get", "buy 3 get", "b2g1", "b3g1"]
mask_noise = df_raw['product_name'].str.lower().str.contains('|'.join(noise_keywords))
produk_noise = df_raw[mask_noise]
print(len(produk_noise))  # harusnya 31

# cek overlap: berapa dari produk noise yang JUGA gak punya rating
overlap = produk_noise[produk_noise['rating'].isna()]
print(len(overlap))  # ini kuncinya — kalau hasilnya 15, dugaan lu BENAR