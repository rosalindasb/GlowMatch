# Coba akses produk dari brand Abib di kategori toner
import requests, json

url = "https://catalog-api1.sociolla.com/v3/products"
headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
params = {
    "filter": json.dumps({"categories.slug": "182-toner", "is_active_in_sociolla": True}, separators=(',',':')),
    "limit": 20, "skip": 0,
}
r = requests.get(url, params=params, headers=headers)
print("Status:", r.status_code)
print("Keys:", [k for k in r.json().keys() if k != "data"])
if r.json().get("data"):
    print("Nama produk:", r.json()["data"][0].get("name"))