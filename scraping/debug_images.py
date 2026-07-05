import requests, json

url = 'https://catalog-api1.sociolla.com/v3/products'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'application/json'}
params = {
    'filter': json.dumps({'categories.slug': '182-toner', 'is_active_in_sociolla': True}, separators=(',',':')),
    'limit': 5, 'skip': 0,
}
r = requests.get(url, params=params, headers=headers)
data = r.json()

for prod in data['data'][:3]:
    print(f"Produk: {prod.get('name')}")
    images = prod.get('images') or []
    print(f"  Jumlah gambar: {len(images)}")
    for i, img in enumerate(images):
        print(f"  [{i}] {img}")
    print()