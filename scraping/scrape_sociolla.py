"""
Scrape produk skincare Sociolla via catalog-api v3.

CHANGELOG:
- [FIX] Base URL: catalog-api.sociolla.com → catalog-api1.sociolla.com/v3
- [FIX] Endpoint: /brands/distinct/products → /v3/products (endpoint brand salah)
- [FIX] Filter pakai categories.slug per kategori
- [FIX] is_pack pakai truthy check — API return integer 0/1
- [FIX] json.dumps pakai separators=(',',':') — tanpa spasi, fix 422 error
- [FIX] Default limit 20 (max API)
- [FIX] Stop condition pakai total dari API
- [ADD] image_url — ambil gambar is_cover pertama dari field images
- [ADD] url_sociolla — link produk di Sociolla (tanpa query string)
- [REMOVE] ingredients — tidak dipakai di model, dihapus dari awal
- [ADD] Debug logging per filter stage
- [ADD] Per-kategori scraping dengan slug mapping lengkap
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import time
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://catalog-api1.sociolla.com/v3/products"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

CATEGORY_SLUGS: Dict[str, str] = {
    "6324-sealing-treatment-gel-cream": "Acne Treatment",
    "182-toner":                         "Toner",
    "2290-essence":                      "Essence",
    "190-face-serum":                    "Face Serum",
    "2291-booster":                      "Essence",
    "2292-ampoule":                      "Face Serum",
    "2277-face-wash":                    "Cleanser",
    "278-cleansing-oil":                 "Cleanser",
    "2278-cleansing-balm":               "Cleanser",
    "2282-cleansing-gel":                "Cleanser",
    "2293-scrub-and-exfoliator":         "Cleanser",
    "2285-micellar-water":               "Cleanser",
    "2294-peeling":                      "Cleanser",
    "2303-mask":                         "Face Mask",
    "178-moisturizer":                   "Moisturizer",
    "2301-sun-care":                     "Sun Care / Sunscreen",
}


def _has_non_latin_script(text: str) -> bool:
    for char in text:
        cp = ord(char)
        if 0x1E00 <= cp <= 0x1EFF: return True
        if 0x0E00 <= cp <= 0x0E7F: return True
        if 0x0600 <= cp <= 0x06FF: return True
        if 0x4E00 <= cp <= 0x9FFF: return True
        if 0xAC00 <= cp <= 0xD7AF: return True
        if 0x3040 <= cp <= 0x30FF: return True
    return False


def product_name_excluded(raw_name: str) -> bool:
    n = raw_name.strip()
    if not n:
        return True
    if _has_non_latin_script(n):
        return True
    lower = n.lower()
    bundle_patterns = (
        r"\b(bundle|gift set|special set|starter set|travel set|value set|"
        r"duo set|trio set|birthday bundle|shopee bundle|lazada bundle|"
        r"paket|gwp|gratis|lebih hemat|b1g1|buy 1 get|clearance|"
        r"not used|offline|mkp|ol|combo|routine|regimen|kit)\b"
    )
    if re.search(bundle_patterns, lower): return True
    if re.search(r"\bfree\b", lower) and re.search(r"\d", lower): return True
    if re.search(r"\[.{1,30}\]", lower): return True
    if re.search(r"\bset\b", lower) and not re.search(
        r"\b(offset|sunset|mindset|closet|basket|cosset|reset|asset)\b", lower
    ): return True
    if lower.startswith("bộ ") or " bộ " in lower: return True
    return False


def html_to_plain(raw: Optional[str]) -> str:
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
    else:
        text = str(raw)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skin_type_from_api(product: Dict) -> str:
    tags = product.get("tags") or []
    if isinstance(tags, list):
        skin_tags = []
        for tag in tags:
            name = (tag.get("name") if isinstance(tag, dict) else str(tag)).lower()
            if any(x in name for x in ["oily", "dry", "sensitive", "combination", "normal", "acne"]):
                skin_tags.append(name)
        if skin_tags:
            return ",".join(skin_tags)
    for attr in (product.get("attributes") or []):
        if isinstance(attr, dict):
            attr_name = (attr.get("name") or "").lower()
            if "skin" in attr_name or "type" in attr_name or "suitable" in attr_name:
                values = attr.get("values") or []
                if values:
                    return ",".join(str(v) for v in values)
    suitable = product.get("suitable_for") or product.get("skin_type") or ""
    return str(suitable) if suitable else ""


def extract_price(product: Dict) -> Optional[float]:
    dc = product.get("default_combination") or {}
    p = dc.get("price")
    if p is not None:
        return float(p)
    mp = product.get("min_price")
    if mp is not None:
        return float(mp)
    combos = product.get("combinations") or []
    prices = [c.get("price") for c in combos if isinstance(c, dict) and c.get("price") is not None]
    return float(min(prices)) if prices else None


def extract_image_url(product: Dict) -> str:
    images = product.get("images") or []
    if not images:
        return ""
    for img in images:
        if isinstance(img, dict) and img.get("is_cover") is True:
            url = img.get("url", "")
            if url:
                return url
    first = images[0]
    if isinstance(first, dict):
        return first.get("url", "")
    return ""


def extract_sociolla_url(product: Dict) -> str:
    """Ambil URL produk di Sociolla, buang query string ?is_preview=true."""
    url = product.get("url_sociolla") or ""
    if url:
        return url.split("?")[0]
    return ""


_filter_stats: Dict[str, int] = {
    "total": 0,
    "skip_pack": 0,
    "skip_name": 0,
    "skip_desc_bundle": 0,
    "passed": 0,
}


def transform_product(product: Dict, mapped_category: str) -> Optional[Dict]:
    if not isinstance(product, dict):
        return None
    _filter_stats["total"] += 1
    if product.get("is_pack"):
        _filter_stats["skip_pack"] += 1
        return None
    raw_name = (product.get("name") or "").strip()
    if product_name_excluded(raw_name):
        _filter_stats["skip_name"] += 1
        return None
    desc_raw = product.get("description") or ""
    desc_plain = html_to_plain(desc_raw)
    if re.search(
        r"\b(bundle ini|bundle terdiri|product bundle|paket ini|terdiri dari|set mist)\b",
        desc_plain.lower()
    ):
        _filter_stats["skip_desc_bundle"] += 1
        return None
    rating_raw = (product.get("review_stats") or {}).get("average_rating")
    rating = float(rating_raw) if rating_raw else None
    brand = (product.get("brand") or {}).get("name", "")
    _filter_stats["passed"] += 1
    return {
        "product_name":  raw_name,
        "brand":         brand.strip(),
        "category":      mapped_category,
        "price":         extract_price(product),
        "rating":        rating,
        "description":   desc_plain,
        "skin_type":     extract_skin_type_from_api(product),
        "image_url":     extract_image_url(product),
        "url_sociolla":  extract_sociolla_url(product),   # [ADD]
    }


def request_with_retry(params: Dict, retries: int = 3) -> Optional[Dict]:
    for attempt in range(1, retries + 1):
        try:
            res = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            if res.status_code == 404:
                return None
            res.raise_for_status()
            return res.json()
        except Exception as e:
            if attempt == retries:
                print(f"  [ERROR] Gagal setelah {retries} percobaan: {e}")
                return None
            wait = 2.0 * attempt
            print(f"  [WARN] Attempt {attempt} gagal, retry {wait:.1f}s...")
            time.sleep(wait)
    return None


def scrape_category(slug: str, label: str, limit: int, sleep_seconds: float) -> List[Dict]:
    rows: List[Dict] = []
    seen_ids: Set[str] = set()
    skip = 0
    page = 0
    total_api = None
    while True:
        page += 1
        params = {
            "filter": json.dumps({
                "categories.slug": slug,
                "is_active_in_sociolla": True,
            }, separators=(',', ':')),
            "limit": limit,
            "skip": skip,
            "sort": "-thirty_days_total_orders _id",
        }
        payload = request_with_retry(params)
        if payload is None:
            break
        products = payload.get("data", [])
        if not products:
            break
        if total_api is None:
            total_api = payload.get("total") or payload.get("count")
        kept = 0
        for prod in products:
            pid = str(prod.get("_id") or prod.get("id", ""))
            if pid and pid in seen_ids:
                continue
            row = transform_product(prod, label)
            if not row:
                continue
            if pid:
                seen_ids.add(pid)
            rows.append(row)
            kept += 1
        print(f"  [Page {page:03d}] skip={skip} fetched={len(products)} kept={kept} "
              f"subtotal={len(rows)} (API total={total_api or '?'})")
        skip += limit
        if total_api is not None and skip >= total_api:
            break
        if len(products) < limit:
            break
        time.sleep(sleep_seconds)
    return rows


def scrape_all(limit: int, sleep_seconds: float) -> pd.DataFrame:
    for k in _filter_stats:
        _filter_stats[k] = 0
    all_rows: List[Dict] = []
    seen_global: Set[str] = set()
    for slug, label in CATEGORY_SLUGS.items():
        print(f"\n[KATEGORI] {label} ({slug})")
        rows = scrape_category(slug, label, limit, sleep_seconds)
        before = len(rows)
        unique_rows = []
        for r in rows:
            key = f"{r['product_name']}|{r['brand']}"
            if key not in seen_global:
                seen_global.add(key)
                unique_rows.append(r)
        after = len(unique_rows)
        all_rows.extend(unique_rows)
        print(f"  → {before} produk, {before - after} duplikat dibuang, {after} unik ditambahkan")
        print(f"  → Total sejauh ini: {len(all_rows)}")
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["product_name", "brand"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape skincare Sociolla via catalog-api v3.")
    parser.add_argument("--output", default="data/sociolla_skincare_raw.csv")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    df = scrape_all(limit=args.limit, sleep_seconds=args.sleep)
    print("\n[DEBUG] Filter breakdown:")
    print(f"  Total diproses     : {_filter_stats['total']}")
    print(f"  Skip (is_pack)     : {_filter_stats['skip_pack']}")
    print(f"  Skip (nama produk) : {_filter_stats['skip_name']}")
    print(f"  Skip (desc bundle) : {_filter_stats['skip_desc_bundle']}")
    print(f"  Lolos filter       : {_filter_stats['passed']}")
    if df.empty:
        print("\n[INFO] Tidak ada data yang lolos filter.")
        return
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Total produk bersih : {len(df)}")
    print(f"[DONE] Disimpan ke         : {args.output}")
    print("\n[INFO] Distribusi kategori:")
    print(df["category"].value_counts().to_string())
    print(f"\n[INFO] Produk dengan image_url    : {(df['image_url'].astype(str).str.strip() != '').sum()}")
    print(f"[INFO] Produk dengan url_sociolla : {(df['url_sociolla'].astype(str).str.strip() != '').sum()}")
    print(f"[INFO] Produk dengan rating       : {df['rating'].notna().sum()}")


if __name__ == "__main__":
    main()