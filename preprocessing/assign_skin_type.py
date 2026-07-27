"""
Assign skin type ke setiap produk berdasarkan keyword
di kolom product_name dan description.

VERSI REVISI 2 — perbaikan dari audit manual terhadap logika negasi & matching:

FIX BARU DI VERSI INI:
1. BUG check_dry() early-return: versi sebelumnya berhenti (return) begitu
   SINONIM PERTAMA dari daftar "dry" ditemukan di teks, tanpa cek sinonim
   lain dulu. Akibatnya kalau sinonim pertama yang ketemu ternyata ternegasi
   (mis. "tidak membuat kulit kering"), fungsi langsung menyimpulkan
   "dry_negated=True" dan TIDAK PERNAH mengecek sinonim lain yang mungkin
   muncul valid/tidak ternegasi di bagian lain teks yang sama (mis. "cocok
   untuk kulit extremely dry"). Terbukti via reproduksi manual:
     teks: "...tidak membuat kulit kering, cocok untuk kulit extremely dry..."
     versi lama -> (False, 'kulit kering', True)   [SALAH]
     versi fix  -> (True, 'extremely dry', False)  [BENAR]
   Fix: lanjutkan scan SEMUA sinonim dulu; baru simpulkan "negated" di akhir
   kalau memang TIDAK ADA satupun sinonim yang unnegated.
2. GENERALISASI NEGASI KE 'oily': risiko yang sama sebenarnya berlaku
   simetris untuk "oily" — deskripsi seperti "tanpa membuat kulit
   berminyak" (klaim produk TIDAK bikin oily, biasa muncul di produk untuk
   kulit kering) sebelumnya akan salah ke-label 'oily' karena kata
   "berminyak" match tanpa cek negasi sama sekali. Sekarang 'oily' dicek
   dengan mekanisme negasi yang sama seperti 'dry' (fungsi generik
   `check_type_with_negation`), pakai daftar pola negasi sendiri
   (OILY_NEGATION_PATTERNS).
   CATATAN: negasi TIDAK diterapkan ke sensitive/combination/normal/acne.
   Alasan: kata kunci di kategori itu kebanyakan sudah berbentuk klaim
   absolut/frasa pasti (mis. "fragrance free", "tanpa pewangi", "hypoallergenic",
   "anti-acne"), bukan bentuk "membuat kulit jadi X" yang lazim dinegasikan
   di copywriting skincare. Menambah negasi di situ berisiko over-engineer
   tanpa temuan kasus nyata yang mendukung, beda dengan oily/dry yang polanya
   sama-sama "membuat kulit (tidak) berminyak/kering" dan sudah terbukti
   ada di deskripsi produk.
3. WORD-BOUNDARY MATCHING: sebelumnya semua non-dry keyword dicek dengan
   substring polos (`kw in text`), sama seperti pendekatan lama di
   preprocessing.py sebelum diperbaiki di sana. Sekarang seluruh matching
   (termasuk oily & dry) pakai regex `\b...\b`, konsisten dengan pendekatan
   di preprocessing.py, supaya kata kunci pendek tidak salah nempel ke
   tengah kata lain yang tidak terkait (meski untuk kata kunci yang ada di
   sini risikonya kecil, ini soal konsistensi & jaga-jaga ke depan kalau
   kata kunci baru ditambahkan).

PERBAIKAN DARI VERSI SEBELUMNYA (dipertahankan):
- Kamus 'sensitive' dipersempit: kata generik ("lembut", "menenangkan",
  "gentle", "soothing", "calming") dibuang karena muncul di hampir semua
  deskripsi skincare apapun tipenya, sehingga sebelumnya bikin sensitive
  over-triggered (65,9% dari katalog kelabel sensitive, feature jadi
  nyaris tidak diskriminatif padahal skin_type berbobot tinggi di CBF).
- Aturan konsolidasi (match >= N tipe murni -> disederhanakan jadi 'all
  skin types') DIHAPUS. Produk yang match banyak tipe TETAP disimpan apa
  adanya, mis. "oily,dry,sensitive,combination" — bukan diringkas.
- Label 'all skin types' HANYA dipakai untuk fallback murni: produk yang
  TIDAK match satupun dari 6 kata kunci di atas.

Output: data/sociolla_skincare_labeled.csv
"""

from __future__ import annotations
import re
import time
import pandas as pd

INPUT_PATH  = "data/sociolla_skincare_clean.csv"
OUTPUT_PATH = "data/sociolla_skincare_labeled.csv"

SKIN_TYPE_KEYWORDS: dict[str, list[str]] = {
    "oily": [
        "oily skin", "kulit berminyak", "berminyak", "oil control", "oil-control",
        "oily face", "minyak berlebih", "pore minimizing", "minimize pore",
        "mattify", "mattifying", "matte finish", "sebum control", "sebum",
        "mengontrol minyak", "mengurangi minyak",
    ],
    "dry": [
        "dry skin", "kulit kering", "dryskin", "extra dry", "very dry",
        "extremely dry", "moisture barrier", "kulit dehidrasi",
        "skin dehidrasi", "dehydrated skin",
    ],
    # kata generik "lembut", "menenangkan", "gentle", "soothing", "calming"
    # DIBUANG — bukan klaim eksklusif sensitive, muncul di hampir semua
    # produk apapun tipenya.
    "sensitive": [
        "sensitive skin", "kulit sensitif", "sensitive", "sensitif",
        "fragrance free", "fragrance-free", "hypoallergenic",
        "no fragrance", "tanpa pewangi", "skin barrier", "barrier repair",
        "untuk kulit sensitif",
    ],
    "combination": [
        "combination skin", "kulit kombinasi", "combination", "kombinasi",
        "combi skin", "t-zone", "t zone",
    ],
    "normal": [
        "normal skin", "kulit normal", "all skin type",
        "all skin types", "semua jenis kulit",
        "cocok untuk semua", "suitable for all",
        "semua tipe kulit", "semua jenis",
    ],
    "acne": [
        "acne", "jerawat", "pimple", "blemish",
        "breakout", "anti-acne", "anti acne",
        "acne-prone", "acne prone", "komedo",
        "blackhead", "whitehead", "bruntusan",
    ],
}

# Pola negasi HANYA didefinisikan untuk tipe yang polanya "membuat kulit
# (tidak) jadi X" — pola yang lazim muncul di copywriting skincare untuk
# oily & dry. Tipe lain (sensitive/combination/normal/acne) sengaja tidak
# diberi pola negasi, lihat alasan di poin 2 pada docstring di atas.
DRY_NEGATION_PATTERNS = [
    r"tanpa membuat kulit kering",
    r"tanpa membuat.{0,20}kering",
    r"tidak membuat.{0,20}kering",
    r"without (making|leaving).{0,20}dry",
    r"non[- ]drying",
    r"without drying",
    r"tidak.{0,25}kering",
    r"tanpa.{0,25}kering",
    r"mencegah.{0,25}kering",
    r"prevent.{0,25}dry",
    r"avoid.{0,25}dry",
]

OILY_NEGATION_PATTERNS = [
    r"tanpa membuat kulit berminyak",
    r"tanpa membuat.{0,20}berminyak",
    r"tidak membuat.{0,20}berminyak",
    r"without (making|leaving).{0,20}oily",
    r"non[- ]greasy",
    r"without.{0,10}greasy",
    r"tidak.{0,25}berminyak",
    r"tanpa.{0,25}berminyak",
    r"mencegah.{0,25}berminyak",
    r"prevent.{0,25}oily",
    r"avoid.{0,25}oily",
]

# Skin type yang punya mekanisme negasi (pakai check_type_with_negation).
# Skin type di luar daftar ini dicek dengan word-boundary match polos
# (lihat match_keywords_plain), tanpa negation-scope checking.
NEGATION_PATTERNS: dict[str, list[str]] = {
    "dry": DRY_NEGATION_PATTERNS,
    "oily": OILY_NEGATION_PATTERNS,
}

ORDER = ["oily", "dry", "sensitive", "combination", "normal", "acne"]
PURE_SKIN_TYPES = ["oily", "dry", "sensitive", "combination", "normal"]
NEGATION_WINDOW = 40  # karakter, dipakai di negation-scope check

# CATATAN: aturan konsolidasi (>= N tipe murni match -> disederhanakan jadi
# 'all skin types') SENGAJA DIHAPUS. Alasan: threshold-nya selalu jadi
# keputusan heuristik yang gak bisa dipertanggungjawabkan secara objektif
# ("kenapa 4, bukan 3?"), dan yang lebih penting, konsolidasi membuang
# informasi (mis. keyword 'oily' yang beneran match hilang dari label)
# padahal skin_type dipakai sebagai fitur berbobot tinggi di TF-IDF. Produk
# yang match banyak tipe sekarang TETAP disimpan apa adanya, mis.
# "oily,dry,sensitive,combination" -- bukan diringkas jadi "all skin types".
# 'all skin types' sekarang HANYA dipakai untuk fallback murni: produk yang
# TIDAK match satupun dari 6 kata kunci di atas (nggak ada keputusan
# ambang sama sekali, batasnya jelas: ada info vs nggak ada info).


# ==============================================================
# HELPER — print & formatting
# ==============================================================
def pct(part: int, total: int) -> str:
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
    items = list(items)
    if not items:
        return
    print(f"  Contoh {label} (menampilkan {min(len(items), max_items)} dari {len(items)}):")
    for item in items[:max_items]:
        print(f"    - {item}")
    if len(items) > max_items:
        print(f"    ... dan {len(items) - max_items} lainnya")


def snippet(text: str, keyword: str, window: int = 35) -> str:
    idx = text.find(keyword)
    if idx == -1:
        return text[:60]
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


# ==============================================================
# WORD-BOUNDARY KEYWORD MATCHING
# ==============================================================
# FIX (v2): sebelumnya keyword non-dry dicek dengan substring polos
# (`kw in text`). Sekarang semua keyword (termasuk multi-kata seperti
# "oily skin") dicek dengan regex \b...\b, supaya konsisten dengan
# pendekatan di preprocessing.py dan lebih aman terhadap kata kunci pendek
# yang berpotensi nempel di tengah kata lain (mis. kalau suatu saat ada
# kata kunci baru yang pendek ditambahkan).
_KEYWORD_REGEX_CACHE: dict[str, "re.Pattern"] = {}


def _keyword_pattern(kw: str) -> "re.Pattern":
    if kw not in _KEYWORD_REGEX_CACHE:
        _KEYWORD_REGEX_CACHE[kw] = re.compile(r"\b" + re.escape(kw) + r"\b", flags=re.IGNORECASE)
    return _KEYWORD_REGEX_CACHE[kw]


def _find_all_indices(text: str, kw: str) -> list[tuple[int, int]]:
    """Return list of (start, end) span untuk tiap kemunculan kw (word-boundary)."""
    return [m.span() for m in _keyword_pattern(kw).finditer(text)]


def _negation_near(text: str, start: int, end: int, negation_patterns: list[str],
                    window: int = NEGATION_WINDOW):
    """Cek pola negasi HANYA di window lokal sekitar kemunculan kata kunci
    (bukan di seluruh teks), supaya negasi di bagian teks yang tidak
    terkait tidak salah nge-cancel klaim yang valid di bagian lain."""
    win_start = max(0, start - window)
    win_end = min(len(text), end + window)
    local = text[win_start:win_end]
    for pat in negation_patterns:
        if re.search(pat, local):
            return True
    return False


def match_keywords_plain(text: str, keywords: list[str]) -> str | None:
    """Word-boundary match TANPA negation-check. Return sinonim pertama yang
    match, atau None kalau tidak ada yang match sama sekali."""
    for kw in keywords:
        if _keyword_pattern(kw).search(text):
            return kw
    return None


def check_type_with_negation(text: str, skin_type: str):
    """
    Generalisasi dari check_dry() lama: cek SEMUA sinonim untuk suatu skin
    type dulu sebelum menyimpulkan "negated". Kalau MINIMAL SATU kemunculan
    (dari sinonim manapun) tidak ternegasi, dianggap match — tidak berhenti
    di sinonim pertama seperti bug versi lama.

    Return (matched: bool, keyword: str|None, all_occurrences_negated: bool)
    """
    keywords = SKIN_TYPE_KEYWORDS[skin_type]
    negation_patterns = NEGATION_PATTERNS.get(skin_type, [])

    found_negated_kw = None
    for kw in keywords:
        spans = _find_all_indices(text, kw)
        if not spans:
            continue
        any_not_negated = any(
            not _negation_near(text, s, e, negation_patterns) for s, e in spans
        )
        if any_not_negated:
            return True, kw, False
        elif found_negated_kw is None:
            found_negated_kw = kw

    if found_negated_kw:
        return False, found_negated_kw, True
    return False, None, False


# ==============================================================
# CORE LOGIC
# ==============================================================
def assign_skin_type_debug(product_name: str, description: str) -> dict:
    text = (str(product_name).lower() + " " + str(description).lower())
    matched: list[str] = []
    keyword_hits: dict[str, str] = {}
    negated_types: list[str] = []  # tipe yang keyword-nya ketemu tapi SEMUA ternegasi

    for skin_type in SKIN_TYPE_KEYWORDS:
        if skin_type in NEGATION_PATTERNS:
            ok, kw, negated = check_type_with_negation(text, skin_type)
            if ok:
                matched.append(skin_type)
                keyword_hits[skin_type] = kw
            elif negated:
                negated_types.append(skin_type)
        else:
            kw = match_keywords_plain(text, SKIN_TYPE_KEYWORDS[skin_type])
            if kw:
                matched.append(skin_type)
                keyword_hits[skin_type] = kw

    combination_auto = False
    if "oily" in matched and "dry" in matched and "combination" not in matched:
        matched.append("combination")
        combination_auto = True

    # 'all skin types' HANYA untuk fallback murni (tidak match satupun
    # keyword). Produk yang match banyak tipe TETAP disimpan apa adanya,
    # tidak ada lagi penyederhanaan/ambang jumlah tipe.
    if not matched:
        label = "all skin types"
    else:
        label = ",".join([s for s in ORDER if s in matched])

    return {
        "label": label,
        "matched_types": matched,
        "keyword_hits": keyword_hits,
        "negated_types": negated_types,  # dulu cuma "dry_negated: bool", sekarang list (dry & oily)
        "combination_auto": combination_auto,
    }


def assign_skin_type(product_name: str, description: str) -> str:
    return assign_skin_type_debug(product_name, description)["label"]


# ==============================================================
# MAIN
# ==============================================================
def main() -> None:
    import os

    start_time = time.time()

    df = pd.read_csv(INPUT_PATH)
    section("STEP 1 — LOAD DATA")
    print(f"[LOAD] {len(df)} produk dari '{INPUT_PATH}'")

    debug_results = [
        assign_skin_type_debug(
            row["product_name"],
            row["description"] if pd.notna(row["description"]) else ""
        )
        for _, row in df.iterrows()
    ]
    df["skin_type"] = [r["label"] for r in debug_results]

    section("STEP 2 — HASIL ASSIGN SKIN TYPE")
    total = len(df)
    print(f"Total produk di-assign : {total}")

    subsection("Distribusi skin type (satu produk bisa punya >1 label)")
    all_types = []
    for st in df["skin_type"]:
        all_types.extend(str(st).split(","))
    type_counts = pd.Series(all_types).value_counts()
    for skin, count in type_counts.items():
        print(f"  {skin:<20}: {count:>5} produk  ({pct(count, total)})")

    # Distribusi jumlah tipe murni yang match per produk -- buat narasi Bab 4
    # soal seberapa umum produk match banyak tipe sekaligus (tanpa ada aturan
    # konsolidasi/ambang lagi, ini murni informasi deskriptif).
    section("STEP 3 — DISTRIBUSI JUMLAH TIPE MURNI YANG MATCH PER PRODUK")
    pure_counts = {}
    for r in debug_results:
        n = len([t for t in r["matched_types"] if t in PURE_SKIN_TYPES])
        pure_counts[n] = pure_counts.get(n, 0) + 1
    for n in sorted(pure_counts):
        print(f"  match {n} tipe murni : {pure_counts[n]:>5} produk  ({pct(pure_counts[n], total)})")

    fallback_n = int((df["skin_type"] == "all skin types").sum())
    section("STEP 4 — PRODUK FALLBACK 'ALL SKIN TYPES' (tidak match keyword apapun)")
    print(f"  Jumlah produk fallback : {fallback_n}  ({pct(fallback_n, total)})")

    # bukti negasi dry & oily (sekarang lokal per window, generalisasi v2)
    section("STEP 5 — BUKTI NEGASI (LOKAL, dry & oily)")
    for t in ["dry", "oily"]:
        negated = [
            row["product_name"] for (_, row), r in zip(df.iterrows(), debug_results)
            if t in r["negated_types"]
        ]
        print(f"\n  Tipe '{t}': {len(negated)} produk dengan SEMUA kemunculan keyword ternegasi")
        show_examples(negated, f"produk dengan '{t}' ternegasi", max_items=5)

    # bukti auto combination
    section("STEP 6 — BUKTI AUTO-LABEL 'COMBINATION'")
    combo_auto = [row["product_name"] for (_, row), r in zip(df.iterrows(), debug_results) if r["combination_auto"]]
    print(f"Jumlah produk auto-label combination: {len(combo_auto)}")
    show_examples(combo_auto, "produk auto-label combination", max_items=5)

    elapsed = time.time() - start_time
    section("SELESAI")
    print(f"Waktu eksekusi : {elapsed:.2f} detik")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Disimpan ke     : {OUTPUT_PATH}")
    print(f"Total berlabel  : {len(df)}")


if __name__ == "__main__":
    main()