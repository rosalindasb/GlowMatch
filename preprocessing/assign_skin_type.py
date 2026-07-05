"""
Assign skin type ke setiap produk berdasarkan keyword
di kolom product_name dan description.

Strategi:
- Cek keyword per skin type di teks gabungan (product_name + description)
- Khusus "dry": ada pengecekan konteks negatif (negasi)
- Produk oily+dry otomatis ditambah label "combination"
- Produk yang tidak ketangkap keyword → "all skin types"

Output: data/sociolla_skincare_labeled.csv

Catatan tambahan (versi laporan):
- Proses assign dijalankan lewat assign_skin_type_debug() yang selain
  mengembalikan label akhir, juga merekam keyword mana yang match, apakah
  negasi "dry" terpicu, dan apakah "combination" ditambahkan otomatis.
- Info debug ini HANYA dipakai untuk laporan di terminal — kolom yang
  disimpan ke CSV tetap sama seperti sebelumnya (skin_type saja).
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
    "sensitive": [
        "sensitive skin", "kulit sensitif", "sensitive", "sensitif",
        "fragrance free", "fragrance-free", "hypoallergenic",
        "calming", "soothing", "menenangkan", "gentle", "lembut",
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

ORDER = ["oily", "dry", "sensitive", "combination", "normal", "acne"]

# "acne" adalah concern (masalah kulit), bukan tipe kulit — jadi tidak dihitung
# dalam aturan konsolidasi ini. Kalau sebuah produk match >= CONSOLIDATION_THRESHOLD
# dari 5 tipe kulit murni di bawah ini, produk itu dianggap generik/cocok untuk
# semua tipe kulit, dan label-nya disederhanakan jadi "all skin types".
PURE_SKIN_TYPES = ["oily", "dry", "sensitive", "combination", "normal"]
CONSOLIDATION_THRESHOLD = 4


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
    """Ambil potongan teks di sekitar keyword pertama kali muncul, untuk konteks."""
    idx = text.find(keyword)
    if idx == -1:
        return text[:60]
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def has_dry_negation(text: str) -> re.Match | None:
    for pat in DRY_NEGATION_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m
    return None


# ==============================================================
# CORE LOGIC — dengan versi debug untuk pelaporan
# ==============================================================
def assign_skin_type_debug(product_name: str, description: str) -> dict:
    """
    Sama seperti assign_skin_type, tapi mengembalikan info tambahan:
    - label            : hasil akhir skin_type (string, dipisah koma)
    - matched_types    : list skin type yang match (SEBELUM konsolidasi)
    - keyword_hits     : {skin_type: keyword_yang_match}
    - dry_negated      : True kalau keyword 'dry' ketemu TAPI dibatalkan oleh negasi
    - dry_negation_info: (keyword_dry, pattern_negasi, teks) kalau dry_negated True
    - combination_auto : True kalau 'combination' ditambahkan otomatis (oily+dry)
    - consolidated      : True kalau label disederhanakan jadi 'all skin types'
                           karena match >= CONSOLIDATION_THRESHOLD tipe kulit murni
    - label_before_consolidation : label yang akan dihasilkan SEBELUM aturan
                           konsolidasi diterapkan (buat perbandingan di laporan)
    """
    text = (str(product_name).lower() + " " + str(description).lower())
    matched: list[str] = []
    keyword_hits: dict[str, str] = {}
    dry_negated = False
    dry_negation_info = None

    for skin_type, keywords in SKIN_TYPE_KEYWORDS.items():
        if skin_type == "dry":
            for kw in keywords:
                if kw in text:
                    neg_match = has_dry_negation(text)
                    if neg_match is None:
                        matched.append(skin_type)
                        keyword_hits["dry"] = kw
                    else:
                        dry_negated = True
                        dry_negation_info = (kw, neg_match.group(0), text)
                    break
        else:
            for kw in keywords:
                if kw in text:
                    matched.append(skin_type)
                    keyword_hits[skin_type] = kw
                    break

    combination_auto = False
    if "oily" in matched and "dry" in matched and "combination" not in matched:
        matched.append("combination")
        combination_auto = True

    label_before_consolidation = (
        ",".join([s for s in ORDER if s in matched]) if matched else "all skin types"
    )

    # ---- aturan konsolidasi: terlalu banyak tipe murni yang match → generik ----
    pure_matched = [t for t in matched if t in PURE_SKIN_TYPES]
    consolidated = len(pure_matched) >= CONSOLIDATION_THRESHOLD
    if consolidated:
        parts = ["all skin types"]
        if "acne" in matched:
            parts.append("acne")
        label = ",".join(parts)
    else:
        label = label_before_consolidation

    return {
        "label": label,
        "matched_types": matched,
        "keyword_hits": keyword_hits,
        "dry_negated": dry_negated,
        "dry_negation_info": dry_negation_info,
        "combination_auto": combination_auto,
        "consolidated": consolidated,
        "label_before_consolidation": label_before_consolidation,
    }


def assign_skin_type(product_name: str, description: str) -> str:
    """Versi ringkas (dipakai kalau hanya butuh label, tanpa debug info)."""
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
    print(f"Kolom skin_type sebelum assign : kosong untuk semua baris "
          f"({df['skin_type'].isna().sum() if 'skin_type' in df.columns else len(df)} dari {len(df)})")

    # ---- jalankan assign dengan debug info ----
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

    # ---- distribusi per skin type (multi-label, produk bisa dihitung >1x) ----
    subsection("Distribusi skin type (satu produk bisa punya >1 label)")
    all_types = []
    for st in df["skin_type"]:
        all_types.extend(str(st).split(","))
    type_counts = pd.Series(all_types).value_counts()
    for skin, count in type_counts.items():
        print(f"  {skin:<20}: {count:>5} produk  ({pct(count, total)})")

    # ---- distribusi jumlah label per produk ----
    subsection("Distribusi jumlah label per produk")
    n_labels = df["skin_type"].apply(lambda s: len(str(s).split(",")))
    label_count_dist = n_labels.value_counts().sort_index()
    for n, count in label_count_dist.items():
        ket = "label" if n > 1 else "label (single)"
        print(f"  {n} {ket:<15}: {count:>5} produk  ({pct(count, total)})")

    # ---- top kombinasi label ----
    subsection("Top 10 kombinasi label paling umum")
    combo_counts = df["skin_type"].value_counts().head(10)
    for combo, count in combo_counts.items():
        print(f"  {combo:<35}: {count:>5} produk  ({pct(count, total)})")

    # ---- fallback all skin types ----
    fallback = (df["skin_type"] == "all skin types").sum()
    print(f"\nProduk fallback 'all skin types' (tidak match keyword apapun) : "
          f"{fallback}  ({pct(fallback, total)})")

    # ---- keyword paling sering trigger per skin type ----
    subsection("Keyword paling sering trigger per skin type")
    keyword_counter: dict[str, dict[str, int]] = {t: {} for t in SKIN_TYPE_KEYWORDS}
    for r in debug_results:
        for skin_type, kw in r["keyword_hits"].items():
            keyword_counter[skin_type][kw] = keyword_counter[skin_type].get(kw, 0) + 1
    for skin_type in ORDER:
        counts = keyword_counter.get(skin_type, {})
        if not counts:
            continue
        top_kw = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join([f"'{k}' ({v}x)" for k, v in top_kw])
        print(f"  {skin_type:<12}: {top_str}")

    # ---- bukti mekanisme negasi "dry" ----
    section("STEP 3 — BUKTI MEKANISME NEGASI 'DRY'")
    negated = [
        (row["product_name"], r["dry_negation_info"])
        for (_, row), r in zip(df.iterrows(), debug_results)
        if r["dry_negated"]
    ]
    print(f"Jumlah produk yang mengandung keyword 'dry' TAPI dibatalkan oleh negasi: {len(negated)}")
    if negated:
        examples = []
        for name, info in negated:
            kw, pattern_matched, text = info
            pattern_display = pattern_matched if len(pattern_matched) <= 60 else pattern_matched[:57] + "..."
            examples.append(
                f"{name}\n"
                f"      keyword ditemukan : '{kw}'  |  pattern negasi match: '{pattern_display}'\n"
                f"      konteks           : \"{snippet(text, kw)}\""
            )
        show_examples(examples, "produk dengan keyword 'dry' yang dinegasikan", max_items=3)
    else:
        print("  Tidak ada produk yang kena kasus negasi pada dataset ini.")

    # ---- bukti auto-label combination ----
    section("STEP 4 — BUKTI AUTO-LABEL 'COMBINATION' (oily + dry)")
    combo_auto = [
        row["product_name"]
        for (_, row), r in zip(df.iterrows(), debug_results)
        if r["combination_auto"]
    ]
    print(f"Jumlah produk yang otomatis ditambah label 'combination' karena oily+dry: {len(combo_auto)}")
    show_examples(combo_auto, "produk dengan auto-label 'combination'", max_items=5)

    # ---- bukti aturan konsolidasi ke "all skin types" ----
    section(f"STEP 5 — BUKTI KONSOLIDASI (match ≥ {CONSOLIDATION_THRESHOLD} tipe kulit murni → 'all skin types')")
    consolidated_list = [
        (row["product_name"], r["label_before_consolidation"], r["label"])
        for (_, row), r in zip(df.iterrows(), debug_results)
        if r["consolidated"]
    ]
    print(f"Jumlah produk yang label-nya disederhanakan jadi 'all skin types' : {len(consolidated_list)}")
    if consolidated_list:
        examples = [f"{name}\n      sebelum : {before}\n      sesudah : {after}"
                    for name, before, after in consolidated_list]
        show_examples(examples, "produk yang dikonsolidasi", max_items=3)
    else:
        print("  Tidak ada produk yang kena aturan konsolidasi pada dataset ini.")

    # ---- contoh random hasil assign, dilengkapi keyword yang match ----
    section("STEP 6 — CONTOH HASIL ASSIGN (RANDOM SAMPLE)")
    sample_n = min(10, len(df))
    sample_idx = df.sample(sample_n, random_state=42).index
    for idx in sample_idx:
        row = df.loc[idx]
        r = debug_results[idx]
        print(f"  [{row['category']}] {str(row['product_name'])[:55]}")
        print(f"    → skin_type     : {row['skin_type']}")
        if r["keyword_hits"]:
            hits_str = ", ".join([f"{t}←'{k}'" for t, k in r["keyword_hits"].items()])
            print(f"    → keyword match : {hits_str}")
        else:
            print(f"    → keyword match : (tidak ada, fallback 'all skin types')")

    elapsed = time.time() - start_time
    section("SELESAI")
    print(f"Waktu eksekusi assign_skin_type : {elapsed:.2f} detik")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Disimpan ke                      : {OUTPUT_PATH}")
    print(f"Total produk berlabel             : {len(df)}")


if __name__ == "__main__":
    main()