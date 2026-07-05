"""
Glowmatch - Halaman Rekomendasi
"""
import pickle
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Rekomendasi · Glowmatch",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,700;1,400;1,500&family=Jost:wght@300;400;500;600&display=swap');

:root {
    --rose:   #D4848E;
    --blush:  #EDB8C0;
    --petal:  #FAE8EC;
    --cream:  #FDF7F2;
    --warm:   #FFFAF7;
    --mauve:  #B5687A;
    --deep:   #6B3A47;
    --text:   #3D2430;
    --muted:  #9A6B78;
    --border: #EDD5DA;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif !important;
    color: var(--text); background: var(--cream);
}
.main { background: var(--cream) !important; }
.block-container { padding: 0 2.5rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── TOPBAR — full width, nutup padding block-container ── */
.topbar {
    background: rgba(253,247,242,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    width: calc(100% + 5rem);
    margin-left: -2.5rem;
    margin-right: -2.5rem;
    margin-top: -1rem;
    padding: 1.5rem 2.5rem 1rem 2.5rem;
    margin-bottom: 1.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.topbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; font-weight: 700; color: var(--deep);
}
.topbar-logo em { font-style: italic; color: var(--rose); }

.topbar-home {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: linear-gradient(135deg, var(--rose), var(--mauve));
    color: white !important; text-decoration: none !important; border-bottom: none !important;
    border-radius: 50px; padding: 0.6rem 1.5rem;
    font-family: 'Jost', sans-serif; font-size: 0.9rem; font-weight: 600;
    box-shadow: 0 4px 14px rgba(180,100,120,0.28);
    transition: all 0.2s ease; white-space: nowrap;
}
.topbar-home:hover {
    opacity: 0.88; transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(180,100,120,0.38);
    color: white !important;
}

.pg-title { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: var(--deep); letter-spacing: -0.3px; margin-bottom: 0.2rem; }
.pg-sub { font-size: 0.88rem; color: var(--muted); font-weight: 300; margin-bottom: 1.8rem; }

.filter-wrap {
    background: var(--warm); border: 1.5px solid var(--border);
    border-radius: 20px; padding: 0.9rem 1.8rem; margin-bottom: 1rem;
    box-shadow: 0 2px 14px rgba(180,100,120,0.06);
    display: flex; align-items: center;
}
.filter-title { font-size: 0.9rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--muted); margin: 0; }

.stSelectbox > div > div {
    background: var(--cream) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; font-family: 'Jost', sans-serif !important; font-size: 0.86rem !important;
}

.stMultiSelect > label { font-size: 0.86rem !important; }
.stMultiSelect [data-baseweb="select"] > div:first-child {
    background: var(--cream) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.86rem !important;
    min-height: 42px !important;
}
.stMultiSelect [data-baseweb="select"] > div:first-child:hover {
    border-color: var(--blush) !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: var(--petal) !important;
    border: 1px solid var(--blush) !important;
    border-radius: 50px !important;
    color: var(--mauve) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.76rem !important;
}
.stMultiSelect span[data-baseweb="tag"] span[role="presentation"] {
    color: var(--mauve) !important;
}

.result-count { font-size: 0.85rem; color: var(--muted); margin-bottom: 1.2rem; }
.result-count strong { color: var(--mauve); font-weight: 600; }

.pcard {
    background: var(--warm); border: 1.5px solid var(--border);
    border-radius: 20px; overflow: hidden; transition: all 0.22s ease;
    box-shadow: 0 2px 10px rgba(180,100,120,0.06);
    display: flex; flex-direction: column;
    height: 395px; margin-bottom: 1rem;
}
.pcard:hover {
    border-color: var(--rose); box-shadow: 0 8px 28px rgba(180,100,120,0.15);
    transform: translateY(-3px);
}
.pcard-img {
    width: 100%; height: 150px; background: var(--petal);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; flex-shrink: 0;
}
.pcard-img img { width: 100%; height: 100%; object-fit: contain; padding: 6px; mix-blend-mode: multiply; }
.pcard-body { padding: 0.85rem 1rem 0; flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.pcard-cat { font-size: 0.6rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--rose); margin-bottom: 0.2rem; }
.pcard-name {
    font-family: 'Playfair Display', serif; font-size: 0.9rem; font-weight: 500;
    color: var(--deep); line-height: 1.3; margin-bottom: 0.15rem;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; flex-shrink: 0;
}
.pcard-brand { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.45rem; flex-shrink: 0; }
.pcard-tags { display: flex; gap: 0.3rem; flex-wrap: wrap; margin-bottom: 0.45rem; flex-shrink: 0; overflow: hidden; max-height: 40px; }
.tag-skin { font-size: 0.6rem; padding: 0.1rem 0.45rem; border-radius: 50px; font-weight: 500; background: #F0EAF8; color: #8B6BA8; border: 1px solid #DDD0EE; white-space: nowrap; }
.pcard-footer {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.5rem 0 0.6rem; border-top: 1px solid var(--border); margin-top: auto; flex-shrink: 0;
}
.pcard-price { font-size: 0.85rem; font-weight: 600; color: var(--deep); }
.pcard-rating { font-size: 0.75rem; color: var(--muted); }
.pcard-btns { display: flex; flex-direction: column; gap: 0.35rem; padding: 0 0 0.9rem; flex-shrink: 0; }
.btn-soc {
    background: transparent; color: var(--deep) !important;
    border: 1.5px solid var(--blush); border-radius: 50px;
    font-family: 'Jost', sans-serif; font-size: 0.78rem; font-weight: 500;
    padding: 0.45rem 0.6rem; cursor: pointer; transition: all 0.2s;
    text-decoration: none !important; text-align: center; display: flex;
    align-items: center; justify-content: center; gap: 0.3rem;
}
.btn-soc:hover { background: var(--petal); color: var(--deep) !important; text-decoration: none !important; }
.btn-soc:visited { color: var(--deep) !important; text-decoration: none !important; }

.rec-banner {
    background: linear-gradient(135deg, #FDF0F4, #F5E8F8);
    border: 2px solid var(--blush); border-radius: 24px;
    padding: 1.5rem 2rem; margin-bottom: 2rem;
    display: flex; align-items: flex-start; gap: 1.5rem;
}
.rb-img {
    width: 120px; height: 120px; border-radius: 16px;
    background: white; border: 1.5px solid var(--blush);
    overflow: hidden; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 2rem;
}
.rb-img img { width: 100%; height: 100%; object-fit: contain; padding: 6px; mix-blend-mode: multiply; }
.rb-info { flex: 1; min-width: 0; }
.rb-lbl { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.25rem; }
.rb-name { font-family: 'Playfair Display', serif; font-size: 1.35rem; font-weight: 700; color: var(--deep); line-height: 1.2; margin-bottom: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rb-brand { font-size: 0.75rem; color: var(--muted); margin-bottom: 0.45rem; }
.rb-meta { display: flex; gap: 1.2rem; flex-wrap: wrap; }
.rb-meta span { font-size: 0.8rem; color: var(--text); }
.rb-meta strong { color: var(--mauve); }
.rb-soc {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--petal); border: 1.5px solid var(--blush);
    border-radius: 50px; padding: 0.3rem 1rem;
    font-size: 0.78rem; font-weight: 500; color: var(--deep) !important;
    text-decoration: none !important; transition: all 0.2s; margin-top: 0.6rem; cursor: pointer;
}
.rb-soc:hover { background: var(--blush); color: var(--deep) !important; text-decoration: none !important; }
.rb-soc:visited { color: var(--deep) !important; text-decoration: none !important; }

.desc-box {
    flex: 1.4; min-width: 220px;
    background: rgba(255,255,255,0.65);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 0.6rem 0.9rem 1.1rem;
    font-size: 0.78rem; color: var(--text); line-height: 1.65;
    height: 140px; overflow-y: scroll; align-self: stretch;
    scrollbar-width: thin; scrollbar-color: var(--blush) transparent;
}
.desc-box::-webkit-scrollbar { width: 5px; }
.desc-box::-webkit-scrollbar-track { background: transparent; border-radius: 10px; }
.desc-box::-webkit-scrollbar-thumb { background: var(--blush); border-radius: 10px; }
.desc-box::-webkit-scrollbar-thumb:hover { background: var(--rose); }
.desc-box-label {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem;
}

.rec-title-row { display: flex; align-items: baseline; gap: 0.8rem; margin-bottom: 1.2rem; }
.rec-title { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; color: var(--deep); }
.rec-sub { font-size: 0.8rem; color: var(--muted); }
.rec-divider { height: 2px; background: linear-gradient(90deg, var(--blush), transparent); border: none; margin: 0 0 1.5rem; }

.recrow {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    align-items: stretch;
    margin-bottom: 1.5rem;
}
.reccard {
    background: var(--warm); border: 1.5px solid var(--border);
    border-radius: 20px; overflow: hidden;
    display: flex; flex-direction: column; height: 100%;
    transition: all 0.2s; box-shadow: 0 2px 10px rgba(180,100,120,0.06);
}
.reccard:hover { border-color: var(--rose); box-shadow: 0 8px 28px rgba(180,100,120,0.15); transform: translateY(-2px); }
.reccard-img {
    width: 100%; height: 130px; background: var(--petal);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative; flex-shrink: 0;
    border-radius: 18px 18px 0 0;
}
.reccard-img img { width: 100%; height: 100%; object-fit: contain; padding: 6px; mix-blend-mode: multiply; }
.rec-rank { position: absolute; top: 0.55rem; left: 0.55rem; background: white; border: 1.5px solid var(--blush); border-radius: 50px; padding: 0.06rem 0.5rem; font-family: 'Playfair Display', serif; font-size: 0.8rem; color: var(--mauve); font-weight: 700; }
.rec-match { position: absolute; top: 0.55rem; right: 0.55rem; background: linear-gradient(135deg, var(--rose), var(--mauve)); border-radius: 50px; padding: 0.1rem 0.55rem; font-size: 0.65rem; color: white; font-weight: 600; }
.reccard-body { padding: 0.85rem 0.95rem 0.5rem; flex: 1; display: flex; flex-direction: column; }
.reccard-cat { font-size: 0.58rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; color: var(--rose); margin-bottom: 0.18rem; flex-shrink: 0; }
.reccard-name { font-family: 'Playfair Display', serif; font-size: 0.88rem; font-weight: 500; color: var(--deep); line-height: 1.3; margin-bottom: 0.12rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; flex-shrink: 0; }
.reccard-brand { font-size: 0.62rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.3rem; flex-shrink: 0; }
.reccard-footer { display: flex; justify-content: space-between; align-items: center; padding-top: 0.45rem; border-top: 1px solid var(--border); margin-top: auto; flex-shrink: 0; }
.reccard-price { font-size: 0.82rem; font-weight: 600; color: var(--deep); }
.reccard-rating { font-size: 0.72rem; color: var(--muted); }
.reccard-soc {
    display: block; text-align: center;
    margin: 0.45rem 0.95rem 0.6rem;
    background: transparent; border: 1.5px solid var(--blush);
    border-radius: 50px; padding: 0.32rem 0.5rem;
    font-size: 0.74rem; font-weight: 500; color: var(--deep) !important;
    text-decoration: none !important; transition: all 0.2s; flex-shrink: 0;
}
.reccard-soc:hover { background: var(--petal); color: var(--deep) !important; text-decoration: none !important; }
.reccard-soc:visited { color: var(--deep) !important; text-decoration: none !important; }

.reason-box {
    background: linear-gradient(135deg, #FDF5F7, #F5EAF8);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 0.45rem 0.6rem; margin: 0.25rem 0; flex-shrink: 0;
}
.reason-label {
    font-size: 0.54rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem;
}
.reason-row {
    display: flex; align-items: flex-start;
    gap: 0.4rem; padding: 0.1rem 0;
    font-size: 0.63rem; line-height: 1.45; color: var(--text);
}
.reason-icon { flex-shrink: 0; width: 14px; text-align: center; }
.reason-key { flex-shrink: 0; color: var(--muted); font-weight: 500; white-space: nowrap; }
.reason-val { color: var(--deep); font-weight: 600; }

.no-result { background: #FFF5F7; border: 1.5px solid var(--blush); border-radius: 16px; padding: 2.5rem; text-align: center; color: var(--muted); }
.no-result-t { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: var(--deep); margin-bottom: 0.5rem; }

.stButton > button {
    background: linear-gradient(135deg, var(--rose), var(--mauve)) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; padding: 0.6rem 1.4rem !important;
    font-family: 'Jost', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 600 !important; box-shadow: 0 4px 14px rgba(180,100,120,0.25) !important;
    transition: all 0.2s ease !important; width: 100% !important;
    white-space: nowrap !important; line-height: 1.2 !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(180,100,120,0.35) !important; }

@media (max-width: 1024px) {
    .block-container { padding: 0 1.5rem 2rem !important; }
    .topbar { width: calc(100% + 3rem); margin-left: -1.5rem; margin-right: -1.5rem; padding: 1.2rem 1.5rem 1rem 1.5rem; }
    .recrow { grid-template-columns: repeat(3, 1fr); }
    .filter-wrap { padding: 1rem; }
}

@media (max-width: 768px) {
    .block-container { padding: 0 1rem 1.5rem !important; }
    .topbar { width: calc(100% + 2rem); margin-left: -1rem; margin-right: -1rem; padding: 1rem; flex-direction: column; gap: 0.8rem; }
    .filter-wrap { display: block; padding: 1rem; }
    .filter-title { margin-bottom: 0.8rem; text-align: center; }
    .rec-banner { flex-direction: column; align-items: center; text-align: center; gap: 1rem; padding: 1.2rem; }
    .rb-meta { justify-content: center; }
    .desc-box { width: 100%; min-width: unset; height: auto; max-height: 180px; }
    .rec-title-row { flex-direction: column; gap: 0.3rem; text-align: center; }
    .recrow { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 480px) {
    .recrow { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ──
MODEL_DIR      = "model"
TFIDF_PATH     = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH    = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
DATAFRAME_PATH = os.path.join(MODEL_DIR, "products_df.pkl")

@st.cache_resource
def load_model():
    with open(TFIDF_PATH,     "rb") as f: vec = pickle.load(f)
    with open(MATRIX_PATH,    "rb") as f: mat = pickle.load(f)
    with open(DATAFRAME_PATH, "rb") as f: df  = pickle.load(f)
    return vec, mat, df

try:
    vectorizer, tfidf_matrix, df = load_model()
except FileNotFoundError:
    st.error("⚠️ Model belum di-build. Jalankan: python model/recommender.py --rebuild")
    st.stop()

# Precompute feature names sekali saja (dipakai buat cari term paling berkontribusi)
FEATURE_NAMES = vectorizer.get_feature_names_out()

ALL_CATEGORIES = sorted(df["category"].unique().tolist())
ALL_SKIN_TYPES = ["oily", "dry", "sensitive", "combination", "normal", "acne", "all skin types"]
CAT_EMOJI = {
    "Acne Treatment":"🎯","Toner":"💧","Essence":"✨",
    "Face Serum":"🧪","Cleanser":"🫧","Face Mask":"🌿",
    "Moisturizer":"🌸","Sun Care / Sunscreen":"☀️",
}

# ── SESSION STATE ──
if "sel_idx"  not in st.session_state: st.session_state.sel_idx  = None
if "rec_res"  not in st.session_state: st.session_state.rec_res  = None
if "cur_page" not in st.session_state: st.session_state.cur_page = 1

# ── HELPERS ──
def stags(s, n=3):
    ts = [x.strip() for x in str(s).split(",")][:n]
    return "".join(f'<span class="tag-skin">{t}</span>' for t in ts)

def get_recs(idx):
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sd   = df.copy(); sd["sim"] = sims
    sd   = sd[sd.index != idx]
    sd   = sd[sd["category"] == df.loc[idx, "category"]]
    return sd.nlargest(5, "sim").reset_index()

# Whitelist kosakata skincare yang BERMAKNA (bahan aktif, manfaat/fungsi,
# skin concern) — bukan blacklist kata generik. Pendekatan ini dipilih
# karena kosakata skincare itu TERBATAS dan bisa dienumerate, sementara
# kata generik/penghubung bahasa Indonesia nyaris tak terbatas (whack-a-mole
# kalau di-blacklist satu-satu). Term hanya ditampilkan sebagai "alasan"
# kalau match daftar ini — bukan sekadar lolos dari daftar larangan.
#
# Silakan tambah entry baru kalau nemu bahan/istilah skincare relevan yang
# belum masuk (cek vocabulary TF-IDF kamu di build_content_features.py).
SKINCARE_VOCAB = {
    # bahan aktif populer
    "niacinamide", "retinol", "vitamin", "hyaluronic", "hyaluronate",
    "ceramide", "ceramides", "salicylic", "glycolic", "lactic", "mandelic",
    "centella", "cica", "snail", "rice", "beras", "green", "tea", "licorice",
    "aloe", "vera", "collagen", "kolagen", "squalane", "panthenol",
    "peptide", "peptida", "antioxidant", "antioksidan", "spf", "sunscreen",
    "zinc", "sulfur", "charcoal", "arang", "kojic", "azelaic", "ferulic",
    "bakuchiol", "probiotic", "prebiotic", "postbiotic", "madecassoside",
    "allantoin", "glutathione", "arbutin", "mugwort", "propolis", "honey",
    "madu", "oat", "oatmeal", "avocado", "alpukat", "cucumber", "timun",
    "chamomile", "argan", "jojoba", "shea", "ginseng", "turmeric", "kunyit",
    "charcoal", "clay", "tanah", "liat", "spirulina", "algae", "rumput",
    "laut", "caffeine", "kafein", "tranexamic", "adenosine", "biome",

    # manfaat / fungsi
    "brightening", "mencerahkan", "cerah", "whitening", "hydrating",
    "melembapkan", "lembap", "moisturizing", "moisturizer", "soothing",
    "menenangkan", "calming", "exfoliating", "eksfoliasi", "cleansing",
    "membersihkan", "purifying", "balancing", "menyeimbangkan", "repairing",
    "memperbaiki", "repair", "protecting", "melindungi", "proteksi",
    "nourishing", "menutrisi", "nutrisi", "smoothing", "menghaluskan",
    "firming", "mengencangkan", "aging", "pore", "pori", "acne", "jerawat",
    "blemish", "noda", "spot", "flek", "kusam", "dull", "brighten",
    "oil", "minyak", "redness", "kemerahan", "sensitive", "sensitif",
    "barrier", "gentle", "lembut", "refreshing", "menyegarkan", "glow",
    "glowing", "radiant", "bercahaya", "blackhead", "komedo", "whitehead",
    "pigmentation", "pigmentasi", "wrinkle", "keriput", "elastisitas",
    "elasticity", "pores", "scar", "bekas", "luka", "hydrate", "hydration",
    "exfoliate", "clarifying", "detox", "renewal", "regenerasi",
    "antiaging", "uv", "meratakan", "bright", "blemishes", 
    "dark", "spots", "protect", "strengthen", "nourish",
    "irritated", "redness", "soothe", "calm", "restore", 
    "menutrisi", "menghidrasi", "menyamarkan", "wash", "exfo", "eksfoliasi", "melembutkan",
}


def top_contributing_terms(idx_a, idx_b, top_k=3):
    """
    Ambil term yang PALING BERKONTRIBUSI terhadap cosine similarity antara
    dua produk, DIBATASI hanya ke kosakata skincare yang bermakna
    (SKINCARE_VOCAB) — bukan term acak yang kebetulan sama-sama muncul.

    Cosine similarity = dot product dari dua vektor TF-IDF (yang sudah
    dinormalisasi). Kontribusi tiap term = tfidf(term, produk_a) *
    tfidf(term, produk_b). Dari term-term yang berkontribusi, kita filter
    dengan whitelist supaya yang muncul di UI benar-benar istilah bahan
    aktif/manfaat, bukan kata penghubung generik.
    """
    vec_a = tfidf_matrix[idx_a].toarray().flatten()
    vec_b = tfidf_matrix[idx_b].toarray().flatten()
    contrib = vec_a * vec_b

    order = contrib.argsort()[::-1]
    terms = []
    for i in order:
        if contrib[i] <= 0:
            break
        term = FEATURE_NAMES[i]
        if term not in SKINCARE_VOCAB:
            continue
        terms.append(term)
        if len(terms) >= top_k:
            break
    return terms


def generate_reasons(input_prod, rec_row, sim_score, idx_input, idx_rec):
    reasons = []

    # 1. KATEGORI
    reasons.append({
        "icon": "📂",
        "key": "Kategori:",
        "val": str(rec_row["category"])
    })

    # 2. SKIN TYPE
    input_skins = set(x.strip().lower() for x in str(input_prod["skin_type"]).split(","))
    rec_skins   = set(x.strip().lower() for x in str(rec_row["skin_type"]).split(","))
    shared      = input_skins & rec_skins
    if "all skin types" in rec_skins:
        reasons.append({"icon": "✅", "key": "Skin type:", "val": "Cocok untuk semua jenis kulit"})
    elif shared:
        val = ", ".join(s.title() for s in sorted(shared)[:4])
        reasons.append({"icon": "✅", "key": "Skin type cocok:", "val": val})
    else:
        reasons.append({"icon": "⚠️", "key": "Skin type:", "val": "Berbeda, cek kesesuaian"})

    # 3. BRAND
    if str(input_prod["brand"]).strip().lower() == str(rec_row["brand"]).strip().lower():
        reasons.append({
            "icon": "🏪",
            "key": "Brand sama:",
            "val": str(rec_row["brand"])
        })

    # 4. TERM PALING BERKONTRIBUSI KE SIMILARITY SCORE
    # (diganti dari "kata kunci serupa" yang tadinya cuma intersection kata
    # mentah diurutkan alfabetis — sekarang benar-benar mencerminkan term
    # yang mendorong similarity score sesuai definisi cosine similarity)
    top_terms = top_contributing_terms(idx_input, idx_rec, top_k=3)
    if top_terms:
        reasons.append({
            "icon": "🏷️",
            "key": "Bahan/manfaat serupa:",
            "val": ", ".join(top_terms)
        })

    return reasons[:4]


# ── TOPBAR — logo kiri, tombol Home kanan ──
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">Glow<em>match</em></div>
    <a class="topbar-home" href="/" target="_self"> 🏠  Home </a>
</div>
""", unsafe_allow_html=True)

# ── PAGE TITLE ──
st.markdown("""
<div class="pg-title">Katalog Produk Skincare</div>
<div class="pg-sub">Gunakan filter untuk menyesuaikan pencarian, lalu klik produk untuk melihat rekomendasi serupa.</div>
""", unsafe_allow_html=True)

# ── FILTER BAR ──
st.markdown('<div class="filter-wrap"><div class="filter-title">🔽 Filter Produk</div>', unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)
with f1:
    sel_cat = st.selectbox("Kategori", ["Semua Kategori"] + ALL_CATEGORIES, key="fcat")
with f2:
    sel_skin = st.multiselect(
        "Skin Type",
        ALL_SKIN_TYPES,
        placeholder="Pilih skin type (bisa lebih dari 1)",
        max_selections=3,
        key="fskin"
    )
with f3:
    sel_price = st.selectbox("Rentang Harga", [
        "Semua Harga", "< Rp50.000", "Rp50.000 – Rp150.000",
        "Rp150.000 – Rp300.000", "Rp300.000 – Rp500.000", "> Rp500.000"
    ], key="fprice")
with f4:
    sel_rating = st.selectbox("Minimum Rating", [
        "Semua Rating", "⭐ 4.0+", "⭐ 4.5+", "⭐ 4.7+", "⭐ 5.0"
    ], key="frating")
st.markdown('</div>', unsafe_allow_html=True)

# ── APPLY FILTERS ──
fdf = df.copy()
if sel_cat != "Semua Kategori":
    fdf = fdf[fdf["category"] == sel_cat]

if sel_skin:
    sel_skin_lower = [s.lower() for s in sel_skin]
    def ok_s(s):
        lst = [x.strip().lower() for x in str(s).split(",")]
        return "all skin types" in lst or any(sk in lst for sk in sel_skin_lower)
    fdf = fdf[fdf["skin_type"].apply(ok_s)]

pm = {"< Rp50.000":(0,50000),"Rp50.000 – Rp150.000":(50000,150000),
      "Rp150.000 – Rp300.000":(150000,300000),"Rp300.000 – Rp500.000":(300000,500000),
      "> Rp500.000":(500000,99999999)}
if sel_price in pm:
    lo, hi = pm[sel_price]
    fdf = fdf[(fdf["price"] >= lo) & (fdf["price"] <= hi)]
rm = {"⭐ 4.0+":4.0,"⭐ 4.5+":4.5,"⭐ 4.7+":4.7,"⭐ 5.0":5.0}
if sel_rating in rm:
    fdf = fdf[fdf["rating"] >= rm[sel_rating]]
fdf = fdf.sort_values("rating", ascending=False).reset_index()

# ── SELECTED PRODUCT + REKOMENDASI ──
if st.session_state.sel_idx is not None:
    if st.session_state.get("scroll_to_top"):
        components.html(
            """<script>
            const parent = window.parent;
            const scrollContainers = parent.document.querySelectorAll('.main, [data-testid="stAppViewContainer"], [data-testid="stMain"]');
            scrollContainers.forEach(c => c.scrollTo({top: 0, behavior: 'smooth'}));
            parent.scrollTo({top: 0, behavior: 'smooth'});
            </script>""",
            height=0
        )
        st.session_state.scroll_to_top = False

    idx     = st.session_state.sel_idx
    prod    = df.loc[idx]
    url     = str(prod.get("image_url", ""))
    soc_url = str(prod.get("url_sociolla", ""))
    ih      = f'<img src="{url}" onerror="this.style.display=\'none\'">' if url.strip() else "🌸"
    soc_btn = f'<a href="{soc_url}" target="_blank" class="rb-soc">🛍️ Lihat di Sociolla</a>' if soc_url else ""

    desc     = str(prod.get("description", "")).strip()
    has_desc = desc and desc.lower() != "nan" and desc != ""

    st.markdown(f"""
    <div class="rec-banner">
        <div class="rb-img">{ih}</div>
        <div class="rb-info">
            <div class="rb-lbl">✦ Produk yang kamu pilih</div>
            <div class="rb-name">{prod['product_name']}</div>
            <div class="rb-brand">{prod['brand']}</div>
            <div style="display:flex;gap:0.3rem;flex-wrap:wrap;margin-bottom:0.4rem">{stags(prod['skin_type'])}</div>
            <div class="rb-meta">
                <span>💰 <strong>Rp{prod['price']:,.0f}</strong></span>
                <span>⭐ <strong>{prod['rating']:.2f}</strong></span>
                <span>📂 <strong>{prod['category']}</strong></span>
            </div>
            {soc_btn}
        </div>
        <div class="desc-box">
            <div class="desc-box-label">📄 Deskripsi Produk</div>
            {desc if has_desc else '<em style="color:var(--muted);font-size:0.75rem">Deskripsi tidak tersedia.</em>'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    recs = st.session_state.rec_res
    if recs is not None and not recs.empty:
        st.markdown("""
        <div class="rec-title-row">
            <span class="rec-title">✨ 5 Produk Paling Mirip</span>
            <span class="rec-sub">berdasarkan kategori, skin type, nama produk, brand &amp; deskripsi</span>
        </div>
        <hr class="rec-divider">
        """, unsafe_allow_html=True)

        cards_html = ""
        for i, row in recs.iterrows():
            ru       = str(row.get("image_url", ""))
            rsoc     = str(row.get("url_sociolla", ""))
            ri       = f'<img src="{ru}" onerror="this.style.display=\'none\'">' if ru.strip() else '<span style="font-size:2rem;opacity:0.2">🌸</span>'
            soc_link = f'<a href="{rsoc}" target="_blank" class="reccard-soc">🛍️ Lihat di Sociolla</a>' if rsoc else ""
            reasons      = generate_reasons(prod, row, row["sim"], idx, row["index"])
            reasons_html = "".join(
                f'<div class="reason-row"><span class="reason-icon">{r["icon"]}</span><span class="reason-key">{r["key"]}</span><span class="reason-val">{r["val"]}</span></div>'
                for r in reasons
            )
            cards_html += f"""<div class="reccard">
<div class="reccard-img">{ri}<div class="rec-rank">#{i+1}</div><div class="rec-match">{row['sim']*100:.1f}%</div></div>
<div class="reccard-body">
<div class="reccard-cat">{CAT_EMOJI.get(row['category'],'')} {row['category']}</div>
<div class="reccard-name">{row['product_name']}</div>
<div class="reccard-brand">{row['brand']}</div>
<div class="pcard-tags">{stags(row['skin_type'])}</div>
<div class="reason-box"><div class="reason-label">💡 Kenapa Direkomendasikan?</div>{reasons_html}</div>
<div class="reccard-footer"><span class="reccard-price">Rp{row['price']:,.0f}</span><span class="reccard-rating">⭐ {row['rating']:.2f}</span></div>
</div>{soc_link}</div>"""

        st.markdown(f'<div class="recrow">{cards_html}</div>', unsafe_allow_html=True)

    _, cb, _ = st.columns([3, 2, 3])
    with cb:
        if st.button("✕ Tutup Rekomendasi", key="close"):
            st.session_state.sel_idx = None
            st.session_state.rec_res = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

# ── CATALOG ──
total = len(fdf)
st.markdown(f'<div class="result-count">Menampilkan <strong>{total}</strong> produk</div>',
            unsafe_allow_html=True)

if fdf.empty:
    st.markdown("""
    <div class="no-result">
        <div class="no-result-t">🌸 Tidak ada produk</div>
        <div>Coba ubah filter untuk melihat lebih banyak produk.</div>
    </div>""", unsafe_allow_html=True)
else:
    per_page    = 20
    total_pages = max(1, (total - 1) // per_page + 1)

    if st.session_state.cur_page > total_pages: st.session_state.cur_page = total_pages
    if st.session_state.cur_page < 1:           st.session_state.cur_page = 1
    page = st.session_state.cur_page

    start   = (page - 1) * per_page
    page_df = fdf.iloc[start: start + per_page]

    cols = st.columns(5)
    for i, (_, row) in enumerate(page_df.iterrows()):
        orig    = row["index"]
        soc_url = str(row.get("url_sociolla", "")).strip()
        with cols[i % 5]:
            url = str(row.get("image_url", ""))
            em  = CAT_EMOJI.get(row["category"], "✿")
            ih  = f'<img src="{url}" onerror="this.style.display=\'none\'">' if url.strip() else '<span style="font-size:2.5rem;opacity:0.2">🌸</span>'

            st.markdown(f"""
            <div class="pcard">
                <div class="pcard-img">{ih}</div>
                <div class="pcard-body">
                    <div class="pcard-cat">{em} {row['category']}</div>
                    <div class="pcard-name">{row['product_name']}</div>
                    <div class="pcard-brand">{row['brand']}</div>
                    <div class="pcard-tags">{stags(row['skin_type'])}</div>
                    <div class="pcard-footer">
                        <span class="pcard-price">Rp{row['price']:,.0f}</span>
                        <span class="pcard-rating">⭐ {row['rating']:.2f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✨ Lihat Rekomendasi", key=f"r_{orig}_{i}"):
                st.session_state.sel_idx  = orig
                st.session_state.rec_res  = get_recs(orig)
                st.session_state.cur_page = page
                st.session_state.scroll_to_top = True
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]:last-of-type {
        align-items: center !important;
        flex-direction: row !important;
        justify-content: center !important;
        flex-wrap: nowrap !important;
        gap: 0.5rem !important;
    }
    div[data-testid="stHorizontalBlock"]:last-of-type > div {
        width: auto !important;
        flex: 0 1 auto !important;
    }

    div[data-testid="stHorizontalBlock"]:last-of-type .stButton > button {
        width: 42px !important; height: 42px !important;
        min-width: 42px !important; min-height: 42px !important;
        max-width: 42px !important; border-radius: 50% !important;
        padding: 0 !important; margin: 0 auto !important;
        font-size: 1.1rem !important; font-weight: 700 !important;
        line-height: 1 !important; border: none !important;
        box-shadow: 0 2px 10px rgba(180,100,120,0.25) !important;
        background: linear-gradient(135deg, #D4848E, #B5687A) !important;
        color: white !important; transition: all 0.2s !important;
        display: flex !important; align-items: center !important; justify-content: center !important;
    }
    div[data-testid="stHorizontalBlock"]:last-of-type .stButton > button:hover { opacity: 0.85 !important; transform: scale(1.08) !important; }
    </style>
    """, unsafe_allow_html=True)

    _, pg_prev, pg_mid, pg_next, _ = st.columns([4.6, 0.6, 0.6, 0.6, 4.6])

    with pg_prev:
        if st.button("←", key="pg_prev", disabled=(page <= 1)):
            st.session_state.cur_page -= 1
            st.rerun()

    with pg_mid:
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:center;">'
            f'<div style="height:42px;padding:0 1.3rem 4px;'
            f'background:linear-gradient(135deg,#FDF0F4,#F8EAF5);'
            f'border:2px solid #EDB8C0;border-radius:50px;'
            f'font-family:\'Playfair Display\',serif;font-size:1.05rem;font-weight:700;'
            f'color:#6B3A47;box-shadow:0 4px 16px rgba(180,100,120,0.12);'
            f'display:flex;align-items:center;justify-content:center;min-width:52px;">'
            f'{page}</div></div>',
            unsafe_allow_html=True
        )

    with pg_next:
        if st.button("→", key="pg_next", disabled=(page >= total_pages)):
            st.session_state.cur_page += 1
            st.rerun()

    st.markdown(f"""
    <div style="text-align:center;margin-top:0.8rem;font-size:0.8rem;color:var(--muted)">
        Menampilkan {start + 1}–{min(start + per_page, total)} dari {total} produk
    </div>
    """, unsafe_allow_html=True)