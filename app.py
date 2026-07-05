"""
Glowmatch - Landing Page
"""
import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Glowmatch · Skincare Recommender",
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
    --sage:   #6B9E65;
    --sage-l: #E8F0E7;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif !important;
    color: var(--text); background: var(--cream);
}
[data-testid="stAppViewContainer"], [data-testid="stApp"], .main { background: var(--cream) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none !important; }

.navbar {
    background: rgba(253,247,242,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 1rem 3rem;
    display: flex; justify-content: space-between; align-items: center;
    position: sticky; top: 0; z-index: 999;
    padding-bottom: 1rem;
}
.nav-logo { font-family: 'Playfair Display', serif; font-size: 1.6rem; font-weight: 700; color: var(--deep); }
.nav-logo em { font-style: italic; color: var(--rose); }
.nav-links { display: flex; gap: 2.5rem; align-items: center;}
.nav-links a { font-size: 0.95rem; font-weight: 500; color: var(--deep); text-decoration: none; letter-spacing: 0.2px; }
.nav-links a:hover { color: var(--mauve); }

.nav-btn {
    display: inline-flex; align-items: center;
    background: linear-gradient(135deg, var(--rose), var(--mauve));
    color: white !important; border: none; border-radius: 50px;
    padding: 0.4rem 1.2rem;   /* ← sedikit dikurangin dari 0.45rem */
    font-family: 'Jost', sans-serif;
    font-size: 0.88rem; font-weight: 600; letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(180,100,120,0.28);
    transition: all 0.2s ease; text-decoration: none;
}
.nav-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(180,100,120,0.38); color: white !important; }

.hero {
    min-height: 90vh; background: var(--cream);
    display: flex; align-items: center;
    padding: 5rem 5rem 6rem; position: relative; overflow: hidden;
}
.hero-blob-1 { position: absolute; width: 550px; height: 550px; background: radial-gradient(circle, rgba(242,196,206,0.55) 0%, transparent 70%); border-radius: 50%; top: -120px; right: -80px; filter: blur(60px); pointer-events: none; }
.hero-blob-2 { position: absolute; width: 380px; height: 380px; background: radial-gradient(circle, rgba(232,213,240,0.45) 0%, transparent 70%); border-radius: 50%; bottom: 30px; right: 380px; filter: blur(60px); pointer-events: none; }
.hero-blob-3 { position: absolute; width: 280px; height: 280px; background: radial-gradient(circle, rgba(212,221,210,0.4) 0%, transparent 70%); border-radius: 50%; top: 180px; left: -60px; filter: blur(50px); pointer-events: none; }
.hero-left { position: relative; z-index: 1; max-width: 560px; padding-bottom: 2rem; }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 0.5rem; background: var(--petal); border: 1px solid var(--blush); border-radius: 50px; padding: 0.35rem 1.1rem; font-size: 0.72rem; font-weight: 600; color: var(--mauve); letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 1.6rem; }
.hero-h1 { font-family: 'Playfair Display', serif; font-size: 4.4rem; font-weight: 700; color: var(--deep); line-height: 1.06; letter-spacing: -1.5px; margin-bottom: 1.5rem; }
.hero-h1 em { font-style: italic; color: var(--rose); font-weight: 400; }
.hero-p { font-size: 1.05rem; color: var(--muted); font-weight: 300; line-height: 1.75; margin-bottom: 2.5rem; max-width: 470px; }
.hero-stats { display: flex; gap: 2.5rem; margin-top: 2.8rem; padding-top: 2rem; border-top: 1px solid var(--border); }
.hstat-num { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 700; color: var(--deep); line-height: 1; }
.hstat-lbl { font-size: 0.75rem; color: var(--muted); font-weight: 400; margin-top: 0.2rem; }
.hero-right { position: absolute; right: 3.5rem; top: 50%; transform: translateY(-50%); display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; z-index: 1; width: 480px; }
.hcard-feat { background: var(--warm); border: 1.5px solid var(--border); border-radius: 16px; padding: 1.2rem; box-shadow: 0 4px 16px rgba(180,100,120,0.06); transition: transform 0.2s; }
.hcard-feat:hover { transform: translateY(-2px); border-color: var(--blush); box-shadow: 0 6px 20px rgba(180,100,120,0.1); }
.hcard-feat-icon { font-size: 1.5rem; display: block; margin-bottom: 0.5rem; line-height: 1; }
.hcard-feat-title { font-family: 'Playfair Display', serif; font-size: 0.95rem; font-weight: 600; color: var(--deep); margin-bottom: 0.3rem; }
.hcard-feat-desc { font-size: 0.75rem; color: var(--muted); line-height: 1.5; }

.stButton > button {
    background: linear-gradient(135deg, var(--rose), var(--mauve)) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; padding: 0.9rem 2.5rem !important;
    font-family: 'Jost', sans-serif !important; font-size: 1rem !important;
    font-weight: 600 !important; letter-spacing: 0.3px !important;
    box-shadow: 0 8px 28px rgba(180,100,120,0.32) !important;
    transition: all 0.25s ease !important; min-width: 200px !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 12px 36px rgba(180,100,120,0.42) !important; }

/* Posisi button CTA di Hero (di kiri bawah stats) */
div[data-testid="stElementContainer"]:has(span#hero-btn-target) + div[data-testid="stElementContainer"] {
    margin-top: -110px !important;
    margin-left: 5rem !important;
    margin-bottom: 45px !important;
    position: relative;
    z-index: 10;
    width: fit-content;
}

.sec-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; color: var(--rose); margin-bottom: 0.8rem; }
.sec-h2 { font-family: 'Playfair Display', serif; font-size: 2.8rem; font-weight: 700; color: var(--deep); line-height: 1.15; letter-spacing: -0.5px; margin-bottom: 1.2rem; }
.sec-h2 em { font-style: italic; color: var(--rose); font-weight: 400; }
.sec-p { font-size: 0.98rem; color: var(--muted); font-weight: 300; line-height: 1.8; max-width: 500px; }

.howto-sec { padding: 6rem 5rem; background: var(--cream); }
.howto-inner { max-width: 1040px; margin: 0 auto; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.5rem; margin-top: 3rem; }
.step { background: var(--warm); border: 1.5px solid var(--border); border-radius: 24px; padding: 1.8rem 1.6rem; transition: all 0.25s ease; position: relative; }
.step:hover { border-color: var(--rose); box-shadow: 0 8px 32px rgba(180,100,120,0.12); transform: translateY(-3px); }
.step-num { font-family: 'Playfair Display', serif; font-size: 3.6rem; font-weight: 700; color: var(--mauve); opacity: 1; line-height: 1; margin-bottom: 0.8rem; display: block; width: fit-content; }
.step-icon { font-size: 2rem; margin-bottom: 0.9rem; display: block; }
.step-title { font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 600; color: var(--deep); margin-bottom: 0.5rem; }
.step-desc { font-size: 0.85rem; color: var(--muted); line-height: 1.6; }
.step-arrow { position: absolute; top: 3.2rem; right: -1.6rem; font-size: 1.5rem; background: linear-gradient(135deg, var(--rose), var(--mauve)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; z-index: 1; }

/* Chart section */
.chart-sec { padding: 5rem 5rem 2rem 5rem; background: var(--warm); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); }
.chart-wrap { max-width: 860px; margin: 0 auto; padding: 1rem 0; }

div[data-testid="stHorizontalBlock"]:has(span#chart-pull-target) {
    margin-top: -490px !important;
    position: relative; z-index: 10; margin-bottom: 3rem !important;
}
div[data-testid="stElementContainer"]:has(span#chart-pull-target-none) + div[data-testid="stElementContainer"] {
    margin-top: -150px !important; position: relative; z-index: 10; margin-bottom: 3rem !important;
}
.chart-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 500; color: var(--deep); margin-bottom: 0.3rem; }
.chart-sub { font-size: 0.82rem; color: var(--muted); margin-bottom: 2rem; }



.cta-sec { padding: 4rem 5rem 1.5rem; background: transparent; text-align: center; }
.cta-inner { max-width: 580px; margin: 0 auto; }
.cta-h2 { font-family: 'Playfair Display', serif; font-size: 3rem; font-weight: 700; color: var(--deep); line-height: 1.15; margin-bottom: 1rem; }
.cta-h2 em { font-style: italic; color: var(--rose); font-weight: 400; }
.cta-p { font-size: 1rem; color: var(--muted); font-weight: 300; margin-bottom: 2.5rem; line-height: 1.75; }

.footer { padding: 1.2rem 5rem; background: var(--deep); display: flex; justify-content: space-between; align-items: center; margin-top: 4rem; }
.footer-logo { font-family: 'Playfair Display', serif; font-size: 1.2rem; color: white; font-style: italic; }
.footer-copy { font-size: 0.75rem; color: rgba(255,255,255,0.38); }

/* Plotly chart override */
.js-plotly-plot .plotly { font-family: 'Jost', sans-serif !important; }

/* Responsive Media Queries */
@media (max-width: 1024px) {
    .navbar { padding: 1rem 2rem; }
    .hero { padding: 4rem 2rem 3rem; min-height: auto; flex-direction: column; align-items: flex-start; }
    .hero-right { position: relative; right: auto; top: auto; transform: none; width: 100%; margin-top: 3rem; grid-template-columns: repeat(2, 1fr); }
    div[data-testid="stElementContainer"]:has(span#hero-btn-target) + div[data-testid="stElementContainer"] {
        margin-top: 0 !important;
        margin-left: 2rem !important;
        margin-bottom: 3rem !important;
    }
    .howto-sec { padding: 4rem 2rem; }
    .steps { grid-template-columns: repeat(2, 1fr); }
    .step-arrow { display: none; }
    .chart-sec { padding: 4rem 2rem 2rem 2rem; }
    .cta-sec { padding: 3rem 2rem 1.5rem; }
    .footer { padding: 1.2rem 2rem; }
}

@media (max-width: 768px) {
    .navbar { padding: 1rem; flex-direction: column; gap: 1rem; }
    .nav-links { gap: 1rem; flex-wrap: wrap; justify-content: center; }
    .hero { padding: 3rem 1.5rem 2rem; }
    .hero-h1 { font-size: 2.8rem; }
    .hero-stats { flex-wrap: wrap; gap: 1.5rem; margin-top: 1.5rem; padding-top: 1.5rem; }
    .hero-right { grid-template-columns: 1fr; margin-top: 2.5rem; }
    div[data-testid="stElementContainer"]:has(span#hero-btn-target) + div[data-testid="stElementContainer"] {
        margin-left: 1.5rem !important;
    }
    .howto-sec { padding: 3rem 1.5rem; }
    .steps { grid-template-columns: 1fr; gap: 1.5rem; }
    .chart-sec { padding: 3rem 1.5rem 2rem 1.5rem; }
    .cta-sec { padding: 2rem 1.5rem 1.5rem; }
    .cta-h2 { font-size: 2.2rem; }
    .footer { padding: 1.5rem; flex-direction: column; gap: 0.5rem; text-align: center; }
}
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA UNTUK CHART ──
@st.cache_data
def load_df():
    MODEL_DIR = "model"
    DATAFRAME_PATH = os.path.join(MODEL_DIR, "products_df.pkl")
    if os.path.exists(DATAFRAME_PATH):
        with open(DATAFRAME_PATH, "rb") as f:
            import pickle
            return pickle.load(f)
    return None

df = load_df()

# ── NAVBAR ──
st.markdown("""
<nav class="navbar">
    <div class="nav-logo">Glow<em>match</em></div>
    <div class="nav-links">
        <a href="#cara-pakai">Cara Pakai</a>
        <a href="#statistik">Statistik</a>
        <a href="/rekomendasi" class="nav-btn">Jelajahi Produk →</a>
    </div>
</nav>
""", unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<section class="hero">
    <div class="hero-blob-1"></div>
    <div class="hero-blob-2"></div>
    <div class="hero-blob-3"></div>
    <div class="hero-left">
        <div class="hero-eyebrow">✦ Sistem Rekomendasi Skincare</div>
        <h1 class="hero-h1">Temukan <em>skincare</em><br>yang tepat<br>untukmu.</h1>
        <p class="hero-p">
            Glowmatch adalah sistem rekomendasi skincare berbasis <strong>Content-Based Filtering (CBF)</strong> yang dibangun menggunakan data 1.866 produk dari Sociolla.<br><br>
            Sistem ini menganalisis kemiripan antar produk berdasarkan kategori, skin type, nama produk, deskripsi, dan brand — lalu menghitung skor kemiripan menggunakan <strong>TF-IDF &amp; Cosine Similarity</strong>.
        </p>
        <div class="hero-stats">
            <div><div class="hstat-num">1.866</div><div class="hstat-lbl">Produk skincare</div></div>
            <div><div class="hstat-num">162</div><div class="hstat-lbl">Brand tersedia</div></div>
            <div><div class="hstat-num">8</div><div class="hstat-lbl">Kategori produk</div></div>
            <div><div class="hstat-num">83.7%</div><div class="hstat-lbl">Akurasi Precision@5</div></div>
        </div>
    </div>
    <div class="hero-right">
        <div class="hcard-feat">
            <span class="hcard-feat-icon">🔬</span>
            <div class="hcard-feat-title">TF-IDF Vectorizer</div>
            <div class="hcard-feat-desc">Mengubah fitur teks produk menjadi representasi vektor numerik yang dapat dibandingkan.</div>
        </div>
        <div class="hcard-feat">
            <span class="hcard-feat-icon">📐</span>
            <div class="hcard-feat-title">Cosine Similarity</div>
            <div class="hcard-feat-desc">Mengukur kemiripan antar produk berdasarkan sudut antara dua vektor fitur.</div>
        </div>
        <div class="hcard-feat">
            <span class="hcard-feat-icon">🧴</span>
            <div class="hcard-feat-title">Skin Type Detection</div>
            <div class="hcard-feat-desc">Mengekstrak skin type otomatis dari deskripsi produk menggunakan keyword extraction.</div>
        </div>
        <div class="hcard-feat">
            <span class="hcard-feat-icon">📊</span>
            <div class="hcard-feat-title">Precision@5 = 83.7%</div>
            <div class="hcard-feat-desc">Dievaluasi terhadap seluruh 1.866 produk menggunakan metrik Precision@5.</div>
        </div>
    </div>
</section>
""", unsafe_allow_html=True)

st.markdown('<span id="hero-btn-target"></span>', unsafe_allow_html=True)
if st.button("🌸 Mulai Sekarang", key="cta1"):
    st.switch_page("pages/rekomendasi.py")

# ── HOW TO ──
st.markdown("""
<section class="howto-sec" id="cara-pakai">
    <div class="howto-inner">
        <div class="sec-eyebrow">Cara Pakai</div>
        <h2 class="sec-h2">Tiga langkah <em>mudah</em><br>menuju kulit sehat.</h2>
        <div class="steps">
        <div class="step">
            <span class="step-num">01</span>
            <span class="step-icon">🔽</span>
            <div class="step-title">Pilih Filter Produk</div>
            <div class="step-desc">Gunakan filter kategori, skin type, rentang harga, dan rating untuk menyesuaikan pencarian dengan kebutuhan kulitmu.</div>
            <span class="step-arrow">→</span>
        </div>
        <div class="step">
            <span class="step-num">02</span>
            <span class="step-icon">👆</span>
            <div class="step-title">Klik Produk Favoritmu</div>
            <div class="step-desc">Temukan produk yang kamu suka dari katalog, lalu klik tombol <strong>"Lihat Rekomendasi"</strong> untuk memulai analisis kemiripan.</div>
            <span class="step-arrow">→</span>
        </div>
        <div class="step">
            <span class="step-num">03</span>
            <span class="step-icon">✨</span>
            <div class="step-title">Dapatkan Rekomendasi</div>
            <div class="step-desc">Sistem menampilkan 5 produk paling mirip lengkap dengan persentase kemiripan, harga, rating, dan skin type.</div>
        </div>
    </div>
    </div>
</section>
""", unsafe_allow_html=True)

# ── CHART SECTION ──
chart_spacer = '<div style="height: 480px;"></div>' if df is not None else '<div style="height: 150px;"></div>'

st.markdown(f"""
<section class="chart-sec" id="statistik">
    <div class="sec-eyebrow">Statistik Dataset</div>
    <h2 class="sec-h2" style="margin-bottom:0.5rem">Distribusi produk<br>per <em>kategori</em>.</h2>
    <p style="font-size:0.9rem;color:var(--muted);font-weight:300;margin-bottom:2.5rem">
        Dari 1.866 produk skincare yang dikumpulkan dari Sociolla,
        berikut sebaran produk di setiap kategori.
    </p>
    {chart_spacer}
</section>
""", unsafe_allow_html=True)

if df is not None:
    import plotly.graph_objects as go

    cat_counts = df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    cat_counts = cat_counts.sort_values("count", ascending=True)

    # Warna gradient rose-mauve
    colors = [
        "#EDB8C0", "#E8A8B2", "#E398A4", "#DE8896",
        "#D97888", "#D4687A", "#C4586A", "#B5485A"
    ]

    fig = go.Figure(go.Bar(
        x=cat_counts["count"],
        y=cat_counts["category"],
        orientation="h",
        marker=dict(
            color=colors[:len(cat_counts)],
            line=dict(width=0),
        ),
        text=cat_counts["count"],
        textposition="outside",
        textfont=dict(family="Jost", size=13, color="#6B3A47"),
        hovertemplate="<b>%{y}</b><br>%{x} produk<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=10, b=10),
        height=380,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(237,213,218,0.5)",
            gridwidth=1,
            showticklabels=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            tickfont=dict(family="Jost", size=13, color="#6B3A47"),
        ),
        hoverlabel=dict(
            bgcolor="#FDF7F2",
            bordercolor="#EDD5DA",
            font=dict(family="Jost", color="#3D2430"),
        ),
        bargap=0.35,
    )

    _, chart_col, _ = st.columns([0.5, 6, 0.5])
    with chart_col:
        st.markdown('<span id="chart-pull-target"></span><div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<span id="chart-pull-target-none"></span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:var(--muted);font-size:0.9rem">
        Grafik akan tampil setelah model di-build.<br>
        Jalankan: <code>python model/recommender.py --rebuild</code>
    </div>
    """, unsafe_allow_html=True)

# ── CTA ──
st.markdown("""
<section class="cta-sec">
    <div class="cta-inner">
        <h2 class="cta-h2">Siap menemukan<br><em>skincare match</em>-mu?</h2>
        <p class="cta-p">Mulai jelajahi 1.866 produk skincare dan dapatkan rekomendasi personal yang sesuai dengan jenis kulitmu.</p>
    </div>
</section>
""", unsafe_allow_html=True)

_, c2, _ = st.columns([2, 1, 2])
with c2:
    if st.button("🌸 Mulai Sekarang", key="cta2"):
        st.switch_page("pages/rekomendasi.py")

# ════════════════════════════════
# FOOTER
# ════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-logo">Glowmatch</div>
    <div class="footer-copy">&#169; 2025 &middot; Sistem Rekomendasi Skincare &middot; Data dari Sociolla</div>
</div>
""", unsafe_allow_html=True)