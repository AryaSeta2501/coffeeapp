"""
streamlit_app.py — Entrypoint & router.

CARA MENJALANKAN: streamlit run streamlit_app.py
(BUKAN app.py lagi — nama entrypoint berubah karena sekarang multi-page)
"""

import streamlit as st
from common import inject_css

st.set_page_config(
    page_title="Coffee Leaf Disease Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# Definisikan 4 halaman sesuai alur: Dashboard -> Upload -> Prediksi -> Hasil
page_dashboard = st.Page("steps/dashboard.py", title="Dashboard", default=True)
page_upload = st.Page("steps/upload.py", title="Upload Gambar")
page_predict = st.Page("steps/predict.py", title="Klasifikasi")
page_result = st.Page("steps/result.py", title="Hasil")

# position="hidden" mematikan sidebar nav bawaan Streamlit sepenuhnya,
# supaya kita bisa pakai nav custom bergaya portofolio di bawah ini.
pg = st.navigation(
    [page_dashboard, page_upload, page_predict, page_result],
    position="hidden",
)

# ── NAV CUSTOM (menggantikan sidebar bawaan) ──
nav_cols = st.columns([2.5, 1.3, 1.1, 1.4, 1])
with nav_cols[0]:
    st.markdown('<div class="nav-logo">COFFEE·DISSEASE·CLASSIFIER</div>', unsafe_allow_html=True)
with nav_cols[1]:
    st.page_link(page_dashboard, label="01 · Dashboard")
with nav_cols[2]:
    st.page_link(page_upload, label="02 · Upload")
with nav_cols[3]:
    st.page_link(page_predict, label="03 · Klasifikasi")
with nav_cols[4]:
    st.page_link(page_result, label="04 · Hasil")

st.markdown('<div style="border-bottom:1px solid #e8e6e1; margin-bottom:2rem;"></div>', unsafe_allow_html=True)

pg.run()