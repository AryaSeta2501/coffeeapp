import streamlit as st
import numpy as np
from common import CLASS_NAMES, CLASS_INFO

st.markdown("""
<div class="section-header">
    <span class="section-num">04</span>
    <span class="section-title">Hasil Klasifikasi</span>
</div>
""", unsafe_allow_html=True)

# Gating: kalau user loncat ke halaman ini lewat nav tanpa prediksi dulu
if "preds" not in st.session_state:
    st.warning("Belum ada hasil klasifikasi.")
    st.page_link("steps/upload.py", label="← Mulai dari Upload Gambar")
    st.stop()

preds = np.array(st.session_state["preds"])
top_idx = int(np.argmax(preds))
top_class = CLASS_NAMES[top_idx]
top_conf = float(preds[top_idx]) * 100

tag_class = "tag-healthy" if top_class == "Healthy" else "tag-disease"
tag_label = "Sehat" if top_class == "Healthy" else "Terindikasi Penyakit"

st.markdown(f"""
<div class="result-card">
    <span class="result-tag {tag_class}">{tag_label}</span>
    <div class="result-title">{top_class}</div>
    <p class="result-desc">{CLASS_INFO.get(top_class, "")}</p>
    <span class="result-confidence">CONFIDENCE — {top_conf:.2f}%</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-header" style="margin-top:2rem;">
    <span class="section-num">—</span>
    <span class="section-title">Distribusi Probabilitas per Kelas</span>
</div>
""", unsafe_allow_html=True)

order = np.argsort(preds)[::-1]
bars_html = ""
for i in order:
    cls = CLASS_NAMES[i]
    pct = float(preds[i]) * 100
    bars_html += f"""
    <div class="prob-row">
        <div class="prob-label-row">
            <span>{cls}</span>
            <span class="prob-value">{pct:.2f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct}%;"></div>
        </div>
    </div>
    """
st.markdown(bars_html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Klasifikasikan Gambar Lain"):
    del st.session_state["image_bytes"]
    del st.session_state["preds"]
    st.switch_page("steps/upload.py")

st.markdown("""
<div class="footer-note">
COFFEE LEAF DISEASE CLASSIFIER — CONVNEXT TINY — JMUBEN DATASET
</div>
""", unsafe_allow_html=True)