import streamlit as st

st.markdown('<div class="eyebrow">Skripsi — Deep Learning Application</div>', unsafe_allow_html=True)
st.markdown("""
<div class="hero-title">Coffee Leaf<br><em>Disease Classifier</em></div>
<p class="hero-desc">
Klasifikasi otomatis penyakit daun kopi menggunakan ConvNeXt Tiny yang dilatih pada JMuBEN Coffee Dataset.
Alur penggunaan: unggah gambar daun kopi → jalankan klasifikasi → lihat hasil
klasifikasi penyakit.
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠ ALAT BANTU RISET / SKRIPSI — bukan alat diagnosis definitif. Prediksi model
dapat keliru, khususnya pada kelas dengan overlap fitur visual tinggi (mis.
Miner vs Cerscospora/Phoma pada evaluasi model ini).
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <span class="section-num">—</span>
    <span class="section-title">Ringkasan Model</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.markdown("""
    <div class="stat-box">
        <span class="stat-num">5</span>
        <span class="stat-label">Kelas Klasifikasi</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-box">
        <span class="stat-num">ConvNeXt-T</span>
        <span class="stat-label">Arsitektur Backbone</span>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-box">
        <span class="stat-num">91.55%</span>
        <span class="stat-label">Akurasi Model Terpilih</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Mulai Klasifikasi →"):
    st.switch_page("steps/upload.py")

st.markdown("""
<div class="footer-note">
COFFEE LEAF DISEASE CLASSIFIER — CONVNEXT TINY — JMUBEN DATASET
</div>
""", unsafe_allow_html=True)