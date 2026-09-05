import streamlit as st
from PIL import Image
import io

st.markdown("""
<div class="section-header">
    <span class="section-num">02</span>
    <span class="section-title">Unggah Gambar Daun</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Pilih gambar daun kopi (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    # Simpan ke session_state supaya bisa dipakai di halaman Prediksi & Hasil
    st.session_state["image_bytes"] = image_bytes

    image = Image.open(io.BytesIO(image_bytes))
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption="Preview gambar", width="stretch")
    with col2:
        st.markdown(
            '<p style="color:var(--gray-600); font-size:0.9rem; margin-top:1rem;">'
            'Gambar berhasil diunggah. Klik tombol di bawah untuk lanjut ke '
            'halaman klasifikasi.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Lanjut ke Klasifikasi →"):
            st.switch_page("steps/predict.py")
else:
    st.info("Unggah gambar daun kopi untuk melanjutkan.")