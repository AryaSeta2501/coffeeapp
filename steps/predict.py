import streamlit as st
from PIL import Image
import io
from common import load_model, predict, MODEL_PATH

st.markdown("""
<div class="section-header">
    <span class="section-num">03</span>
    <span class="section-title">Jalankan Klasifikasi</span>
</div>
""", unsafe_allow_html=True)

# Gating: kalau user loncat ke halaman ini lewat nav tanpa upload gambar dulu
if "image_bytes" not in st.session_state:
    st.warning("Belum ada gambar yang diunggah.")
    st.page_link("steps/upload.py", label="← Kembali ke Upload Gambar")
    st.stop()

image = Image.open(io.BytesIO(st.session_state["image_bytes"]))

col1, col2 = st.columns([1, 1], gap="large")
with col1:
    st.image(image, caption="Gambar yang akan diklasifikasi", width="stretch")
with col2:
    st.markdown(
        '<p style="color:var(--gray-600); font-size:0.9rem; margin-top:1rem;">'
        'Gambar siap diproses. Klik tombol di bawah untuk menjalankan model '
        'klasifikasi ConvNeXt Tiny.</p>',
        unsafe_allow_html=True,
    )
    run = st.button("Jalankan Klasifikasi →")

if run:
    with st.status("Menjalankan klasifikasi...", expanded=True) as status:
        st.write("🔹 Memuat model...")
        model, err = load_model(MODEL_PATH)

        if err is not None:
            status.update(label="Gagal memuat model", state="error", expanded=True)
        else:
            st.write("🔹 Melakukan preprocessing gambar...")
            st.write("🔹 Menjalankan inferensi...")
            preds = predict(model, image)
            status.update(label="Klasifikasi selesai ✓", state="complete", expanded=False)

    if err is not None:
        st.markdown(f"""
        <div class="disclaimer error">
        ✕ Gagal memuat model dari <code>{MODEL_PATH}</code>.<br>
        Error: {err}<br><br>
        Pastikan MODEL_PATH di common.py sudah diarahkan ke file model yang
        benar dan model tersebut tersedia di direktori deploy.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Simpan hasil prediksi (sebagai list, bukan np.array, karena
        # session_state idealnya berisi objek yang mudah di-serialize)
        st.session_state["preds"] = preds.tolist()
        st.switch_page("steps/result.py")