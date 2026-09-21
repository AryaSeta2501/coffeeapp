"""
common.py — Konfigurasi dan fungsi bersama untuk semua halaman.

>>> WAJIB DIVERIFIKASI (lihat komentar di masing-masing bagian):
- MODEL_PATH
- CLASS_NAMES (urutan harus sama dengan train_generator.class_indices)
- preprocess_image() (fungsi preprocessing harus identik dengan training)
"""

import streamlit as st
import numpy as np
from PIL import Image

# ============================================================
# KONFIGURASI
# ============================================================

# >>> WAJIB DIISI: repo model kamu di Hugging Face, format "username/nama-repo"
HF_REPO_ID = "yuliseta/coffeeapp"
 
# >>> WAJIB DIISI: nama file model PERSIS seperti di repo HF kamu
# (bukan checkpoint Eksperimen 4 yang dicurigai data leakage, kecuali
# sudah dikonfirmasi aman)
HF_FILENAME = "best_model_phase2.keras"
 
# Kalau repo HF kamu PRIVATE, isi token di Streamlit Cloud lewat menu
# Settings -> Secrets dengan format:
#   HF_TOKEN = "hf_xxxxxxxxxxxx"
# Kalau repo PUBLIC, biarkan ini apa adanya — tidak akan dipakai.
def _get_hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None

# >>> VERIFIKASI: urutan HARUS sama dengan train_generator.class_indices
CLASS_NAMES = ["Cerscospora", "Healthy", "Leaf_rust", "Miner", "Phoma"]

# Dikonfirmasi dari error runtime model sebelumnya: expected shape (224,224,3)
IMG_SIZE = (224, 224)

CLASS_INFO = {
    "Cerscospora": (
        "Penyakit bercak daun akibat jamur Cercospora coffeicola yang ditandai "
        "bercak coklat kehitaman pada daun. Pencegahan dan pengendalian: "
        "pangkas bagian yang terinfeksi, jaga sirkulasi udara, hindari kelembapan "
        "berlebih, lakukan sanitasi kebun, dan gunakan fungisida sesuai anjuran bila diperlukan."
    ),

    "Healthy": (
        "Daun kopi sehat tanpa gejala penyakit. Pertahankan kesehatan tanaman "
        "dengan pemupukan seimbang, penyiraman yang cukup, sanitasi kebun, "
        "serta lakukan pemantauan rutin terhadap hama dan penyakit."
    ),

    "Leaf_rust": (
        "Penyakit karat daun akibat jamur Hemileia vastatrix yang ditandai "
        "bercak kuning-oranye pada permukaan bawah daun. Pencegahan dan pengendalian: "
        "gunakan varietas tahan, lakukan pemangkasan untuk meningkatkan sirkulasi udara, "
        "bersihkan daun yang terinfeksi, dan aplikasikan fungisida sesuai rekomendasi jika diperlukan."
    ),

    "Miner": (
        "Kerusakan akibat larva pengorok daun (leaf miner) yang membentuk lorong "
        "di dalam jaringan daun. Pencegahan dan pengendalian: "
        "petik dan musnahkan daun yang terserang, pelihara musuh alami, "
        "lakukan monitoring rutin, dan gunakan insektisida yang direkomendasikan "
        "apabila serangan telah melewati ambang kendali."
    ),

    "Phoma": (
        "Penyakit bercak daun akibat jamur Phoma spp. yang ditandai bercak "
        "nekrotik pada tepi atau ujung daun. Pencegahan dan pengendalian: "
        "pangkas bagian yang terinfeksi, hindari genangan dan kelembapan tinggi, "
        "jaga kebersihan kebun, serta gunakan fungisida sesuai anjuran bila diperlukan."
    ),
}


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(repo_id: str = HF_REPO_ID, filename: str = HF_FILENAME):
    """Cache resource — model hanya di-download & di-load sekali per proses
    server (tersimpan di cache Hugging Face lokal setelah pertama kali),
    dipakai bersama lintas halaman dan lintas sesi."""
    import tensorflow as tf
    from huggingface_hub import hf_hub_download
 
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=_get_hf_token(),  # None kalau repo public, itu tidak masalah
        )
    except Exception as e:
        return None, f"Gagal download dari Hugging Face ({repo_id}/{filename}): {e}"
 
    try:
        model = tf.keras.models.load_model(local_path, compile=False)
        return model, None
    except Exception as e:
        return None, f"Model berhasil didownload tapi gagal di-load Keras: {e}"


def preprocess_image(pil_img: Image.Image, target_size=IMG_SIZE):
    """
    >>> VERIFIKASI PALING PENTING <<<
    Asumsi: ConvNeXt Tiny butuh tf.keras.applications.convnext.preprocess_input.
    Ganti kalau training kamu pakai preprocessing lain.
    """
    from tensorflow.keras.applications.convnext import preprocess_input

    img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, pil_img: Image.Image):
    x = preprocess_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    return preds


# ============================================================
# CSS — gaya senada portofolio (DM Sans / DM Mono / Playfair Display)
# ============================================================

def inject_css():
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@400;500&display=swap" rel="stylesheet">

    <style>
    :root {
        --black: #0e0e0e;
        --white: #fafaf8;
        --gray-100: #f4f3f0;
        --gray-200: #e8e6e1;
        --gray-400: #a8a49c;
        --gray-600: #6b6760;
        --gray-800: #2c2b28;
        --teal: #0f6e56;
        --teal-soft: #e1f5ee;
        --amber: #854f0b;
        --amber-soft: #faeeda;
        --coral: #993c1d;
        --coral-soft: #faece7;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: var(--black);
    }

    .stApp { background: var(--white); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1000px;
    }

    /* ── CUSTOM TOP NAV (menggantikan sidebar default Streamlit) ── */
    .custom-nav-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid var(--gray-200);
        margin-bottom: 2rem;
    }
    .nav-logo {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: var(--gray-600);
    }

    [data-testid="stPageLink"] {
        font-family: 'DM Mono', monospace !important;
    }
    [data-testid="stPageLink"] p {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase;
        color: var(--gray-600) !important;
    }
    [data-testid="stPageLink"]:hover p {
        color: var(--black) !important;
    }

    .eyebrow {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        color: var(--gray-400);
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .eyebrow::before {
        content: '';
        display: inline-block;
        width: 28px; height: 1px;
        background: var(--gray-400);
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-weight: 400;
        font-size: clamp(2rem, 4.5vw, 3rem);
        line-height: 1.15;
        letter-spacing: -0.01em;
        color: var(--black);
        margin-bottom: 0.75rem;
    }
    .hero-title em { font-style: italic; color: var(--gray-400); }
    .hero-desc {
        font-size: 1rem;
        color: var(--gray-600);
        max-width: 620px;
        line-height: 1.8;
        font-weight: 300;
        margin-bottom: 1.5rem;
    }

    .section-header {
        display: flex;
        align-items: baseline;
        gap: 1.5rem;
        margin: 1.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--gray-200);
    }
    .section-num {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--gray-400);
        letter-spacing: 0.08em;
    }
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        font-weight: 400;
        color: var(--black);
        letter-spacing: -0.01em;
    }

    .stat-box {
        border: 1px solid var(--gray-200);
        padding: 1.25rem;
        text-align: left;
    }
    .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 400;
        color: var(--black);
        display: block;
        margin-bottom: 0.3rem;
    }
    .stat-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--gray-400);
    }

    [data-testid="stFileUploader"] {
        border: 1px solid var(--gray-200);
        padding: 1.5rem;
        background: var(--gray-100);
    }
    [data-testid="stFileUploader"] section { background: transparent; border: none; }

    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 400;
        letter-spacing: 0.04em;
        background: var(--black);
        color: var(--white);
        border: 1px solid var(--black);
        border-radius: 0;
        padding: 0.6rem 1.75rem;
        transition: all 0.22s ease;
    }
    .stButton > button:hover { background: transparent; color: var(--black); }

    .result-card {
        border: 1px solid var(--gray-200);
        padding: 2rem;
        margin-top: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .result-tag {
        font-family: 'DM Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border: 1px solid;
        display: inline-block;
        width: fit-content;
    }
    .tag-healthy { border-color: var(--teal); color: var(--teal); background: var(--teal-soft); }
    .tag-disease { border-color: var(--coral); color: var(--coral); background: var(--coral-soft); }
    .result-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 400;
        color: var(--black);
    }
    .result-desc { font-size: 0.9rem; color: var(--gray-600); line-height: 1.7; font-weight: 300; }
    .result-confidence {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: var(--gray-400);
        letter-spacing: 0.06em;
    }

    .prob-row { display: flex; flex-direction: column; gap: 0.3rem; margin-bottom: 0.85rem; }
    .prob-label-row { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--gray-800); }
    .prob-value { font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--gray-400); }
    .prob-bar-bg { height: 4px; background: var(--gray-200); width: 100%; position: relative; }
    .prob-bar-fill { position: absolute; left: 0; top: 0; height: 100%; background: var(--gray-800); }

    .disclaimer {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        color: var(--amber);
        background: var(--amber-soft);
        border: 1px solid var(--amber);
        padding: 0.9rem 1.2rem;
        margin-top: 1rem;
        line-height: 1.6;
    }
    .disclaimer.error {
        color: var(--coral); background: var(--coral-soft); border-color: var(--coral);
    }

    .footer-note {
        font-family: 'DM Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        color: var(--gray-400);
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--gray-200);
    }
    </style>
    """, unsafe_allow_html=True)