"""
app.py
------
Streamlit demo app — upload an image, get a ballet position prediction.

Run:
    streamlit run app.py
"""

import base64
import os
import tempfile

import numpy as np
import streamlit as st

from src.extraction.pose_extractor import PoseExtractor
from src.preprocessing.normalizer import extract_all_features
from src.models.classifier import BalletClassifierBase

st.set_page_config(
    page_title="Ballet Pose Classifier",
    page_icon="🩰",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------- styling ---

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --blush:      #FDF7F5;
    --blush-deep: #F6E7E4;
    --ivory:      #FFFCFB;
    --rose:       #C98A96;
    --rose-deep:  #A6636F;
    --rose-soft:  #E8C6C8;
    --ink:        #4A3B3D;
    --muted:      #9C8285;
    --line:       #F0DEDA;
    --shadow:     0 8px 30px rgba(184, 140, 143, 0.10);
    --radius:     20px;
}

/* Page canvas ---------------------------------------------------------- */
.stApp {
    background:
        radial-gradient(1100px 600px at 12% -8%,  #FBEDEA 0%, transparent 60%),
        radial-gradient(900px 520px at 92% 4%,    #F7EFEA 0%, transparent 55%),
        linear-gradient(180deg, var(--blush) 0%, #FBF3F1 100%);
    background-attachment: fixed;
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    color: var(--ink);
}

/* Hide default Streamlit chrome (no sidebar, no menu, no footer) --------- */
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

.block-container {
    max-width: 1040px;
    padding: 3.2rem 1.6rem 4.5rem;
}

/* Header --------------------------------------------------------------- */
.hero { text-align: center; margin-bottom: 2.6rem; }
.hero-icon {
    font-size: 2.1rem;
    display: block;
    margin-bottom: 0.4rem;
    opacity: 0.9;
}
.hero-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(2.6rem, 6vw, 3.9rem);
    font-weight: 500;
    letter-spacing: 0.01em;
    line-height: 1.08;
    margin: 0;
    color: var(--ink);
}
.hero-title em {
    font-style: italic;
    color: var(--rose-deep);
}
.hero-subtitle {
    font-size: 1rem;
    font-weight: 300;
    color: var(--muted);
    margin: 0.9rem auto 0;
    max-width: 30rem;
    line-height: 1.6;
}
.hero-rule {
    width: 78px;
    height: 1px;
    margin: 1.6rem auto 0;
    background: linear-gradient(90deg, transparent, var(--rose-soft), transparent);
    position: relative;
}
.hero-rule::after {
    content: "";
    position: absolute;
    top: -2px; left: 50%;
    width: 5px; height: 5px;
    margin-left: -2.5px;
    border-radius: 50%;
    background: var(--rose-soft);
}

/* Cards ---------------------------------------------------------------- */
.card {
    background: var(--ivory);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.6rem 1.7rem 1.8rem;
    margin-bottom: 1.4rem;
}
.card-title {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--rose);
    margin-bottom: 1.1rem;
}

/* Upload card ---------------------------------------------------------- */
.upload-intro {
    text-align: center;
    margin: 0 0 0.2rem;
}
.upload-intro h3 {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 500;
    color: var(--ink);
    margin: 0 0 0.35rem;
}
.upload-intro p {
    font-size: 0.86rem;
    font-weight: 300;
    color: var(--muted);
    margin: 0;
}

[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, #FFFDFC 0%, #FDF4F2 100%);
    border: 1px dashed var(--rose-soft);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--rose);
    box-shadow: 0 10px 34px rgba(184, 140, 143, 0.16);
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
}
[data-testid="stFileUploader"] section > div { color: var(--ink); }
[data-testid="stFileUploader"] small { color: var(--muted) !important; }
[data-testid="stFileUploader"] svg { fill: var(--rose); color: var(--rose); }

/* Browse / secondary buttons */
[data-testid="stFileUploader"] button,
.stButton > button,
.stDownloadButton > button {
    background: var(--rose-deep) !important;
    color: #FFF8F7 !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.42rem 1.35rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(166, 99, 111, 0.24) !important;
    transition: transform 0.15s ease, background 0.2s ease !important;
}
[data-testid="stFileUploader"] button:hover,
.stButton > button:hover {
    background: var(--rose) !important;
    transform: translateY(-1px);
}

/* Uploaded-file chip */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #FDF4F2;
    border-radius: 12px;
    padding: 0.35rem 0.6rem;
}

/* Preview card --------------------------------------------------------- */
.preview-frame {
    border-radius: 14px;
    overflow: hidden;
    background: var(--blush-deep);
    line-height: 0;
}
.preview-frame img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 14px;
}
.preview-caption {
    font-size: 0.76rem;
    font-weight: 300;
    color: var(--muted);
    text-align: center;
    margin-top: 0.85rem;
    letter-spacing: 0.02em;
}

/* Results card --------------------------------------------------------- */
.pred-eyebrow {
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.pred-name {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: clamp(1.9rem, 4.2vw, 2.5rem);
    font-weight: 600;
    line-height: 1.12;
    color: var(--rose-deep);
    margin: 0;
}
.pred-confidence {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    background: #FBEDEB;
    border: 1px solid var(--rose-soft);
    color: var(--rose-deep);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.pred-divider {
    height: 1px;
    background: var(--line);
    margin: 1.4rem 0 1.15rem;
}
.prob-heading {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--rose);
    margin-bottom: 0.9rem;
}

/* Soft fade at the bottom of the scrollable probability list */
.prob-wrap { position: relative; }
.prob-wrap::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 0;
    height: 42px;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255,252,251,0) 0%, var(--ivory) 88%);
}
.prob-list {
    max-height: 340px;
    overflow-y: auto;
    padding-right: 0.45rem;
}
.prob-list::-webkit-scrollbar { width: 5px; }
.prob-list::-webkit-scrollbar-track { background: transparent; }
.prob-list::-webkit-scrollbar-thumb {
    background: var(--rose-soft);
    border-radius: 999px;
}

.prob-row { margin-bottom: 0.78rem; }
.prob-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.82rem;
    font-weight: 300;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.prob-row.is-top .prob-meta {
    font-weight: 600;
    color: var(--rose-deep);
}
.prob-value { font-variant-numeric: tabular-nums; }
.prob-track {
    height: 7px;
    border-radius: 999px;
    background: #F5E9E7;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #E3BEC2, #CE97A1);
}
.prob-row.is-top .prob-fill {
    background: linear-gradient(90deg, #D9A3AC, var(--rose-deep));
    box-shadow: 0 1px 6px rgba(166, 99, 111, 0.28);
}

/* Notices -------------------------------------------------------------- */
.notice {
    background: var(--ivory);
    border: 1px solid var(--line);
    border-left: 3px solid var(--rose);
    border-radius: 14px;
    padding: 0.95rem 1.15rem;
    font-size: 0.88rem;
    font-weight: 300;
    color: var(--ink);
    box-shadow: var(--shadow);
}
.notice strong { font-weight: 600; color: var(--rose-deep); }

.placeholder {
    border: 1px dashed var(--rose-soft);
    border-radius: var(--radius);
    background: rgba(255, 253, 252, 0.6);
    padding: 2.6rem 1.5rem;
    text-align: center;
    font-size: 0.88rem;
    font-weight: 300;
    color: var(--muted);
}

/* Spinner */
.stSpinner > div { border-top-color: var(--rose) !important; }
</style>
"""


# ------------------------------------------------------------ label pretty ---

# Display-only prettification of the raw class labels (French accents, casing).
_WORD_STYLE = {
    "a": "à",
    "la": "la",
    "seconde": "seconde",
    "degage": "Dégagé",
    "derriere": "Derrière",
    "plie": "Plié",
    "passe": "Passé",
    "penche": "Penché",
}


def display_name(raw_label: str) -> str:
    """Turn 'degage_a_la_seconde' into 'Dégagé à la seconde'."""
    return " ".join(_WORD_STYLE.get(w, w.title()) for w in str(raw_label).split("_"))


# ------------------------------------------------------------- UI sections ---

def html(*parts: str) -> None:
    """Emit raw HTML. Parts are joined without indentation, since Streamlit's
    Markdown parser turns indented lines into code blocks."""
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_header() -> None:
    html(
        '<div class="hero">',
        '<span class="hero-icon">🩰</span>',
        '<h1 class="hero-title">Ballet <em>Pose</em> Classifier</h1>',
        '<p class="hero-subtitle">Upload a photo of a ballet position '
        'and the model will identify it.</p>',
        '<div class="hero-rule"></div>',
        "</div>",
    )


def render_uploader():
    """Draw the upload card and return the uploaded file (or None)."""
    html(
        '<div class="upload-intro">',
        "<h3>Begin with a photograph</h3>",
        "<p>JPG, JPEG or PNG &nbsp;·&nbsp; a single dancer, full body in frame</p>",
        "</div>",
    )
    return st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )


def render_preview_card(image_bytes: bytes, mime_type: str) -> None:
    """Show the uploaded image inside a rounded card."""
    encoded = base64.b64encode(image_bytes).decode()
    html(
        '<div class="card">',
        '<div class="card-title">Image Preview</div>',
        '<div class="preview-frame">',
        f'<img src="data:{mime_type};base64,{encoded}" alt="Uploaded ballet pose" />',
        "</div>",
        '<div class="preview-caption">Your uploaded photograph</div>',
        "</div>",
    )


def render_results_card(class_names, proba) -> None:
    """Show the predicted position plus a bar for every class probability."""
    ranked = sorted(zip(class_names, proba), key=lambda pair: -pair[1])
    top_label, top_prob = ranked[0]

    rows = []
    for index, (label, prob) in enumerate(ranked):
        top_class = " is-top" if index == 0 else ""
        # A hairline minimum width keeps near-zero bars from disappearing.
        width = max(float(prob) * 100, 1.2)
        rows.append(
            f'<div class="prob-row{top_class}">'
            f'<div class="prob-meta"><span>{display_name(label)}</span>'
            f'<span class="prob-value">{prob:.1%}</span></div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width: {width:.2f}%"></div>'
            f"</div></div>"
        )

    html(
        '<div class="card">',
        '<div class="card-title">Results</div>',
        '<div class="pred-eyebrow">Detected Position</div>',
        f'<h2 class="pred-name">{display_name(top_label)}</h2>',
        f'<div class="pred-confidence">{top_prob:.1%} confidence</div>',
        '<div class="pred-divider"></div>',
        '<div class="prob-heading">All Class Probabilities</div>',
        f'<div class="prob-wrap"><div class="prob-list">{"".join(rows)}</div></div>',
        "</div>",
    )


def render_notice(message: str, title: str = "") -> None:
    heading = f"<strong>{title}</strong><br/>" if title else ""
    html(f'<div class="notice">{heading}{message}</div>')


def render_placeholder(message: str) -> None:
    html(f'<div class="placeholder">{message}</div>')


# ------------------------------------------------------------------ model ---

@st.cache_resource
def load_model():
    """Load the most recent trained model and its label classes (cached)."""
    saved_dir = "models/saved"
    pkl_files = [f for f in os.listdir(saved_dir) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError("No trained model found. Run training first.")
    latest = max(pkl_files, key=lambda f: os.path.getmtime(os.path.join(saved_dir, f)))
    model = BalletClassifierBase.load(os.path.join(saved_dir, latest))
    classes = np.load("data/splits/label_classes.npy", allow_pickle=True)
    return model, classes


def predict(image_bytes: bytes, model):
    """Extract pose features from the image bytes and return class probabilities.

    Returns None when no human pose can be detected.
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        extractor = PoseExtractor()
        result = extractor.extract(tmp_path)
        if not result.success:
            return None
        angles = extract_all_features(result.keypoints)
        return model.predict_proba([angles])[0]
    finally:
        os.unlink(tmp_path)


# -------------------------------------------------------------------- page ---

st.markdown(STYLES, unsafe_allow_html=True)
render_header()

try:
    model, class_names = load_model()
    model_loaded = True
except Exception as exc:  # noqa: BLE001 — surface any load failure to the user
    render_notice(f"{exc} Run training first, then reload this page.", "Model not loaded")
    model_loaded = False

uploaded_file = render_uploader()

if uploaded_file and model_loaded:
    image_bytes = uploaded_file.getvalue()
    left, right = st.columns(2, gap="large")

    with left:
        render_preview_card(image_bytes, uploaded_file.type or "image/jpeg")

    with right:
        with st.spinner("Reading the pose..."):
            proba = predict(image_bytes, model)

        if proba is None:
            render_notice(
                "We couldn't detect a human pose in this image. "
                "Try a clearer photo with the full body in frame.",
                "No pose detected",
            )
        else:
            render_results_card(class_names, proba)

elif model_loaded:
    render_placeholder("Your image preview and results will appear here.")
