"""EyeCLIP demo — research prototype, not a diagnostic tool.

Upload one fundus photograph and see its similarity to four reference classes learned from 40 labelled
RFMiD images (seed-0 10-shot). All runs on CPU from committed assets; the uploaded image is processed in
memory and never written to disk.

    pip install -r requirements-demo.txt && streamlit run app/streamlit_app.py
Assets are read from app/assets/ (override with EYECLIP_DEMO_ASSETS, e.g. for a test copy).
"""
import base64
import io
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ASSETS = Path(os.environ.get("EYECLIP_DEMO_ASSETS", Path(__file__).resolve().parent / "assets"))
MODEL_ID, PRETRAINED = "ViT-B-16-quickgelu", "openai"
RFMiD_SOURCE_URL = "https://doi.org/10.5281/zenodo.7505822"
DISCLAIMER = "Research prototype. Not a diagnostic tool."
DISCLAIMER_FULL = "Research prototype. Not a diagnostic tool. Scores are similarity values, not calibrated probabilities."
SHIFT_NOTE = ("Cross-dataset tests (RFMiD to FIVES) showed that decision thresholds shift across devices and "
              "hospitals, so scores may be unreliable on other sources.")
SYNTHETIC_NOTE = ("Synthetic placeholder assets: the reference images and embeddings in this build are generated, "
                  "so no value on this page reflects RFMiD data.")


@st.cache_resource
def load_assets():
    meta = json.loads((ASSETS / "meta.json").read_text())
    ref = np.load(ASSETS / "reference.npz", allow_pickle=False)
    sup = np.load(ASSETS / "support.npz", allow_pickle=False)
    return meta, ref, sup


@st.cache_resource
def load_model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID, pretrained=PRETRAINED)
    return model.eval(), preprocess


@st.cache_resource
def load_classifier():
    """Fit on the support set once: classifier and class prototypes."""
    from sklearn.linear_model import LogisticRegression

    meta, ref, sup = load_assets()
    classes = meta["classes"]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sup["emb"], sup["labels"])
    proto = np.stack([sup["emb"][sup["labels"] == c].mean(axis=0) for c in classes])
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    return clf, proto


def embed(image, model, preprocess):
    """Same path as the notebook: open_clip preprocess, encode_image, float, L2 normalise."""
    import torch

    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(x)
    f = f.float()
    return (f / f.norm(dim=-1, keepdim=True)).numpy()[0]


THEMES = {
    "Light": {
        "bg": "#F5F8FC",
        "bg_glow": "rgba(14,159,173,0.09)",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFD",
        "text": "#17243B",
        "text_secondary": "#53647D",
        "muted": "#5B6B80",
        "border": "#DCE4EF",
        "accent": "#0E9FAD",
        "accent_soft": "#E6F7F9",
        "accent_text": "#0B7285",
        "warn_bg": "#FFF7E8",
        "warn_text": "#8A5A00",
        "warn_border": "#F3D9A6",
        "danger_bg": "#FDECEE",
        "danger_text": "#A1263B",
        "danger_border": "#F4C7CF",
        "ok_bg": "#ECF8F1",
        "ok_text": "#1E6B3F",
        "ok_border": "#BFE3CC",
        "shadow": "0 8px 24px rgba(23,36,59,0.06)",
        "chart_bg": "transparent",
        "chart_grid": "#E6EDF5",
        "chart_text": "#53647D",
        "chart_muted": "#8A99AD",
        "img_bg": "#E5E9EF",
        "uploader_bg": "#FBFCFE",
        "button_bg": "#FFFFFF",
        "button_text": "#17243B",
        "button_border": "#DCE4EF",
        "scheme": "light",
    },
    "Dark": {
        "bg": "#0B1220",
        "bg_glow": "rgba(53,195,209,0.13)",
        "surface": "#111C2E",
        "surface_alt": "#17243A",
        "text": "#F1F5FB",
        "text_secondary": "#B3C0D4",
        "muted": "#94A3B8",
        "border": "#293850",
        "accent": "#35C3D1",
        "accent_soft": "rgba(53,195,209,0.14)",
        "accent_text": "#7FE7F1",
        "warn_bg": "rgba(245,158,11,0.14)",
        "warn_text": "#FCD34D",
        "warn_border": "rgba(245,158,11,0.28)",
        "danger_bg": "rgba(244,63,94,0.14)",
        "danger_text": "#FDA4AF",
        "danger_border": "rgba(244,63,94,0.28)",
        "ok_bg": "rgba(34,197,94,0.14)",
        "ok_text": "#86EFAC",
        "ok_border": "rgba(34,197,94,0.28)",
        "shadow": "0 14px 40px rgba(0,0,0,0.40)",
        "chart_bg": "transparent",
        "chart_grid": "#243247",
        "chart_text": "#B3C0D4",
        "chart_muted": "#94A3B8",
        "img_bg": "#08101D",
        "uploader_bg": "#17243A",
        "button_bg": "#17243A",
        "button_text": "#F1F5FB",
        "button_border": "#293850",
        "scheme": "dark",
    },
}

BRAND_MARK = (
    "<svg class='brand-mark' viewBox='0 0 32 32' aria-hidden='true' focusable='false'>"
    "<circle cx='16' cy='16' r='10' fill='none' stroke='currentColor' stroke-width='1.7'/>"
    "<circle cx='16' cy='16' r='3.2' fill='var(--accent)'/>"
    "<circle cx='16' cy='3.5' r='1.5' fill='var(--accent)'/>"
    "<circle cx='28.5' cy='16' r='1.5' fill='var(--accent)'/>"
    "<circle cx='16' cy='28.5' r='1.5' fill='var(--accent)'/>"
    "<circle cx='3.5' cy='16' r='1.5' fill='var(--accent)'/>"
    "</svg>"
)


def inject_styles(theme: str) -> None:
    t = THEMES[theme]
    css = f"""
    <style>
    :root {{
      --bg: {t['bg']}; --surface: {t['surface']}; --surface-alt: {t['surface_alt']};
      --bg-glow: {t['bg_glow']};
      --text: {t['text']}; --text-secondary: {t['text_secondary']}; --muted: {t['muted']};
      --border: {t['border']}; --accent: {t['accent']}; --accent-soft: {t['accent_soft']};
      --accent-text: {t['accent_text']}; --warn-bg: {t['warn_bg']}; --warn-text: {t['warn_text']};
      --warn-border: {t['warn_border']}; --danger-bg: {t['danger_bg']}; --danger-text: {t['danger_text']};
      --danger-border: {t['danger_border']}; --ok-bg: {t['ok_bg']}; --ok-text: {t['ok_text']};
      --ok-border: {t['ok_border']}; --shadow: {t['shadow']}; --chart-bg: {t['chart_bg']};
      --chart-grid: {t['chart_grid']}; --chart-text: {t['chart_text']}; --chart-muted: {t['chart_muted']};
      --img-bg: {t['img_bg']}; --uploader-bg: {t['uploader_bg']}; --button-bg: {t['button_bg']};
      --button-text: {t['button_text']}; --button-border: {t['button_border']};
    }}
    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stCaptionContainer"] p,
    .stCaption, .brand-sub, .badge, .body-text, .secondary-text, .disclaimer,
    .status-chip, .ref-score {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .stApp {{
      background: radial-gradient(ellipse 62% 35% at 88% 0%, var(--bg-glow) 0%, transparent 72%), var(--bg);
      color: var(--text); color-scheme: {t['scheme']};
    }}
    [data-testid="stHeader"] {{
      background: transparent !important; background-color: transparent !important;
    }}
    [data-testid="stToolbar"] {{ background: transparent !important; }}
    [data-testid="stDecoration"] {{ background: transparent !important; }}
    .block-container {{ max-width: 1400px; padding-left: 2rem !important; padding-right: 2rem !important; }}
    h1,h2,h3,h4,h5,h6,.heading-font,.brand-title,.section-title,.result-class,.ref-label {{
      font-family: 'Manrope', 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
      background: var(--surface); border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {{ color: var(--muted) !important; }}
    [data-testid="stSidebar"] label {{ color: var(--text) !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {{
      color: var(--text-secondary);
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li strong {{
      color: var(--text);
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] {{
      border: none; border-bottom: 1px solid var(--border); border-radius: 0;
      background: transparent; overflow: visible; box-shadow: none;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details {{
      background: transparent; border: none !important; border-radius: 0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
      border-top: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary {{
      background: transparent !important; color: var(--text) !important;
      padding-left: 0 !important; padding-right: 0 !important;
      border-bottom: none;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover {{
      background: transparent !important; color: var(--accent-text) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary p,
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary svg {{
      color: var(--text) !important; fill: var(--text) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
      background: var(--surface) !important; color: var(--text-secondary);
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] span:not(.badge) {{
      color: var(--text-secondary) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] strong {{
      color: var(--text) !important;
    }}
    .app-header {{
      display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; flex-wrap:wrap;
      padding: 0 0 1rem 0; margin: 0 0 1rem 0; border-bottom: 1px solid var(--border);
    }}
    .brand-title {{
      font-family: 'Manrope','Inter',sans-serif; font-size: 32px; line-height: 1.15; font-weight: 800;
      color: var(--text); margin: 0; letter-spacing: -0.02em;
    }}
    .brand-title-row {{
      display:flex; align-items:center; gap:.75rem; flex-wrap:wrap;
    }}
    .brand-mark {{ width:36px; height:36px; flex:0 0 auto; color:var(--text); }}
    .sidebar-brand {{
      display:flex; align-items:center; gap:.55rem; margin-bottom:.1rem;
      font-family:'Manrope','Inter',sans-serif; font-size:22px; font-weight:800; color:var(--text);
    }}
    .sidebar-brand .brand-mark {{ width:28px; height:28px; }}
    .brand-line {{
      display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; margin-top:.35rem;
    }}
    .brand-sub {{
      color: var(--text-secondary); font-size: 16px; line-height: 1.5; margin: 0; font-weight: 400;
    }}
    .badge {{
      display:inline-flex; align-items:center; gap:.4rem; padding:.3rem .65rem; border-radius:999px;
      background: var(--accent-soft); color: var(--accent-text) !important; font-size:12px; font-weight:600;
      border:1px solid var(--border); white-space:nowrap;
    }}
    .section-title {{
      font-family: 'Manrope','Inter',sans-serif; font-size: 21px; font-weight: 700; color: var(--text);
      margin: 0 0 .55rem 0; letter-spacing: 0; text-transform: none;
    }}
    .panel {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
      padding: 1rem 1.1rem; box-shadow: var(--shadow);
    }}
    .panel + .panel {{ margin-top: 1rem; }}
    .meta-label {{
      font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em;
      margin-bottom: .25rem;
    }}
    .result-class {{
      font-family: 'Manrope','Inter',sans-serif; font-size: 34px; font-weight: 800; color: var(--text);
      line-height: 1.1; margin: .1rem 0 .4rem 0;
    }}
    .result-muted {{ color: var(--text-secondary); font-size: 16px; line-height: 1.5; }}
    .body-text {{ color: var(--text-secondary); font-size: 16px; line-height: 1.55; }}
    .secondary-text {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    .disclaimer {{
      background: var(--accent-soft); border: 1px solid var(--border); color: var(--text);
      padding: .8rem .95rem; border-radius: 12px; font-size: 14px; line-height: 1.5;
    }}
    .status-chip {{
      display:inline-block; padding:.34rem .7rem; border-radius:999px; font-size:13px; font-weight:600;
      margin:.25rem .4rem .1rem 0; line-height:1.3;
    }}
    .chip-warn {{ background: var(--warn-bg); color: var(--warn-text); border:1px solid var(--warn-border); }}
    .chip-danger {{ background: var(--danger-bg); color: var(--danger-text); border:1px solid var(--danger-border); }}
    .chip-ok {{ background: var(--ok-bg); color: var(--ok-text); border:1px solid var(--ok-border); }}
    .ref-grid-card {{
      background: var(--surface); border:1px solid var(--border); border-radius: 12px; padding: .65rem;
      height: 100%; transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
    }}
    .ref-grid-card:hover {{ transform: translateY(-2px); border-color: var(--accent); box-shadow: var(--shadow); }}
    .ref-img-wrap {{
      width: 100%; aspect-ratio: 4 / 3; background: var(--img-bg); border-radius: 10px;
      display:flex; align-items:center; justify-content:center; overflow:hidden; border:1px solid var(--border);
      margin-bottom: .55rem;
    }}
    .ref-img-wrap img {{
      width: 100%; height: 100%; object-fit: contain; object-position: center; display:block;
    }}
    .ref-label {{
      font-family: 'Manrope','Inter',sans-serif; font-size: 16px; font-weight: 700; color: var(--text); margin: 0;
      line-height: 1.2;
    }}
    .ref-score {{ font-size: 13px; color: var(--muted); margin-top: .2rem; }}
    /* Bordered container card: st.container(border=True, key="viz_card") */
    .st-key-viz_card {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      border-radius: 14px !important;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    /* Streamlit widgets theme adaptation */
    div[data-testid="stFileUploader"] section {{
      background: var(--accent-soft) !important; border: 1px dashed var(--accent) !important; border-radius: 14px !important;
      transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stFileUploader"] section:hover {{ border-color: var(--accent) !important; }}
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {{
      color: var(--text-secondary) !important;
    }}
    /* Uploaded file chip (filename/remove) */
    div[data-testid="stFileChip"] {{
      background: var(--surface-alt) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
      color: var(--text) !important;
      transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stFileChip"]:hover {{ border-color: var(--accent) !important; }}
    div[data-testid="stFileChip"] [data-testid="stFileChipName"] {{
      color: var(--text) !important;
    }}
    div[data-testid="stFileChip"] svg {{ color: var(--text-secondary) !important; fill: currentColor !important; }}
    /* File-type tile inside the chip is painted with a fixed dark fill; make it theme-aware */
    div[data-testid="stFileChip"] > div:first-of-type {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      color: var(--text-secondary) !important;
    }}
    div[data-testid="stFileChip"] [data-testid="stFileChipDeleteBtn"] button {{
      background: transparent !important; border: 1px solid transparent !important;
      border-radius: 8px !important; color: var(--text-secondary) !important;
      transition: background-color 150ms ease, color 150ms ease, border-color 150ms ease;
    }}
    div[data-testid="stFileChip"] [data-testid="stFileChipDeleteBtn"] button:hover {{
      background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important;
    }}
    div[data-testid="stFileUploader"] button {{
      background: var(--button-bg) !important; color: var(--button-text) !important;
      border: 1px solid var(--button-border) !important;
    }}
    .stButton > button {{
      background: var(--button-bg); color: var(--button-text); border: 1px solid var(--button-border);
      border-radius: 10px; font-weight: 500; transition: all 180ms ease;
    }}
    .stButton > button:hover {{ border-color: var(--accent); color: var(--text); background: var(--surface-alt); }}
    .stButton > button:active {{ transform: translateY(1px); }}
    [data-testid="stExpander"] {{
      border: 1px solid var(--border); border-radius: 14px; background: var(--surface); overflow: hidden;
      margin-top: .25rem;
    }}
    [data-testid="stMain"] [data-testid="stExpander"] {{ margin-top: 1rem !important; }}
    /* The native <details> frame is square-cornered; the outer expander supplies the card edge */
    [data-testid="stExpander"] details {{
      background: transparent; border: none !important; border-radius: 0 !important;
    }}
    [data-testid="stExpander"] details > summary {{
      background: var(--surface-alt) !important; color: var(--text) !important;
    }}
    [data-testid="stExpander"] details > summary p,
    [data-testid="stExpander"] details > summary span,
    [data-testid="stExpander"] details > summary svg {{
      color: var(--text) !important; fill: var(--text) !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
      background: var(--surface) !important; color: var(--text-secondary);
      border-top: 1px solid var(--border);
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] li,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] span {{
      color: var(--text-secondary) !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] strong {{
      color: var(--text) !important;
    }}
    [data-testid="stAlert"] {{ border-radius: 12px; transition: opacity 220ms ease; }}
    /* Native alert/tooltip/popover surfaces follow the app tokens (fixes light-blue box in dark mode) */
    [data-testid="stAlertContainer"] {{
      background: var(--accent-soft) !important; color: var(--accent-text) !important;
      border: 1px solid var(--border) !important; border-radius: 12px !important;
    }}
    [data-testid="stAlertContainer"] p,
    [data-testid="stAlertContainer"] span,
    [data-testid="stAlertContainer"] div,
    [data-testid="stAlertContainer"] a {{
      color: var(--accent-text) !important;
    }}
    [data-testid="stAlertContainer"] svg {{ fill: var(--accent-text) !important; color: var(--accent-text) !important; }}
    [data-testid="stTooltipContent"], [data-baseweb="tooltip"] {{
      background: var(--surface-alt) !important; color: var(--text) !important;
      border: 1px solid var(--border) !important; border-radius: 8px !important;
    }}
    [data-testid="stTooltipContent"] * {{ color: var(--text) !important; }}
    [data-testid="stTooltipIcon"], [data-testid="stTooltipHoverTarget"] {{
      background: transparent !important; color: var(--text-secondary) !important;
    }}
    [data-testid="stTooltipIcon"] button {{
      background: transparent !important; border: none !important; padding: 0 !important;
      color: var(--muted) !important; transition: color 150ms ease;
    }}
    [data-testid="stTooltipIcon"] button:hover {{ color: var(--text) !important; }}
    [data-testid="stTooltipIcon"] svg {{
      fill: none !important; stroke: currentColor !important; color: var(--muted) !important;
    }}
    [data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"] {{
      background: var(--surface) !important; color: var(--text) !important;
      border-color: var(--border) !important;
    }}
    [data-baseweb="notification"] {{ background: var(--surface-alt) !important; color: var(--text) !important; border-color: var(--border) !important; }}
    /* Main (hamburger) menu popover: Streamlit paints it with the config theme */
    [data-testid="stMainMenuPopover"] {{
      background: var(--surface) !important; color: var(--text) !important;
      border: 1px solid var(--border) !important; border-radius: 10px !important;
    }}
    [data-testid="stMainMenuPopover"] * {{ color: var(--text) !important; }}
    [data-testid="stMainMenuPopover"] [data-testid^="stMainMenuItem"] {{
      background: transparent !important; border-radius: 8px !important;
    }}
    [data-testid="stMainMenuPopover"] [data-testid^="stMainMenuItem"]:hover,
    [data-testid="stMainMenuPopover"] [data-testid^="stMainMenuItem"]:focus-visible {{
      background: var(--surface-alt) !important;
    }}
    [data-testid="stNotificationContentInfo"] p,
    [data-testid="stNotificationContentWarning"] p,
    [data-testid="stNotificationContentSuccess"] p,
    [data-testid="stNotificationContentError"] p {{ color: inherit !important; }}
    div[data-testid="stCaptionContainer"], .stCaption {{ background: transparent !important; }}
    div[data-testid="stCaptionContainer"] p, .stCaption {{ color: var(--muted) !important; font-size: 13px !important; }}
    /* The uploaded image carries the full card edge, including its lower corners. */
    .st-key-image_card [data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
    .st-key-image_card [data-testid="stElementContainer"] {{ margin: 0 !important; }}
    .st-key-image_card [data-testid="stMarkdown"] {{ margin-bottom: 0 !important; }}
    .st-key-image_card img {{
      width: 100% !important; height: auto !important; aspect-ratio: 4 / 3;
      max-height: min(60vh, 480px) !important; object-fit: contain !important; object-position: center;
      background: var(--img-bg);
      border-radius: 14px !important;
      border: 1px solid var(--border) !important;
      transition: border-color 160ms ease;
    }}
    .st-key-image_card:hover img {{ border-color: var(--accent) !important; }}
    /* Uploaded/displayed images: preserve aspect ratio, no distortion/cropping */
    [data-testid="stImage"] {{ display: flex; justify-content: center; align-items: center; }}
    [data-testid="stImage"] img {{
      border-radius: 12px; width: auto !important; max-width: 100% !important; height: auto !important;
      max-height: min(70vh, 560px); object-fit: contain !important; object-position: center;
      background: var(--img-bg); display: block; margin-left: auto; margin-right: auto;
    }}
    [data-testid="stVegaLiteChart"], [data-testid="stVegaLiteChart"] > div, [data-testid="stArrowVegaLiteChart"] {{
      background: transparent !important;
    }}
    /* Vega tooltip shown when hovering a chart mark (Streamlit injects its own,
       fixed light-theme colours for #vg-tooltip-element). */
    #vg-tooltip-element {{
      background-color: var(--surface-alt) !important;
      color: var(--text) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      box-shadow: var(--shadow) !important;
      padding: 8px 10px !important;
    }}
    #vg-tooltip-element * {{ color: inherit !important; }}
    [data-testid="stVegaLiteChart"] svg {{ background: transparent !important; }}
    /* Element hover toolbar (chart: copy spec / download / fullscreen) */
    [data-testid="stElementToolbar"] {{
      background: transparent !important;
      border: none !important;
      box-shadow: none !important;
      padding: 0 !important;
    }}
    /* Only the inner chip gets a surface, so the toolbar stays compact */
    [data-testid="stElementToolbarButtonContainer"] {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
    }}
    [data-testid="stElementToolbarButton"] {{ background: transparent !important; border: none !important; }}
    [data-testid="stElementToolbarButton"] button,
    [data-testid="stElementToolbar"] button,
    [data-testid="stBaseButton-elementToolbar"] {{
      background: transparent !important; color: var(--text-secondary) !important;
      border: 1px solid transparent !important; border-radius: 6px !important;
      transition: background-color 150ms ease, color 150ms ease;
    }}
    [data-testid="stElementToolbar"] button:hover,
    [data-testid="stBaseButton-elementToolbar"]:hover {{
      background: var(--surface) !important; color: var(--text) !important;
    }}
    [data-testid="stElementToolbar"] svg,
    [data-testid="stElementToolbar"] [data-testid="stIconMaterial"],
    [data-testid="stBaseButton-elementToolbar"] svg {{
      color: inherit !important; fill: currentColor !important;
    }}
    /* Expanded (fullscreen) element view: Streamlit paints it with the config
       backgroundColor, which stays light when the in-app theme is Dark. */
    [data-testid="stFullScreenFrame"] {{
      background: var(--bg) !important;
      color: var(--text) !important;
    }}
    [data-testid="stFullScreenFrame"] [data-testid="stElementToolbarButtonContainer"] {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
    }}
    [data-testid="stFullScreenFrame"] [data-testid="stBaseButton-headerNoPadding"] {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      color: var(--text-secondary) !important;
    }}
    [data-testid="stFullScreenFrame"] [data-testid="stIconMaterial"] {{
      color: var(--text-secondary) !important;
    }}
    /* The chart's "Show data" grid is a glide-data-grid canvas painted with the
       config theme (always light here); a canvas cannot be recoloured by CSS, so
       hide that action instead of showing a white grid in dark mode. Values stay
       visible as labels next to each bar. */
    [data-testid="stElementToolbar"] button[aria-label="Show data"] {{ display: none !important; }}
    /* Sidebar collapse/expand control visibility */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarExpandButton"] button,
    [data-testid="collapsedControl"] button,
    button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="stBaseButton-header"] {{
      color: var(--text) !important;
      background: var(--surface-alt) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      transition: background-color 160ms ease, border-color 160ms ease, color 160ms ease;
    }}
    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarExpandButton"] button svg,
    [data-testid="collapsedControl"] button svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-header"] svg {{
      color: var(--text) !important; fill: currentColor !important; stroke: currentColor !important;
    }}
    /* Icon-font glyphs carry an inline dark colour, so target the glyph itself */
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarExpandButton"] [data-testid="stIconMaterial"],
    [data-testid="collapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="stHeader"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarHeader"] [data-testid="stIconMaterial"] {{
      color: var(--text) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarExpandButton"] button:hover,
    [data-testid="collapsedControl"] button:hover,
    button[data-testid="stBaseButton-headerNoPadding"]:hover,
    button[data-testid="stBaseButton-header"]:hover {{
      background: var(--surface) !important; border-color: var(--accent) !important; color: var(--text) !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:focus-visible,
    [data-testid="collapsedControl"] button:focus-visible,
    button[data-testid="stBaseButton-headerNoPadding"]:focus-visible,
    button[data-testid="stBaseButton-header"]:focus-visible {{
      outline: 2px solid var(--accent) !important; outline-offset: 2px;
    }}
    [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
      background: transparent;
    }}
    /* Theme toggle: icon-only capsule with the selected mode as the thumb */
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) > div[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {{
      align-items: flex-end !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] {{
      gap: 2px; background: var(--border); border: 1px solid var(--border); border-radius: 999px; padding: 3px;
      width: fit-content; transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label {{
      margin: 0 !important; padding: 0 !important; border-radius: 50% !important;
      background: transparent; border: 1px solid transparent !important; min-height: 0 !important;
      width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
      transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked) {{
      background: var(--surface); border: 1px solid var(--border) !important;
      box-shadow: 0 1px 3px rgba(23,36,59,.18) !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label p {{
      font-size: 0; font-weight: 500; color: var(--text-secondary) !important; line-height: 1;
      display: inline-flex; align-items: center; justify-content: center; transition: color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked) p {{
      color: var(--text) !important; font-weight: 600;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label p::before {{
      content: ""; width: 17px; height: 17px; display: inline-block; background-color: currentColor;
      -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center; mask-position: center;
      -webkit-mask-size: contain; mask-size: contain; transition: opacity 180ms ease; opacity: .7;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked) p::before {{ opacity: 1; }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] > div:nth-child(1) label p::before {{
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cpath d='M12 2v2.2M12 19.8V22M4.93 4.93l1.56 1.56M17.51 17.51l1.56 1.56M2 12h2.2M19.8 12H22M4.93 19.07l1.56-1.56M17.51 6.49l1.56-1.56'/%3E%3C/svg%3E");
      mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cpath d='M12 2v2.2M12 19.8V22M4.93 4.93l1.56 1.56M17.51 17.51l1.56 1.56M2 12h2.2M19.8 12H22M4.93 19.07l1.56-1.56M17.51 6.49l1.56-1.56'/%3E%3C/svg%3E");
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] > div:nth-child(2) label p::before {{
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z'/%3E%3C/svg%3E");
      mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z'/%3E%3C/svg%3E");
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) [data-testid="stWidgetLabel"] {{ display:none; }}
    /* The SVG icons above replace the native radio dot */
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label[data-testid="stRadioOption"] > div > div:first-child {{
      display: none !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label[data-testid="stRadioOption"] > div {{
      gap: 0 !important; padding-left: 0 !important; margin-left: 0 !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:focus-visible) {{
      outline: 2px solid var(--accent) !important; outline-offset: 2px;
    }}
    /* Subtle appearance animations */
    @keyframes eyeFadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .panel, .st-key-viz_card, .ref-grid-card, [data-testid="stAlert"] {{
      animation: eyeFadeIn 220ms ease;
    }}
    @media (min-width: 641px) and (max-width: 1024px) {{
      .block-container {{ padding-left: 1.5rem !important; padding-right: 1.5rem !important; }}
    }}
    @media (max-width: 640px) {{
      .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
      .brand-mark {{ width:32px; height:32px; }}
      .brand-title {{ font-size: 26px; }}
      .brand-line {{ gap: .4rem; }}
      .badge {{ font-size: 11px; padding: .24rem .55rem; }}
      .section-title {{ font-size: 19px; }}
      .result-class {{ font-size: 26px; }}
      [data-testid="stImage"] img {{ max-height: 38vh !important; }}
      .st-key-viz_card {{ padding: 0.85rem !important; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }}
      .panel:hover, .ref-grid-card:hover {{ transform: none !important; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def status_chip(text: str, kind: str = "warn", title: str = "") -> str:
    cls = {"warn": "chip-warn", "danger": "chip-danger", "ok": "chip-ok"}.get(kind, "chip-warn")
    title_attr = f" title='{title}'" if title else ""
    return f"<span class='status-chip {cls}'{title_attr}>{text}</span>"


def lollipop(df, top, theme: str, ood_thr: float | None = None):
    t = THEMES[theme]
    lo, hi = float(df["sim"].min()) - 0.02, float(df["sim"].max()) + 0.02
    # Keep the OOD threshold visible when it sits near the observed similarities; skip it when it
    # would stretch the axis so far that the four classes collapse together.
    show_threshold = ood_thr is not None and (lo - 0.12) <= ood_thr <= (hi + 0.12)
    if show_threshold:
        lo, hi = min(lo, ood_thr - 0.015), max(hi, ood_thr + 0.015)
    df = df.assign(floor=lo)
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort="-x", axis=alt.Axis(title=None, labelFontSize=13, labelColor=t["text_secondary"], tickColor=t["border"], domainColor=t["border"])),
        x=alt.X(
            "sim:Q",
            scale=alt.Scale(domain=[lo, hi], zero=False),
            axis=alt.Axis(
                title="Cosine similarity (prototype)",
                format=".3f",
                tickCount=5,
                labelFontSize=12,
                titleFontSize=13,
                labelColor=t["chart_text"],
                titleColor=t["chart_text"],
                tickColor=t["chart_grid"],
                domainColor=t["chart_grid"],
            ),
        ),
        tooltip=[alt.Tooltip("class:N", title="Class"), alt.Tooltip("sim:Q", title="Cosine similarity", format=".4f")],
    )
    highlight = t["accent"]
    muted = t["chart_muted"]
    color = alt.condition(alt.datum["class"] == top, alt.value(highlight), alt.value(muted))
    stem = base.mark_rule(strokeWidth=3).encode(x="floor:Q", x2="sim:Q", color=color)
    dot = base.mark_circle(size=170, opacity=1).encode(color=color)
    label = base.mark_text(align="left", dx=10, fontSize=12, fontWeight=500).encode(
        text=alt.Text("sim:Q", format=".3f"), color=color
    )
    layers = stem + dot + label
    if show_threshold:
        thr_df = pd.DataFrame({"x": [ood_thr]})
        layers = layers + alt.Chart(thr_df).mark_rule(
            color=t["danger_text"], strokeDash=[4, 4], strokeWidth=1.5
        ).encode(x=alt.X("x:Q"))
        layers = layers + alt.Chart(thr_df).mark_text(
            text=f"OOD {ood_thr:.3f}", align="left", dx=5, baseline="top", fontSize=11, color=t["danger_text"]
        ).encode(x=alt.X("x:Q"), y=alt.value(4))
    chart = (
        layers
        .properties(height=240, background=t["chart_bg"], padding={"left": 8, "right": 24, "top": 8, "bottom": 8})
        .configure_view(stroke=None, fill=t["chart_bg"])
        .configure_axis(grid=False)
    )
    return chart, lo, hi


@st.cache_data(show_spinner=False)
def thumb_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def render_sidebar(meta, ref, sup, unc_thr, ood_thr):
    with st.sidebar:
        st.markdown(f"<div class='sidebar-brand'>{BRAND_MARK}<span>EyeCLIP</span></div>", unsafe_allow_html=True)
        st.caption("Reference Similarity Explorer")
        st.divider()

        classes = list(meta["classes"])
        ref_counts = {c: sum(str(lab) == c for lab in ref["labels"]) for c in classes}
        sup_counts = {c: sum(str(lab) == c for lab in sup["labels"]) for c in classes}
        license_short = meta.get("license", "").split(";")[0].strip()

        with st.expander("Reference dataset", expanded=True):
            st.markdown(
                " ".join(f"<span class='badge'>{c}</span>" for c in classes), unsafe_allow_html=True
            )
            st.markdown(f"**{meta['reference']['n']} images** · {len(classes)} classes")
            st.caption(f"{meta['reference']['split'].title()} split · single-label · 224 px thumbnails")
            st.caption("Per class: " + " · ".join(f"{c} {ref_counts[c]}" for c in classes))
            if license_short:
                st.caption(f"Source: [{license_short}]({RFMiD_SOURCE_URL})")
            st.caption("Used only as retrieval targets; never for fitting.")

        with st.expander("Model", expanded=False):
            st.markdown(f"**{meta['model']}**")
            st.caption(f"{meta.get('pretrained', 'openai')} weights · {meta['classifier']}")
            st.caption(f"Prototype: {meta['prototype']}")

        with st.expander("Support set", expanded=False):
            st.markdown(f"**{meta['support']['n']} images** · {meta['support']['shot']} per class")
            st.caption("Per class: " + " · ".join(f"{c} {sup_counts[c]}" for c in classes))
            st.caption(f"Seed {meta['support']['seed']} · class-balanced sample")
            st.caption("Fits the logistic-regression classifier and the class prototypes. No reference image is used for fitting.")

        with st.expander("Thresholds", expanded=True):
            st.markdown(f"**Review · {unc_thr:.3f}**")
            st.caption("Flag for review when top-class confidence falls below this.")
            st.markdown(f"**Domain shift · {ood_thr:.3f}**")
            st.caption("Warn when top-1 reference similarity falls below this.")


def main():
    favicon = Path(__file__).resolve().parent / "assets" / "eyeclip-icon.svg"
    st.set_page_config(
        page_title="EyeCLIP — Reference Similarity Explorer",
        page_icon=str(favicon),
        layout="wide",
    )

    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "Light"
    # Apply styles early based on current session theme (before most widgets render content)
    inject_styles(st.session_state.get("ui_theme", "Light"))

    if not (ASSETS / "meta.json").exists():
        st.error("Demo assets are missing. Run `scripts/export_demo_assets.py` and commit `app/assets/`.")
        st.stop()

    meta, ref, sup = load_assets()
    classes = meta["classes"]
    unc_thr, ood_thr = meta["uncertainty"]["threshold"], meta["ood"]["threshold"]

    # Header with theme toggle (top-right)
    header_left, header_right = st.columns([5, 1.35], vertical_alignment="top")
    with header_left:
        st.markdown(
            f"""
            <div class="app-header" style="border-bottom:none;margin-bottom:0;padding-bottom:0;">
              <div>
                <div class="brand-title-row">
                  {BRAND_MARK}
                  <div class="brand-title">EyeCLIP</div>
                  <span class="badge">Research prototype</span>
                </div>
                <div class="brand-line">
                  <span class="brand-sub">Reference similarity explorer · Compare a retinal photograph with four labelled classes.</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.markdown('<div class="theme-anchor"></div>', unsafe_allow_html=True)
        st.radio(
            "Theme",
            options=["Light", "Dark"],
            horizontal=True,
            key="ui_theme",
            label_visibility="collapsed",
        )

    theme = st.session_state.get("ui_theme", "Light")
    # Re-inject if theme changed this run to ensure latest tokens apply
    inject_styles(theme)

    st.markdown("<div style='border-bottom:1px solid var(--border);margin:0.25rem 0 1rem 0;'></div>", unsafe_allow_html=True)

    render_sidebar(meta, ref, sup, unc_thr, ood_thr)

    left_col, right_col = st.columns([1.15, 1.5], gap="medium")

    with left_col:
        st.markdown("<div class='section-title'>Start with an image</div>", unsafe_allow_html=True)
        up = st.file_uploader(
            "Upload a retinal photograph (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            key="fundus_upload",
            help="The uploaded image is processed in memory and never written to disk.",
        )
        if up is None:
            image = None
        else:
            image = Image.open(up)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            with st.container(key="image_card"):
                st.markdown(
                    f"<img src='data:image/png;base64,{b64}' alt='Uploaded fundus photograph' "
                    f"style='width:100%;height:auto;display:block;border-radius:12px;border:1px solid var(--border);"
                    f"background:var(--img-bg);' />",
                    unsafe_allow_html=True,
                )

    with right_col:
        if image is None:
            st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="panel">
                  <div class="meta-label">Status</div>
                  <div class="result-class" style="font-size:28px;font-weight:750;">Ready to compare</div>
                  <div class="body-text">Add one retinal photograph to see the model’s class prediction, how it compares with each class, and the closest reference images.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            st.info(SHIFT_NOTE)
        else:
            with st.spinner("Loading model..."):
                model, preprocess = load_model()
                clf, proto = load_classifier()
            with st.spinner("Analyzing image..."):
                emb = embed(image, model, preprocess)
                sims = proto @ emb                                    # cosine to each class prototype
                sim = ref["emb"] @ emb                                # cosine to each reference image
                order = np.argsort(-sim)
                proba = clf.predict_proba(emb[None])[0]
                pred = str(clf.classes_[int(np.argmax(proba))]); conf = float(proba.max())
                top1_sim = float(sim[order[0]])
                ood, flagged = top1_sim < ood_thr, conf < unc_thr

            st.markdown("<div class='section-title'>Prediction and classification</div>", unsafe_allow_html=True)
            panel_parts = ["<div class='panel'>", "<div class='meta-label'>Closest reference class</div>"]
            if ood:
                panel_parts.append("<div class='result-class' style='opacity:.8;'>No reliable match</div>")
                panel_parts.append("<div class='body-text' style='margin-bottom:.6rem;'>No reference class is a close match for this image.</div>")
            else:
                panel_parts.append(f"<div class='result-class'>{pred}</div>")
                panel_parts.append(
                    f"<div class='body-text' style='margin-bottom:.25rem;'>Logistic regression confidence (uncalibrated): <strong>{conf:.3f}</strong></div>"
                )
            panel_parts.append("<div class='secondary-text' style='margin-bottom:.7rem;'>Uncalibrated confidence from logistic regression on 40 support images. Not a calibrated disease probability.</div>")
            chips = []
            if ood:
                chips.append(status_chip(
                    "Domain shift suspected", "danger",
                    title="Top-1 reference similarity is below the OOD threshold, so the image may come from another device or dataset, or not be a fundus photograph.",
                ))
            if flagged:
                chips.append(status_chip(
                    "Below review threshold", "warn",
                    title="Top-class confidence is below the review threshold, so this case should be checked by a person.",
                ))
            if not ood and not flagged:
                chips.append(status_chip("Within reference range", "ok"))
            if chips:
                panel_parts.append("<div>" + "".join(chips) + "</div>")
            panel_parts.append("</div>")
            st.markdown("".join(panel_parts), unsafe_allow_html=True)

            if ood:
                with st.expander("Closest class anyway"):
                    st.markdown(
                        f"<div class='heading-font' style='font-size:1.25rem;font-weight:800;color:var(--text);margin:.15rem 0 .35rem;letter-spacing:-.01em;'>"
                        f"{pred}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"Logistic regression confidence (uncalibrated): **{conf:.3f}**")
                    st.caption("Uncalibrated confidence from logistic regression on 40 support images. Not a calibrated disease probability.")

            with st.container(border=True, key="viz_card"):
                st.markdown("<div class='section-title' style='margin-bottom:.35rem;'>Similarity visualization</div>", unsafe_allow_html=True)
                df = pd.DataFrame({"class": classes, "sim": sims}).sort_values("sim", ascending=False)
                chart, lo, hi = lollipop(df, top=pred, theme=theme, ood_thr=ood_thr)
                st.altair_chart(chart, width="stretch")
                st.caption(
                    f"Cosine similarity of the uploaded image to each class prototype (L2-normalised mean support embedding). "
                    f"Highlighted class: {pred}. Dashed line marks the OOD threshold ({ood_thr:.3f}). X-axis range shown: {lo:.3f}–{hi:.3f}. "
                    "Similarity values are not calibrated probabilities and are distinct from logistic regression confidence."
                )

    if image is None:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>The reference set</div>", unsafe_allow_html=True)
        st.caption(
            "Four labelled classes form the comparison set. Upload an image to explore its closest matches."
        )
        class_cards = []
        for cls in classes:
            match = [int(j) for j, lab in enumerate(ref["labels"]) if str(lab) == cls]
            src = thumb_data_uri(str(ASSETS / "thumbs" / f"{ref['stems'][match[0]]}.jpg")) if match else ""
            img_html = f"<img src='{src}' alt='{cls} reference image' />" if src else "<div class='secondary-text'>Image unavailable</div>"
            class_cards.append(
                "<div class='ref-grid-card'>"
                f"<div class='ref-img-wrap'>{img_html}</div>"
                f"<div class='ref-label'>{cls}</div>"
                f"<div class='ref-score'>{len(match)} references</div>"
                "</div>"
            )
        class_cols = st.columns(len(class_cards), gap="medium")
        for c, card_html in zip(class_cols, class_cards):
            with c:
                st.markdown(card_html, unsafe_allow_html=True)

    if image is not None:
        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Most similar reference images</div>", unsafe_allow_html=True)
        cards = []
        for idx in order[:5]:
            thumb = ASSETS / "thumbs" / f"{ref['stems'][idx]}.jpg"
            src = thumb_data_uri(str(thumb))
            label = str(ref['labels'][idx])
            score = float(sim[idx])
            img_html = f"<img src='{src}' alt='{label} reference image' />" if src else "<div class='secondary-text'>Image unavailable</div>"
            cards.append(
                "<div class='ref-grid-card'>"
                f"<div class='ref-img-wrap'>{img_html}</div>"
                f"<div class='ref-label'>{label}</div>"
                f"<div class='ref-score'>Similarity: {score:.3f}</div>"
                "</div>"
            )
        cols = st.columns(5, gap="medium")
        for c, card_html in zip(cols, cards):
            with c:
                st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        st.info(SHIFT_NOTE)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    st.divider()
    if meta.get("synthetic"):
        st.warning(SYNTHETIC_NOTE)
    st.caption(DISCLAIMER_FULL)


main()
