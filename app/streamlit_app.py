"""EyeCLIP demo — research prototype, not a diagnostic tool.

Upload one fundus photograph and see its similarity to four reference classes learned from 40 labelled
RFMiD images (seed-0 10-shot). All runs on CPU from committed assets; the uploaded image is processed in
memory and never written to disk.

    pip install -r requirements-demo.txt && streamlit run app/streamlit_app.py
Assets are read from app/assets/ (override with EYECLIP_DEMO_ASSETS, e.g. for a test copy).
"""
import base64
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import open_clip
from PIL import Image
from sklearn.linear_model import LogisticRegression

ASSETS = Path(os.environ.get("EYECLIP_DEMO_ASSETS", Path(__file__).resolve().parent / "assets"))
MODEL_ID, PRETRAINED = "ViT-B-16-quickgelu", "openai"
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
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID, pretrained=PRETRAINED)
    return model.eval(), preprocess


@st.cache_resource
def load_classifier():
    """Fit on the support set once: classifier and class prototypes."""
    meta, ref, sup = load_assets()
    classes = meta["classes"]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sup["emb"], sup["labels"])
    proto = np.stack([sup["emb"][sup["labels"] == c].mean(axis=0) for c in classes])
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    return clf, proto


def embed(image, model, preprocess):
    """Same path as the notebook: open_clip preprocess, encode_image, float, L2 normalise."""
    x = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        f = model.encode_image(x)
    f = f.float()
    return (f / f.norm(dim=-1, keepdim=True)).numpy()[0]


THEMES = {
    "Light": {
        "bg": "#F5F8FC",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFD",
        "text": "#17243B",
        "text_secondary": "#53647D",
        "muted": "#718096",
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
        "img_bg": "#0F172A",
        "uploader_bg": "#FBFCFE",
        "button_bg": "#FFFFFF",
        "button_text": "#17243B",
        "button_border": "#DCE4EF",
    },
    "Dark": {
        "bg": "#0B1220",
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
        "chart_bg": "#111C2E",
        "chart_grid": "#243247",
        "chart_text": "#B3C0D4",
        "chart_muted": "#94A3B8",
        "img_bg": "#08101D",
        "uploader_bg": "#17243A",
        "button_bg": "#17243A",
        "button_text": "#F1F5FB",
        "button_border": "#293850",
    },
}


def inject_styles(theme: str) -> None:
    t = THEMES[theme]
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;750;800&display=swap');
    :root {{
      --bg: {t['bg']}; --surface: {t['surface']}; --surface-alt: {t['surface_alt']};
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
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    .stApp {{ background: var(--bg); color: var(--text); }}
    [data-testid="stHeader"] {{
      background: var(--bg) !important; background-color: var(--bg) !important;
    }}
    [data-testid="stToolbar"] {{ background: transparent !important; }}
    [data-testid="stDecoration"] {{ background: transparent !important; }}
    .block-container {{ padding-top: 2.5rem !important; padding-bottom: 2rem; max-width: 1400px; }}
    h1,h2,h3,.heading-font {{ font-family: 'Manrope', 'Inter', sans-serif; }}
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
      border: 1px solid var(--border); border-radius: 10px; background: var(--surface); overflow: hidden;
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details {{
      background: var(--surface);
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary {{
      background: var(--surface-alt) !important; color: var(--text) !important;
      border-bottom: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover {{
      background: var(--surface-alt) !important;
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
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] span {{
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
    .brand-sub {{
      color: var(--text-secondary); font-size: 15px; line-height: 1.5; margin-top: 0.35rem; max-width: 760px;
      font-weight: 400;
    }}
    .badge {{
      display:inline-flex; align-items:center; gap:.4rem; padding:.3rem .65rem; border-radius:999px;
      background: var(--accent-soft); color: var(--accent-text); font-size:12px; font-weight:600;
      border:1px solid var(--border); white-space:nowrap;
    }}
    .section-title {{
      font-family: 'Manrope','Inter',sans-serif; font-size: 19px; font-weight: 700; color: var(--text);
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
    .result-muted {{ color: var(--text-secondary); font-size: 15px; line-height: 1.5; }}
    .body-text {{ color: var(--text-secondary); font-size: 15px; line-height: 1.55; }}
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
    .upload-empty {{
      border: 1px dashed var(--border); border-radius: 14px; padding: 1.2rem 1.2rem;
      background: var(--surface-alt); color: var(--text-secondary);
    }}
    .upload-empty strong {{ color: var(--text); font-family: 'Manrope','Inter',sans-serif; font-size: 18px; font-weight: 700; }}
    .upload-empty p {{ margin: .35rem 0 0 0; font-size: 14px; color: var(--text-secondary); }}
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
    /* Consistent card containers (incl. similarity viz bordered container) */
    [data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] {{
      border-radius: 14px !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background: var(--surface) !important;
      border: 1px solid var(--border) !important;
      box-shadow: var(--shadow);
      padding: 0.2rem 0.2rem;
      transition: box-shadow 200ms ease, border-color 200ms ease, transform 200ms ease;
      animation: fadeIn 320ms ease both;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
      border-color: var(--accent);
      box-shadow: var(--shadow);
      transform: translateY(-2px);
    }}
    .panel {{
      background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
      padding: 1rem 1.1rem; box-shadow: var(--shadow);
      transition: box-shadow 200ms ease, border-color 200ms ease, transform 200ms ease;
    }}
    .panel:hover {{ border-color: var(--accent); box-shadow: var(--shadow); transform: translateY(-2px); }}
    /* Streamlit widgets theme adaptation */
    div[data-testid="stFileUploader"] section {{
      background: var(--uploader-bg) !important; border: 1px dashed var(--border) !important; border-radius: 14px !important;
      transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stFileUploader"] section:hover {{ border-color: var(--accent) !important; }}
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] p {{
      color: var(--text-secondary) !important;
    }}
    /* Uploaded file chip (filename/size/remove) */
    div[data-testid="stFileUploaderFile"] {{
      background: var(--surface-alt) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
      padding: 0.45rem 0.55rem !important;
      color: var(--text) !important;
      transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stFileUploaderFile"]:hover {{ border-color: var(--accent) !important; }}
    div[data-testid="stFileUploaderFile"] * {{ color: inherit; }}
    div[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"],
    div[data-testid="stFileUploaderFile"] p, div[data-testid="stFileUploaderFile"] span {{
      color: var(--text) !important;
    }}
    div[data-testid="stFileUploaderFile"] small,
    div[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileSize"] {{
      color: var(--muted) !important;
    }}
    div[data-testid="stFileUploaderFile"] button,
    div[data-testid="stFileUploaderFile"] svg {{
      color: var(--text-secondary) !important;
      fill: currentColor !important;
    }}
    div[data-testid="stFileUploaderFile"] button {{
      border-radius: 8px !important;
      border: 1px solid transparent !important;
      background: transparent !important;
      transition: background-color 150ms ease, color 150ms ease, border-color 150ms ease;
    }}
    div[data-testid="stFileUploaderFile"] button:hover {{
      background: var(--surface) !important;
      border-color: var(--border) !important;
      color: var(--text) !important;
    }}
    div[data-testid="stFileUploaderFile"] {{
      background: var(--surface-alt) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
      padding: .35rem .5rem !important;
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
      box-shadow: var(--shadow); transition: box-shadow 200ms ease, border-color 200ms ease, transform 200ms ease;
      margin-top: .25rem;
    }}
    [data-testid="stExpander"]:hover {{
      border-color: var(--accent); box-shadow: var(--shadow); transform: translateY(-2px);
    }}
    [data-testid="stExpander"] details {{ background: var(--surface); }}
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
    [data-baseweb="notification"] {{ background: var(--surface-alt) !important; color: var(--text) !important; border-color: var(--border) !important; }}
    [data-testid="stNotificationContentInfo"] p,
    [data-testid="stNotificationContentWarning"] p,
    [data-testid="stNotificationContentSuccess"] p,
    [data-testid="stNotificationContentError"] p {{ color: inherit !important; }}
    div[data-testid="stCaptionContainer"], .stCaption {{ background: transparent !important; }}
    div[data-testid="stCaptionContainer"] p, .stCaption {{ color: var(--muted) !important; font-size: 13px !important; }}
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
    [data-testid="stVegaLiteChart"] svg {{ background: transparent !important; }}
    /* Sidebar collapse/expand control visibility */
    [data-testid="stSidebarCollapseButton"] button,
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
    [data-testid="collapsedControl"] button svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-header"] svg {{
      color: var(--text) !important; fill: currentColor !important; stroke: currentColor !important;
    }}
    [data-testid="stSidebarCollapseButton"] button:hover,
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
    /* Theme toggle (compact segmented control in header, no emoji, outline icons) */
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] {{
      gap: 2px; background: var(--surface-alt); border: 1px solid var(--border); border-radius: 8px; padding: 2px;
      width: fit-content; transition: border-color 180ms ease, background-color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label {{
      margin: 0 !important; padding: .22rem .55rem !important; border-radius: 6px !important;
      background: transparent; border: 1px solid transparent !important; min-height: 0 !important;
      transition: background-color 180ms ease, border-color 180ms ease, color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked) {{
      background: var(--surface); border: 1px solid var(--border) !important; box-shadow: none !important;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label p {{
      font-size: 12px; font-weight: 500; color: var(--text-secondary) !important; line-height: 1.2;
      display: inline-flex; align-items: center; gap: .35rem; transition: color 180ms ease;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked) p {{
      color: var(--text) !important; font-weight: 600;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label p::before {{
      content: ""; width: 14px; height: 14px; display: inline-block; background-color: currentColor;
      -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat; -webkit-mask-position: center; mask-position: center;
      -webkit-mask-size: contain; mask-size: contain; transition: transform 220ms ease, opacity 180ms ease; opacity: .9;
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:nth-of-type(1) p::before {{
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cpath d='M12 2v2.2M12 19.8V22M4.93 4.93l1.56 1.56M17.51 17.51l1.56 1.56M2 12h2.2M19.8 12H22M4.93 19.07l1.56-1.56M17.51 6.49l1.56-1.56'/%3E%3C/svg%3E");
      mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4'/%3E%3Cpath d='M12 2v2.2M12 19.8V22M4.93 4.93l1.56 1.56M17.51 17.51l1.56 1.56M2 12h2.2M19.8 12H22M4.93 19.07l1.56-1.56M17.51 6.49l1.56-1.56'/%3E%3C/svg%3E");
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:nth-of-type(2) p::before {{
      -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z'/%3E%3C/svg%3E");
      mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.8A8.5 8.5 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z'/%3E%3C/svg%3E");
    }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) div[role="radiogroup"] label:has(input:checked):nth-of-type(1) p::before {{ transform: rotate(20deg); }}
    div[data-testid="stHorizontalBlock"]:has(.theme-anchor) [data-testid="stWidgetLabel"] {{ display:none; }}
    /* Subtle appearance animations */
    @keyframes eyeFadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .panel, [data-testid="stVerticalBlockBorderWrapper"], .ref-grid-card, .upload-empty, [data-testid="stAlert"] {{
      animation: eyeFadeIn 220ms ease;
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


def status_chip(text: str, kind: str = "warn") -> str:
    cls = {"warn": "chip-warn", "danger": "chip-danger", "ok": "chip-ok"}.get(kind, "chip-warn")
    return f"<span class='status-chip {cls}'>{text}</span>"


def lollipop(df, top, theme: str):
    t = THEMES[theme]
    lo, hi = float(df["sim"].min()) - 0.02, float(df["sim"].max()) + 0.02
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
                gridColor=t["chart_grid"],
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
    chart = (
        (stem + dot + label)
        .properties(height=240, background=t["chart_bg"], padding={"left": 8, "right": 24, "top": 8, "bottom": 8})
        .configure_view(stroke=None, fill=t["chart_bg"])
        .configure_axis(grid=True)
    )
    return chart, lo, hi


@st.cache_data(show_spinner=False)
def thumb_data_uri(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def render_sidebar(meta, unc_thr, ood_thr):
    with st.sidebar:
        st.markdown("<div style=\"font-family:'Manrope','Inter',sans-serif;font-size:22px;font-weight:800;margin-bottom:.1rem;color:var(--text);\">EyeCLIP</div>", unsafe_allow_html=True)
        st.caption("Reference Similarity Explorer")
        st.divider()

        with st.expander("Reference dataset", expanded=True):
            st.markdown(f"**Source split:** {meta['reference']['split']} (single-label: {meta['reference']['single_label']})")
            st.markdown(f"**Reference images:** {meta['reference']['n']}")
            st.markdown(f"**Classes:** {', '.join(meta['classes'])}")

        with st.expander("Model information", expanded=False):
            st.markdown(f"**Model:** {meta['model']}")
            st.markdown(f"**Pretrained:** {meta.get('pretrained', 'openai')}")
            st.markdown(f"**Classifier:** {meta['classifier']}")
            st.markdown(f"**Prototype:** {meta['prototype']}")

        with st.expander("Support set", expanded=False):
            st.markdown(f"**Shots per class:** {meta['support']['shot']}")
            st.markdown(f"**Support images:** {meta['support']['n']}")
            st.markdown(f"**Seed:** {meta['support']['seed']}")

        with st.expander("Thresholds", expanded=True):
            st.markdown(f"**Review threshold (confidence):** {unc_thr:.6f}")
            st.caption(meta["uncertainty"]["rule"])
            st.markdown(f"**OOD threshold (top-1 reference cosine):** {ood_thr:.6f}")
            st.caption(meta["ood"]["rule"])

        with st.expander("Research prototype status", expanded=True):
            st.markdown("<span class='badge'>Research prototype</span>", unsafe_allow_html=True)
            st.caption(DISCLAIMER_FULL)
            if meta.get("synthetic"):
                st.warning(SYNTHETIC_NOTE)


def main():
    st.set_page_config(page_title="EyeCLIP — Reference Similarity Explorer", layout="wide")

    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "Light"
    if "upload_key" not in st.session_state:
        st.session_state["upload_key"] = 0

    # Apply styles early based on current session theme (before most widgets render content)
    inject_styles(st.session_state.get("ui_theme", "Light"))

    if not (ASSETS / "meta.json").exists():
        st.error("Demo assets are missing. Run `scripts/export_demo_assets.py` and commit `app/assets/`.")
        st.stop()

    meta, ref, sup = load_assets()
    model, preprocess = load_model()
    clf, proto = load_classifier()
    classes = meta["classes"]
    unc_thr, ood_thr = meta["uncertainty"]["threshold"], meta["ood"]["threshold"]

    # Header with theme toggle (top-right)
    header_left, header_right = st.columns([5, 1.35], vertical_alignment="top")
    with header_left:
        st.markdown(
            """
            <div class="app-header" style="border-bottom:none;margin-bottom:0;padding-bottom:0;">
              <div>
                <div class="brand-title">EyeCLIP — Reference Similarity Explorer</div>
                <div class="brand-sub">Few-shot retinal image similarity exploration using CLIP-style embeddings and a labelled reference set.</div>
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
        st.markdown(
            "<div style='display:flex;justify-content:flex-end;margin-top:.25rem;'><span class='badge'>Research prototype</span></div>",
            unsafe_allow_html=True,
        )

    theme = st.session_state.get("ui_theme", "Light")
    # Re-inject if theme changed this run to ensure latest tokens apply
    inject_styles(theme)

    st.markdown("<div style='border-bottom:1px solid var(--border);margin:0.25rem 0 1rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER_FULL}</div>", unsafe_allow_html=True)
    st.write("")

    render_sidebar(meta, unc_thr, ood_thr)

    left_col, right_col = st.columns([1.05, 1.6], gap="large")

    with left_col:
        st.markdown("<div class='section-title'>Image upload workspace</div>", unsafe_allow_html=True)
        up = st.file_uploader(
            "Upload a fundus photograph (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            key=f"uploader_{st.session_state['upload_key']}",
            help="The uploaded image is processed in memory and never written to disk.",
        )
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if up is not None and st.button("Remove / replace image", use_container_width=True):
                st.session_state["upload_key"] += 1
                st.rerun()
        with btn_col2:
            st.caption("Supported: PNG, JPG, JPEG")

        if up is None:
            st.markdown(
                """
                <div class="upload-empty">
                  <strong>No image uploaded yet</strong>
                  <p>Upload a retinal fundus photograph to compare it against the reference classes. The image is processed in memory only and is not saved.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            image = None
        else:
            image = Image.open(up)
            import io, base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            st.markdown(
                f"<div style='max-width:100%;max-height:min(70vh,560px);display:flex;align-items:center;justify-content:center;"
                f"background:var(--img-bg);border-radius:12px;overflow:hidden;'>"
                f"<img src='data:image/png;base64,{b64}' style='max-width:100%;max-height:min(70vh,560px);"
                f"width:auto;height:auto;object-fit:contain;display:block;border-radius:12px;' "
                f"alt='Uploaded fundus photograph' /></div>",
                unsafe_allow_html=True,
            )
            st.caption("Uploaded image · processed in memory, not saved")

    with right_col:
        if image is None:
            st.markdown("<div class='section-title'>Analysis</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="panel">
                  <div class="meta-label">Status</div>
                  <div class="result-class" style="font-size:28px;font-weight:750;">Awaiting image</div>
                  <div class="body-text">Upload a fundus photograph to view the closest reference class, confidence, similarity visualization, and most similar reference images.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
            st.info(SHIFT_NOTE)
        else:
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
                chips.append(status_chip("Image unlike reference set (possible device/domain shift or non-fundus image)", "danger"))
            if flagged:
                chips.append(status_chip("Flag for expert review — confidence below review threshold", "warn"))
            if not ood and not flagged:
                chips.append(status_chip("Within demo reference/threshold range", "ok"))
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

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("<div class='section-title' style='margin-bottom:.35rem;'>Similarity visualization</div>", unsafe_allow_html=True)
                df = pd.DataFrame({"class": classes, "sim": sims}).sort_values("sim", ascending=False)
                chart, lo, hi = lollipop(df, top=pred, theme=theme)
                st.altair_chart(chart, use_container_width=True)
                st.caption(
                    f"Cosine similarity of the uploaded image to each class prototype (L2-normalised mean support embedding). "
                    f"Highlighted class: {pred}. X-axis range shown: {lo:.3f}–{hi:.3f}. "
                    "Similarity values are not calibrated probabilities and are distinct from logistic regression confidence."
                )

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
    st.caption(DISCLAIMER_FULL)


main()
