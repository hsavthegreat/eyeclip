"""EyeCLIP demo — research prototype, not a diagnostic tool.

Upload one fundus photograph and see its similarity to four reference classes learned from 40 labelled
RFMiD images (seed-0 10-shot). All runs on CPU from committed assets; the uploaded image is processed in
memory and never written to disk.

    pip install -r requirements-demo.txt && streamlit run app/streamlit_app.py
Assets are read from app/assets/ (override with EYECLIP_DEMO_ASSETS, e.g. for a test copy).
"""
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
SHIFT_NOTE = ("Cross-dataset tests (RFMiD to FIVES) showed that decision thresholds shift across devices and "
              "hospitals, so scores may be unreliable on other sources.")
SYNTHETIC_NOTE = ("Synthetic placeholder assets: the reference images and embeddings in this build are generated, "
                  "so no value on this page reflects RFMiD data.")
# navy / cyan / light blue / slate / amber; the tint #EAF3FB is the theme's secondary background
NAVY, CYAN, LIGHT, SLATE, AMBER = "#0B1B3A", "#22B8D6", "#8FBDE8", "#5B7090", "#F2A65A"

st.set_page_config(page_title="EyeCLIP demo (research prototype)", layout="wide")
st.markdown(f"<style>[data-testid='stImage'] img{{border-radius:14px}}"
            f"[data-testid='stMetricValue']{{color:{NAVY}}}</style>", unsafe_allow_html=True)


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


def chip(text, fg=NAVY, bg=AMBER):
    st.markdown(f"<span style='display:inline-block;padding:.35rem .7rem;border-radius:999px;"
                f"background:{bg};color:{fg};font-size:.86rem;line-height:1.25'>{text}</span>",
                unsafe_allow_html=True)


def lollipop(df, top):
    lo, hi = float(df["sim"].min()) - 0.02, float(df["sim"].max()) + 0.02
    df = df.assign(floor=lo)
    base = alt.Chart(df).encode(
        y=alt.Y("class:N", sort="-x", axis=alt.Axis(title=None, labelColor=NAVY, labelFontSize=13)),
        x=alt.X("sim:Q", scale=alt.Scale(domain=[lo, hi], zero=False),
                axis=alt.Axis(title="cosine similarity", format=".2f", labelColor=SLATE, titleColor=SLATE)))
    color = alt.condition(alt.datum["class"] == top, alt.value(CYAN), alt.value(SLATE))
    stem = base.mark_rule(strokeWidth=3).encode(x="floor:Q", x2="sim:Q", color=color)
    dot = base.mark_circle(size=190, stroke=LIGHT, strokeWidth=1.5).encode(color=color)
    label = base.mark_text(align="left", dx=14, fontSize=12, color=NAVY).encode(text=alt.Text("sim:Q", format=".3f"))
    return (stem + dot + label).properties(height=200, width="container"), lo, hi


def main():
    st.title("EyeCLIP — reference similarity explorer")
    st.caption(DISCLAIMER + " Scores are similarity values, not calibrated probabilities.")
    if not (ASSETS / "meta.json").exists():
        st.error("Demo assets are missing. Run `scripts/export_demo_assets.py` and commit `app/assets/`.")
        st.stop()

    meta, ref, sup = load_assets()
    model, preprocess = load_model()
    clf, proto = load_classifier()
    classes = meta["classes"]
    unc_thr, ood_thr = meta["uncertainty"]["threshold"], meta["ood"]["threshold"]

    st.sidebar.header("Reference set")
    st.sidebar.write(f"{meta['model']}")
    st.sidebar.write(f"{meta['reference']['n']} RFMiD train images, {meta['support']['n']}-image "
                     f"{meta['support']['shot']}-shot support set; review threshold {unc_thr:.3f}; OOD {ood_thr:.3f}")
    if meta.get("synthetic"):
        st.sidebar.warning(SYNTHETIC_NOTE)

    up = st.file_uploader("Upload a fundus photograph (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload an image to see its similarity to the reference classes. Nothing is stored.")
        st.info(SHIFT_NOTE)
        return

    image = Image.open(up)
    emb = embed(image, model, preprocess)
    sims = proto @ emb                                    # cosine to each class prototype
    sim = ref["emb"] @ emb                                # cosine to each reference image
    order = np.argsort(-sim)
    proba = clf.predict_proba(emb[None])[0]
    pred = str(clf.classes_[int(np.argmax(proba))]); conf = float(proba.max())
    top1_sim = float(sim[order[0]])
    ood, flagged = top1_sim < ood_thr, conf < unc_thr

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.image(image, caption="Uploaded image · processed in memory, not saved", width=300)
    with right:
        if ood:
            with st.container(border=True):
                st.markdown(f"<div style='font-size:1.55rem;font-weight:600;color:{SLATE}'>No reliable match</div>"
                            f"<div style='color:{SLATE};font-size:.9rem'>No reference class is a close match for "
                            f"this image.</div>", unsafe_allow_html=True)
            with st.expander("Closest class anyway"):
                st.markdown(f"**{pred}** · confidence {conf:.2f}")
                st.caption("Uncalibrated confidence from logistic regression on 40 support images.")
        else:
            with st.container(border=True):
                st.markdown(f"<div style='color:{SLATE};font-size:.82rem;letter-spacing:.04em;"
                            f"text-transform:uppercase'>Closest reference class</div>"
                            f"<div style='font-size:2rem;font-weight:600;color:{NAVY};line-height:1.15'>{pred}</div>"
                            f"<div style='color:{SLATE};font-size:.95rem'>confidence {conf:.2f}</div>",
                            unsafe_allow_html=True)
                st.caption("Uncalibrated confidence from logistic regression on 40 support images.")

        if ood:                                           # out-of-distribution message first
            chip("Image looks unlike the reference set (possible different device, non-fundus image)")
        if flagged:
            chip("Flag for expert review — confidence is below the Cell 8 threshold for this demo.")

        df = pd.DataFrame({"class": classes, "sim": sims}).sort_values("sim", ascending=False)
        chart, lo, hi = lollipop(df, top=pred)
        with st.container(border=True):
            st.subheader("Similarity (not a probability)")
            st.altair_chart(chart)
            st.caption(f"Cosine similarity to each class prototype. The x-axis spans {lo:.3f}–{hi:.3f}: the "
                       f"minimum and maximum class similarity plus a small margin, so dots need no zero baseline. "
                       f"Values are not probabilities and are not comparable across models or datasets.")

    with st.container(border=True):
        st.subheader("Most similar reference images")
        cols = st.columns(5, gap="small")
        for c, idx in zip(cols, order[:5]):
            with c:
                thumb = ASSETS / "thumbs" / f"{ref['stems'][idx]}.jpg"
                if thumb.exists():
                    st.image(str(thumb), width=140)
                st.markdown(f"**{ref['labels'][idx]}**")
                st.caption(f"stem {ref['stems'][idx]} · similarity {float(sim[idx]):.3f}")

    st.info(SHIFT_NOTE)
    st.divider()
    st.caption(DISCLAIMER)


main()
