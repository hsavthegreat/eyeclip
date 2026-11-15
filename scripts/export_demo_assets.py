#!/usr/bin/env python3
"""Export the assets the Streamlit demo needs, from a machine that has the RFMiD images and the
cached CLIP embeddings. Run it once, then commit app/assets/.

    python scripts/export_demo_assets.py \
        --data-root /path/to/RFMiD2_0 \
        --embeddings-dir /path/to/embeddings \
        --manifests-dir manifests \
        --out app/assets

Writes:
  app/assets/reference.npz   CLIP ViT-B/16 embeddings/labels/stems for the 228 train-split
                             single-label WNL, DR, CRS and RT images
  app/assets/support.npz     the seed-0 10-shot support set (40 images), same fields
  app/assets/thumbs/*.jpg    224 px JPEG thumbnails of the reference images
  app/assets/meta.json       model, classes, the Cell 8 uncertainty rule + threshold, OOD threshold
  app/assets/ATTRIBUTION.md  dataset attribution for the redistributed thumbnails

Embeddings must be the cached CLIP ones written by cell 4, in manifest_cache.csv row order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

MAIN = ["WNL", "DR", "CRS", "RT"]
MODEL_KEY = "clip_vit_b_16_openai"
MODEL_NAME = "CLIP ViT-B/16 (ViT-B-16-quickgelu, OpenAI weights)"
SEED, SHOT, THUMB = 0, 10, 224
UNCERTAINTY_PERCENTILE = 20          # Cell 8: confidence < 20th percentile of support confidences
OOD_PERCENTILE = 5                   # 5th percentile of nearest-other-reference cosine

ATTRIBUTION = """# Attribution

The thumbnails in `thumbs/` are 224 px derivatives of retinal fundus photographs from
**RFMiD 2.0** (Retinal Fundus Multi-disease Image Dataset), used under **CC BY 4.0**.

- Source: https://ieee-dataport.org/datasets/retinal-fundus-multi-disease-image-dataset-rfmid
- Reference: Pachade S, Porwal S, Thulkar D, et al. "Retinal Fundus Multi-Disease Image Dataset
  (RFMiD): A Dataset for Multi-Disease Detection Research." Data 2021;6(2):14.
  https://doi.org/10.3390/data6020014
- Changes: resized to 224 px on the long side and re-encoded as JPEG. No other modification.
- The dataset is **not** redistributed here: only the 228 reduced thumbnails needed by the demo are
  included, and the licence requires the attribution above to accompany them.

The embedding arrays (`reference.npz`, `support.npz`) are derived features computed with OpenAI's
CLIP ViT-B/16 weights (MIT licence) and carry no image content beyond 512-dimensional vectors.
"""


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def nearest_other_cosine(emb: np.ndarray) -> np.ndarray:
    """Cosine of each row to its closest row that is not itself."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    return sim.max(axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="RFMiD 2.0 root, holding the rel_path prefixes")
    ap.add_argument("--embeddings-dir", required=True, help="directory with %s.npy from cell 4" % MODEL_KEY)
    ap.add_argument("--manifests-dir", default="manifests")
    ap.add_argument("--out", default="app/assets")
    args = ap.parse_args()

    data_root, emb_dir = Path(args.data_root), Path(args.embeddings_dir)
    out = Path(args.out)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(Path(args.manifests_dir) / "manifest_cache.csv")
    sa = pd.read_csv(Path(args.manifests_dir) / "support_assignments.csv")
    all_emb = np.load(emb_dir / f"{MODEL_KEY}.npy")
    assert all_emb.shape[0] == len(manifest), f"embeddings {all_emb.shape} vs manifest {len(manifest)}"
    all_emb = l2(all_emb.astype(np.float32))

    # reference pool: train-split single-label images of the four closed-set classes
    ref_mask = (manifest["orig_split"] == "train") & (manifest["n_pos"] == 1) & manifest["single_class"].isin(MAIN)
    ref = manifest[ref_mask].reset_index(drop=True)
    ref_emb = all_emb[manifest.index[ref_mask].to_numpy()]
    assert len(ref) == 228, f"expected 228 reference images, got {len(ref)}"
    assert ref["stem"].is_unique and ref["single_class"].isin(MAIN).all()
    print(f"reference: {len(ref)} images, " + ", ".join(f"{c}={int((ref['single_class'] == c).sum())}" for c in MAIN))

    # support: the seed-0 10-shot assignment, one row per class
    sup = sa[(sa["seed"] == SEED) & (sa["shot"] == SHOT) & sa["class"].isin(MAIN)]
    assert sorted(sup["class"]) == sorted(MAIN), f"support classes {list(sup['class'])} != {MAIN}"
    stem_to_row = {str(s): i for i, s in enumerate(manifest["stem"].astype(str))}
    rows, labels, stems = [], [], []
    for _, r in sup.iterrows():
        stems_c = [s for s in str(r["support"]).split(";") if s]
        assert len(stems_c) == SHOT, f"{r['class']}: {len(stems_c)} support stems, expected {SHOT}"
        for s in stems_c:
            i = stem_to_row[s]
            rows.append(i); labels.append(r["class"]); stems.append(s)
    assert len(rows) == len(MAIN) * SHOT == 40, len(rows)
    print(f"support: {len(rows)} images, {dict(pd.Series(labels).value_counts().astype(int))}")

    np.savez_compressed(out / "reference.npz", emb=ref_emb, labels=np.asarray(ref["single_class"], dtype=str),
                        stems=np.asarray(ref["stem"], dtype=str))          # dtype=str: no pickled object arrays
    np.savez_compressed(out / "support.npz", emb=all_emb[rows], labels=np.asarray(labels, dtype=str),
                        stems=np.asarray(stems, dtype=str))

    # thumbnails (RFMiD is CC BY 4.0; attribution written alongside)
    for r in ref.itertuples():
        dst = out / "thumbs" / f"{r.stem}.jpg"
        if dst.exists():
            continue
        with Image.open(data_root / r.rel_path) as im:
            im = im.convert("RGB")
            im.thumbnail((THUMB, THUMB))
            im.save(dst, "JPEG", quality=85)
    n_thumbs = len(list((out / "thumbs").glob("*.jpg")))
    assert n_thumbs == len(ref), f"{n_thumbs} thumbnails for {len(ref)} reference images"
    print(f"thumbnails: {n_thumbs} at <= {THUMB} px")

    # thresholds: Cell 8's rule, but the support confidences here are in-sample
    sup_emb, sup_lab = all_emb[rows], np.array(labels)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sup_emb, sup_lab)
    support_conf = clf.predict_proba(sup_emb).max(axis=1)
    unc_threshold = float(np.percentile(support_conf, UNCERTAINTY_PERCENTILE))
    ood_threshold = float(np.percentile(nearest_other_cosine(ref_emb), OOD_PERCENTILE))

    meta = {
        "model": MODEL_NAME, "model_key": MODEL_KEY, "pretrained": "openai",
        "classes": MAIN,
        "classifier": "logistic regression, C=1.0, max_iter=1000, fitted on the support set only",
        "prototype": "cosine similarity to the L2-normalised mean of that class's support embeddings",
        "reference": {"split": "train", "single_label": True, "n": int(len(ref))},
        "support": {"seed": SEED, "shot": SHOT, "n": len(rows)},
        "uncertainty": {
            "rule": "flag for expert review when the classifier's top class confidence is below the "
                    f"{UNCERTAINTY_PERCENTILE}th percentile of the support-set confidences (Cell 8's rule)",
            "percentile": UNCERTAINTY_PERCENTILE,
            "threshold": round(unc_threshold, 6),
            "support_confidences": "in-sample",
            "note": "Here the support confidences come from the same 40 images the classifier was fitted on, "
                    "so they are optimistic and this threshold is lower (flags less often) than Cell 8's, "
                    "where they are 5-fold out-of-fold.",
        },
        "ood": {
            "rule": "warn when the top-1 cosine similarity to any reference image is below the "
                    f"{OOD_PERCENTILE}th percentile of each reference image's cosine similarity to its "
                    "nearest OTHER reference image",
            "percentile": OOD_PERCENTILE,
            "threshold": round(ood_threshold, 6),
        },
        "license": "RFMiD 2.0, CC BY 4.0; thumbnails are 224 px derivatives, see ATTRIBUTION.md",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    print(f"thresholds: uncertainty {unc_threshold:.4f} (p{UNCERTAINTY_PERCENTILE} of support confidences, "
          f"min {support_conf.min():.4f}), OOD {ood_threshold:.4f} (p{OOD_PERCENTILE} nearest-other cosine)")
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
