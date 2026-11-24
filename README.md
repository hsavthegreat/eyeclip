# EyeCLIP: few-shot recognition of rare eye diseases (research prototype)

Pilot study: can frozen CLIP-style image/text embeddings recognise rare retinal conditions from 1, 5 or 10
labelled examples per class? Not a diagnostic tool; no clinical validation.

## Status
**Done:** data audit and matching, fixed evaluation protocol, frozen embeddings for 3 models, few-shot evaluation
(4 classes); per-image predictions, bootstrap CIs, selective prediction, retrieval and a nearest-neighbour
near-duplicate audit (cell 8 of the notebook); cross-dataset test on FIVES, WNL vs DR (cells 10-12); Streamlit demo.

## Corrections
**14 November 2026 — support-sampling bug in cell 5.** The support lookup selected every row for a seed
(`support_assignments[seed == seed]`) and sliced `[:shot]`, but that table holds one row per
(seed, class, shot) and the rows are nested, so 1-, 5- and 10-shot examples were stacked: a "10-shot" run
trained on 64 rows covering 40 distinct images, with each 1-shot image weighted three times (44 rows / 32
images at "5 shots"). The lookup now filters on `shot` as well, giving 20 rows at 5 shots and 40 at 10.
Cell 8's dict keyed by class happened to keep the last (10-shot) row, so its 10-shot outputs were unaffected;
cell 5's numbers changed. **All results published before this fix, including any shared at mid-project, are
superseded by the tables below**, and every file in `results/` was regenerated.

## Data
RFMiD 2.0 (CC BY 4.0), 854 usable images after matching. Closed 4-class task: WNL (normal), DR, CRS, RT.
Support examples come from the training split only; evaluation uses validation+test (153 images:
WNL 105, DR 28, CRS 11, RT 9). See DATASETS.md.

## Method
Frozen encoders (ResNet-50 ImageNet baseline, CLIP ViT-B/16, BiomedCLIP); L2-normalised embeddings cached;
classifiers: zero-shot text prompts, prototype (cosine to class mean), logistic regression, hybrid
(prototype + text). 1/5/10 shots, 10 seeds. Logistic regression on raw embeddings; prototype, hybrid and
zero-shot on embeddings centred on the training-split mean.
Metrics: macro-F1, balanced accuracy, per-class recall with eval counts. Nothing is tuned on the evaluation set.

## Results (pilot; mean ± std over 10 support-sampling seeds; logistic regression on raw embeddings; prototype/hybrid/zero-shot on embeddings centred on the training-split mean; macro-F1)
| Model | Method | 5-shot | 10-shot |
|---|---|---|---|
| CLIP ViT-B/16 | logistic | 0.828 ± 0.060 | 0.882 ± 0.022 |
| CLIP ViT-B/16 | prototype | 0.808 ± 0.090 | 0.870 ± 0.011 |
| CLIP ViT-B/16 | hybrid | 0.810 ± 0.094 | 0.884 ± 0.024 |
| BiomedCLIP | logistic | 0.752 ± 0.063 | 0.828 ± 0.035 |
| BiomedCLIP | prototype | 0.760 ± 0.059 | 0.796 ± 0.045 |
| BiomedCLIP | hybrid | 0.752 ± 0.053 | 0.791 ± 0.039 |
| ResNet-50 (baseline) | logistic | 0.758 ± 0.060 | 0.819 ± 0.019 |
| ResNet-50 (baseline) | prototype | 0.776 ± 0.058 | 0.830 ± 0.026 |

Zero-shot text prompts: macro-F1 0.12–0.16 (CLIP) and 0.42–0.45 (BiomedCLIP), depending on centring. 1-shot results
are very noisy (std 0.1–0.2). Source: results/summary_all_runs.csv, results/recalls_all_runs.csv.

**Uncertainty.** Paired bootstrap over the 10 seeds (2,000 resamples, macro-F1 recomputed per seed and then averaged)
for 10-shot logistic regression: CLIP 0.882 [0.799, 0.949], BiomedCLIP 0.828 [0.748, 0.891], ResNet-50 0.819
[0.727, 0.891]; CLIP − ResNet-50 +0.063 [−0.005, 0.137], BiomedCLIP − ResNet-50 +0.009 [−0.057, 0.080].
Retrieval P@1 over the 228-image training pool is 0.92 (CLIP and BiomedCLIP) and 0.95 (ResNet-50). Retrieval is
markedly weaker for CRS, at 0.36–0.45 and below every other class, so nearest-neighbour retrieval should not be
relied on for that condition.

**Reading the results.** Few-shot adaptation helps a great deal over zero-shot. CLIP has the higher point estimate at
5–10 shots, mainly because of DR (all 28 eval images recalled in every seed, vs a mean of 0.76–0.79 for the other
models), but with 153 evaluation images the bootstrap interval for CLIP − ResNet-50 includes zero, so
the ordering of the two encoders is not established here. This comparison is **in-dataset only**: it does not
carry over to another dataset, where the ordering reverses on point estimates (see Cross-dataset test). For the rarer CRS and RT (11 and 9 eval images), intervals
overlap and no method is clearly better.

## Duplicate sensitivity (10-shot logistic)
`retrieval_nearest.csv` flags one evaluation image, stem 886 (test split, WNL), at cosine 1.000 to training
image stem 798 (train split, WNL) under all three encoders: the same image sits on both sides of the split.
The main protocol is left unchanged; this is a sensitivity check that drops the eval-side copy, leaving 152
evaluation images (WNL 104). Mean ± std over the 10 seeds:

| Model | 153 images | 152 images (duplicate dropped) | change |
|---|---|---|---|
| CLIP ViT-B/16 | 0.882 ± 0.022 | 0.882 ± 0.022 | −0.00003 |
| BiomedCLIP | 0.828 ± 0.035 | 0.828 ± 0.035 | −0.00008 |
| ResNet-50 (baseline) | 0.819 ± 0.019 | 0.819 ± 0.019 | −0.00011 |

All three models classify the duplicated image correctly in all 10 seeds, so dropping it changes only the
WNL denominators and no conclusion depends on it.

## Cross-dataset test (RFMiD → FIVES, WNL vs DR only)
Two classes only, because WNL and DR are the only classes shared with FIVES. Support examples are RFMiD
**train** images taken from `support_assignments.csv`, one row per (seed, class, shot) with the support size
asserted equal to the shot; shots 1/5/10, seeds 0-9. Three methods: logistic regression on raw embeddings
(C=1.0, max_iter=1000, fitted on the two classes only), prototype with embeddings centred on the RFMiD
train-split mean (that same mean applied unchanged to FIVES), and zero-shot text prompts for the two CLIP
models (uncentred, WNL vs DR only). **Nothing is fitted or tuned on FIVES.** All features are cached: the
RFMiD ones from cell 4, the FIVES ones from cell 10, which reuses cell 4's encoders, preprocessing, fp16
autocast and L2 normalisation.

Evaluation sets: `rfmid_ref` = RFMiD validation+test single-label WNL and DR, n = 133 (WNL 105, DR 28);
`fives_all` = all 400 FIVES images (WNL 200, DR 200); `fives_q1` = the FIVES `quality == 1` subset, n = 322
(WNL 178, DR 144), reported separately as a sensitivity check. AUROC uses DR as the positive class with each
method's own score (logistic probability, prototype cosine DR − WNL, zero-shot logit difference DR − WNL).
Source: `results/cross_dataset.csv` (means and per-seed std), `results/cross_dataset_ci.csv` (2,000 bootstrap
resamples, `BOOT_SEED = 0`), figure in `results/fig_cross_dataset.png`, code in cell 12 of the notebook.

**Macro-F1** (mean ± std over the 10 support-sampling seeds):

| Model | Method | RFMiD ref, 5-shot | RFMiD ref, 10-shot | FIVES, 5-shot | FIVES, 10-shot | FIVES q=1, 10-shot |
|---|---|---|---|---|---|---|
| ResNet-50 (baseline) | logistic | 0.887 ± 0.045 | 0.903 ± 0.047 | 0.638 ± 0.068 | 0.655 ± 0.079 | 0.575 ± 0.095 |
| ResNet-50 (baseline) | prototype | 0.962 ± 0.027 | 0.970 ± 0.022 | 0.779 ± 0.020 | 0.779 ± 0.026 | 0.724 ± 0.033 |
| CLIP ViT-B/16 | logistic | 0.998 ± 0.005 | 0.999 ± 0.004 | 0.665 ± 0.128 | 0.614 ± 0.106 | 0.574 ± 0.108 |
| CLIP ViT-B/16 | prototype | 0.988 ± 0.012 | 0.990 ± 0.011 | 0.505 ± 0.115 | 0.496 ± 0.128 | 0.467 ± 0.137 |
| CLIP ViT-B/16 | zero-shot | 0.169 ± 0.000 | 0.169 ± 0.000 | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.309 ± 0.000 |
| BiomedCLIP | logistic | 0.925 ± 0.029 | 0.933 ± 0.022 | 0.626 ± 0.048 | 0.640 ± 0.035 | 0.603 ± 0.033 |
| BiomedCLIP | prototype | 0.936 ± 0.026 | 0.940 ± 0.023 | 0.664 ± 0.029 | 0.672 ± 0.009 | 0.633 ± 0.006 |
| BiomedCLIP | zero-shot | 0.522 ± 0.000 | 0.522 ± 0.000 | 0.700 ± 0.000 | 0.700 ± 0.000 | 0.707 ± 0.000 |

**AUROC** (DR positive, mean over seeds):

| Model | Method | RFMiD ref, 5-shot | RFMiD ref, 10-shot | FIVES, 5-shot | FIVES, 10-shot | FIVES q=1, 10-shot |
|---|---|---|---|---|---|---|
| ResNet-50 (baseline) | logistic | 0.999 | 1.000 | 0.899 | 0.901 | 0.873 |
| ResNet-50 (baseline) | prototype | 0.999 | 0.999 | 0.891 | 0.892 | 0.863 |
| CLIP ViT-B/16 | logistic | 1.000 | 1.000 | 0.843 | 0.851 | 0.816 |
| CLIP ViT-B/16 | prototype | 1.000 | 1.000 | 0.843 | 0.851 | 0.817 |
| CLIP ViT-B/16 | zero-shot | 0.471 | 0.471 | 0.624 | 0.624 | 0.647 |
| BiomedCLIP | logistic | 0.994 | 0.995 | 0.867 | 0.870 | 0.872 |
| BiomedCLIP | prototype | 0.993 | 0.994 | 0.878 | 0.882 | 0.892 |
| BiomedCLIP | zero-shot | 0.749 | 0.749 | 0.724 | 0.724 | 0.724 |

**Fraction of images predicted DR, 10-shot** (true DR share: 0.21 on the reference, 0.50 on FIVES, 0.45 on
FIVES q=1):

| Model | Method | RFMiD ref | FIVES | FIVES q=1 |
|---|---|---|---|---|
| ResNet-50 (baseline) | logistic | 0.154 | 0.209 | 0.123 |
| ResNet-50 (baseline) | prototype | 0.198 | 0.323 | 0.232 |
| CLIP ViT-B/16 | logistic | 0.211 | 0.611 | 0.575 |
| CLIP ViT-B/16 | prototype | 0.217 | 0.881 | 0.867 |
| CLIP ViT-B/16 | zero-shot | 0.992 | 1.000 | 1.000 |
| BiomedCLIP | logistic | 0.196 | 0.213 | 0.142 |
| BiomedCLIP | prototype | 0.210 | 0.252 | 0.169 |
| BiomedCLIP | zero-shot | 0.632 | 0.407 | 0.342 |

**10-shot change and intervals** (bootstrap over evaluation images, 2,000 resamples, per-seed metrics recomputed
and averaged inside each resample; "drop" = reference minus FIVES, from independent resamples):

| Model | Method | FIVES macro-F1 [95% CI] | macro-F1 drop [95% CI] | FIVES AUROC [95% CI] |
|---|---|---|---|---|
| ResNet-50 (baseline) | logistic | 0.655 [0.613, 0.695] | +0.247 [+0.172, +0.315] | 0.901 [0.868, 0.931] |
| ResNet-50 (baseline) | prototype | 0.779 [0.743, 0.817] | +0.190 [+0.142, +0.237] | 0.892 [0.858, 0.924] |
| CLIP ViT-B/16 | logistic | 0.614 [0.585, 0.641] | +0.386 [+0.357, +0.414] | 0.851 [0.815, 0.884] |
| CLIP ViT-B/16 | prototype | 0.496 [0.462, 0.528] | +0.495 [+0.459, +0.531] | 0.851 [0.815, 0.885] |
| CLIP ViT-B/16 | zero-shot | 0.333 [0.309, 0.354] | −0.165 [−0.215, −0.115] | 0.624 [0.567, 0.681] |
| BiomedCLIP | logistic | 0.640 [0.594, 0.684] | +0.293 [+0.223, +0.358] | 0.870 [0.836, 0.904] |
| BiomedCLIP | prototype | 0.672 [0.623, 0.719] | +0.267 [+0.202, +0.333] | 0.882 [0.847, 0.913] |
| BiomedCLIP | zero-shot | 0.700 [0.654, 0.744] | −0.180 [−0.277, −0.086] | 0.724 [0.672, 0.774] |

**Reading it.** Few-shot accuracy does not transfer: in-dataset 10-shot macro-F1 is 0.90-1.00 and on FIVES it is
0.50-0.78, with every few-shot drop interval excluding zero. Ranking transfers partly: AUROC falls from about
1.0 to 0.85-0.90, well above chance, so the encoders still order DR above normal while the decision threshold
they learned does not carry over. The threshold shifts in both directions — ResNet-50 and BiomedCLIP under-call
DR on FIVES (0.21-0.32 predicted, DR recall 0.38-0.61) and CLIP over-calls it (0.61-0.88 predicted, WNL recall
0.20-0.55) — and CLIP zero-shot predicts DR for essentially every image on both datasets. CLIP's in-dataset
edge does not carry over: ResNet-50 has the best point estimates on FIVES (AUROC 0.901/0.892, prototype
macro-F1 0.779), but the AUROC intervals overlap with CLIP's and BiomedCLIP's, so the safe claim is that CLIP
has no advantage on FIVES, not that it is worse. BiomedCLIP zero-shot scores a higher FIVES macro-F1 (0.700)
than its own few-shot methods (0.640/0.672) while ranking worse (AUROC 0.724), a reminder that macro-F1 alone
is misleading under this shift.

**Caveats.** (1) *What the interval is.* The bootstrap resamples evaluation images and covers that sampling
uncertainty for the 10-seed mean. It does not cover the choice of support images: on FIVES the 10-shot per-seed
macro-F1 std is 0.106 for CLIP logistic and 0.128 for CLIP prototype, against CIs 0.043-0.056 wide (ResNet-50
logistic 0.079, BiomedCLIP logistic 0.035, BiomedCLIP prototype 0.009), so for CLIP the CI understates
support-selection uncertainty; `results/cross_dataset.csv` reports both and they are not interchangeable. (2) *Causes are untested.* A
different DR severity mix, camera and colour differences, and label definitions are all plausible, and none
is distinguished by these results — they are hypotheses. (3) *Quality subset.* The flagged images are 56 of
200 DR against 22 of 200 WNL, so `quality == 1` is more WNL-heavy (0.45 DR vs 0.50), and DR recall is lower
on that subset for every few-shot method (ResNet-50 logistic 0.397 → 0.258, prototype 0.610 → 0.483;
BiomedCLIP logistic 0.384 → 0.291, prototype 0.443 → 0.339; CLIP logistic 0.767 → 0.719, prototype 0.966 →
0.955). Report as an observation only. (4) *Scope.* Two classes, one target dataset, one 10-seed support
protocol. The FIVES features were extracted on CPU with fp16 autocast disabled, while the cached RFMiD
features come from cell 4; the Step 11 invariance check re-embedded 30 RFMiD images through the cell 10 path
and recovered the cached vectors (minimum cosine 0.999994, gate 0.999), so the two sides are not confounded
by device or precision, but a GPU rerun of cell 11 has not been done.

## Limitations
Small evaluation sets for rare classes; one dataset; 4 classes after dropping multi-label images; patient-level
independence of splits not verified; similarity scores are not calibrated probabilities; no clinical validation.
One evaluation image (stem 886, WNL, test split) is an exact duplicate of a training image (stem 798, WNL, train
split) — cosine 1.000 under all three encoders — so it leaks into training, though the sensitivity check above
puts its effect on macro-F1 at 0.0001 or less.
The nearest-neighbour audit (top-1 cosine of each eval image to the training pool; written by cell 8 to
`retrieval_nearest.csv` and `fig_similarity.png` in the results directory) can flag a possible duplicate or
same-patient image but cannot prove one: it uses no patient IDs, and a high cosine may also be two different
eyes of similar appearance.

## Reproduce
Open notebooks/eyeclip.ipynb in Colab (GPU), mount Drive, place RFMiD 2.0 under
/content/drive/MyDrive/eyeclip/data/dataset_main/RFMiD2_0, run cells in order. Cell 6 runs on CPU from the cached embeddings.
The bootstrap CIs in `results/bootstrap_ci.csv` depend on eval-row order (resampling picks row positions), so they
reproduce exactly only if `predictions.csv` is read in its saved row order; sorting the rows before resampling
shifts the interval bounds by up to about 0.006.

## Run the demo
A Streamlit app that shows an uploaded fundus photograph's similarity to four reference classes, using a
logistic-regression classifier fitted on the 40-image seed-0 10-shot support set. It runs on CPU from a clean
clone: the committed assets in `app/assets/` (CLIP ViT-B/16 embeddings for 228 RFMiD train images, the support
set, 224 px thumbnails, `meta.json`) mean no dataset download and no GPU are needed. The uploaded image is
processed in memory and never written to disk.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-demo.txt      # CPU torch, streamlit, open_clip
streamlit run app/streamlit_app.py        # then open the printed localhost URL
```

On first launch open_clip downloads the ~350 MB ViT-B/16 weights from Hugging Face; after that the app starts
offline. The app reports a *Closest reference class* with an uncalibrated confidence, per-class cosine
similarities explicitly labelled "Similarity (not a probability)", the five most similar reference images, a
"Flag for expert review" banner when confidence is below the Cell 8 threshold, and an out-of-distribution
warning when the top-1 similarity is below the 5th percentile of reference-to-reference nearest-neighbour
similarity. The thresholds live in `app/assets/meta.json` with the rule they came from: the demo's support
confidences are in-sample (the classifier is fitted on those same 40 images), so its threshold is optimistic
and flags less often than Cell 8's out-of-fold version.

To rebuild the assets where the images and cached embeddings exist:

```bash
python scripts/export_demo_assets.py --data-root /path/to/RFMiD2_0 \
    --embeddings-dir /path/to/embeddings --manifests-dir manifests --out app/assets
```

The thumbnails are 224 px derivatives of RFMiD 2.0 images and are redistributed under its CC BY 4.0 licence;
`app/assets/ATTRIBUTION.md` carries the citation and is written by the export script.

## Licences
Code: MIT. Data: see DATASETS.md.
