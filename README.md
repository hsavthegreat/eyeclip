# EyeCLIP: few-shot recognition of rare eye diseases (research prototype)

Pilot study: can frozen CLIP-style image/text embeddings recognise rare retinal conditions from 1, 5 or 10
labelled examples per class? Not a diagnostic tool; no clinical validation.

## Status
**Done:** data audit and matching, fixed evaluation protocol, frozen embeddings for 3 models, few-shot evaluation
(4 classes); per-image predictions, bootstrap CIs, selective prediction, retrieval and a nearest-neighbour
near-duplicate audit (cell 8 of the notebook).
**Planned:** cross-dataset test, Streamlit demo.

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
the ordering of the two encoders is not established here. For the rarer CRS and RT (11 and 9 eval images), intervals
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

## Licences
Code: MIT. Data: see DATASETS.md.
