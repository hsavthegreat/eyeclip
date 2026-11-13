# EyeCLIP: few-shot recognition of rare eye diseases (research prototype)

Pilot study: can frozen CLIP-style image/text embeddings recognise rare retinal conditions from 1, 5 or 10
labelled examples per class? Not a diagnostic tool; no clinical validation.

## Status
**Done:** data audit and matching, fixed evaluation protocol, frozen embeddings for 3 models, few-shot evaluation
(4 classes); per-image predictions, bootstrap CIs, selective prediction, retrieval and a nearest-neighbour
near-duplicate audit (cell 8 of the notebook).
**Planned:** cross-dataset test, Streamlit demo.

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
| CLIP ViT-B/16 | logistic | 0.840 ± 0.073 | 0.857 ± 0.037 |
| CLIP ViT-B/16 | prototype | 0.833 ± 0.089 | 0.837 ± 0.065 |
| CLIP ViT-B/16 | hybrid | 0.837 ± 0.091 | 0.847 ± 0.070 |
| BiomedCLIP | logistic | 0.772 ± 0.042 | 0.815 ± 0.046 |
| BiomedCLIP | prototype | 0.764 ± 0.028 | 0.801 ± 0.054 |
| BiomedCLIP | hybrid | 0.749 ± 0.059 | 0.793 ± 0.044 |
| ResNet-50 (baseline) | logistic | 0.781 ± 0.051 | 0.800 ± 0.040 |
| ResNet-50 (baseline) | prototype | 0.788 ± 0.045 | 0.798 ± 0.037 |

Zero-shot text prompts: macro-F1 0.12–0.16 (CLIP) and 0.42–0.45 (BiomedCLIP), depending on centring. 1-shot results
are very noisy (std 0.1–0.2). Source: results/summary_all_runs.csv, recalls_all_runs.csv.

**Reading the results.** Few-shot adaptation helps a great deal over zero-shot. CLIP features are somewhat ahead of the
ImageNet baseline at 5–10 shots, mainly because of DR (28/28 eval images recalled in every seed vs 0.75–0.91 for the others).
For the rarer CRS and RT (11 and 9 eval images), intervals overlap and no method is clearly better.

## Limitations
Small evaluation sets for rare classes; one dataset; 4 classes after dropping multi-label images; patient-level
independence of splits not verified; similarity scores are not calibrated probabilities; no clinical validation.
The nearest-neighbour audit (top-1 cosine of each eval image to the training pool; written by cell 8 to
`retrieval_nearest.csv` and `fig_similarity.png` in the results directory) can flag a possible duplicate or
same-patient image but cannot prove one: it uses no patient IDs, and a high cosine may also be two different
eyes of similar appearance.

## Reproduce
Open notebooks/eyeclip.ipynb in Colab (GPU), mount Drive, place RFMiD 2.0 under
/content/drive/MyDrive/eyeclip/data/dataset_main/RFMiD2_0, run cells in order. Cell 6 runs on CPU from the cached embeddings.

## Licences
Code: MIT. Data: see DATASETS.md.
