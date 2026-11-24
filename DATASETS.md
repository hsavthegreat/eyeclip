# Dataset sources and provenance

This project uses RFMiD 2.0 for its few-shot experiments and FIVES for the cross-dataset test. The raw
dataset archives are not committed. Counts below distinguish the upstream releases from the matched data
recorded in this repository's manifests.

## RFMiD 2.0

- **Source:** [Zenodo record 7505822](https://doi.org/10.5281/zenodo.7505822), version 1, published 2023-01-05.
- **License:** CC BY 4.0, as declared in the [Zenodo record metadata](https://zenodo.org/api/records/7505822).
- **Citation:** Panchal, S., Naik, A., Kokare, M., Pachade, S., Naigaonkar, R., Phadnis, P., & Bhange, A.
  (2023). *Retinal Fundus Multi-Disease Image Dataset (RFMiD) 2.0* [Data set]. Zenodo.
  https://doi.org/10.5281/zenodo.7505822.
- The release describes 860 images and 51 disease labels, with multi-label cases and train, validation, and
  test splits. The original label CSVs contain 509 / 177 / 174 rows by split.
- **Matched images:** [`manifests/manifest_cache.csv`](manifests/manifest_cache.csv) records 854 unique
  image paths: 507 train, 177 validation, and 170 test. The recorded audit found 860 image files with 860
  unique stems. Matching is split-folder-aware: an image is used only with the label row for its own split.
  Six label rows have no matching image in their own split folder:
  train IDs 89 and 216; test IDs 151, 191, 405, and 490. IDs 191, 216, and 490 also appear in another
  split's CSV, but are not cross-matched. Six image files have no label row: train IDs 79 and 222; test IDs
  251, 291, 404, and 489. These lists are from the notebook's recorded audit output.
- Of the 854 matched rows, 663 have exactly one positive label, 137 have two, 44 have three, and 10 have
  four. The experiments use only the 663 exactly-one-positive rows.
- **Four-class task:** WNL (normal), DR (diabetic retinopathy), CRS (cataract), and RT (refractive error).
  Selection requires at least 20 single-positive examples overall and 10 in train. Support examples come
  from train; validation and test are combined for evaluation. No evaluation examples are used for tuning.

| Class | Train | Validation | Test | Total | Evaluation |
|---|---:|---:|---:|---:|---:|
| WNL | 156 | 53 | 52 | 261 | 105 |
| DR | 42 | 14 | 14 | 70 | 28 |
| CRS | 16 | 4 | 7 | 27 | 11 |
| RT | 14 | 3 | 6 | 23 | 9 |

Counts are from [`manifests/class_table.csv`](manifests/class_table.csv). BRVO has 17 single-positive examples
and is kept as a secondary class; the other 265 single-positive rows belong to unused classes. This is a
closed four-class experiment, not a 51-class evaluation.

**Distribution in this repository:** Full-size RFMiD images are not included. The demo includes 228 resized
JPEG thumbnails under `app/assets/thumbs/`; their source and modifications are documented in
[`app/assets/ATTRIBUTION.md`](app/assets/ATTRIBUTION.md).

**Split limitation:** Patient identifiers are not provided in the release, so patient-level separation across
splits cannot be verified.

## FIVES

- **Source:** [Figshare record 19688169, version 1](https://doi.org/10.6084/m9.figshare.19688169.v1),
  published 2022-05-01. The [record metadata](https://api.figshare.com/v2/articles/19688169) declares
  CC BY 4.0 and requests citation of the paper below.
- **Citation:** Jin, K., Huang, X., Zhou, J., et al. (2022). FIVES: A Fundus Image Dataset for Artificial
  Intelligence based Vessel Segmentation. *Scientific Data*, 9, 475.
  https://doi.org/10.1038/s41597-022-01564-3.
- The release contains 800 color fundus images across healthy, AMD, DR, and glaucoma classes. This project
  uses 400: 200 normal images mapped to WNL and 200 DR images. AMD, glaucoma, and vessel masks are not used.
- The distributed split is retained: 150 WNL and 150 DR in train; 50 WNL and 50 DR in test. The matched
  rows are listed in [`manifests/manifest_B.csv`](manifests/manifest_B.csv).
- Only WNL and DR overlap with the four RFMiD experiment classes, so the cross-dataset test covers those two
  classes only.
- The quality workbook supplies illumination/color (IC), blur, and low-contrast (LC) flags. An image is
  marked quality 0 if any flag is 0. Of the 400 used images, 78 are flagged (56 DR, 22 WNL); all remain in
  the main test. The quality 1 subset contains 322 images (144 DR, 178 WNL) and is reported separately as a
  sensitivity check. Flag rates differ by class and split: train DR 42, test DR 14, train WNL 5, test WNL 17.
- The FIVES RAR contains the quality workbook. In the recorded environment, stock `unrar`, `unar`, and 7-Zip
  did not extract that entry; `unrar-nonfree` did. The workbook was 30,824 bytes with MD5
  `c50a0a62b149f485a3023bec9131c558`. Six pairs of byte-identical images were found, all in the unused AMD
  class, so none of the 400 WNL/DR images are duplicate pairs.

**Split limitation:** Patient identifiers are not provided in the release, so patient-level separation across
splits cannot be verified.

## Pretrained models

The encoders are frozen; none is fine-tuned in this project. The sources below were checked on 2026-11-24.
Repository or model-card licenses are recorded as published; they do not establish a separate license for
every upstream dataset or derived artifact.

| Model used | Source and terms recorded |
|---|---|
| ResNet-50 ImageNet baseline | [Torchvision model documentation](https://pytorch.org/vision/stable/models.html) says pretrained models may have their own licenses or restrictions. The [Torchvision code license](https://github.com/pytorch/vision/blob/main/LICENSE) is BSD 3-Clause; the docs do not specify separate terms for this checkpoint. |
| CLIP ViT-B/16 | OpenAI weights loaded through OpenCLIP as `ViT-B-16-quickgelu` (`pretrained="openai"`). The [OpenAI CLIP repository](https://github.com/openai/CLIP) is MIT-licensed; its [license file](https://github.com/openai/CLIP/blob/main/LICENSE) does not separately identify checkpoint terms. |
| BiomedCLIP | [Microsoft model card](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) lists MIT. Its intended-use section limits use to research and reproducibility and says deployed use cases are out of scope. |
