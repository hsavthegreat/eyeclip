# Attribution

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
