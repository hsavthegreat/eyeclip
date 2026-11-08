# Datasets

## RFMiD 2.0 (used)
- Source: https://zenodo.org/record/7505822 (DOI 10.5281/zenodo.7505822)
- Licence: CC BY 4.0 (attribution required). Citation: VERIFY from the Zenodo record.
- 860 colour fundus images, 51 disease labels, multi-label, official train/validation/test split
  (509 / 177 / 174 label rows). Images are NOT redistributed in this repo.
- Data issues in the official release, handled by dropping (not renaming): 6 label rows have no
  image in their own split folder (train 89, 216; test 151, 405, 191, 490; some IDs are duplicated
  across split CSVs). Result: 854 matched images. Orphan image files with no label row: VERIFY (paste list).
- Experiment subset: exactly-one-positive images (663). Main classes: WNL 261, DR 70, CRS 27, RT 23 images
  (train/val/test counts in manifests/class_table.csv). BRVO (17) is a secondary class only.
- Not verified: patient identifiers are not available, so both eyes of one patient could appear in different splits.

## Pretrained models (frozen, not fine-tuned)
- ResNet-50, ImageNet weights (torchvision): licence/terms VERIFY
- CLIP ViT-B/16, OpenAI weights, loaded via OpenCLIP as ViT-B-16-quickgelu: licence VERIFY
- BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): licence VERIFY

## Planned for the cross-dataset test (not yet used)
- ODIR-5K (licence not stated on the challenge page: VERIFY) and/or IDRiD (CC BY 4.0).
