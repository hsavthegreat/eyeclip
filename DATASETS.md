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

## FIVES (dataset B, cross-dataset test only)
- Source: https://doi.org/10.6084/m9.figshare.19688169 (v1). Cite: Jin, K., Huang, X., Zhou, J.
  *et al.* FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation.
  *Sci Data* **9**, 475 (2022). https://doi.org/10.1038/s41597-022-01564-3 (PMCID PMC9352679).
  Citation verified: the Figshare record asks for exactly this paper.
- Licence: CC BY 4.0, attribution required (verified on the Figshare record; the paper is CC BY 4.0
  as well). Images are NOT redistributed in this repo.
- 800 colour fundus images from the Second Affiliated Hospital of Zhejiang University, Hangzhou;
  Topcon TRC-NW8, 50 degree field centred on the macula; 2048x2048 PNG, RGB; 200 per class
  (healthy, AMD, DR, glaucoma).
- Used: 400 of 800 — 200 normal -> WNL, 200 diabetic retinopathy -> DR (class letters N and D).
  AMD (A) and glaucoma (G) images and the vessel masks in `Ground truth/` are not used.
  Distributed split kept as-is: train 150 DR + 150 WNL, test 50 DR + 50 WNL.
  Manifest: `manifests/manifest_B.csv` (rebuilt from raw data by notebook cell 9).
- Class overlap with RFMiD 2.0 is only WNL and DR, so CRS and RT cannot be tested cross-dataset;
  the cross-dataset comparison is limited to WNL and DR.
- Quality flags (IC, Blur, LC) from the dataset's `Quality Assessment.xlsx`; quality = 0 if any of
  the three is 0. 78 of 400 flagged (56 DR, 22 WNL). All images kept in the main test; the
  quality = 1 subset (322: 144 DR, 178 WNL) is reported as a sensitivity check. The flagged rate is
  uneven by split and class (train DR 42, test DR 14, train WNL 5, test WNL 17), so filtering on
  quality would not be class- or split-neutral.
- Data issues: the quality workbook ships inside the distributed `.rar`, and stock unrar / unar /
  7-Zip fail on that entry — extract it with `unrar-nonfree` (30,824 bytes, md5
  c50a0a62b149f485a3023bec9131c558). Six pairs of images are byte-identical duplicates, all class A,
  so none are among the 400 used and there is no cross-split leakage in D/N.
- Not verified: patient identifiers are not available, so both eyes of one patient could appear in
  different splits.

## Pretrained models (frozen, not fine-tuned)
- ResNet-50, ImageNet weights (torchvision): licence/terms VERIFY
- CLIP ViT-B/16, OpenAI weights, loaded via OpenCLIP as ViT-B-16-quickgelu: licence VERIFY
- BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224): licence VERIFY

## Cross-dataset test
- FIVES (above). ODIR-5K (licence not stated on the challenge page: VERIFY) and IDRiD (CC BY 4.0)
  were the earlier candidates and are not used.
