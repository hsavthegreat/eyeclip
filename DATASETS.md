
# Datasets and licences

> **Template (Stage 0).** One record per dataset used in EyeCLIP.
> Fill every field from the dataset's **official pages and licence text**, never from memory.
> If a term cannot be confirmed, write **VERIFY** instead of guessing.
> This file is a working record, not legal advice.

## Summary

| # | Dataset | Version / release | Access | Licence (name + link) | Obtained on | Verified on |
|---|---------|-------------------|--------|-----------------------|-------------|-------------|
| 1 | `<name>` | `<e.g. v1.0 / 2019 challenge>` | `<open / registration / DUA>` | `<name + URL>` | `YYYY-MM-DD` | `YYYY-MM-DD` |

*(One row per dataset; keep in sync with the records below.)*

## Records

*(Copy this block for each dataset.)*

### `<DATASET NAME>`

- **Official page / download:** `<URL>`
- **Citation (as required by the source):** `<BibTeX from the official page>`
- **Licence:** `<licence name>` — `<URL to the licence text>`
  - Key terms for this project (quote or closely paraphrase): `<redistribution allowed? derivatives such as embeddings? research-only? attribution required?>`
  - Anything unclear: **VERIFY**
- **Access conditions:** `<direct download / free account + agreement / signed data use agreement>`
- **Obtained on:** `YYYY-MM-DD`
- **Stored at (Drive only — never committed):** `/content/drive/MyDrive/eyeclip/data/<name>/`
- **What we use:** `<images / labels / official splits / which classes>`
- **Classes used in this project:** `<list + per-class image counts — fill after the Stage 1 audit>`
- **Known risks:** `<duplicate images; patient overlap with other sources; class-definition mismatches; label noise; anything that could leak between support and test>`

## Ground rules

1. **Never commit datasets or embeddings to this repo.** `.gitignore` excludes `data/`, `embeddings/`, `*.npy`, `*.pt`, `*.pth` and Drive paths.
2. Cached embeddings are **derivatives of the datasets** — whether they may be shared outside the project is governed by each dataset's licence (**VERIFY per dataset before sharing**).
3. Record the date each dataset was obtained; sources and licence texts change over time.
4. If a dataset's terms turn out to forbid this use, remove the dataset from the project and note here when and why.
