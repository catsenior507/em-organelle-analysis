# DETR Training Readiness Report

**English** ｜ [中文](DETR-TRAINING-READINESS.zh-CN.md)

> Server: `youruser@your-server-host:22`, environment `dachuang`
> Date: 2026-10-31
> Conclusion: **the training pipeline has been verified end-to-end on real data**; only the "more data" step remains.

---

## 1. Verified facts (all measured, not inferred)

### 1.1 Environment

| Item | Value |
|---|---|
| GPU | 2× RTX 3080, 10 018 MiB each, compute capability 8.6 |
| torch | **2.6.0+cu118**, `cuda_avail=True`, GPU matmul passes |
| Driver | 455.38 (native CUDA 11.1, but the CUDA 11.8 runtime is available) |
| transformers | 5.20.2 + timm (the default DETR backbone in 5.x requires timm) |
| DETR weights | `facebook/detr-resnet-50`, downloaded via `HF_ENDPOINT=https://hf-mirror.com` (333 MB / 37 s) |

### 1.2 Data

| Item | Value |
|---|---|
| Label source | Fetched directly from S3 per chunk (**the 42 GB zip was not downloaded**) |
| Downloaded label volumes | 6 crops × mito/er = 12 volumes (300³ / 200³) |
| Crops in `jrc_mus-kidney` that contain both mito+er | **17–20** (different counting conventions; 17 by manifest, 20 by S3 directory scan) |
| EM planes | Taken on demand from `em/fibsem-uint8/s0` (8 nm), **already aligned with the labels** |
| Exported 2D slices | 13 (crop129 five layers 300×300, crop140 five layers 100×100, etc.), 202 instances |

### 1.3 Key alignment rules (verified)

```
EM origin (voxel) = translation(nm) / 8
EM extent (voxel) = shape * voxel_size / 8
label downsample factor = 8 / crop_voxel    (4 nm labels → 2×; 8 nm labels → 1×)
```

Measured for `jrc_mus-kidney`: all 17 mito+er crops **fall inside the EM extent**;
the EM planes extracted for crop129 are 100% non-zero (dense tissue), which shows the mapping is correct.

### 1.4 Training smoke test (real slices, not synthetic)

```
13 slices, 202 instances (mito 48, er 154)
model 41.5 M parameters, classification head rebuilt from COCO 91 classes to 3 (2 classes + no-object)
batch=1, size=480 : peak VRAM   831–1123 MiB
batch=2, size=480 : peak VRAM       1123 MiB  → 7–9× headroom at 85% of VRAM
loss 5.58 → 3.69 (4 steps, no tuning, only proving that it can decrease)
```

**Conclusion: 10 GB of VRAM is enough to fine-tune DETR; the batch size can go to 8–14 (480px) or 4–6 (640px).**

---

## 2. What is still missing: data volume

13 slices are **far from enough** for training. Fine-tuning DETR from COCO weights to a new
domain typically requires **thousands** of images and on the order of **200 epochs**. There are
currently only 13, which amounts to zero.

### Feasible expansion paths (ordered by cost)

| Step | Number of slices | Estimated data acquisition time |
|---|---|---|
| All 17–20 crops of `jrc_mus-kidney` × 30 layers per crop | ~500–600 | about 10–20 minutes |
| Extend to other datasets (`jrc_mus-liver` 23, `jrc_hela-2` 26, etc.) | ~2,000–7,000 | several hours (S3 is slow) |

Why data acquisition is cheap: a 100×100 4 nm crop covers only
**1 chunk** (128×128) in the 8 nm EM, so each crop needs only 2–4 chunks. What is really expensive is a 300³ crop
(needs 45 chunks).

---

## 3. Methodological caveats (they affect the conclusions and must go into the paper)

1. **The label resolution does not match the EM**: of the 17 mito+er crops, **16 have 4 nm labels**,
   while the only available EM is **8 nm**. The labels therefore have to be downsampled 2× to align pixel by pixel.
   Downsampling uses "max within block" (any foreground makes it foreground), which is safer for the fine ER network than averaging —
   averaging systematically drops thin structures.

2. **Downsampling merges instances**: mito labels are **instance IDs** (0–5), and after 2× downsampling adjacent instances
   may merge. The current approach first takes a per-class binary mask and then downsamples, so the training target is
   "the foreground region of each class" and instance boundaries are lost. For an instance-level target, one should instead
   **re-run connected-component labeling on the downsampled mask** (`slice_dataset.py` already emits boxes by connected component,
   so in practice a "merged blob" is treated as one instance).

3. **The classes are not the same across crops**: `crop1026`/`crop1027` have `mito_ribo`, `er_mem_all`
   but **not** `mito` and `er`. Earlier, taking the first N crops in alphabetical order caused all of them to 404.
   Crops are now filtered by "the directory actually contains the required classes".

4. **Only 8 nm EM exists**: `em/fibsem-uint8` has only s0=8 nm as the lower resolution limit of a single array.
   If 4 nm images must later be strictly aligned with 4 nm labels, another data source is required.

---

## 4. Reproduction commands

```bash
# on the server
conda activate dachuang
cd ~/dachuang_organelle
export HF_HOME=~/dachuang_organelle/models/hf
export HF_ENDPOINT=https://hf-mirror.com

# 1) labels (6 crops already downloaded; extend to 17 with --max-crops)
python scripts/fetch_labels.py --dataset jrc_mus-kidney --labels mito,er \
    --out data/labels --eligible-cache data/eligible_crops_jrc_mus-kidney.json \
    --max-crops 17

# 2) EM 2D slices (taken from the 8 nm EM by each crop's physical extent; labels are downsampled and aligned automatically)
python scripts/fetch_em_slices.py --manifest data/train_crop_manifest.csv \
    --labels data/labels --dataset jrc_mus-kidney \
    --z-stride 10 --max-slices 30 --out data/slices

# 3) training smoke test (verified)
python scripts/smoke_train_detr.py --slices data/slices --classes mito,er \
    --iters 6 --batch 1 --size 480
```

---

## 5. Recommended configuration for formal training (based on measured VRAM)

| Parameter | Recommended value | Basis |
|---|---|---|
| size (short side) | 480 | 640 also runs (288 MiB forward), but 480 is faster and allows a larger batch |
| batch | 8 | measured batch=2 uses 1123 MiB; 85% of VRAM allows 7–9× |
| epoch | 150–200 | the value commonly used in the DETR fine-tuning literature |
| lr | 1e-4 (backbone 1e-5) | DETR fine-tuning convention: smaller for the backbone |
| gradient clipping | 0.1 | measured grad_norm reaches 1372 without clipping, so clipping is mandatory |
| split | **grouped by crop**, never a random split by slice | adjacent layers of the same crop are highly similar, so a random split would severely overestimate performance |
| augmentation | horizontal/vertical flip + 90° rotation | EM images have no orientation prior, so rotation is safe |

**Grouped splitting is the easiest mistake to make**: adjacent z layers of the same crop are almost the same image.
With a random split, the validation set is contaminated with near neighbours from the training set and the metrics are inflated. Splitting must be grouped by crop.
