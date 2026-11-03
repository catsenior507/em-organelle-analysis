# Golgi annotation data: what already exists online

**English** ｜ [中文](GOLGI-DATA-SOURCES.zh-CN.md)

> Background: there is no time to annotate it ourselves, so we need to find Golgi EM annotations in public data.
> Search date 2026-11-02. **All URLs were verified reachable (2026-11-02).**

---

## 1. First choice: ASEM (the Kirchhausen lab at Harvard)

**This is currently the best-matching resource** — it provides both a **trained Golgi model** and
**Golgi ground-truth volumes**, and it is EM (FIB-SEM).

| Item | Content |
|---|---|
| Full name | Automated Segmentation of Intracellular Substructures in EM |
| Institution | Kirchhausen Lab, Harvard Medical School |
| Paper | Gallusser et al., *J Cell Biol* 222(2):e202208005 (2023) |
| License | **CC BY-SA 4.0** (commercial use allowed; requires attribution + share-alike) |
| Access | public S3 bucket, **no AWS account needed** |
| Registry page | https://registry.opendata.aws/asem-project/ |
| Code | https://github.com/kirchhausenlab/incasem (**not on PyPI**) |

### 1.1 Pretrained models (70 MB each, already downloaded to `models/asem/`)

```
models/pretrained_checkpoints/model_checkpoint_1837_golgi_CF.pt    ← Golgi
models/pretrained_checkpoints/model_checkpoint_1841_er_CF.pt
models/pretrained_checkpoints/model_checkpoint_1847_mito_CF.pt
models/pretrained_checkpoints/model_checkpoint_1675_mito_HPF.pt
models/pretrained_checkpoints/model_checkpoint_2000_np_HPF.pt      nuclear pore
models/pretrained_checkpoints/model_checkpoint_1986_ccp_HPF.pt     clathrin
models/pretrained_checkpoints/model_checkpoint_2661_clathrin.pt
models/pretrained_checkpoints/model_checkpoint_2632_eres_cop.pt
```

**CF = chemical fixation, HPF = high-pressure freezing + freeze substitution.** There is one set of models per preparation method.
Which one our own images belong to has to be determined — that is the precondition for choosing a model.

### 1.2 Golgi ground-truth data (the key part)

Of the 26 datasets, **only 4 come with `labels/golgi`**:

| Dataset | Annotated volumes | golgi volume shape | Chunks | Estimated size |
|---|---|---|---|---|
| 46 | er / golgi / mito | [1921, 1100, 4600] | ~5184 | **about 40 GB** |
| 58 | er / golgi / mito | [1982, 1000, 4752] | ~4864 | **about 40 GB** |
| **61** | er / golgi / mito | [210, 283, 371] | **18** | **about 150 MB** |
| **64** | er / golgi / mito | [284, 204, 424] | **24** | **about 200 MB** |

> **The golgi annotations in 61 and 64 are only a subregion of the raw volume** (raw is 1805×1400×4800),
> so they are small and practical. 46/58 are whole-volume annotations and far too large.

**These three classes are annotated together (er + golgi + mito), which means this data can be used for a
real multi-class interaction analysis** — exactly the piece our own data lacks.

### 1.3 An architectural issue that must be noted

The ASEM models are **3D U-Nets** (5.84 M parameters, 3D convolutions),
whereas our images are **single 2D sections**. Feeding one section straight into a 3D model works badly.

Two routes:

* **A (recommended)**: **cut the ASEM golgi ground-truth volume into 2D sections**
  and train a 2D model with our existing `train_seg.py`. The pipeline already exists; lowest risk.
* **B**: if our numbered images (e.g. 11_0002, 11_0003…) really are **consecutive z slices**,
  3–5 of them can be stacked into a small volume to run a 3D model. This requires first confirming that they are consecutive.

### 1.4 Checkpoint architecture (decoded; can be rebuilt offline)

`model_checkpoint_1837_golgi_CF.pt` is a **plain state_dict** (not TorchScript),
with key names in the classic `pytorch-3dunet` style:

```
unet.l_conv.0..3   encoder  32→64→128→256, 2×(Conv3d3³+ReLU) per block, MaxPool3d(2) between blocks
unet.r_conv.0.2    decoder  up(256)→256 ++ skip(l_conv.2=128) = 384 → conv → 128
unet.r_conv.0.1    decoder  up(128)→128 ++ skip(l_conv.1= 64) = 192 → conv →  64
unet.r_conv.0.0    decoder  up( 64)→ 64 ++ skip(l_conv.0= 32) =  96 → conv →  32
final_conv_pass.conv_pass.0   32 → 2
```

**Every individual tensor shape has been checked, so it can be rebuilt offline without the `incasem` package**
(that package is not on PyPI and can only be installed from GitHub, and GitHub is not directly reachable from either the local machine or the server).

> Note: the state_dict holds only conv weights/biases, **no BatchNorm parameters**,
> which means this version is a 3D U-Net without BN.

---

## 2. Second choice: CellMap / OpenOrganelle (already in use)

CellMap's 73 classes include `golgi`. But:

* its form is **per-crop figshare entries** (e.g. `jrc_mus-kidney` crop164),
  so it has to be pulled crop by crop, which is fragmented
* our experience is that **CellMap single-modality models transfer poorly** (mito zero-shot Dice 0.1516),
  so even if a golgi model were trained, transferring it to our data would most likely be unsatisfactory

**Conclusion**: CellMap's golgi can serve as supplementary training data, but should not be the first choice.

---

## 3. Options ruled out

| Resource | Why it does not fit |
|---|---|
| MitoEM 2.0 | mitochondria only |
| EMPIAR-11695 | has Golgi annotations, but 434 GB, and it is Cal51 cell line FIB-SEM |
| nanotomy | 2D large-field EM, annotated but mostly pathology; Golgi coverage uncertain |
| various fluorescence Golgi datasets | wrong modality (we are EM) |

---

## 4. Measured access performance

| Path | Speed | Note |
|---|---|---|
| Local machine → ASEM S3 | **about 47 kB/s** | a 70 MB model took **25 minutes** |
| Server → S3 | 0.12–0.37 MB/s | measured earlier on CellMap |

**Conclusion: pulling large files from S3 is slow.** The golgi annotations of 61/64 (about 150–200 MB each)
would take 1–1.5 hours at 47 kB/s, so **pulling them on the server is recommended** (3–8× faster),
or pull only the needed subregion chunks.

---

## 5. Suggested next steps

In order of value:

| Priority | Action | Rationale |
|---|---|---|
| **1** | **wait until your own Golgi annotations are uploaded**, then train a 2D model with the existing pipeline | same imaging conditions, so the transfer problem is minimal |
| **2** | if our own annotations are too few: pull golgi ground truth from **61/64** and cut it into 2D for supplementary training | same modality (FIB-SEM), correct class |
| 3 | use the ASEM golgi model as a **control baseline** (requires rebuilding the 3D U-Net first) | gives a "general model, zero-shot" reference, analogous to MitoNet for mitochondria |
| 4 | if our images are confirmed to be consecutive z slices, then consider using a 3D model directly | continuity must be verified first |

**Key reminder**: first determine whether our own images are **CF or HPF** preparations.
The ASEM golgi model exists only in a CF version (`1837_golgi_CF`),
so if we are HPF, applying it directly will be less effective.
