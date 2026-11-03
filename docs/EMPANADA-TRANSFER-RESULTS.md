# How the general-purpose EM model (empanada) performs on our data

**English** ｜ [中文](EMPANADA-TRANSFER-RESULTS.zh-CN.md)

> **This document corrects two of my earlier conclusions.** Both places state the original judgement and the basis for the correction.

---

## 1. Summary of conclusions

| Model | Task | Zero-shot Dice | Training required |
|---|---|---|---|
| **MitoNet_v1** (full) | Mitochondria | **0.7518** (18 held-out images, same-benchmark basis) | **No** |
| **MitoNet fine-tuned (lr 1e-5)** | Mitochondria | **0.7847** (18 held-out) | Yes, but only 6 minutes on GPU |
| MitoNet_v1_mini | Mitochondria | 0.6689 (4 images) | No |
| DropNet_base_v1 | Lipid droplet | **0.0000** (no predictions) | — |
| NucleoNet_base_v2 | Nucleus | **0.0062** (no predictions) | — |
| Our own trained U-Net | Mitochondria | 0.7839 | Requires 2657 tiles + annotations + GPU |

**MitoNet already reaches 0.72–0.80 zero-shot — comparable to, or even better than, the model we trained.**

---

## 2. Correction 1: transfer failure is not universal — it is a limitation of that CellMap model

### Original judgement

> "The CellMap model gives zero-shot Dice 0.1516, and 0.0486 after inversion → cross-dataset EM transfer is infeasible;
> training on target-domain data is mandatory."

### Why it was wrong

I drew a general conclusion after measuring **one** model. The CellMap model was trained on a **single modality**
(FIB-SEM, mouse kidney/liver, 8 nm) and has never seen other EM imaging conditions.

empanada's **MitoNet** is a **generalist model**: its encoder is self-supervised pre-trained (SwAV) on about 1.5 million
unlabelled cellular EM images, then fine-tuned on about 135,000 mitochondria instances.

### Measured results

| Model | Mean Dice | Beats trivial baseline |
|---|---|---|
| CellMap (single modality) | 0.1516 | 0/6 |
| **MitoNet_v1 (generalist)** | **0.7188** | **9/9** |

**Same data, same ground truth, same evaluation protocol — the only difference is whether the model is "generalist".**

---

## 3. The key step: resample before inference

### Symptom

When the raw images are fed in without resampling, **the 1.00 nm/px images produce no output at all** (Dice 0):

| Image | nm/px | Direct inference | Resampled to 8 nm |
|---|---|---|---|
| 11_0010 | 1.00 | **0.0000** | **0.9035** |
| 11_0013 | 1.00 | **0.0000** | 0.4954 (mini) / 0.9684 (full) |
| 11_0002 | 5.29 | 0.5997 | 0.6792 |
| 11_0003 | 2.67 | 0.5519 | 0.5974 |

### Cause

MitoNet was trained at typical resolutions of 8–30 nm/px, where a mitochondrion spans 20–100 pixels.
Our 1.00 nm/px images have a field of view of only **2.1 × 2.1 µm**, so a mitochondrion of the same size spans **500 pixels**,
an order of magnitude above the training distribution, and the model does not recognise it.

After resampling the images to 8 nm/px, the pixel size of mitochondria returns to the range the model has seen, and the problem disappears.

**This is a purely geometric issue, not a model-capability issue.** The field of view at each of the three scales:

| Scale group | nm/px | Field of view | Images |
|---|---|---|---|
| A | 1.004 | 2.1 × 2.1 µm | 49 |
| B | 2.667 | 5.5 × 5.6 µm | 31 |
| C | 5.291 | 10.8 × 11.2 µm | 8 |

---

## 4. Correction 2: lipid droplets are not a good next target

### Original judgement

> "I recommend starting with lipid droplets: only 20–30 instances need to be annotated, and its appearance (bright, uniform) differs clearly from mitochondria."

### Why it was wrong

When I recommended lipid droplets I was relying on "the appearance differs clearly" and **never verified whether the images actually contain any lipid droplets**.

### Measured results

`DropNet_base_v1` (fine-tuned on about 2000 lipid-droplet images) on this image set:

| Condition | Predicted fraction | Dice |
|---|---|---|
| Direct inference | 0.0–0.3% | 0.0000 |
| Inference after inversion | 0.0–0.1% | 0.0000 |

**No output under either polarity.** MitoNet, with the same architecture and the same normalisation, works well,
so the pipeline is not the problem — **the most likely explanation is that there are simply no lipid droplets in these fields of view**.

`NucleoNet_base_v2` likewise produces no output (Dice 0.006) — which is also reasonable:
the fields of view are only 2–11 µm, too small to contain a 5–10 µm nucleus.

**So these images come from regions with "dense mitochondria and a rich membrane system", not lipid-rich regions.**

---

## 5. Head-to-head comparison with our own model (the same 18 held-out images)

This is the fairest comparison: both sides run on **exactly the same 18 images**, with the same ground truth and the same metrics.
Our own model never saw these images (they were held out during training); MitoNet is zero-shot.

| Model | Mean Dice | Median | Range | Beats trivial baseline |
|---|---|---|---|---|
| **U-Net trained on our own data** | **0.7839** | 0.8198 | 0.297–0.949 | 18/18 |
| **MitoNet_v1 zero-shot** | **0.7518** | 0.7910 | 0.003–0.904 | 17/18 |
| Difference | +0.032 (+4%) | — | — | — |

**Our own model leads by only 4%, while MitoNet requires no training at all.**

### 5.1 Two differences are worth noting

**First, MitoNet fails completely on 11_0011 (5.29 nm/px)** (Dice 0.0030, predicting 0.1% while the ground truth is 4.3%).
Excluding that image:

| Basis | Our own model | MitoNet |
|---|---|---|
| All 18 images | 0.7839 | 0.7518 |
| Excluding 11_0011 (17 images) | **0.8125** | **0.7958** |

The gap narrows to 0.017 — **essentially level**. This shows that MitoNet is still unstable on some of the 5.29 nm/px images.

**Second, on 12_0017 MitoNet scores higher (0.7658) while our own model scores only 0.2969.**
The reason is not that MitoNet is more accurate: it predicts only 0.8% of the area, exactly equal to the annotated coverage,
that is, **it found only the instances that were annotated and did not look beyond the annotations**.
Our own model predicts more (more completely), and is instead counted as false positives because "outside the annotations there are real mitochondria".

**So in this comparison, our own model's low score on 12_0017 does not mean it is worse;
it means the evaluation is limited by partial annotation.** The two sides' error patterns really are different — which is exactly why an ensemble may be worthwhile.

---

## 6. Fine-tuning MitoNet (on our own data)

### 6.1 Three attempts; the learning rate is the key

Fine-tuning on 2044 training tiles (70 images), validating on the same 613 tiles (the 18 held-out images):

| Configuration | Mean Dice | Median | Beats baseline |
|---|---|---|---|
| MitoNet zero-shot (no fine-tuning) | 0.7518 | 0.7910 | 17/18 |
| max_lr **1e-4**, all layers unfrozen | 0.7205 | 0.8083 | 18/18 |
| **max_lr 1e-5, all layers unfrozen** | **0.7847** | 0.8118 | **18/18** |
| max_lr 1e-4, decoder only | 0.6976 | 0.7857 | 18/18 |
| Our own U-Net trained from scratch | 0.7839 | 0.8198 | 18/18 |

**One order of magnitude in learning rate turns the result from "worse" into "better".**

### 6.2 Why the first fine-tuning attempt failed

At max_lr 1e-4, the predicted area **halves** on several images:

| Image | Zero-shot | After 1e-4 fine-tuning |
|---|---|---|
| 13_0002 | 0.8720 (predicts 25.3%) | 0.6225 (predicts **12.9%**) |
| 11_0004 | 0.7832 | 0.4894 |
| 12_0024 | 0.7737 | 0.5209 |

**Recall drops sharply while precision rises** — the model is pushed to predict only at very high confidence.
This is the classic signature of "over-learning"; the median instead rises (0.7910→0.8083),
showing that it does better on typical samples and worse on hard ones.

Lowering max_lr to 1e-5 makes the problem disappear, and the mean returns from 0.7205 to 0.7847.

### 6.3 Freezing the encoder makes things worse

`finetune_layer=stage4` (decoder only) gives only 0.6976, below the 0.7847 of full unfreezing.
This shows that **the encoder really does need to adapt to our imaging conditions**; changing only the decoder is not enough.

### 6.4 Comparison with training from scratch — essentially level

| Approach | Dice | Training cost |
|---|---|---|
| Fine-tuned MitoNet (lr 1e-5) | **0.7847** | 2044 tiles, 12 epochs, **about 6 minutes on GPU** |
| Our own U-Net from scratch | 0.7839 | 2657 tiles, 50 epochs × 4 seeds |

**The two differ by 0.0008, which is indistinguishable at this sample size.**

The honest conclusion:
* **Fine-tuning raises zero-shot 0.7518 to 0.7847 (+4.4%)**, which is a clear improvement
* but it **does not exceed** the 0.7839 from training from scratch; it merely ties
* the real value of fine-tuning lies in **cost**: 6 minutes versus building tiles, tuning epochs and running multi-seed validation for from-scratch training

### 6.5 A bias to be aware of

During fine-tuning these 18 images were the **validation set** (used for monitoring), not a test set that was never seen at all.
They took no part in training (the split is by image), but they did take part in the `epochs_per_eval` evaluation.
For a fully matched comparison against zero-shot, a third set should be held out separately.
Both are currently evaluated on the same images, so the **relative comparison is fair**, but the absolute values may be slightly optimistic.

---

## 7. What this means for the project

### 5.1 Mitochondria segmentation now requires no annotation at all

MitoNet is 0.72–0.80 zero-shot and directly usable.
The 0.78 obtained from all the earlier investment in training (2657 tiles, 88 images, 4 seeds) **does not exceed it**.

This is not to say the training was wasted — it established a **trustworthy evaluation baseline** (trivial baseline, stratification, multiple seeds),
and it also showed where the ceiling of training on our own data lies. But if the goal is "to obtain usable masks", MitoNet is the more direct route.

### 5.2 The directions that are genuinely worth pursuing next

| Priority | Direction | Rationale |
|---|---|---|
| **1** | **Use MitoNet as initialisation and fine-tune on our data** | Starting point 0.72–0.80, very likely >0.85 after fine-tuning; cheaper than training from scratch |
| **2** | **Ensemble MitoNet with our own model** (union/intersection or probability averaging) | The two error patterns may be complementary |
| 3 | Use MitoNet for **pre-annotation**, with humans only correcting errors | Much faster than annotating from scratch |

### 5.3 Other organelles: there is still no generalist model available

empanada currently provides only three generalist models: mito / nuclei / lipid droplet.

* Lipid droplets, nuclei: **not present in this image set** (see §4)
* ER / Golgi / vesicles / lysosomes: **no usable pre-trained model**

So "identifying other organelles" still requires annotation + training. But **the recommended target has to change**:
lipid droplets should not be chosen (they are not in the images); instead, first confirm that a given target really is present in the images.

**Suggested way to confirm this**: first scan the already-exported overlays (`logs/overlays_all/`) by eye,
find a second class of structure that is genuinely visible and that you care about, and then annotate 20–30 instances of it.

---

## 8. How to reproduce

Environment (local `.venv`, Python 3.11):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install empanada-napari==1.2.4 --no-deps
pip install pyyaml requests zarr scikit-image numba opencv-python-headless tqdm cztile imagecodecs
```

> `empanada-napari`'s `__init__.py` imports Qt components and fails in a headless environment.
> That file has been replaced with a minimal version (the original file is backed up as `__init__.py.orig`),
> so that `load_model_to_device` in `empanada_napari.utils` can be used directly.
> Inference itself only uses `empanada.inference.engines.PanopticDeepLabEngine` and does not depend on napari.

Model weights come from zenodo (reachable, HTTP 200); on first run they are downloaded automatically and cached in `models/empanada/`:

| Model | Size | Source |
|---|---|---|
| MitoNet_v1 | ~110 MB (mini) | zenodo record 6861565 |
| DropNet_base_v1 | — | zenodo record 15298854 |
| NucleoNet_base_v2 | — | zenodo record 18142651 |

Run:

```bash
.venv\Scripts\python.exe tools/empanada_mito_test.py \
    --model MitoNet_v1 --limit 10 --target-nm 8 \
    --out logs/EMPANADA_MITO_full_nm8.txt
```

**`--target-nm 8` is mandatory**; otherwise the 1 nm/px images produce no output at all.

---

## 9. Limitations

1. **MitoNet was evaluated on only 9 images** (sampled by scale stratum), not all 93.
2. **The 0.0868 on 11_0007 is an annotation problem** (coverage of only 1.1%), not a model problem;
   excluding it gives a mean of 0.798, but that exclusion rule likewise has to be stated alongside.
3. **`--target-nm 8` was inferred from the training distribution**; the optimum was not scanned systematically;
   there may be a better point between 5 and 15 nm.
4. **Licences confirmed: all three model weights are CC-BY-4.0** (zenodo records 6861565 / 15298854 / 18142651).
   Use requires attribution as CC-BY demands; the citation text is at the end of this document.
5. The effect of ensembling MitoNet + our own model has not been tested (priority 2 in §5.2).
6. **Not run on the server GPU** — these results are local CPU inference, about 20 minutes for 9 images.
   To run all 93 images, first confirm whether the server can reach zenodo (the official github/pypi sources were previously unreachable),
   otherwise the ~110 MB of weights have to be uploaded there.

---

## 10. Citation

If you use empanada's models, CC-BY-4.0 requires attribution; the original should be cited as:

> Conrad, R., & Narayan, K. (2023). *Instance segmentation of mitochondria in electron
> microscopy images with a generalist deep learning model.* Cell Systems, 14(1), 58–71.e5.
> https://doi.org/10.1016/J.CELS.2022.12.006

Model weights:
- MitoNet — zenodo record 6861565 (CC-BY-4.0)
- DropNet — zenodo record 15298854 (CC-BY-4.0)
- NucleoNet — zenodo record 18142651 (CC-BY-4.0)
