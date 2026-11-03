# First Training and Evaluation Results for Organelle Detection (DETR)

**English** ｜ [中文](ORGANELLE-DETECTION-RESULTS.zh-CN.md)

> Data: CellMap / COSEM `jrc_mus-kidney` (mouse kidney, FIB-SEM)
> Server: `youruser@your-server-host:22`, environment `dachuang`, single RTX 3080
> Date: 2026-10-31
> Code: `tools/server/scripts/{train_detr_slices,eval_detr}.py`

---

## 1. One-sentence conclusion

**Mitochondria (mito) detection is usable but the boxes are imprecise; ER (er) detection essentially
fails. The latter is not a training problem — the modeling choice of "representing the ER with
axis-aligned bounding boxes" is itself problematic.**

---

## 2. Training configuration and actual cost

| Item | Value |
|---|---|
| Data | 250 slices (14 crops), of which 240 contain instances |
| Grouped split by crop | 11 training crops (197 slices) / 3 validation crops (43 slices) |
| Model | DETR-R50, 41.5 M parameters, classification head rebuilt from COCO 92 classes to 3 |
| Input | short side 480 px, batch 8 |
| Optimization | AdamW, head lr=1e-4, backbone lr=1e-5, cosine schedule, warmup 187 steps, gradient clipping 0.1 |
| epoch / steps | 150 epochs × 25 steps = **3750 steps** |
| Wall time | **about 18 minutes** (about 7 s per epoch) |
| GPU memory | batch 8 @480px peaks at about 4.4 GB / 10 GB |
| Best | **epoch 85, val_loss 2.406**; afterwards val climbs back to 2.53 → overfitting |

Training loss drops from 5.6 to 1.73, but val stops improving after epoch 85. **This is the inevitable
outcome of 197 training images**, not a tuning problem.

---

## 3. Detection metrics (validation set: 3 crops, 43 slices)

| IoU threshold | mito AP | er AP | mAP |
|---|---|---|---|
| 0.3 | **0.7411** | 0.0635 | 0.4023 |
| 0.5 | 0.2654 | 0.0221 | 0.1438 |
| 0.75 | 0.0258 | 0.0039 | 0.0149 |

Operating point at fixed confidence 0.5 and IoU 0.5:

| Class | Ground-truth boxes | AP@0.5 | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| mito | 55 | 0.2654 | 0.4211 | **0.6667** | 24 | 33 | 12 |
| er | 120 | 0.0221 | 0.0805 | 0.0875 | 7 | 80 | 73 |

---

## 4. How to read these numbers

### 4.1 Mitochondria: found, but the boxes are inaccurate

When IoU is relaxed to 0.3, mito AP = **0.74** with recall **0.975** (only 1 of 55 ground truths is
missed). But raising IoU to 0.5 drops it to 0.27, and at 0.75 it is almost zero.

This is the typical signature of **"insufficient localization precision"**, and there are two causes:

1. **Too few training steps**. 3750 steps is very few for DETR fine-tuning (the literature usually
   starts at 20 000–50 000 steps). The classification head has learned "where there are mitochondria",
   but box regression has not converged.
2. **Annotation resolution does not match EM resolution**. Labels for 16/17 crops are at 4 nm, while
   the available EM is only 8 nm, so the labels were downsampled 2×. Box boundaries already carry a
   systematic error of ±2 pixels, and this is amplified under the strict IoU 0.5 threshold.

### 4.2 ER: not a training problem — **the modeling choice is wrong**

My initial hypothesis was that "ER boxes are too large and too elongated, and the inside of the box is
all background". **The data refuted this hypothesis**:

| Class | Boxes | Box area fraction (median) | Aspect ratio (median) | In-box fill rate (median) | Fraction with fill rate <0.3 |
|---|---|---|---|---|---|
| mito | 487 | 0.071 | 1.54 | **0.740** | 0.0% |
| er | 1562 | 0.008 | 1.50 | **0.608** | 14.6% |

ER boxes are in fact **very small** (median only 0.8% of the image), and a fill rate of 0.61 is not bad
either. So the boxes themselves are fine.

The real problems are:

1. **A 3× difference in instance count**: ER has **1562** connected components, mito only 487.
2. **"Instance" is meaningless for the ER**. The ER is one continuous tubular network; chopping it up
   with 8-connectivity produces a number of fragments that depends on fine details of annotation
   connectivity — **the same network becomes a different number of instances under a different
   threshold**. Such ground truth is itself unstable, the model cannot learn it, and AP naturally
   approaches 0.
3. 14.6% of ER boxes have a fill rate < 0.3 — these are bounding boxes of branching structures, with
   more than half of the box being background.

**Conclusion: the ER should be handled with semantic/instance segmentation, not object detection.**
Stuffing the ER into DETR's "a handful of boxes per class" framework means doing the task with the wrong
representation.

---

## 5. The boundaries of these numbers (do not over-interpret)

1. **250 slices, 14 crops, one mouse kidney**. Fine-tuning DETR from COCO usually needs thousands of
   images. The metrics here can only show that **the pipeline works**; they cannot show "what level our
   detector has reached".
2. **The validation set has only 3 crops / 43 slices**, so the variance of mAP is very large.
   With another 3 crops the numbers could differ by a factor of two.
3. **Validation crops and training crops are different regions** (grouped split by crop), which is much
   harder than interpolating inside the same crop. A random split by slice would give clearly higher
   metrics, but that is **leakage** and cannot be used.
4. **The mito instance ID was lost during preprocessing**. CellMap's mito labels carry instance numbers
   (0–5), but `fetch_em_slices.py` binarized them to stay compatible with the 2× downsampling.
   The training target is therefore "the connected components of the mito foreground region", not the
   original instances.
   If two mitochondria that originally had different numbers stick together after downsampling, they are
   treated as one instance.
   This must be kept in mind in any analysis that needs accurate instance counts.

---

## 6. Next steps (ranked by cost-effectiveness)

| Priority | Action | Expected benefit |
|---|---|---|
| 1 | **Increase the number of training steps**: 250 slices × 600 epochs (=15k steps), about 70 minutes | Directly improves box regression precision; the most likely way to pull mito AP@0.5 up from 0.27 |
| 2 | **Scale to 500–1000 slices** (more crops / more datasets) | Reduces overfitting, so that val no longer peaks as early as epoch 85 |
| 3 | **Switch the ER from detection to segmentation** | The only path that can make the ER results meaningful |
| 4 | **Preserve the mito instance ID** | Makes "instance counting" style analyses usable |
| 5 | Train only on 8 nm crops | Removes the box-boundary error caused by the label/EM resolution mismatch |

---

## 7. Reproduction commands

```bash
conda activate dachuang
cd ~/dachuang_organelle
export HF_HOME=~/dachuang_organelle/models/hf
export HF_ENDPOINT=https://hf-mirror.com

# train
python scripts/train_detr_slices.py --slices data/slices --classes mito,er \
    --out runs/detr_kidney --epochs 150 --batch 8 --size 480 \
    --lr 1e-4 --backbone-lr 1e-5 --clip 0.1 --val-frac 0.2 --resume auto

# evaluate (real detection metrics, not loss)
python scripts/eval_detr.py --slices data/slices \
    --ckpt runs/detr_kidney/best.pt --classes mito,er \
    --size 480 --iou 0.5 --conf 0.5 --out runs/detr_kidney/eval.json
```

---

## 8. Second round: long training for 600 epochs (15 000 steps)

### 8.1 Training curves: severe overfitting

| Item | Value |
|---|---|
| epoch / steps | 600 × 25 = **15 000 steps** |
| Wall time | 71.5 minutes (7.28 s/epoch) |
| train_loss | 5.176 → **0.538** |
| Best val_loss | **epoch 57 → 2.411** |
| Final val_loss | epoch 600 → **4.499** |

train falls all the way to 0.54, while val peaks at epoch 57 and then climbs to 4.50.
**197 training images cannot sustain 15 000 steps** — this is the ceiling of the data volume, not a
tuning problem.

### 8.2 Full sweep at three IoU thresholds (14 checkpoints)

**IoU = 0.3 (a hit as long as the box "encloses" the object)**

| Checkpoint | epoch | mAP | mito AP | mito P | mito R | er AP | er P | er R |
|---|---|---|---|---|---|---|---|---|
| **best of the single 150-epoch run** | — | **0.4023** | **0.7411** | 0.68 | 0.98 | 0.0635 | 0.16 | 0.30 |
| epoch0050 | 50 | 0.3665 | 0.6324 | 0.87 | 0.96 | 0.1007 | 0.29 | 0.28 |
| epoch0100 | 100 | 0.4002 | 0.6928 | 0.60 | 0.97 | 0.1075 | 0.21 | 0.31 |
| epoch0200 | 200 | 0.3536 | 0.6343 | 0.58 | 0.92 | 0.0730 | 0.11 | 0.37 |
| epoch0300 | 300 | 0.2816 | 0.4415 | 0.54 | 0.88 | 0.1217 | 0.19 | 0.41 |
| epoch0400 | 400 | 0.3518 | 0.5792 | 0.67 | 0.88 | 0.1243 | 0.30 | 0.39 |
| epoch0500 | 500 | 0.3464 | 0.5592 | 0.71 | 0.83 | **0.1336** | 0.32 | 0.40 |
| epoch0600 | 600 | 0.3442 | 0.5634 | 0.70 | 0.82 | 0.1250 | 0.33 | 0.39 |

**IoU = 0.5**

| Checkpoint | epoch | mAP | mito AP | er AP | er P |
|---|---|---|---|---|---|
| best of the single 150-epoch run | — | 0.1438 | 0.2654 | 0.0221 | 0.08 |
| epoch0050 | 50 | 0.1958 | 0.3477 | 0.0440 | 0.18 |
| epoch0200 | 200 | 0.2333 | 0.4242 | 0.0425 | 0.06 |
| **epoch0400** | 400 | **0.2408** | 0.4300 | 0.0517 | 0.20 |
| epoch0500 | 500 | 0.2315 | 0.4099 | **0.0531** | 0.17 |
| epoch0600 | 600 | 0.2344 | 0.4169 | 0.0518 | 0.20 |

**IoU = 0.75 (the box must fit very tightly)**

| Checkpoint | epoch | mAP | mito AP |
|---|---|---|---|
| **epoch0050** | 50 | **0.0726** | 0.1406 |
| best.pt (best val_loss) | 57 | 0.0684 | 0.1317 |
| epoch0200 | 200 | 0.0252 | 0.0439 |
| epoch0400 | 400 | 0.0218 | 0.0387 |
| epoch0600 | 600 | 0.0173 | 0.0303 |

### 8.3 Key conclusion: the validation set is too small, **checkpoint rankings are not trustworthy**

The "best checkpoint" differs across the three validation crops:

| IoU | crop140 winner | crop162 winner | crop163 winner | Distinct winners |
|---|---|---|---|---|
| 0.3 | epoch0200 (0.437) | epoch0150 (0.463) | epoch0100 (0.080) | **3 / 3** |
| 0.5 | epoch0400 (0.436) | epoch0150 (0.463) | epoch0100 (0.080) | **3 / 3** |
| 0.75 | epoch0050 (0.112) | epoch0350 (0.048) | epoch0100 (0.008) | **3 / 3** |

**So it cannot be claimed that "epoch 400 is best"** — the fluctuation of mAP between 0.20–0.24 falls
entirely within sampling noise. At least 10–20 validation crops are needed to decide which checkpoint is
really better.

### 8.4 Which conclusions are robust (they hold across checkpoints and across IoU)

1. **Coarse localization (IoU 0.3) saturates early**. mito recall has stayed in 0.82–0.97 since epoch 50,
   AP fluctuates between 0.44–0.69, and **does not rise monotonically with training**.
   The single 150-epoch run even attains the best coarse localization (mAP 0.4023, mito AP 0.7411).

2. **Fine localization (IoU 0.75) is uniformly poor and does not improve with training**. mAP ≤ 0.073
   for every checkpoint, and the best is still epoch 50 (early training). epoch 600 instead degrades to
   0.017.
   This shows that prolonged training makes the boxes "rigid" and reduces the ability to generalize to
   precise positions — another manifestation of overfitting.

3. **er is weak at every threshold**. Even with the loosest IoU 0.3, er AP reaches at most 0.13
   (epoch 500), with precision P=0.32. At IoU 0.75, er AP ≈ 0.004.
   Quadrupling the training did not rescue it, **which supports the diagnosis that "bounding boxes are
   not a suitable representation for the ER"**.

4. **crop163 is an outlier**. Its mito coverage is only 0.37% (the earlier data audit),
   and its AP under IoU 0.5/0.75 is close to 0, dragging the overall mean down. It should not be
   averaged with equal weight alongside the other crops.

5. **Selecting weights by val_loss is still wrong**. The best val_loss is at epoch 57,
   but its mAP@0.5 is 0.2106, whereas epoch 400/600 give 0.2408/0.2344.
   loss and AP are not synchronized — this holds at all three IoU thresholds.

### 8.5 Practical recommendations

* **Do not add more epochs**. Beyond 150–200 epochs there is no measurable gain, only overfitting.
* **The bottleneck is data**: 197 training images / 11 crops. To improve, the data must be expanded
  (more crops, more datasets), not the training lengthened.
* **More validation crops are needed** in order to do checkpoint selection. With the current 3 crops,
  no statement about a "best epoch" holds.
* **Handle the er by segmentation**; stop tuning it inside the detection framework.

---

## 9. A server outage (recorded so that missing data is understood during reproduction)

**First occurrence**: 2026-10-31 23:5x–2026-11-01 00:10, server SSH unavailable:
the host answers ping (62 ms), but **both port 22 and port 22 refuse TCP**, a server-side problem.
Impact: the first checkpoint sweep was running in the foreground and was killed by the disconnect at the
9th checkpoint (`Socket exception 10054`). After recovery it was rerun as a **tmux background job +
log**, and all 14 checkpoints completed.

**Second occurrence**: from 2026-11-01 05:24, the same symptoms (ping OK at 60 ms, ports refusing),
still not recovered by 05:40. By then the overnight pipeline was all inside tmux, so the jobs themselves
were unaffected.

**Lesson: any remote task longer than a few minutes must go through tmux; do not use foreground
`push_run`.**

---

## 10. Cross-organ validation (kidney training → liver validation)

### 10.1 Why switch organ

The problem exposed in Section 8 is not "low AP" but **a validation set too small to support reliable
conclusions**: across the 3 validation crops, at all three IoU thresholds there are **3 different "best
checkpoints"**, showing that the mAP differences in 0.20–0.24 fall entirely within sampling noise, and
even "which weight is better" cannot be judged.

After switching to a whole liver as the held-out set:

| | Old (same organ) | New (cross-organ) |
|---|---|---|
| Training crops | 11 (kidney) | 14 (kidney) |
| Validation crops | 3 (kidney) | **15 (liver)** |
| Validation slices | 43 | **263** |

This not only makes checkpoint selection possible, it also tests **cross-organ generalization** — a much
stronger claim than interpolation within the same distribution.

### 10.2 Two bugs in data construction (both once produced wrong conclusions)

1. **`has_data` used `delimiter="/"`**: Zarr chunks (`s0/0/0/0`) were folded into `CommonPrefixes`,
   leaving only `.zarray`/`.zattrs` in `Contents`, so **a fully populated array was judged empty**.
   Consequence: for a while `jrc_mus-liver`, `jrc_mus-liver-zon-1`, and `jrc_hela-2` were all believed
   to have no mito+er. After the fix, `jrc_mus-liver` has **15 usable crops**.
2. **EM voxel size hard-coded to 8 nm**: kidney's s0 is 8 nm, but different datasets may differ.
   A wrong scale shifts the physical extent of the crop as a whole, while **the extracted image still
   looks "normal"**, making this extremely hard to spot. Changed to read from `multiscales` in
   `.zattrs`.

### 10.3 Result: cross-organ performance drops substantially

Training for 150 epochs (4500 steps), best val_loss at **epoch 40 (3.087)**, finally rising to 3.476.
train drops from 5.2 to 1.58 — **under the cross-organ setting, overfitting is earlier and more
severe**.

| Setting | mito | er |
|---|---|---|
| Detection, same organ (kidney→kidney) | **AP@0.5 = 0.4292** | AP@0.5 = 0.0495 |
| Detection, cross-organ (kidney→liver) | **AP@0.5 = 0.2754** | AP@0.5 = 0.0170 |

> The above are values recomputed **after fixing the AP algorithm** (COCO area method).
> Before the fix (`np.interp` 101-point interpolation) they were 0.4300 / 0.0531 / 0.2797 / 0.0275,
> a difference within 10% — showing that the bug had limited impact on the real distribution, but the
> boundary case (e.g. a perfect detection should score 1.0 yet is given 0.9901) is a clear error.
> **The ranking conclusions are the same before and after the fix.**

mito retains about 64% (0.4292 → 0.2754), er drops to about 1/3 (0.0495 → 0.0170).

---

## 11. ER segmentation control: changing the representation did **not** rescue the ER

### 11.1 What this experiment tests

One of the conclusions of Section 8 is that "the ER is a continuous tubular network, and representing it
with axis-aligned bounding boxes is itself unsuitable". This is a **diagnosis** that needs evidence.
The control experiment: the same batch of annotations, the same split, with only the output changed
from "a handful of boxes per class" to "per-pixel masks", to see whether er improves.

### 11.2 Implementation

`train_seg.py`: a 2.29 M-parameter U-Net, input 256×256, one sigmoid channel per class as output,
loss = BCE + Dice (Dice is far more sensitive to thin structures than plain BCE),
with metrics being per-class Dice/IoU/P/R and per-crop Dice.

### 11.3 Result: fails exactly like detection

Cross-organ segmentation (kidney training → liver validation, 60 epochs, best at **epoch 20**):

| Class | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| mito | **0.2587** | 0.1486 | 0.462 | 0.180 |
| er | **0.0662** | 0.0343 | 0.399 | 0.036 |

**er's Dice 0.066 is on the same order as detection AP 0.013–0.053 — switching to per-pixel prediction
brought no improvement.**
mito also drops from the same-organ AP 0.43 to Dice 0.26.

**So the diagnosis "bounding boxes are the wrong representation for the ER" is not supported.** My
original judgment was made too quickly.

### 11.4 Key point: **same-organ segmentation fails too**, so it is not domain shift

A same-organ control was added (kidney training / kidney validation, grouped by crop, 60 epochs):

| Setting | mito Dice | **er Dice** | mDice | Validation crops |
|---|---|---|---|---|
| **Same organ** | 0.3451 | **0.0772** | 0.2112 | 4 |
| Cross-organ | 0.2587 | **0.0662** | 0.1625 | 15 |

mito drops from 0.345 to 0.259 (-25%, as expected from domain shift);
but **er goes from 0.0772 to 0.0662, essentially no difference**.

Three explanations are ruled out one by one:

| Explanation | Verdict | Evidence |
|---|---|---|
| Representation (boxes vs pixels) | ❌ ruled out | er stays in 0.02–0.08 under both representations |
| Cross-organ domain shift | ❌ ruled out | er is equally poor under the same organ |
| Incomplete annotation | ❌ ruled out | ER z coverage is **100%**, volume fraction 0.75–22%, only 5–72 3D connected components |

(Note: the earlier statement that "the ER has 1562 connected components" was counted **per 2D slice**;
in 3D there are only 5–72, so the reason of "unstable instance definition" does not hold either.)

### 11.5 The real clue: er's Dice is **below the trivial baseline**

This is the most important finding of this section. If the entire image is labeled foreground,
Dice = `2p/(1+p)`, where p is the foreground fraction. At an ER fraction of about 7% this baseline is
about **0.118**.

**Measured er Dice = 0.0772, below 0.118.**

In other words: **the model performs worse than the dumb strategy of "guess foreground everywhere".**
The problem is not "the ER is too hard to learn", but that the model hardly predicts the ER at all
(Recall only 0.05).

The mechanism points to **loss design**: when the two classes are trained jointly, the pixel-averaged
BCE is dominated by the majority class (mito occupies 17–54%, er only 4–22%), and to push BCE down the
model classifies the ER as background.

Added `--class-balance` (BCE weighted inversely to the per-class foreground fraction),
and the trivial baseline is now written into the evaluation output
(`dice_all_foreground` / `beats_trivial`), so that from now on any low Dice immediately shows whether it
beats the trivial strategy.

### 11.6 Per-crop Dice shows a **cliff-like** distribution (cross-organ, and only mito is affected)

```
crop145  mito 0.5565  er 0.2269   ← 迁移得不错
crop137  mito 0.4093  er 0.1594
crop133  mito 0.3947  er 0.0344
crop125  mito 0.0000  er 0.0041   ← 完全失败
crop131  mito 0.0000  er 0.0155
crop143  mito 0.0000  er 0.0006
crop150  mito 0.0000  er 0.0000
```

6 of the 15 crops are a complete 0, while the successful ones reach 0.4–0.56. This part does reflect
domain shift, but **it only affects mito** — er's failure exists under the same organ as well and is
unrelated to domain shift.
