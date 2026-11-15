# nnU-Net 2d vs our ResNet50-UNet: under a like-for-like comparison the conclusion flipped twice

**English** ｜ [中文](NNUNET-RESULTS.zh-CN.md)

> The first pass used nnU-Net's own `foreground_mean` and gave 0.2096, which looked far below our
> 0.3643. Recomputing with the same `dice()` gave 0.5162, which looked far above us instead.
> **Both are wrong.** Split by whether the ground truth is empty, the real answer is:
> **on tiles that actually contain Golgi, our model is better.**

---

## 1. Three numbers, three conclusions

| Metric | nnU-Net | Ours | Apparent conclusion |
|---|---|---|---|
| nnU-Net's own `foreground_mean` | 0.2096 | 0.3643 | nnU-Net is far behind |
| Same `dice()`, per-case mean | **0.5162** | 0.4036 | nnU-Net is much better |
| **Split by whether the ground truth is empty** | see below | see below | **we are better** |

**Same 281 tiles, same models, only the evaluation metric changed, and the conclusion went from
"far behind" to "much better" and then to the reverse.**
This is the same problem this project keeps running into: **on this dataset, the metric decides
the conclusion more than the model does.**

## 2. Why the first two are both wrong

Of the 281 validation tiles, **129 (46%) have empty ground truth** — those tiles contain no Golgi
to begin with.

On an empty tile, "predict all background" is a perfect score (Dice = 1).
**A model that never dares to predict foreground therefore gains a large advantage in the overall
mean.**

And nnU-Net's `foreground_mean` uses yet another treatment: it handles empty cases differently
from this project's `dice()`, so the same predictions yield 0.2096.

Both numbers are dominated by "does it call the tile empty", not by "how well does it segment".

## 3. Split by whether the ground truth is empty (the real comparison)

| Split | n | nnU-Net 1000 epochs | nnU-Net 302 epochs | ResNet50-UNet |
|---|---|---|---|---|
| **Ground truth empty** | 129 | **0.9612** | 0.8450 | 0.5504 |
| 　of which correctly called empty | — | **124 / 129** | 109 / 129 | 71 / 129 |
| **Ground truth non-empty** | 152 | 0.2174 | 0.2372 | **0.2791** |
| 　of which Dice = 0 | — | 96 / 152 | 90 / 152 | **66 / 152** |
| 　coverage > 2% | 129 | 0.2499 | 0.2777 | **0.3285** |
| 　coverage > 8% (sufficient annotation) | 97 | 0.2891 | 0.3341 | **0.4051** |
| **Pooled Dice (positive samples only)** | 152 | 0.3912 | 0.4263 | **0.4760** |

### 3.1 Training to 1000 epochs makes segmentation worse

The continuation ran from epoch 302 to the full 1000 (rc=0, not a timeout), and nnU-Net's own
validation Dice went from **0.2096 to 0.2105** — 394 more epochs bought 0.0009, so it had long
since reached a plateau.

But split the same way, the changes go in opposite directions:

* **Empty/non-empty discrimination improves a great deal**: empty tiles 0.8450 → **0.9612**,
  correctly called empty 109 → **124 / 129**
* **Segmentation gets worse**: positive samples 0.2372 → **0.2174**, coverage >8% 0.3341 →
  **0.2891**

That is, **the longer it trains, the more it leans towards "there is no Golgi here".**
This is consistent with its loss setup: Dice+CE with `batch_dice=True`, and 46% of the tiles are
empty, so the gradient is dominated by large numbers of empty patches and the decision boundary
is pushed steadily towards the conservative.

The phenomenon is worth recording in itself: **on this dataset, more thorough training does not
mean better segmentation**.

**Conclusions:**

1. **nnU-Net is better at judging "is there Golgi in this image"** — empty tiles 0.8450 against
   0.5504; of 129 empty tiles it correctly calls 109 empty, where we get only 71.
   It also leans more towards an empty prediction on positive samples (90/152 called empty,
   against our 66/152).
2. **Our model is better at drawing the shape** — on every stratum where the ground truth is
   non-empty we are better: 0.2791 against 0.2372, 0.3285 against 0.2777, 0.4051 against 0.3341.
   Pooled Dice 0.4760 against 0.4263.

In other words: **nnU-Net is a conservative detector; we are a more accurate segmenter.**

## 4. Why this happens

Both reasons point to the nature of this dataset:

**First, 46% of the tiles are negative samples.** The data construction kept tiles without Golgi
as negatives (which is right, otherwise the model would over-predict). But it also makes "calling
a tile empty" an ability with a very large weight.

**Second, annotation coverage in the positive samples is extremely low.** Only 97 of the positive
samples have coverage >8%, and the correlation between coverage and Dice is **+0.730** — the
sparser the annotation, the harder the task for any model.

nnU-Net's loss is Dice + CE with `batch_dice=True`. When large numbers of empty patches take part
in a batch, its decision boundary is pushed towards "more conservative", so it calls more tiles
empty and draws fewer masks.

## 5. What this means

### 5.1 Stop saying "nnU-Net is the stronger baseline"

**On this dataset**, it is worse than our existing ResNet50-UNet at the actual segmentation task.
Saying "nnU-Net is the standard baseline so it ought to be better" applies a general conclusion to
a specific setting where it does not hold.

### 5.2 But its "empty/non-empty ability" is real, and useful

109/129 against 71/129 is a large gap. If the pipeline were changed to "decide presence first,
then segment", nnU-Net's discrimination part is worth borrowing.

### 5.3 What the paper should say

The existing paper's conclusion that "changing the encoder gives +0.05" is **unaffected** — that
was our own three-seed comparison. But §3.3 of the paper could add a sentence: external standard
baselines did not beat the self-trained model on this data, because this dataset is 46% empty
tiles with sparsely annotated positive samples.

## 6. Still not done

1. **nnU-Net has been run to the full 1000 epochs** (302 epochs in the first round, then `--c`
   continuation to 999). Results for both epoch counts are in the table above, and the conclusion
   points the same way.
2. **Only fold 0 was run**, not 5 folds.
3. **Our model used seed 0 only** (`runs/cem_B/best.pt`), not a three-seed mean.

## 7. Reproduction

```bash
# nnU-Net: convert -> preprocess -> train -> validate
bash tools/server/scripts/launch_nnunet.sh
bash tools/server/scripts/launch_nnunet_final.sh

# like-for-like per-case comparison (runs our model inference)
python scripts/compare_nnunet_vs_ours.py --limit 0

# split by whether ground truth is empty
python scripts/split_compare.py

# rule out transpose/flip misalignment (this is what excluded it)
python scripts/check_orientation.py --n 4
```

## 8. Five traps when installing nnU-Net (all of them fail silently)

1. The `pip` on PATH points at a **broken Python 2.7 script from mgltools** (rc=126) → use
   `python -m pip`
2. `connected-components-3d` needs C++17 and the system g++ 6.3.0 cannot compile it → install the
   prebuilt wheel first
3. **`pip install nnunetv2` replaces the cu118 torch with a CUDA 12 build**, while the driver only
   goes up to 11.1 → force the rollback with `torch==2.6.0+cu118` (with the suffix)
4. **The `nnunet` environment has no libstdc++ by default**, and `ndindex` needs GLIBCXX_3.4.21 →
   install `libstdcxx-ng` and export `LD_LIBRARY_PATH`
5. **The Triton backend of torch.compile fails to compile**, and the error hides inside a
   dataloader thread, so from outside you only see "background workers are no longer alive" →
   `export nnUNet_compile=0` (this switch is not in `--help`, you have to read the
   `_do_i_compile()` source)
