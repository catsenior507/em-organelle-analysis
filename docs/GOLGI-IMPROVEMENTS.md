# Golgi: a record of three improvement steps

**English** ｜ [中文](GOLGI-IMPROVEMENTS.zh-CN.md)

> Goal: take the Golgi from "51 in-house figure panels, Dice 0.48" to something more usable.
> The three steps are: ① handle the text artifacts ② bring in the public ASEM data ③ choose a route that can be applied at scale.
>
> **This document records the actual result of each step, including the failures and the decisions to abandon a direction.**

---

## 1. Text artifacts: investigated, quantified, and deliberately not masked

### 1.1 The problem

These original Golgi images are **figure panels from published papers**, and they have black annotation text burned into them
("Golgi Cisternae", "Mitochondria") plus arrows. The text sits on top of real tissue, so the GT
there is not trustworthy.

### 1.2 Five detection attempts, all of them failed

| Criterion | Result | Why it failed |
|---|---|---|
| Gray value < threshold | At a threshold of 20 the foreground is 1.15%, but the median connected component is only **2 pixels** | JPEG breaks the strokes into scattered dots that never connect |
| Black-hat transform (thin dark structures) | **masks 44%** | EM is full of thin dark **membranes**, and text cannot be separated from them along the "thin and dark" dimension |
| Dark + **low** local variance | only 0.04–0.5% | the sign is reversed: **the inside of a mitochondrion is dark and flat**, so it gets picked out instead |
| Dark + **high** local variance + black hat | **47–58%** | membranes satisfy this too |
| Dark + small component + **spatial clustering** | 20% | dark cytoplasm texture clusters too |
| The above with tighter parameters (amin=20, radius=30, nbr=6) | 2.3% | visual inspection confirmed it: **it marks dark tissue blobs, not the text** |

**Conclusion: image statistics alone cannot separate "text that was drawn on" from "membrane that was photographed".**
The root cause is that the text in these images is **dark lettering on a dark background**, while TEM tissue is itself full of dark structures.

### 1.3 A more decisive move: quantify how much contamination there is

Instead of guessing whether to mask, measure how much GT the text actually contaminates:

| Metric | Mean | Median |
|---|---|---|
| Share of dark pixels inside the GT region | 6.1% | **1.8%** |
| **Share of dark pixels that belong to GT** | 8.4% | **2.6%** |

The second row is the key one: **only about 2.6% of the dark pixels (text included) fall on the Golgi**,
and all the rest are in the background. But GT in the background is 0 to begin with, so the text being learned as background
**agrees with the ground truth, and therefore is not contamination**.

On top of that, Golgi cisternal stacks are themselves dark in TEM, so part of that 1.8% is real Golgi.

### 1.4 The decision

**No automatic text masking.** Reasoning:
1. All the usable detectors are unreliable (all five attempts failed), and the cost of a wrong mask (deleting real tissue)
   far exceeds the benefit of a correct one (improving at most about 2% of the area)
2. The upper bound on contamination has been quantified (about 2%), and that is an acceptable level of annotation noise
3. Keeping this conclusion has value in itself — it stops anyone from retrying the same set of criteria later

> The related scripts are kept in `tools/analyze_text_artifacts.py` (threshold calibration),
> `tools/detect_text_artifacts.py` (the five criteria and their parameters),
> `tools/quantify_text_contamination.py` (contamination quantification).
> Their value lies in **recording what does not work**.

### 1.5 But the mechanism itself is built and verified

Even though this dataset cannot use it, **the "ignore region" mechanism is implemented and its unit tests pass** —
because the problem it solves is far bigger than text artifacts: **incomplete annotation** (see Section 10).

The implementation (`train_seg.py`):

| Stage | What it does |
|---|---|
| Data | When `SegDataset` finds `ignore.png` in a slice directory it additionally returns a `(1,H,W)` ignore channel; **without that file the behaviour is exactly as before** (still a 3-tuple) |
| Loss | `seg_loss(..., ignore=)`: BCE is weighted by `1-ignore` and then **divided by the number of kept pixels** (which preserves the scale); for Dice, numerator and denominator are both multiplied by `1-ignore` |
| Degenerate case | When a whole batch is ignored it returns "a zero value that carries the graph", the gradient is exactly 0, and `does not require grad` is not triggered |
| Augmentation | On flip/rotation the ignore map is transformed in sync with the image and the GT |
| Unified layer | If `unified.py` finds `<id>_ignore.png` or `ignore/<id>.png` in the annotation directory, it carries it out together with the slice |

Verification (`src/test_ignore_mask.py`, all 12 assertions pass):

```
[2] 被忽略像素的梯度必须精确为 0
  PASS  忽略区内梯度全为 0        [max|g|=0.00e+00]
  PASS  忽略区外梯度不为 0        [max|g|=8.14e-05]
  PASS  忽略后整体梯度与不忽略不同  [max|Δg|=1.29e-04]
  PASS  忽略全图时梯度全为 0       [max|g|=0.00e+00]
[3] ignore=None 时必须与旧行为完全一致
  PASS  ignore=None 与不传该参数结果相同  [0.679907 vs 0.679907]
```

**Why verify with gradients rather than loss values**: soft Dice is degenerate on an "all-negative" region
(`num=0`, `den≈p.sum()`, Dice is identically 0), so even with a full ignore
the loss value does not drop to 0 — but that is a **constant offset that does not change the gradient**.
My first version asserted "the loss should be close to 0" and two items failed; switching to gradients gave an exact test.

**What is still missing before it can be used**: a **trustworthy source of "unannotated regions"**.
That is the same problem as annotation itself — you cannot use the model's own predictions to define what it should ignore
(that would freeze the model's current errors in place). A workable approach is for a human to mark out metadata such as "I only annotated the left half of this image",
expressed as `ignore.png`. **The mechanism is ready; it is waiting for annotation metadata.**

---

## 2. Public ASEM data: downloaded successfully, filling in ground truth for several classes

### 2.1 Source and licence

ASEM (the Kirchhausen lab at Harvard, Gallusser et al., JCB 2023),
**CC BY-SA 4.0**, public S3 bucket, no account needed. See `docs/GOLGI-DATA-SOURCES.md` for details.

### 2.2 What was downloaded

| Dataset | Volume | Shape (z,y,x) | Foreground share | Download size |
|---|---|---|---|---|
| 61 (BSC-1 cells, **chemical fixation, CF**) | golgi | 210×283×371 | 6.61% | 1.1 MB labels + 52.8 MB raw |
| 64 | golgi | 284×204×424 | 9.31% | 1.2 MB + 39.5 MB |
| 64 | er | 241×476×528 | 20.03% | 242 MB |
| 64 | mito | 241×476×528 | 9.16% | 232 MB |

> The label volumes are extremely sparse and compress down to the 1 MB order; the raw data is what dominates the size.

### 2.3 Alignment: a problem that has to be handled

A label volume is usually only a **sub-region** of the raw volume, and different label volumes have different ROIs.
The alignment offset comes from subtracting the `.zattrs` `offset` values (nm) and dividing by `resolution`:

| Dataset / volume | raw index offset |
|---|---|
| 61 / golgi | [1000, 410, 2597] |
| 64 / golgi | [777, 291, 954] |
| 64 / er, mito | [400, 326, 2212] (both share the same ROI) |

**Pitfall we hit**: at first `raw.npy` was named only by dataset (`raw.npy`),
but ds64's golgi and its er/mito sit in two different ROIs,
so the raw generated by the golgi run was reused for er/mito,
and the shapes did not match ((284,204,424) vs (241,476,528)), which put everything out of alignment.
Now raw is named after the "first label volume" (`raw__golgi.npy`, `raw__er.npy`).

### 2.4 Conversion to 2D training slices

| Source | Slices | Tiles |
|---|---|---|
| asem61 / golgi | 165 | **660** |
| asem64 / golgi | 196 | **392** |
| asem64 / er | 241 | 2169 |
| asem64 / mito | 213 | 1916 |

**ASEM provides 1052 golgi training tiles in total**, comparable to our own 976.

**Pitfall we hit**: ds64's golgi ROI is 284×**204**×424, so the cross-sections cut along z are 204×424,
and the short side 204 < tile 256. The original implementation's `if g.shape != (tile, tile): continue`
meant that **not a single tile came out of the whole volume** (positive and negative were both 0). After switching to padding we got 392.

### 2.5 A bonus capability: multi-class interaction data

ds64's **er and mito share the same ROI** (241×476×528),
which is the first time the project has had **co-registered, alignable multi-class annotations** —

* Before this, our own data had `organelle_type` "M" throughout, so only same-class adjacency networks were possible
* Now real **mito–ER contact** statistics are possible (one of the core goals of the project)

---

## 3. A route that can be applied at scale

### 3.1 Why not use the CF-specific pretrained model

ASEM published only a `golgi_CF` (chemical fixation) model, and it needs the `incasem` package
(not on PyPI, installable only from GitHub, unreachable from both the local machine and the server).
More importantly: **a specialised model is tied to one preparation protocol**, and it can fail as soon as the samples change, so it does not scale.

### 3.2 The chosen route

**Train our own 2D model + merge data from multiple sources**, for these reasons:
* A 2D model is structurally the same as our existing pipeline, so all the tooling is reusable
* It does not depend on a particular preparation protocol, and the more diverse the data sources the more robust it is
* Adding data only means adding one training root; no new code is needed

### 3.3 Implementation: `--train-roots`

`train_seg.py` gained `--train-roots` (multiple root directories, comma separated),
which merges slices from different sources into one training set.

**A key detail**: different sources can have **crops with the same name** (both datasets have `crop129`).
Without distinguishing them, grouping the split by crop treats same-named crops from different sources as the same one,
which causes **cross-source leakage**. The implementation tags every source with `source_tag`,
so the grouping key becomes `tag::crop`.

### 3.4 Results: it holds up engineering-wise, not statistically

After running it, the wording in 3.2 has to be corrected. **"The more diverse the data sources, the more robust" is only half true.**

| Level | Does it hold? | Evidence |
|---|---|---|
| **Engineering**: adding a data source is cheap | **Yes** | just one more `--train-roots` entry, no new code |
| **Engineering**: one model serves several domains | **Yes** | the merged model gets 0.2679 on our own held-out set and 0.80/0.85 on the ASEM training sets, both of which are fine |
| **Statistical**: adding external data makes this domain more accurate | **No** | in the 5-fold paired validation best is −0.014 and last5 is +0.009, opposite signs (see 5.7) |
| **Statistical**: cross-domain transfer | **No** | cross-domain Dice drops to 1/3–1/8 (see 11) |

So the route should be described like this:
**it makes "supporting more data sources" cheap, but it does not guarantee that "more data sources make the current task more accurate".**
The latter has been refuted by cross-validation.

---

## 4. Appendix: real defects encountered along the way

| # | Defect | Symptom | Fix |
|---|---|---|---|
| 1 | `raw.npy` named only by dataset | ds64's er/mito and golgi use different ROIs, so they were misaligned | name by label volume: `raw__<label>.npy` |
| 2 | skip outright when the section is smaller than a tile | the whole ds64 golgi volume gave 0 tiles | edge padding |
| 3 | `find` does not follow symlinks | the split directory was judged empty and training aborted | `find -L` |
| 4 | the split directory had too few levels | `glob(root/*/*/z*)` matched nothing and all 38 in-house images were dropped | add a level for the dataset name |
| 5 | editing files with Python inside PowerShell | newlines were translated to CRLF and bash reported a `\r` syntax error | use the `write` tool, or `newline='\n'` |
| 6 | waiting on `pgrep -f "train_seg.py..."` | it **matched the waiting loop itself** (the pattern is on its own command line) and waited forever | wait on a log/artefact file instead |
| 7 | relative paths in `upload.py` | it uploaded to the home directory `~/scripts/`, while the script looked in `~/dachuang_organelle/scripts/` → **the session died on start-up with an empty log** | anchor relative paths to the project root |
| 8 | nested `python -c "..."` inside tmux | the layers of quoting were mangled and the session exited silently | use plain-shell `grep -c` |
| 9 | SFTP does not expand `~` | `mkdir` (through the shell) succeeded but `sftp.put` raised FileNotFoundError | substitute the absolute home directory explicitly |

**The common lesson**: all of these look like logic bugs but are really environment/quoting/path problems,
and **they fail quietly** (session exits, empty log, return 0),
which makes them harder to spot than a real computation error. So for any long-running task
you must first confirm **the session is actually alive** (`tmux ls` + a log with content in it), and only then wait for the result.

---

## 5. Experimental results and conclusions

### 5.1 A methodological correction: first, which numbers are not comparable

| Experiment | Validation set | Training data | Dice |
|---|---|---|---|
| First run (earlier) | 317 tiles, `--val-frac 0.2` | 51 in-house images | **0.4849** |
| Merged training | 281 tiles, hold out 1 in every 5 images | 41 in-house images + ASEM 1052 | **0.3879** |

**These two rows cannot be subtracted directly.** The validation slices differ and so does the number of training images (51 vs 41).
Putting them side by side and saying "bringing in ASEM dropped Dice by 0.10" is **wrong**.

So a **control arm on the same split** was added (in-house only, 41 images, 281 validation tiles),
and the run was repeated as **3 seeds × 2 arms**, judged by mean ± standard deviation,
because a single training run's validation curve swings enormously (measured on adjacent epochs: 0.359 → 0.195 → 0.353).

> The actual numbers are in `logs/GOLGI_MULTISEED.txt` (generated once training finished).

### 5.2 The same-split control arm (single run)

| Arm | Training tiles | Dice | P | R |
|---|---|---|---|---|
| Control: in-house only | 1515 | **0.3605** | 0.295 | 0.462 |
| Experiment: in-house + ASEM | 2567 | **0.3879** | 0.348 | 0.438 |
| **Difference** | +1052 | **+0.0274** | | |

On this single run, bringing in ASEM is a **positive gain** (+0.027), and precision clearly improves (0.295 → 0.348)
while recall drops slightly — which fits the expectation that "external data supplies more varied positive shapes".

But a single run's difference still does not settle the question, because:

### 5.3 A worse problem: split variance far exceeds seed variance

| Experiment | Training images | Validation tiles | Dice |
|---|---|---|---|
| In-house only, split B (random 20%) | 41 | 317 | **0.4849** |
| In-house only, split A (hold out 1 in 5) | 38 | 281 | **0.3605** |

**Same data, same model, same hyper-parameters — changing only the split makes a difference of 0.12.**
And there are only 48 images in total — which means **any single-point Dice is unreliable**.

### 5.4 The metric convention: `last5` is 6× more stable than `best`

Repeating over 3 seeds exposed another problem:

| Statistic | Values for the 3 seeds | Mean ± std |
|---|---|---|
| `best` (best epoch) | 0.3623 / 0.3457 / 0.3506 | 0.3529 ± **0.0085** |
| `last5` (mean of the last 5 epochs) | 0.3239 / 0.3213 / 0.3088 | 0.3180 ± **0.0080** |

The standard deviations alone do not differ much, but `best` **picks the maximum over 60 epochs**,
which carries its own selection bias (it is 60 comparisons treated as 1).
It jumps between seeds with less regularity, and it is more optimistic.

So **the report uses `last5` as its primary metric**, and `best` only for reference.

> Note: `last5` is systematically about 0.03–0.05 lower than `best`.
> That is not "worse performance", it is a different statistic. The two must not be compared against each other.

### 5.5 Multi-seed results: the difference is the same size as seed noise, **no improvement can be claimed**

Full results (3 seeds per arm, identical validation set):

| Arm | best-of-epochs | last5 |
|---|---|---|
| In-house only | 0.3529 ± 0.0085 | **0.3180 ± 0.0080** |
| In-house + ASEM | 0.3686 ± 0.0203 | **0.3432 ± 0.0232** |
| **Difference** | **+0.0157** | **+0.0252** |

The individual last5 values for the three seeds:

* In-house only: 0.3239 / 0.3213 / 0.3088 — very consistent (std 0.008)
* In-house + ASEM: **0.3699 / 0.3322 / 0.3275** — large spread (std 0.023)

> **One transcription error that has been corrected**: this section originally said `0.3389`, which was seed2's **intermediate value
> at 51/60 epochs**, not the final result (the final value is 0.3275).
> It was found by the `tools/verify_golgi_numbers.py` audit — it compares every `0.xxxx`
> in the document against the logs one by one, and `0.3389` appears in no artefact at all.
> This is the same class of error as the early `tools/cv_partial.py` folding unfinished folds into the paired difference:
> **treating an in-progress number as a result**.

**Criterion**: the difference (+0.025) is the same order as the merged arm's seed standard deviation (0.023), so
**one cannot claim that bringing in ASEM helps, nor that it harms**.

This conclusion was bought with multiple seeds. Looking only at the first seed gives
"merged arm 0.3699 against in-house 0.3180, an improvement of +0.052" — a **beautiful but wrong** conclusion.
The second seed immediately pushes it down to 0.33.

**This may be the single most important methodological lesson**: on this dataset of only 48 images,
an experiment with `n=1` is not enough to support any conclusion.

### 5.6 The remaining possibilities

The difference is positive but not significant, and there are three explanations:

1. A real effect exists but is very small and needs more data to detect
2. There is no effect and the observed +0.025 is noise
3. A positive effect exists but domain differences cancel part of it (ASEM is BSC-1 cells, chemical fixation,
   FIB-SEM; ours are TEM figure panels)

Telling these apart needs **more seeds** (lower standard error) or **5-fold cross-validation**
(which also lowers split variance and shows whether the direction is consistent fold by fold). Both were done; see 5.7.
**The result is that explanation 2 stands up best**: the stricter design pushes the effect down to +0.009 or flips it negative.

### 5.7 5-fold cross-validation: the conclusion **depends on the statistic**, so no effect can be claimed

Split by image into 5 folds, take turns using each fold as the validation set, run both arms 5 times each:

| Arm | best (mean ± std) | last5 (mean ± std) |
|---|---|---|
| In-house only | 0.4242 ± 0.0506 | 0.3127 ± 0.0865 |
| In-house + ASEM | 0.4103 ± 0.0238 | 0.3219 ± 0.0748 |

Per-fold paired difference (subtract the two arms within the same fold, which removes the difficulty differences between folds):

| Statistic | Mean paired difference | Folds with positive value | Direction implied |
|---|---|---|---|
| **best** | **−0.0139** ± 0.0334 | 3/5 | ASEM is **slightly worse** |
| **last5** | **+0.0092** ± 0.0161 | 4/5 | ASEM is **slightly better** |

**The two statistics give opposite signs.**

That is not a detail; it is the crux of the whole ablation: if you look at the results first and then pick the statistic,
reporting last5 when you want "it works" and best when you want "it does not" — **that is p-hacking**.

The honest conclusion is: **at this data volume, an effect of ASEM has not been demonstrated.**
The optimistic estimates (+0.027 / +0.025) shrink to +0.009, or flip to −0.014, under the stricter design.

### 5.8 A genuine finding by the way: adding data reduced the variance

Look at the standard deviations:

| Arm | best std | last5 std |
|---|---|---|
| In-house only | 0.0506 | 0.0865 |
| In-house + ASEM | **0.0238** | **0.0748** |

**The merged arm's fold-to-fold spread is about half that of the in-house arm** (best: 0.0238 against 0.0506).

In other words, adding ASEM **did not raise the mean, but it made the results far more stable**.
That agrees with the intuition that "data diversity improves robustness", and it is the only benefit of this route that the data supports.

> Note: 5 folds is a very small sample, so "smaller variance" could be a coincidence and cannot be treated as a conclusion.

### 5.9 Criteria (after tightening)

* A single-point difference **is not evidence** — split variance can reach 0.12
* An unpaired difference of seed means is not sensitive enough (+0.025 drowns in a 0.023 spread)
* **A paired fold comparison** is the most sensitive, but the conclusion must be robust to the statistic; here it is not robust -> no conclusion
* Any report must **give both best and last5**, otherwise it amounts to picking the favourable metric

> Data: `logs/GOLGI_MULTISEED.txt`, `logs/GOLGI_CV.txt`
> Summary: `logs/GOLGI_STATUS.md` (generated from the artefacts by `tools/golgi_report.py`)

---

## 6. Configuration improvement: an "invisible no-op"

To improve the Golgi Dice, the first thing tried was `--class-balance` (weight BCE inversely by the foreground share),
for a straightforward reason: the Golgi occupies only part of a tile, so BCE is dominated by the background.

After the run the log said:

```
类别前景比例 [0.1763] -> BCE 权重 [1.0]
```

**The weight is 1.0, which means no weighting at all.** The reason is in the implementation:

```python
inv = 1.0 / np.clip(fg, 1e-3, None)   # 单类：inv = 1/0.1763 = 5.67
w = inv / inv.mean()                   # 5.67 / 5.67 = 1.0
```

Normalising to a mean of 1 is meant to "keep the overall loss scale unchanged", but it also makes
**the weight identically 1 in single-class training** — this option only does anything with **relative weighting across several classes**.

This is a "looks like it works, actually does nothing" trap: the option has a sensible name and the log prints it,
and only by noticing that `[1.0]` can you tell that it did nothing at all.

### 6.1 Emphasising the foreground in a single class needs another mechanism

| Mechanism | Description | Status |
|---|---|---|
| BCE's `pos_weight` | weights only the positive class, no normalisation | **not implemented**, worth adding |
| Foreground over-sampling | raise the probability of tiles containing foreground when sampling | not implemented |
| Raising the Dice weight | Dice optimises overlap directly and naturally ignores the background | **already supported**: `--dice-weight` |

So the only knobs actually available are the learning rate, the epoch count and the Dice weight. The two configurations that were actually run (results in 6.2):

* **v2**: lr 1e-4 + 100 epochs (lower learning rate, longer training)
* **v3**: lr 3e-4 + `--dice-weight 0.7` + 100 epochs (more emphasis on overlap)

> Results are in `logs/GOLGI_CONFIG_SWEEP.txt`. If neither shows a clear improvement,
> the bottleneck is not the optimisation configuration but the data volume (48 images) itself.

### 6.2 Results: neither direction helped

| Configuration | epochs | best (@ep) | last5 |
|---|---|---|---|
| baseline: lr 3e-4 + 60 epochs | 60 | 0.3879 (@27) | **0.3614** |
| v2: lr 1e-4 + 100 epochs | 100 | 0.3836 (@77) | 0.3516 |
| v3: Dice weight 0.7 + 100 epochs | 100 | 0.3801 (@61) | **0.3651** |

Taking `last5` as the measure:

* **v2 (lower learning rate) is actually worse** −0.010
* **v3 (higher Dice weight) is roughly flat** +0.004, entirely within seed noise (0.02)

**Conclusion: the optimisation configuration is not the bottleneck.** All three configurations sit in the range 0.35–0.37 on `last5`,
which says that on this dataset the model's convergence level is limited by something else.

This also rules out a common misjudgement: seeing a jittery curve and concluding that "tuning the hyper-parameters will improve things a lot".
In reality the jitter stays jitter, and **the expected value was not raised by tuning**.

> Side note: v2 pushes the best epoch from baseline's 27 out to 77,
> which shows that lowering the learning rate really does make convergence slower and later — so `best` is slightly higher,
> but `last5` did not improve. This is another example of **why you cannot look only at best**.


---

## 7. Scale or annotation quality? — an analysis that nearly went wrong

The images vary enormously in pixel size (from 885×1273 to 6000×6192), so "inconsistent scale" is a natural suspect for the main obstacle.
So the **equivalent diameter** of Golgi connected components was used as a relative scale indicator (biologically a cisternal stack is about 1–2 µm,
so the diameter should be roughly stable).

At first glance the result is alarming:

| Statistic | Value |
|---|---|
| Median equivalent diameter | 264.8 px |
| **Interquartile range (middle 50%)** | **190 – 346 px, only 1.8×** |
| Full range | 12.7 – 1354 px, **107×** |

Looking only at "max/min = 107×", the conclusion would be "the scale is severely inconsistent and must be normalised first".
**But that is wrong.** The middle 50% differ by only 1.8×, which says **the main body of the scale is in fact quite consistent**;
the 107× comes entirely from the extremes:

* **12.7 px** (about 127 pixels of area) — an annotation fragment, very probably not a complete Golgi
* **1354 px** (about 1.44 million pixels) — several adjacent Golgi merged into **one blob** after hole filling,
  or a large region of the whole image enclosed by a single contour

In other words, these two extremes reflect **annotation quality problems** (fragments, merged blobs),
not differences in imaging scale.

### 7.1 The corrected conclusion

| Suspect | Verdict | Evidence |
|---|---|---|
| Inconsistent imaging scale | **not the main obstacle** | the middle 50% differ by only 1.8× |
| Annotation fragments (regions that are too small) | **is a problem** | fragments on the order of 12.7 px exist |
| Annotation blobs (neighbours merged) | **is a problem** | merged blobs on the order of 1354 px exist |

**What is worth doing, therefore, is cleaning the annotations** (raise the minimum region area, and split merged blobs morphologically where necessary),
not resampling everything to a uniform scale.

### 7.2 A methodological lesson

A ratio of extremes like "max/min" is extremely sensitive to a handful of outliers, and drawing conclusions from it easily misleads.
You should look at the **interquartile range** or the quantile distribution — in this case the two measures give opposite verdicts
("differs by 107×" against "differs by only 1.8×").

> Script: `tools/golgi_scale_analysis.py`, output `logs/golgi_scale.json`.

---

## 8. Cleaning the annotations: done, then found to have no effect, so it was not run

Following the conclusion of Section 7, the annotations were cleaned (`tools/clean_golgi_labels.py`):
connected components with area < 500 px were removed and the rest kept.

Cleaning results:

| Metric | Before | After |
|---|---|---|
| Total connected components | 300 | **101** (199 fragments removed) |
| Total area | 11 148 317 px | 11 145 711 px (**−0.02%**) |
| Images left empty after cleaning | — | 5 (they held only fragments) |

**199 of the 300 connected components are fragments, but together they cover only 0.02% of the area.**

The comparison after rebuilding the tiles accordingly (`ourdata_golgi_clean`):

| Version | Positive tiles | Negative tiles | Images |
|---|---|---|---|
| Original | 976 | 820 | 51 |
| Cleaned | **975** | **819** | 48 |

**The tile count differs by exactly 1.** The reason is clear: whether a tile is positive depends on the GT share (≥1%),
and the fragments are far too small either to turn a negative tile positive or to change which tile a positive belongs to.
Inside a positive tile, the fragments contribute only a tiny number of noise pixels.

**So this experiment cannot produce a measurable difference, and it was not run.** This is 25 minutes of GPU time
saved by "analysis first" — work out the size of the effect before deciding whether to train.

> The dataset `ourdata_golgi_clean` is kept in the manifest,
> so it can be used directly if a more aggressive cleaning is adopted later (splitting merged blobs, redrawing boundaries).

---

## 9. The remaining hypothesis: the target itself is ambiguous

Every configuration stalls at `last5 ≈ 0.32–0.34` (v2's lower learning rate and v3's higher Dice weight made no material difference).
The data volume is only 48 images, but the more likely limit is **the definition of the target itself**:

The green contour marks out "the region where the Golgi is", not a solid body bounded by an outer membrane the way a mitochondrion is.
A hand-drawn region **has no exact counterpart in the image** — where the boundary falls depends on the person drawing it,
not on some objective edge in the image. That implies **an irreducible error floor**.

This hypothesis can be judged with a cheap test: **how much does Dice rise once the boundary tolerance is relaxed**.
If Dice improves substantially once a few pixels of boundary deviation are tolerated, the error comes mainly from the boundary definition,
not from the model failing to learn what a Golgi looks like.

> The test works like this: morphologically dilate/erode the GT and the prediction (tolerance 1–5 px),
> and compute the Dice of "counted correct within tolerance". If at 3 px tolerance Dice rises from 0.33 to 0.6+,
> the report should switch to "tolerant Dice" instead of hard Dice, with the reason explained in the document.

---

## 10. Conclusion: the real cause of the low scores is **incomplete annotation**, not the model

### 10.1 The tolerance test: boundary fuzziness is not the main cause

Whole-image inference was run on all 53 images (`runs/golgi_v3/best.pt`), and the pixel-level Dice on the 11 held-out images is **0.3023**.
Then Dice was computed under boundary tolerance:

| Tolerance | Dice | Relative to hard Dice |
|---|---|---|
| 0 px | 0.3023 | — |
| 1 px | 0.3102 | +0.008 |
| 2 px | 0.3172 | +0.015 |
| 3 px | 0.3237 | +0.021 |
| 5 px | 0.3359 | +0.034 |

**Relaxing the boundary by 5 pixels gains only 0.034.** That means the error is **not** "the overall position is right but the boundary is drawn inaccurately",
so the "ambiguous target" hypothesis of Section 9 is rejected.

### 10.2 The per-image data exposed the real pattern

Looking at the per-image results revealed a very strong pattern:

| Image | GT coverage | Predicted coverage | Hard Dice |
|---|---|---|---|
| 24780 | 9.9% | 10.7% | **0.712** |
| 24776 | 9.6% | 11.7% | **0.697** |
| 36264 | 3.7% | 4.3% | 0.529 |
| 7335 | 1.7% | **14.6%** | **0.022** |
| 37185 | 0.6% | **8.4%** | **0.062** |

**The closer the predicted area is to the GT area, the higher the Dice.** And every one of the low-scoring images is
"GT is very sparse, yet the model predicts a large region" — prediction/GT runs as high as 8–10×.

### 10.3 Stratified statistics confirm it

Stratified by GT coverage (all 53 images):

| GT coverage | Images | Mean Dice | Prediction/GT |
|---|---|---|---|
| **<3%** | 17 | **0.0900** | **10.8×** |
| 3–8% | 18 | 0.4370 | 1.77× |
| 8–20% | 15 | **0.6046** | 0.91× |
| ≥20% | 3 | **0.7139** | — |

Correlations:

* GT coverage vs Dice: **r = +0.730**
* log(prediction/GT) vs Dice: **r = −0.668**

**Monotonic stratification, and the direction matches the mitochondria side exactly** (mitochondria is
`0.2969 / 0.7837 / 0.7987 / 0.8504`, for `<3% / 3–8% / 8–20% / ≥20%`).

### 10.4 Visual check: the model itself is right

The overlay images were downloaded and checked by hand (`docs/figures/golgi_pred/`):

* **24780 (Dice 0.712)**: the red contour fits the Golgi cisternal stacks precisely;
  the large stack right of centre, and the ones lower left, upper left and upper right, are all marked correctly. **The prediction is biologically right.**
* **37185 (Dice 0.062)**: the GT covers only 0.6% (almost nothing is annotated),
  while the model outlines a large number of small vesicles and tubular structures over the whole image, all of which count as false positives.

### 10.5 An ambiguity that has to be admitted

For images like 37185, **the metric cannot distinguish two explanations**:

1. The image really does contain many Golgi-derived vesicles that were not annotated -> **incomplete annotation**
2. The model mistakes ER/other vesicles for Golgi -> **model error**

The two produce **exactly the same** low Dice. I cannot assert which one it is from the metric alone.

There are three reasons for leaning towards "incomplete annotation", but none of them is a proof:

* prediction/GT reaches 10.8×, while on images with coverage ≥8% the ratio falls to 0.91×
* the stratification is monotonic (r = +0.730), and the direction agrees with the independent mitochondria result
* the outlined structures really do look morphologically like Golgi-derived membranes (rather than random noise)

**The only way to really distinguish them is to re-annotate those few images** — which is where the recommendation in 10.7 comes from.

### 10.6 What can therefore be concluded

| Claim | Does it hold? |
|---|---|
| The Golgi can be segmented and the result is usable | **Yes** (Dice 0.60–0.71 when annotation is sufficient, and the overlay shapes are correct) |
| A whole-image mean Dice of 0.30 means the model is bad | **No** — it is dragged down by 17 sparsely annotated images |
| Improving Dice requires tuning hyper-parameters | **No** (Section 6 shows v2/v3 both had no effect) |
| Improving Dice requires more/complete annotation | **most likely yes**, but the ambiguity of 10.5 is not ruled out |

### 10.7 A concrete recommendation for the next step

Rather than tuning the model further, fill in the annotations — especially the 17 images with coverage <3%.
A rough estimate from the stratified numbers: if they were brought up to the ≥8% level, the overall mean Dice would rise from 0.30 to **above 0.5**.

> Script: `tools/golgi_dice_vs_coverage.py`
> Output: `logs/golgi_dice_vs_coverage.txt` (per-image detail)
> Overlays: `docs/figures/golgi_pred/` (53 images, red = predicted contour)

---

## 11. Cross-domain evaluation: almost no transfer between the two domains

### 11.1 Why this was done

We have two Golgi sources — our own TEM figure panels and ASEM's FIB-SEM volume data.
Training on our own data and evaluating on ASEM answers the question directly:
**has the model learned the general morphology of the Golgi, or the specific appearance of our particular images?**

That decides whether the "merge data from multiple sources" route can scale or not.

### 11.2 Results

Using the model trained **on our own data only** (`runs/seg_golgi_own`, tile-level Dice, threshold 0.5):

| Data source | Tiles | Dice | GT coverage | Predicted coverage |
|---|---|---|---|---|
| In-house held-out (**in-domain**) | 281 | **0.2636** | 12.15% | 19.02% |
| ASEM ds61 (**cross-domain**) | 660 | **0.0779** | 11.05% | 3.68% |
| ASEM ds64 (**cross-domain**) | 392 | **0.2242** | 12.17% | 26.45% |

**Cross-domain Dice falls to between 1/3 (ds64) and 1/8 (ds61) of the in-domain value.**

Note that ds61 and ds64 also behave differently (0.078 against 0.224), which shows that "domain" is not a single factor —
within ASEM data itself, different imaging/preparation conditions differ a great deal.

### 11.3 What this explains

| Observation | Explanation |
|---|---|
| Merging ASEM brings only +0.025 (within noise) | the two domains differ too much, so little is transferable |
| in-domain performance did not drop after merging | model capacity is enough to hold both domains (0.2679 against 0.2636) |
| cross-domain is almost unusable | what was learned is **domain-specific features**, not general morphology |

### 11.4 Impact on the "scales up" route

One claim has to be corrected: the value of this route is **not "using public data to raise accuracy on our own data"**
(Section 5 already showed the gain is within noise), but rather:

1. **One model can serve several domains at once** without damaging any of them
2. A new data source only needs one more training root (`--train-roots`), so it really does scale engineering-wise
3. **But accuracy in a new domain must be evaluated in that domain**; cross-domain transfer cannot be assumed

In other words: scaling holds at the **engineering level** (adding a data source is cheap),
and does not hold at the **statistical level** (adding another domain does not make this domain more accurate).

> Script: `tools/eval_crossdomain.py`
> Usage: `--ckpt <weights> --roots <list of tile root directories> --classes golgi`

---

## 12. Excluding sparsely annotated images: the intuition was wrong

### 12.1 Hypothesis

Section 10 showed that the 17 images with coverage <3% have prediction/GT as high as 10.8×.
A natural inference is that **these images provide harmful supervision** — their background tiles teach the model
"these structures should not be predicted". So a control was run: **train only on images with coverage ≥3%**.

### 12.2 A clean comparison (the same high-coverage held-out images, 440 tiles)

| Model | Dice | GT coverage | Predicted coverage |
|---|---|---|---|
| Trained on everything (sparse images included) | **0.3445** | 22.69% | 14.53% |
| Trained on high coverage only | 0.3210 | 22.69% | **32.93%** |

**Dropping the sparse images is actually worse** (−0.024), and the new model **over-predicts badly**
(32.93% against a GT of 22.69%), while the full model is **on the conservative side** (14.53% against 22.69%).

### 12.3 Conclusion: the hypothesis is refuted, but it matters to understand why

The intuition is wrong because it ignores one positive role of sparsely annotated images:
**they supply a large number of real background tiles** that help the model calibrate "what is not a Golgi".
Once they are removed, the share of positives in the training set rises and the model learns to "predict more",
and so over-predicts on the held-out images.

So the conclusion of Section 10 needs to be sharpened:

| Claim | Verdict |
|---|---|
| Sparsely annotated images **lower the mean metric** | **Yes** (their own scores are low, which drags the mean down) |
| Sparsely annotated images **provide harmful supervision** | **No** (Dice is worse under the same conditions once they are removed) |
| Adding annotations raises the **mean metric** | **Yes** (it pulls the low-scoring images up) |
| Adding annotations raises **model quality** | **not shown** (this experiment actually suggests they are useful) |

This is an important distinction: **a low metric** and **harmful training** are two different things.
Deleting low-scoring samples as "bad data" can lose their positive effect
and also lose the chance to reflect data quality honestly.

### 12.4 So what levers are left

Every direction tried so far:

| Lever | Result |
|---|---|
| Lower the learning rate / train longer | no effect (6.2) |
| Raise the Dice weight | no effect (6.2) |
| Add external data (ASEM) | not demonstrated (5.7) |
| Clean annotation fragments | does not change the tile makeup (8) |
| Drop sparsely annotated images | **worse** (12.2) |
| Boundary tolerance | limited gain (10.1) |

**The only lever with clear evidence is "completing the annotations"** — it directly closes
the 0.09 against 0.71 stratified gap from Section 10.3, and it does not depend on any model-side tuning.

> Data: `logs/train_golgi_hicov.log`, `logs/GOLGI_HICOV.txt`
> Model: `runs/golgi_hicov/`








