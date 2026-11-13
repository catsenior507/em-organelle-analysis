# Mixed training with OpenOrganelle: four arms, no improvement

**English** ｜ [中文](GOLGI-OPENORGANELLE-RESULTS.zh-CN.md)

> Mixing public volume-EM data into our Golgi training set does not help. Four arms, each against
> a same-budget control. **B − control = −0.0057, against a two-sigma seed band of 0.0160** — the
> effect is indistinguishable from zero, and the point estimate is negative.

---

## 1. Why mix at all

Our Golgi data is small and partially annotated. OpenOrganelle has Golgi annotation in volume EM,
CC0 licensed. The obvious move is to add it. This document records what happened when we did it
properly, with controls.

## 2. The domain gap, measured before training

Before any training, we compared the two distributions:

| Property | OpenOrganelle tiles | Our TEM tiles |
|---|---|---|
| Mean intensity | 172.7 | 140.8 |
| Standard deviation | 12.3 | 56.9 |
| Fraction below 64 | **0 %** | 10.8 % |

The public data has almost no dark range: its histogram is narrow and bright, ours is wide with a
substantial dark tail. Per-tile 1–99 percentile stretching brings OpenOrganelle to mean 149.4, SD
66.0 — close to ours in the first two moments.

This matters because it predicts a specific failure mode: a model trained on the public data has
never seen our background distribution.

## 3. Four arms

All arms use the same 281-tile validation set and the same augmentation.

| Arm | Training data | Epochs | last5 Dice |
|---|---|---|---|
| control | ours only | 60 | **0.3245** |
| **B** | ours + OpenOrganelle, percentile-normalised | 60 | **0.3188** |
| B′ | ours + OpenOrganelle, un-normalised | 13 (stopped) | 0.0009 |
| C | OpenOrganelle, cross-cell-type | 40 | 0.0072 |
| A | OpenOrganelle only, then fine-tune on ours | 15 | 0.0242 |

**B − control = −0.0057**, against a 2σ seed band of 0.0160. Not an improvement; not a detectable
harm either.

**B′ collapsed.** Without the percentile normalisation the model never learned anything — 0.0009
after 13 epochs, at which point we stopped it. This is the clearest single piece of evidence that
the intensity distributions are genuinely incompatible: the same architecture on the same task
fails outright when the public tiles arrive in their native range.

**C and A barely learn at all.** Training on OpenOrganelle alone and evaluating on our data gives
0.0072; pretraining there then fine-tuning on ours gives 0.0242. Both are far below the 0.3245
control, and below the trivial "predict all foreground" baseline.

## 4. What the arms establish

1. The public data **does** carry learnable Golgi information — B is not catastrophic, unlike B′.
2. But that information does not transfer: the best mixed arm only matches the control.
3. The transfer failure is **distributional**, not architectural: normalising the input range is
   the difference between 0.0009 and 0.3188.

## 5. Why this became a methodology story rather than a result

An earlier analysis of the same question used a single mixed-training run and reported +0.027.
Under paired cross-validation that number reversed sign (best-epoch −0.0139, last5 +0.0092). The
lesson is recorded in the paper's discussion: a single-point difference is not evidence, and the
split variance on this dataset (up to 0.12 Dice) is larger than any effect we were chasing.

## 6. Reproduction

```bash
bash tools/server/scripts/launch_golgi_oo.sh      # arms B, B', C, A
bash tools/server/scripts/launch_controls.sh      # control
```

See also `GOLGI-FEWSHOT-RESULTS.md` for the four follow-up routes (few-shot fine-tuning, learning
rate control, multi-class pretraining) — all negative — and `CEM500K-RESULTS.md` for the one route
that worked, which turned out to be the encoder rather than the data.
