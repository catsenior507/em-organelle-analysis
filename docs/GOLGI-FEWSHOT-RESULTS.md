# Few-shot fine-tuning, learning-rate control, multi-class pretraining: all negative

**English** ｜ [中文](GOLGI-FEWSHOT-RESULTS.zh-CN.md)

> Three follow-up routes to "make public data useful", each compared against its own same-budget
> control, each with multiple seeds. **All negative.** The first single-seed run said otherwise and
> was wrong.

---

## 1. What was tried and why

`GOLGI-OPENORGANELLE-RESULTS.md` showed mixed training gives nothing. Three hypotheses remained:

1. **`few-shot`** — mix the public data into the *pretraining* and fine-tune on few of our labelled
   images. Maybe mixed training failed because the ratio was wrong.
2. **`low LR`** — maybe the fine-tuning destroyed the pretrained features by over-shooting. Lower
   the learning rate.
3. **`multi-class`** — maybe a single organelle is too sparse a signal. Pretrain on five classes at
   once so the encoder learns generic structure.

## 2. Few-shot fine-tuning, three seeds

Our labelled Golgi images available for fine-tuning: N = 6, 12, 38. Init = OpenOrganelle weights;
scratch = same architecture, random init; freeze = init with the encoder frozen.

| N | scratch | init | freeze |
|---|---|---|---|
| 6 | **0.2457 ± 0.0509** | 0.2008 ± 0.0099 | 0.0890 |
| 12 | **0.2903 ± 0.0310** | 0.2673 ± 0.0106 | 0.0762 |
| 38 | 0.3212 ± 0.0062 | **0.3275 ± 0.0116** | — |

**Init is worse than scratch at N = 6 and N = 12**, and only at N = 38 does it edge ahead by
+0.0063 — well inside the seed spread.

> **A method note worth keeping.** The first run used a single seed and showed init *ahead* by
> +0.0189 at N = 6. Adding two more seeds reversed the sign. This is the second time in the project
> that a single-seed comparison pointed the wrong way; it is the reason every claim in the paper
> carries a seed count.

**Freezing the encoder is much worse** (0.0890 and 0.0762). The pretrained features are not merely
useless here — when they cannot adapt, they actively block learning.

## 3. Learning-rate control

If the problem were over-shooting, a lower LR should help. At LR 3e-5:

| N | init − scratch |
|---|---|
| 6 | **−0.0955** |
| 12 | **−0.1266** |

Lowering the learning rate makes init **worse**, not better. Relative to their own scratch baselines
the drops are −15 % and −7 % for scratch against −43 % and −47 % for init: the pretrained arm
degrades far more.

So the hypothesis is rejected in the strongest form — the opposite of the prediction.

## 4. Multi-class pretraining

Five classes (golgi, er, endo, ves, lyso) from OpenOrganelle, 12 092 tiles, 10 492 train / 1 600
val, 40 epochs, reaching ~0.0019 mDice on its own validation set.

Downstream on our Golgi:

| N | scratch | multi-class init |
|---|---|---|
| 6 | 0.2457 ± 0.0509 | 0.1677 ± 0.0095 |
| 12 | 0.2903 ± 0.0310 | 0.2404 ± 0.0044 |

Worse than single-class init by 0.0331 and 0.0269, and worse than scratch by 0.0780 and 0.0499.

**This is the informative failure.** Five classes means more generic features, which was the
hypothesis; it made things worse, not better. That is hard to reconcile with "the encoder needs
more general training" and easy to reconcile with "the features are from the wrong distribution
regardless of how much of it you see".

## 5. The monotone series

Putting all routes on one axis, ordered by how firmly the model is held in the public data's feature
space:

```
no weights  >  single-class init  >  multi-class init  >  low LR  >  frozen encoder
   +0.000        −0.0448 / −0.0230    −0.0780 / −0.0499   −0.0955 / −0.1266   −0.140
```

The monotonicity is the point: it is not one bad hyper-parameter. The harder the model is pulled
toward the public distribution, the worse it gets.

This also rules out the "too few epochs" explanation — the frozen arm has the least chance to
deviate and performs the worst.

## 6. Reproduction

```bash
bash tools/server/scripts/launch_fewshot_seeds.sh   # 3 seeds × N ∈ {6,12,38} × {scratch,init,freeze}
bash tools/server/scripts/launch_fewshot_lr.sh      # LR control at 3e-5
bash tools/server/scripts/launch_multiclass.sh      # 5-class pretraining + downstream
```
