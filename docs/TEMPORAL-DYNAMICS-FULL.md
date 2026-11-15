# S-BIAD712 in full: conclusions from 32 fields, and one self-refutation

**English** ｜ [中文](TEMPORAL-DYNAMICS-FULL.zh-CN.md)

> Contact dynamics were run on all **32 time-series files** (25–100 frames, 7 organelles,
> 79.87 nm/px). **Conclusion: the single-field observation does not hold.** At optical resolution
> most organelle pairs are **persistently interleaved**, with no temporal structure beyond static
> co-localisation.

---

## 1. From one field to the full set: what changed, in one sentence

`TEMPORAL-DATA-FOUND.md` wrote, on the basis of **one** file (`BR1_N08`):

> For most class pairs enrichment ≈ 1.0 (static interleaving), and **the only pair with real
> dynamics is mitochondrion–lysosome** (contact probability 0.36, CV 1.74, 5 formations and 5
> resolutions).

After running all 32 fields, that sentence **is wrong**:

| Class pair | Files with dynamics | Share |
|---|---|---|
| mitochondria:golgi | 7 / 32 | **22%** |
| golgi:lysosome | 2 / 32 | 6% |
| **mitochondria:lysosome** | **2 / 32** | **6%** |

Mitochondrion–lysosome is far from being "the only" pair; it does not even make the top three. In
the single file its CV was 1.74, while the median over 32 fields is only 0.40 — that one field
happened to be an extreme value.

**This is the 4th time in this project that a single-point conclusion has been overturned by more
fields or more seeds**; the first three were ASEM's +0.027, the Golgi small-sample +0.0189, and
the CEM500K single-seed positive effect. The methodological lesson is already in the paper; this
is the 4th illustration.

## 2. Full results

32 files × 7 class pairs = 224 (file, class pair) measurements. Judging a pair "dynamic" requires
all of: contact probability < 0.95, temporal CV > 1.0, and at least 2 independent formation
events.

| Class pair | Median contact probability | Median CV | Median enrichment | With dynamics |
|---|---|---|---|---|
| mitochondria:er | 1.00 | 0.09 | 1.04 | **0 / 32** |
| er:golgi | 1.00 | 0.13 | 1.01 | **0 / 32** |
| er:lysosome | 1.00 | 0.18 | 1.00 | **0 / 32** |
| er:lipid_droplet | 1.00 | 0.06 | 1.06 | **0 / 32** |
| mitochondria:golgi | 1.00 | 0.44 | 0.85 | 7 / 32 (22%) |
| golgi:lysosome | 1.00 | 0.45 | 0.95 | 2 / 32 (6%) |
| mitochondria:lysosome | 1.00 | 0.40 | 0.80 | 2 / 32 (6%) |

### 2.1 The four ER-containing class pairs are completely static

`mitochondria:er`, `er:golgi`, `er:lysosome`, `er:lipid_droplet` are judged static in **all 32
fields**: contact probability 1.00 (in contact in every frame), temporal CV 0.06–0.18 (almost no
fluctuation), median enrichment 1.00–1.06 (no difference from the time-shift null model).

**Meaning**: the ER is a tubular network spread through the cell, so almost any other organelle
always has some part adjacent to it. At 80 nm/pixel with threshold segmentation, "the ER is
adjacent to X" is an almost tautologically true proposition, and it carries no temporal
information.

### 2.2 The three pairs with dynamics all involve the Golgi or the lysosome

`mitochondria:golgi` (22%), `golgi:lysosome` (6%), `mitochondria:lysosome` (6%) are the only three
pairs. Their median enrichment is 0.80–0.95, **below 1** — i.e. less contact than the time-shift
null model, which says these structures are mostly **staggered** rather than persistently apposed.

**Meaning**: the Golgi and the lysosome are smaller structures whose position changes more
readily, so their relationship to other organelles changes over time; the ER and mitochondria are
pervasive large structures with a constant adjacency relationship.

### 2.3 But even "with dynamics" is only 22%, under a quarter

Even the highest, `mitochondria:golgi`, shows switch-like behaviour in only 7 of 32 fields. In the
other 25 fields the same pair is in persistent contact.

So the more accurate statement is not "a few pairs have dynamics", but:
**at optical resolution, contact in this dataset is almost entirely explained by spatial
interleaving; the real switch events are a few class pairs in a few fields.**

## 3. What this means for the project

### 3.1 Good news: dynamics can actually be computed for the first time

`TEMPORAL-DATA-SEARCH.md` originally concluded "there is no data with a time axis, so dynamics
cannot be established". Now, with S-BIAD712, the contact time series, the formation/resolution
events and the contact probabilities have all been computed. This part is **from nothing to
something**.

### 3.2 Bad news: at optical resolution the measurable dynamics are very limited

The contact criterion is "two threshold segmentations adjacent within 1 pixel (≈80 nm)". On a
pervasive structure like the ER this criterion is almost tautologically true, so "contact"
degrades into "they are in the same field".

To measure real membrane contact site dynamics you need:
* finer segmentation (membrane-level, not whole-class masks);
* higher resolution (EM-level, but EM has no time axis);
* or probes that specifically label contact sites (this project has none).

### 3.3 Against the EM results

| Data | Resolution | Time axis | Can answer |
|---|---|---|---|
| OpenOrganelle EM | 4 nm | none | **what** the membrane spacing is (the 10–30 nm MCS window) |
| S-BIAD712 fluorescence | 80 nm | 25–100 frames | **whether** contact changes over time |

The two are complementary rather than substitutes. This project **cannot** claim "we measured the
dynamics of membrane contact sites" — only that "at optical resolution, contact between most
organelle pairs is static interleaving".

## 4. Method and implementation

```bash
# 1) 下载全部 32 个时序（27.1 GB，并行 6 连接，实际约 90 分钟）
python tools/fetch_biad712.py --workers 6

# 2) 逐文件动力学（1 分钟/文件，支持续跑）
python tools/dynamics_biad712.py --paths "data/biad712/*.czi" --dilate 1

# 3) 汇总
python tools/summarise_biad712.py
```

### 4.1 Two performance traps

1. **The number of null-model shifts is the main cost**. The first version computed all T−1
   shifts (100 frames → 99 × 7 class pairs = 693 dilate+intersect operations), and 5 files took
   30 minutes. After limiting to **at most 30 equally spaced shifts** it got about 6× faster
   (1 minute per file). 30 is already enough to estimate the median of the null distribution.
2. **stdout is block-buffered once redirected**, so no progress was visible and there was no way
   to tell whether it had hung. Every print now has `flush=True`, and JSON is written **as each
   file finishes**, so an interruption can resume from existing results.

### 4.2 One data quality problem

In some files **peroxisomes** segment empty in several frames, and `center_of_mass` returns NaN.
The summary uses `nanmedian` and skips them explicitly, so a single NaN does not contaminate a
whole column of statistics.

## 5. Limitations

1. **Time calibration is still undetermined**. The XML says `Interval = 1 ms`, while scanning one
   frame takes 1.9 s, which is physically impossible; and the CZI has no per-frame timestamps.
   **So all quantities are in units of "frames", any reported nm/s assumes a 1.90 s interval, and
   it cannot be compared with absolute speeds in the literature.**
2. **Threshold segmentation stands in for manual annotation**, so absolute contact amounts cannot
   be compared with EM results.
3. **The 32 files come from only 3 biological replicates** (13 / 8 / 11 fields), not 32
   independent samples.
4. **The three thresholds for "with dynamics"** (probability < 0.95, CV > 1.0, formations ≥ 2) are
   chosen by hand. Changing them would change the specific percentages, but the "0/32" conclusion
   for the four pairs is insensitive to them — their median CV is only 0.06–0.18, far from 1.0.
