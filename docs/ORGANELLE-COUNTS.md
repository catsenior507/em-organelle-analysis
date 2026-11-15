# Organelle counts across neuronal differentiation: the Golgi stays flat, lipid droplets double

**English** ｜ [中文](ORGANELLE-COUNTS.zh-CN.md)

> The same instance-level annotations (S-BIAD2983) answer a more basic question than
> interaction: **how does the number of each organelle change across four stages of
> neuronal differentiation?** The answer: **the Golgi barely changes, while mitochondria,
> peroxisomes and lipid droplets rise substantially.**

---

## 1. Why the normalisation is necessary

Counting instances directly gives a tidy but wrong conclusion. Raw medians:

| Organelle | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| mitochondria | 124 | 281 | 349 | 234 |
| Golgi | 36 | 40 | 57 | 47 |
| lysosome | 230 | 468 | 596 | 279 |
| peroxisome | 135 | 289 | 269 | 285 |
| lipid droplet | 120 | 148 | 137 | 390 |

Everything appears to increase. But **the fraction of the field the cell occupies also
changes**:

| | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| cell occupancy (median) | 2.9% | 4.0% | 4.9% | 3.7% |

The cell grows from 2.9% to 4.9%, a factor of 1.7. Without removing that factor, "the cell
got bigger" reads as "there are more organelles".

## 2. After normalisation

Counts per 1% of cell occupancy (medians):

| Organelle | day7 | day14 | day21 | day28 | relative to day7 |
|---|---|---|---|---|---|
| mitochondria | 43 | 67 | 69 | 71 | **+65%** |
| **Golgi** | 13 | 10 | 16 | 13 | **flat** |
| lysosome | 81 | 126 | 140 | 83 | rises then falls |
| peroxisome | 47 | 82 | 70 | 96 | **+104%** |
| lipid droplet | 39 | 42 | 60 | 105 | **+169%** |

### 2.1 Three things that can be said

**First, Golgi density is flat.** 13, 10, 16, 13 -- no trend across four stages. This runs
against the intuition that the Golgi expands during neuronal differentiation, and it is a
separate matter from this project's finding that Golgi segmentation is hard: **hard to
segment is not the same as few or changing in number**.

**Second, mitochondria and peroxisomes rise monotonically.** Mitochondria 43 to 71 (+65%),
peroxisomes 47 to 96 (+104%). Neuronal differentiation involves neurite outgrowth, with
rising energy and lipid-metabolism demand, which is consistent with both.

**Third, lipid droplets surge at day28.** 39 to 105 (+169%), against 39, 42, 60 for the three
earlier stages. Late accumulation of lipid droplets is a point-specific change.

**Lysosomes are the only non-monotonic case**: 81, 126, 140, 83, peaking at day21 and falling
after.

## 3. What this result can and cannot support

**Only 24 fields**: six at day7, eight at day14, five at day21, five at day28.

**They come from different imaging batches.** The four stages were acquired on different dates
(between November 2023 and February 2024), and the biological replicates differ in
composition (day7 spans BR1-5, day21 only BR2-4). **Batch effects cannot be separated from
differentiation effects.**

So this is **descriptive only, with no significance claim**. The percentages above are ratios
of field medians without confidence intervals, because five to eight fields cannot support
interval estimation.

To turn it into a claim that can be asserted, one would need:
1. tens of fields per stage;
2. multiple replicates within a single batch;
3. or a dataset that is **same-batch across stages**.

## 4. Relation to the contact analysis

Both use the same instance annotations but answer different questions:

* **Contact analysis** (`INSTANCE-INTERACTION.md`): who is adjacent to whom, and whether the
  contact count exceeds geometric expectation. The conclusion is that apart from the Golgi,
  every pair is explained by geometry.
* **This document**: how many individuals of each organelle there are per field, and how that
  changes with differentiation. The conclusion is that Golgi density is constant while
  mitochondria, peroxisomes and lipid droplets rise.

Together they give a coherent picture: **the Golgi's abundance is stable across
differentiation, while the energy- and lipid-related organelles keep increasing.**

## 5. Reproduction

```bash
# uses only cached segmentation files, no re-download
python tools/organelle_counts.py
```

The script prints the raw medians, the normalised densities and the cell occupancy, and
reports field counts and biological replicates per stage.
