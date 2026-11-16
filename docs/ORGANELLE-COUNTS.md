# Organelle counts across neuronal differentiation: 342 fields

**English** ｜ [中文](ORGANELLE-COUNTS.zh-CN.md)

> The instance-level annotations in S-BIAD2983 answer a more basic question than
> interaction: **how does the number of each organelle change across four stages of
> neuronal differentiation?**
>
> **All five organelles rise significantly in density, the Golgi by 2.9x and lipid
> droplets by 1.9x; only lysosomes are non-monotonic, peaking at day14 and falling after.**

---

## 1. A conclusion that 24 fields led astray

The first version of this analysis used only 24 fields, five to eight per stage, and concluded:

> Golgi density is flat: 13, 10, 16, 13 -- no trend across four stages.

**That conclusion was wrong.** With 342 fields, 79 to 92 per stage, the normalised Golgi
density is **5, 8, 13, 13** -- a **2.9-fold increase**.

The reason is simple: five to eight fields per stage, against enormous between-field
variation (Golgi instance counts per field range from 16 to 62). **A trend-free finding from a
small sample is often just noise covering the trend.**

The correction is kept here because it is more useful than the result itself.

## 2. Why normalisation by cell occupancy is necessary

Counting instances directly gives a tidy but wrong conclusion. Raw medians:

| Organelle | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| Golgi | 16 | 30 | 57 | 62 |
| mitochondria | 159 | 267 | 342 | 356 |
| lysosome | 220 | 549 | 575 | 391 |
| peroxisome | 172 | 258 | 324 | 342 |
| lipid droplet | 62 | 89 | 145 | 153 |

Everything appears to increase. But **the fraction of the field the cell occupies also
changes**:

| | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| cell occupancy (median) | 3.2% | 4.0% | 4.5% | 4.5% |

The cell grows from 3.2% to 4.5%, a factor of 1.4. Without removing that factor, "the cell got
bigger" reads as "there are more organelles".

## 3. Result after normalisation (342 fields)

Counts per 1% of cell occupancy (medians):

| Organelle | day7 | day14 | day21 | day28 | Kruskal-Wallis p | day7 to day28 |
|---|---|---|---|---|---|---|
| **Golgi** | 5 | 8 | 13 | 13 | **3.9e-17** | **2.9x** |
| mitochondria | 50 | 64 | 76 | 79 | 1.0e-21 | 1.6x |
| lysosome | 71 | 134 | 126 | 90 | 2.4e-31 | 1.3x (non-monotonic) |
| peroxisome | 54 | 64 | 73 | 79 | 3.2e-15 | 1.5x |
| lipid droplet | 20 | 23 | 31 | 39 | 2.7e-02 | 1.9x |

Pairwise comparisons (Mann-Whitney U, day7 against day28) are all significant:

| Organelle | p | median change |
|---|---|---|
| Golgi | 7.6e-12 | 5 to 13 |
| mitochondria | 6.5e-16 | 50 to 79 |
| lysosome | 2.4e-05 | 71 to 90 |
| peroxisome | 1.5e-11 | 54 to 79 |
| lipid droplet | 3.6e-03 | 20 to 39 |

### 3.1 Three things that can be said

**First, the Golgi grows the most, 2.9-fold.** This matches the expectation that neuronal
differentiation brings rising secretory demand and Golgi expansion. Note that this is a
separate matter from the project's other finding: that Golgi **segmentation is hard** (Dice
0.31) describes annotation sparsity, not low abundance.

**Second, mitochondria and peroxisomes rise monotonically.** Mitochondria 50 to 79 (1.6x),
peroxisomes 54 to 79 (1.5x). Neuronal differentiation involves neurite outgrowth, with rising
energy and lipid-metabolism demand, consistent with the function of both.

**Third, lysosomes are the only non-monotonic case: 71, 134, 126, 90.** They peak at day14 and
fall after, ending at day28 below half the day14 value. This shape differs from the other four.

### 3.2 A property of the data: the ER is a single instance

The `er` channel counts **exactly 1 in all 342 fields**. That channel annotates the **entire
connected ER network** rather than being split into individuals. So the ER's "instance count"
has no biological meaning and cannot be compared alongside the others, which is why it is not
in the table above.

## 4. How far this result reaches

**It holds statistically**: 79 to 92 fields per stage, Kruskal-Wallis p between 1e-15 and
1e-31, and every pairwise comparison significant. It is not small-sample chance.

**But it cannot be attributed to differentiation**: the four stages come from **different
acquisition dates** (November 2023 to February 2024), and the biological replicate composition
differs (day7 spans BR1-5, day21 only BR2-4).

So the tests answer "**these fields differ in distribution**", not "**differentiation caused
the difference**". Batch and differentiation effects cannot be separated in this dataset.

Settling causation would need **same-batch, cross-stage** samples, which this data does not
have.

## 5. Reproduction

```bash
# server side: stream download, count, discard; disk usage stays constant
python tools/server/scripts/biad2983_stream.py --out logs/BIAD2983_COUNTS.json

# aggregation and tests
python tools/server/scripts/counts_report2.py
```

Streaming is essential: all 342 fields would be 54 GB, and the server's `/home` has only 23 GB
free. The script downloads a field, counts it, and releases it immediately, so disk usage stays
at roughly one field throughout.
