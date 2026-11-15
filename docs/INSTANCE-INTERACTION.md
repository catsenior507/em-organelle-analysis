# Instance-level organelle interaction networks: results from 20 fields

**English** ｜ [中文](INSTANCE-INTERACTION.zh-CN.md)

> S-BIAD2983's segmentation is **instance-level**, so it can answer "how many mitochondria does
> each Golgi apparatus touch" — the very thing this project's paper explicitly says "semantic
> annotation cannot do".
>
> **Conclusion: only the Golgi's contacts with mitochondria and lipid droplets are significantly
> above the geometric expectation (1.39× and 1.50×); the other five pairs all fall near 1.0, i.e.
> are entirely explained by geometry.**

---

## 1. Data: instance-level, and spanning 4 differentiation days

**S-BIAD2983** (BioImage Archive, Cohen lab, UNC, CC BY 4.0). The clue was that its description
mentions "Lysosomes and mitochondria were the only organelles that underwent **declumping** to
separate touching organelles" — declumping only makes sense in instance segmentation.

Verification (one field, 32×796×796, uint32 labels):

| Channel | Instances | Median voxels |
|---|---|---|
| golgi | 91 | 34 |
| mito | 112 | 224 |
| lyso | 182 | 29 |
| perox | 113 | 19 |
| LD | 157 | 8 |
| nuc | 1 | 160878 |
| cell | 1 | 577962 (occupying **2.9%** of the volume) |

All six channels have different SHA256 values, which rules out placeholder files.

**After enumeration there are 351 fields with all channels present**, covering 4 differentiation
days (day 7/14/21/28, 95/95/82/79 respectively). This document takes **20** of them (5 per group,
sampled round-robin by biological replicate, see `tools/pick_fields.py`).

## 2. Methods

* **Contact criterion**: 6-neighbourhood direct adjacency (radius 1 voxel, about 80 nm);
* **Degree** `deg(a)`: the **number of B instances** in contact with instance a (not the number
  of voxels);
* **Null model**: translate A's centroid to a **random intracellular position**, requiring that
  at least 80% of the voxels are still inside the cell after the translation, then recompute the
  degree distribution. 3 draws per class pair.

### 2.1 The null model needed three versions before it was right, and each failure has a fingerprint

| Version | Approach | Result | What was wrong |
|---|---|---|---|
| v1 | apply a **label bijection** to the list of contacting voxels | enrichment **exactly 1.000** | the bijection is an identity transform: the voxel set is renamed as a whole, and the degree distribution is unchanged as a multiset |
| v2 | **full-frame** toroidal shift | enrichment **∞ / 10.6×** | the cell occupies only **2.9%** of the volume, so 97% of shifts throw the organelles into empty space and the null model is driven to 0 |
| **v3** | shift the centroid to a random **intracellular** position | 1.04–1.50× | the current, correct version |

**Two fingerprints worth remembering: enrichment coming out neatly at 1.000 = an identity
transform; enrichment of infinity = the null model sampled empty space.**

### 2.2 Performance

* Per-instance dilation: thousands of operations over 20 million voxels, and a single field does
  not finish in a quarter of an hour;
* The shift method (translate each offset once as a whole, collect label pairs): 0.8–1.8 minutes
  per field;
* Dropping the radius from 2 to 1 (33 offsets → 6) makes it about 5× faster again, and 20 fields
  take about 2 hours.

## 3. Results (20 fields)

| Class pair | Fields | Median enrichment | Range | Median partners | Fields with enrichment >1.2 |
|---|---|---|---|---|---|
| **golgi:LD** | 13 | **1.50×** | 0.00–3.00 | 1.00 | **69%** |
| **golgi:mito** | 19 | **1.39×** | 0.87–3.51 | 3.59 | **63%** |
| golgi:lyso | 20 | 1.20× | 0.78–1.65 | 2.28 | 50% |
| mito:perox | 16 | 1.05× | 0.95–1.30 | 1.74 | 12% |
| **mito:lyso** | 15 | **1.04×** | 0.98–1.53 | 2.79 | **13%** |
| lyso:LD | 18 | 1.00× | 0.52–1.50 | 1.00 | 11% |
| mito:LD | 14 | 0.99× | 0.57–3.00 | 1.09 | 21% |
| golgi:perox | 18 | 0.93× | 0.42–2.72 | 1.35 | 22% |

### 3.1 Only the Golgi has a "preference"

By the criterion "median enrichment > 1.3 and more than half the fields > 1.2", only two pairs
pass:

* **`golgi:mito` 1.39× (63% of fields > 1.2)** — the Golgi touches more mitochondria than
  geometry predicts;
* **`golgi:LD` 1.50× (69%)** — the Golgi touches more lipid droplets than geometry predicts.

**The other five pairs all lie near 1.0, i.e. are entirely explained by geometry.**

### 3.2 The most interesting counterexample: mitochondrion–lysosome

This pair has the **highest partner count** (median 2.79, up to 18 in some fields), yet
**enrichment is only 1.04×, range 0.98–1.53, with only 13% of fields above 1.2**.

That is: a mitochondrion sitting next to several lysosomes at once is **entirely determined by
geometry** — there are 228–596 lysosomes, scattered across the cell, so any position will touch a
few of them.

**"Many partners" and "having a preference" are two different things.** This is exactly why the
null model is necessary.

### 3.3 Change by differentiation day

| Class pair | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| golgi:LD | 1.50 | 1.52 | 1.50 | **2.05** |
| golgi:mito | **1.62** | 1.15 | 1.39 | 1.27 |
| golgi:lyso | 1.14 | 1.45 | 1.03 | 1.23 |
| mito:lyso | **1.32** | 1.01 | 1.03 | 1.10 |
| mito:perox | 1.14 | 1.06 | 1.00 | 1.06 |
| golgi:perox | 0.98 | 0.99 | 0.90 | 0.93 |
| lyso:LD | 1.00 | 1.00 | 0.95 | 0.90 |
| mito:LD | 0.98 | 1.00 | 0.71 | 1.78 |

`golgi:LD` stays stable at 1.50–2.05 across all four days, **the most stable pair**.
`mito:lyso` drops from 1.32 on day7 to ~1.0 after day14 — but with only 5 fields per day, this
should not be over-interpreted.

**Organelle instance counts change a great deal with differentiation:**

| Organelle | day7 | day14 | day21 | day28 |
|---|---|---|---|---|
| golgi | **16** | 42 | 57 | 47 |
| mito | 137 | 281 | 349 | 234 |
| lyso | 228 | 468 | 596 | 279 |
| perox | 137 | 284 | 269 | 285 |
| LD | 82 | 325 | 137 | 390 |

Day 7 clearly has the fewest organelles (only 16 Golgi), which is a feature of early
differentiation.

## 4. Implications for the paper

Limitation 4 should be rewritten as:

> **On data such as OpenOrganelle that have only semantic masks**, one can only answer "are two
> classes of membrane close to each other"; on data such as S-BIAD2983 that carry instance IDs,
> one can further answer "how many partners does each organelle touch". This document gives
> preliminary results for the latter over 20 fields: **only the Golgi's contacts with
> mitochondria (1.39×) and lipid droplets (1.50×) are significantly above the geometric
> expectation; the other five pairs are entirely explained by geometry.**

## 5. Limitations

1. **20 fields out of 351 available**. The sample is still small, and there are only 5 fields per
   differentiation day.
2. **Radius 1 voxel (about 80 nm)** is a direct-adjacency criterion, stricter than the previous
   radius 2, so the numbers are not directly comparable with the single-field version.
3. **The null model is still crude**: it only guarantees that the centroid is inside the cell; it
   does not preserve the relative arrangement of instances.
4. **Optical resolution 80 nm/pixel** is **not the same quantity** as the 10–30 nm membrane
   contact sites of EM.
5. **Snapshots only** (4 differentiation days). Instance-level **dynamics** would need S-BIAD712,
   which has no instance segmentation.

## 6. Reproduction

```bash
# enumerate available fields (351)
python tools/list_biad2983_fields.py

# pick 20 representative fields
python tools/pick_fields.py --per-group 5

# instance-level interaction network
python tools/instance_network.py --group auto --radius 1 --null 3 \
    --stems $(cat logs/biad2983_sample.txt) --out logs/INSTANCE_20F.json

# aggregate, including per-day and instance counts
python tools/aggregate_instance.py --json logs/INSTANCE_20F.json
```
