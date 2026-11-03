# Mitochondria–ER contact sites: first results on real data

**English** ｜ [中文](CONTACT-3D-RESULTS.zh-CN.md)

> Data: CellMap / COSEM `jrc_mus-kidney` (mouse kidney, FIB-SEM annotations)
> Compute location: the `dachuang` environment on `youruser@your-server-host`
> Generated: 2026-10-30
> Code: `tools/interaction/contact_analysis_3d.py`
> Raw output: server `~/dachuang_organelle/data/contact_3d_v4/`

---

## 1. These numbers are real, not a synthetic test

| Item | Value |
|---|---|
| Dataset | `jrc_mus-kidney` (COSEM / CellMap, CC-BY-4.0) |
| Crops containing both `mito` and `er` annotations | **20** (33 crops in total) |
| Analysed here | **5** (crop129/140/141/147/148) |
| Skipped | crop146 (voxel count below threshold) |
| Voxel size | crop129 = **8 nm**; the rest = **4 nm** |
| Volume shape | 300³ or 200³ |
| Data source | S3 `janelia-cosem-datasets`, fetched directly by chunk (**the 42 GB zip was not downloaded**) |

---

## 2. Metric definitions (state them this way when writing the paper)

- **Strict contact** (`tol = 0 nm`): the number of occasions where a mitochondrial surface voxel and an ER voxel **share a face**
  (6-adjacency). This is the criterion for "the two membranes are directly connected in the segmentation result".
- **Tolerance contact** (`tol = 16/32/64 nm`): the number of faces by which mitochondrial surface voxels "reach" the ER
  within a given physical distance. Specified in **nanometres** rather than voxels, because the 8 nm and 4 nm data
  must be able to be compared on one and the same scale.
- **`frac`**: the share of the mitochondrial total surface voxels accounted for by those faces (dimensionless, comparable across volumes).
- **Null model**: the ER is translated as a whole within the volume on a **torus**, and FFT cross-correlation computes
  the contact value for **all n³ translations at once**, giving the **exact** null distribution
  (300³ → 27 million translations; 200³ → 8 million).
  `p = #{translations : contact ≥ observed} / n³`, with **no sampling error**.

---

## 3. Results

### 3.1 Strict contact is almost non-existent; apposition appears at the 32–64 nm scale

| crop | Voxel size | Mitochondrial surface voxels | Strict contact 0 nm | 16 nm | 32 nm | 64 nm |
|---|---|---|---|---|---|---|
| 129 | 8 nm | 339,296 | **9,369** (2.76%) | 17,255 (5.09%) | 29,504 (8.70%) | 54,281 (16.0%) |
| 140 | 4 nm | 179,052 | **0** | 63 (0.04%) | 13,001 (7.26%) | 27,835 (15.5%) |
| 141 | 4 nm | 64,843 | **0** | 0 | 891 (1.37%) | 9,314 (14.4%) |
| 147 | 4 nm | 211,756 | **0** | 5,146 (2.43%) | 10,279 (4.85%) | 19,378 (9.15%) |
| 148 | 4 nm | 86,200 | **0** | 3,406 (3.95%) | 10,793 (12.5%) | 26,968 (31.3%) |

**How to read this**: 4 of the 5 crops have **no** mito–ER contact **at all at zero distance**; but widening the tolerance
to 32–64 nm, 9%–31% of the mitochondrial surface "reaches" the ER.

This is consistent with the known geometry of **mitochondria-associated membranes (MAMs)**: between the ER and the
mitochondrial outer membrane there is a **10–30 nm gap**, rather than fusion. At 4/8 nm voxels the segmentation masks
cannot resolve this gap as "zero-distance contact", so the strict-contact criterion returns 0; only a nanometre-scale
tolerance reveals the apposition.

### 3.2 Mitochondria and ER spatially **avoid each other**; this is not co-localisation

| crop | Observed count | Null mean | z | p |
|---|---|---|---|---|
| 129 | 9,369 | 30,984 | **−3.94** | ≈1.0 |
| 140 | 0 | 9,523 | **−4.61** | 1.0 |
| 141 | 0 | 7,019 | **−1.63** | 1.0 |
| 147 | 0 | 9,350 | **−3.30** | 1.0 |
| 148 | 0 | 15,975 | **−2.94** | 1.0 |

In all 5 crops the observed contact is **below** the expectation under random translation, with z from −1.63 to −4.61.

**This is not "no interaction"**; rather, in these kidney tissue blocks mitochondria and ER occupy **different spatial
regions** (ER tends to be perinuclear, mitochondria tend to be basal), so moving the ER to a random position makes it
more likely to collide with mitochondrial surfaces.

`p ≈ 1.0` because this is a **one-sided enrichment test** (asking "is it closer than random?"); the answer is no.
To test "is it significantly farther", one should correspondingly look at the negative z values.

---

## 4. Limitations that must go into the paper (these are not pleasantries; they are the boundaries of this result)

1. **The sample size is very small**: 5 crops, from **1 dataset, the kidney of 1 mouse**. Any cross-organ / cross-species
   conclusion does not hold.
2. **The torus null model assumes periodicity**: volume boundaries are treated as connected to the opposite side. For
   structures near a boundary this artificially makes the two sides "join up". Quantified on synthetic data: compared
   with the "translate then dilate" definition, the boundary effect is ≤ 0.92% of faces.
3. **8 nm and 4 nm cannot be pooled**: crop129 has 8 nm voxels and cannot resolve gaps <16 nm.
   Its "strict contact 9,369" is very likely just 8 nm sampling smearing a <16 nm gap into a connection.
   **Cross-resolution comparison can only use the nm-scale tolerance columns, never the strict-contact column.**
4. **Segmentation itself may create gaps**: annotators tend not to let the masks of two organelles touch,
   which systematically turns real contact into a 1–2 voxel gap. The tolerance profile mitigates this to some extent,
   but cannot eliminate it.
5. **This is contact geometry, not functional interaction.** To speak of "interaction", temporal co-variation or
   molecular-mechanism evidence is also required. This project **has no** real time series (see `docs/INTEGRITY-NOTES.md`).
6. **The null model only translates, it does not change shape.** It tests "positional preference" and cannot rule out
   the explanation that "the two shapes themselves evolved to fit each other".

---

## 5. Three pitfalls we hit and fixed in the methodology (all of them affect the conclusions and are worth recording)

### 5.1 The null model degenerated into "nothing changed"

The first version of the null model used "replay only at legal positions inside the volume". In the real data `er` has
**23–33 connected components scattered through the whole volume**, so the bounding box is almost the whole volume
→ the legal translation range collapses to 0 → all 50 samples equal the original position
→ the null distribution is **exactly identical** to the observation (`obs == null_median == null_max`, p identically 1.0).

The degeneration was **reproduced** with a 48³ synthetic scene and automatic `degenerate_null` detection was added.
Switching to torus translation removed the degeneration.

### 5.2 Tolerance expressed as a voxel count → not comparable across resolutions

v2 used `tol=8 voxels`: 64 nm for the 8 nm data, but only 32 nm for the 4 nm data.
The profiles of the three crops were simply not on the same physical scale. From v3 on it is specified in **nm**.

### 5.3 The brute-force null model would not run → switched to an exact FFT solution

6 volumes × 200 relocations × 300³ voxels would not finish in ten minutes.
Exploiting "translation and dilation commute", the problem reduces to a single cross-correlation:

```
contact(shift) = Σ_x S_A(x) · D(x − shift),   D = dilate(B, 1)
```

A single FFT gives the contact values for **all 27 million translations**, fast and with an exact p value.
It has been verified item by item that the FFT and brute-force implementations are **exactly equal** under the same
definition; the two definitions (dilate then translate vs translate then dilate) differ by ≤ 0.92% on periodic boundaries.

---

## 6. Reproduction steps

```bash
# on the server
conda activate dachuang
cd ~/dachuang_organelle

# 1) fetch labels (concurrent, about 2 min per volume)
python scripts/fetch_labels.py --dataset jrc_mus-kidney --labels mito,er \
    --out data/labels \
    --eligible-cache data/eligible_crops_jrc_mus-kidney.json --max-crops 6

# 2) contact analysis (exact FFT null model + nm tolerance)
python scripts/contact_analysis_3d.py --labels data/labels --pairs mito:er \
    --out data/contact_3d_v4 --manifest data/train_crop_manifest.csv \
    --tol-nm 0,16,32,64 --tol 1 --null-mode fft
```

---

## 7. Next steps (ordered by value)

1. **Expand to 20 crops** (every usable crop in jrc_mus-kidney) and raise n from 5 to 20 —
   only then can one speak of "the regularities in this tissue".
2. **Expand to multiple datasets**: `jrc_mus-liver` (23 crops), `jrc_hela-2` (26)
   — only a cross-cell-type comparison has biological meaning.
3. **Replace the strict-contact criterion with "inter-membrane distance"**: use a Euclidean distance transform to obtain
   the distribution of nearest distances from the ER to the mitochondrial surface, rather than counting faces at fixed
   tolerances. This directly reports "median MAM gap = X nm", which is more convincing than the current profile.
4. **Add a shape-controlled null model**: at present only translation is applied, so "the shapes themselves are
   complementary" cannot be ruled out. Random rotation + translation, or random ellipsoids with the same volume
   fraction, could be considered.
