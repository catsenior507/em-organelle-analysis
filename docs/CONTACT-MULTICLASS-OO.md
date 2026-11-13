# Five-class organelle contact analysis: no enrichment, globally or locally

**English** ｜ [中文](CONTACT-MULTICLASS-OO.zh-CN.md)

> 5 organelle classes × 4 public volumes × 15 class pairs, with a shape-preserving null at both
> the whole-field and the block level.
> **Result: 0 of 55 whole-field measurements significant, 0 of 976 blocks significant.**
> Organelles occupy distinct territories. Two self-corrections are recorded at the end.

---

## 1. Setup

### 1.1 Data

OpenOrganelle ground-truth crops (CC0), same field of view, 4 nm isotropic voxels:

| Volume | Cell type | Extent | Classes with real annotation |
|---|---|---|---|
| `jrc_hela-2/crop4` | HeLa | 476×600×600 | golgi, er, endo, ves, lyso |
| `jrc_jurkat-1/crop37` | Jurkat | 400³ | golgi, er, endo, ves, lyso |
| `jrc_macrophage-2/crop39` | Macrophage | 400³ | golgi, er, endo, ves, lyso |
| `jrc_hela-2/crop48` | HeLa | 512³ | golgi nearly empty |

Labels are **semantic, not instance**: each class is one binary mask with no object numbering.

### 1.2 Preparation

`tools/prep_oo_labels.py` builds a per-volume per-class symlink tree plus a manifest carrying the
voxel size, so downstream tools address `<root>/<dataset>/<crop>/<class>.npy` and never hard-code a
path.

## 2. Method

### 2.1 Contact counting

For a class pair A:B at tolerance t, dilate each A instance by t voxels and intersect with B; the
intersection volume, converted to area, is the contact.

The tolerance must not be chosen by eye. We run a **distance scan** from 0 to 40 nm and read the
window that matches the biological definition.

### 2.2 Null model

Contact voxel counts are not evidence on their own: a larger organelle touches more of everything.
To test whether two classes **prefer** to be adjacent, compare against "same shapes, same
abundances, different positions".

The null is a **toroidal shift**: translate one class rigidly under periodic boundary conditions,
recount, repeat over all shifts. In 3D an FFT cross-correlation yields the contact count for all
n³ shifts at once, giving an exact null distribution with no sampling error.

**A uniform-random null is wrong here and we tried it first.** When we placed each class uniformly
at random in the volume, every class pair showed an enrichment ratio of 0.00–0.05 — apparently
strong exclusion, actually the null destroying the tissue's own territorial structure. That
experiment is the reason the shape-preserving null is used.

### 2.3 Local variant

The whole-field null is dominated by large-scale territorial separation, so we also run a **block
version**: cut the volume into cubes of side L (default 96 voxels = 384 nm, stride L/2), keep only
blocks where both classes are substantially present, and run the same test inside each.

The pooled ratio is computed as `Σobserved / Σnull expectation`, not as a median across blocks —
a median is dragged to zero by the many blocks where both classes exist but are far apart, losing
all resolving power.

## 3. Results

### 3.1 Whole field

| Metric | Value |
|---|---|
| Measurements | **55** |
| Significant at p < 0.05 | **0** |
| Median p | 1.000 |
| Median z | **−1.90** (range −5.83 to −0.28) |
| Median enrichment ratio | 0.000 (maximum 0.006) |

Every measurement has z < 0: observed contact is **below** the shape-preserving expectation.

### 3.2 Local blocks

| Block size | Tolerance | Blocks | With contact > 0 | Significant |
|---|---|---|---|---|
| 64³ | 16 nm | 789 | 69 | 0 |
| 64³ | 24 nm | 789 | 191 | 0 |
| 96³ | 16 nm | 775 | 115 | 0 |
| 96³ | 24 nm | 775 | 242 | 0 |

Pooled enrichment ratio 0.00–0.01 across all configurations. **976 of 976 blocks not significant.**

### 3.3 Distance profiles

Median contact faces across volumes at each tolerance:

| Class pair | 0 nm | 8 nm | 16 nm | 24 nm | 32 nm | 40 nm |
|---|---|---|---|---|---|---|
| endo–er | 48 | 66 | 1 077 | 4 770 | 7 196 | **10 694** |
| er–mito | 0 | 0 | 438 | 3 162 | 6 858 | **10 573** |
| golgi–ves | 0 | 8 | 593 | 2 201 | 5 896 | **10 474** |
| er–lyso | 0 | 0 | 28 | 1 771 | 5 325 | **8 452** |
| er–golgi | 78 | 94 | 223 | 1 128 | 3 911 | **7 278** |
| golgi–mito | 0 | 0 | 0 | 0 | 0 | 0 |

Counts by tolerance across all 55 pairs: at 0 nm 9 pairs have any contact (1 057 faces total), at
8 nm 13 pairs (1 960), at 16 nm 37 pairs (24 786), at 24 nm 45 pairs (103 718), at 40 nm 46 pairs
(375 958). **More than two orders of magnitude between 8 and 40 nm.**

That rise sits inside the 10–30 nm window that defines membrane contact sites. Membranes closer
than about 10 nm fuse, so "two membranes at zero distance" is rare by construction.

## 4. The two negative results mean spatial organisation, not "membranes cannot touch"

This distinction matters and is easy to get wrong.

The enrichment analysis asks whether contact **amount** exceeds a shape-preserving null. It does
not — organelles are territorially segregated within the cell.

The distance profile asks at what **separation** the nearest contacts occur. They occur at 12–40 nm.

Both can be true. Territories describe large-scale arrangement; the profile describes local nearest
distance. Together: **organelles avoid each other on average, yet where they do meet they meet at
the distance a contact site requires.**

The per-pair distance profiles above are the primary measurement. Direction of the conclusion is
unaffected by which tolerance is quoted.

## 5. Two self-corrections

### 5.1 "16 nm is the annotation precision limit" — withdrawn

Our first reading of the 16 nm floor was that semantic masks trace the organelle interior, leaving
the inter-membrane space unannotated, so "masks adjacent" is not "membranes touching". To test
this, we recomputed using the membrane channels already present in the data (`*_mem`).

**The result was identical** (16 nm: 96 against 96; 24 nm: 1 335 against 1 335). Investigating why:
`golgi_mem` has 4 711 271 foreground voxels, exactly the intersection of `golgi` (10 487 793) with
itself — i.e. **the membrane mask is a subset of the class mask and traces the outer surface**. Seen
from outside, distance to the membrane and distance to the class are the same number, so the two
profiles must coincide.

The original judgement was too strong and is withdrawn: the mask boundary **is** the membrane
boundary.

### 5.2 "Contact appears from 12 to 16 nm" — withdrawn

The scan has only six levels: 0, 8, 16, 24, 32, 40 nm. **There is no 12 nm measurement.** The
original sentence cited a level that does not exist.

Also, "essentially zero below 10 nm" overstates the data: 9 pairs do have direct contact at 0 nm,
the largest being 574 shared faces. Replaced with the measured counts in section 3.3.

## 6. Limitations

1. **Semantic rather than instance annotation.** Only "are the two membranes close", never "how many
   partners does this organelle have".
2. **The 16 nm floor is set by annotation granularity.** Finer contact analysis needs membrane-level
   annotation.
3. **Four volumes, three cell types.** Results cannot be extrapolated to our own specimens, which
   are chemically fixed BSC-1 and different in contrast.
4. **Contact polarity was initially left open** — since answered in `CONTACT-POLARITY.md`: contact is
   strongly aggregated (median 52× a uniform surface null) but spread over many patches.

## 7. Reproduction

```bash
bash tools/server/scripts/launch_contact3d_oo.sh     # whole field
bash tools/server/scripts/launch_contact_local.sh    # block version
bash tools/server/scripts/launch_contact_mem.sh      # membrane-channel check
```
