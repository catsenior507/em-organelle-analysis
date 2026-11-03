# Multi-class organelle interaction analysis: the first time it is actually feasible

**English** ｜ [中文](MULTICLASS-CONTACTS.zh-CN.md)

> The project goal is "reconstruction of organelle interaction networks and analysis of emergent features".
> Until now we **could only build same-class adjacency networks**, because our own annotations are all single-class
> (`organelle_type`: 543 "M" + 27 empty). This document records the cross-class interaction analysis done after
> obtaining **two classes of ground truth in the same field of view** for the first time.

---

## 1. Data conditions

In ASEM dataset 64, `labels/er` and `labels/mito` fall on the **same ROI**:

| Item | Value |
|---|---|
| Volume | 241 × 476 × 528 |
| Voxel | 5 nm isotropic |
| Alignment offset (relative to raw) | [400, 326, 2212] (identical for both) |
| mito foreground | 9.16% |
| er foreground | 20.03% |
| Sample | BSC-1 cells, chemical fixation (CF) |

**This is the first time the project has two classes of annotation that are in the same field of view and already registered.**
The previous alternative (annotating the two classes separately in the same field of view) required extra manual annotation,
whereas here public data can be used directly to validate the method and toolchain.

> This must be stated: this is **external data**, not our own sample.
> It shows the tools work; it is not a biological conclusion about our sample.

---

## 2. A threshold problem that completely inverted the conclusion

The first time we ran the contact analysis, the criterion was "dilate by 2 voxels = 10 nm", and the result was **0 contacts**.
It looked like the data were broken (both classes occupy 9–20% of the volume — how can they not touch at all?).

After diagnosis, the truth was that **the threshold was wrong**:

| Decision distance | Contact overlap voxels | Share of mito surface |
|---|---|---|
| 0 nm | 0 | 0.00% |
| 5 nm | 0 | 0.00% |
| 10 nm | **0** | **0.00%** |
| 15 nm | 0 | 0.00% |
| 20 nm | 5 502 | 2.45% |
| **25 nm** | **15 058** | **4.60%** |
| 30 nm | 29 156 | 7.34% |
| 40 nm | 70 366 | 12.92% |

The closest distance between the two classes lies between **15–25 nm** — and in the literature
**the classic definition of an endoplasmic reticulum–mitochondria contact site (MCS) is a membrane spacing of 10–30 nm**.
Using 10 nm lands just outside the lower edge of that window, so the count came out as 0.

**Lesson**: the distance threshold in contact analysis cannot be chosen by intuition; it must
① be set from the biological definition and ② come with the full scan listed so a reader can check it.
The current script defaults to 25 nm and also outputs the 0–40 nm scan.

---

## 3. Results (25 nm criterion)

| Metric | Value |
|---|---|
| Contact overlap voxels | 15 058 |
| Contact volume | 0.0019 µm³ |
| mito surface voxels | 224 964 |
| **mito surface in contact** | **10 349 (4.60%)** |
| mito surface area (est.) | 5.6241 µm² |
| Contact area (est.) | 0.2587 µm² |

**4.60% of the mito surface is in contact with ER (within 25 nm).**
That is a reportable magnitude — in the literature, ER–mitochondria contact typically covers a single-digit percentage of the mitochondrial surface.

> The areas are **estimates**: computed as "each contact surface voxel contributes 1 voxel face" (5 nm × 5 nm).
> This is a common approximation, but no true surface meshing was done, so it can only serve as an order-of-magnitude reference.

---

## 4. Two limitations that must be stated clearly

### 4.1 The labels are binary semantics, not instances

Both label sets take values in `{0, 1}`. "Instances" can therefore only be approximated by **connected components**:

| Class | Connected components | Largest component |
|---|---|---|
| mito | 3 | 3.53 M voxels |
| er | 15 | 11.60 M voxels |

Mitochondria are inherently **network-like** structures, so "3 connected components" does not mean "3 mitochondria".
The report cannot say "each mitochondrion contacts N ER objects on average" — that requires true instance segmentation.

### 4.2 The two classes are mutually exclusive

The overlap voxels are **0**, which means a voxel can belong to only one class when annotating.
That is reasonable in itself, but it means **the contact region is forced into a binary assignment**:
at a real contact the two membranes are apposed, so one side must necessarily be assigned to one of the classes.
The 4.60% figure will therefore vary slightly with annotation habits and cannot be treated as a precise physical quantity.

---

## 5. What it means for our own data

### 5.1 What can be done now

* analyse any **registered multi-class annotation** (our own or public) with the same toolchain
* obtain the core interaction metric "what fraction of one class's surface is in contact with the other class"

### 5.2 What is still missing

| Gap | Explanation |
|---|---|
| Multi-class annotation of our own sample | currently only mitochondria and Golgi, and not in the same field of view |
| True instance segmentation | a connected-component approximation is used, so network-like structures merge |
| Three-class interactions involving Golgi | requires golgi + mito + er in the same field of view; the golgi of ASEM ds64 is in **another ROI** |

**The last one is worth noting**: the golgi ROI of ds64 is [777, 291, 954],
while er/mito is [400, 326, 2212] — **the two do not overlap**.
So this dataset cannot currently be used for a same-field-of-view golgi–mito–er analysis.
Three-class interactions would require annotating more of our own data, or finding another dataset.

---

## 6. Reproduction

```bash
# on the server
bash scripts/run_multiclass_contacts.sh
```

Output: `data/network_multiclass/ds64.json` (contains the scan table and the results at the chosen threshold).

Script: `tools/network/multiclass_contacts.py`
