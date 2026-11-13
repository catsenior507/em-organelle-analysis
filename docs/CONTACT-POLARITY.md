# Contact polarity: contacts are strongly aggregated but not localised

**English** ｜ [中文](CONTACT-POLARITY.zh-CN.md)

> Answers the open question left at the end of `CONTACT-MULTICLASS-OO.md`.
> Data: 4 volumes × 5 organelle classes × 15 class pairs, tolerance 24 nm (6 voxels).

---

## 1. The question

Earlier analysis answered two things: whether the contact **amount** exceeds a shape-preserving
null (it does not — 55 of 55 not significant), and at what **distance** contacts appear (a steep
rise over 16–40 nm, matching the membrane contact site definition).

Neither answers the third question: **where contacts occur, how are they arranged on the organelle
surface?**

This is not a detail. Membrane contact sites are not scattered at random over an organelle: ER–mitochondria
contacts concentrate at junctions between the mitochondrial surface and the ER tubule network,
positioned by tethering protein complexes. If the measurement can only report "contact is 0.3% of
the surface", it cannot distinguish two quite different biologies:

- **(a)** the whole surface occasionally comes close — consistent with incidental proximity;
- **(b)** a few regions stay close — consistent with a specialised interface.

## 2. Method

For each (volume, class pair A:B):

1. take the **surface voxels** of A (A voxels with at least one non-A 6-neighbour);
2. **contact voxels** = surface voxels within 24 nm of B;
3. label the contact voxels with 6-connectivity to get **contact patches**;
4. statistic: **largest patch share** = largest patch size / total contact voxels.

Reading: a localised interface gives few large patches and a high share; a uniform spread gives
many small fragments and a low share.

**Null model**: draw the same number of voxels uniformly at random from A's surface and recompute
the same statistic, 100 times. This null preserves two things — total contact amount, and the
surface's own shape and adjacency — so "the surface is connected and will naturally form
contiguous runs" is absorbed by the null, and only spatial aggregation can explain a remaining gap.
One-sided p.

> **One performance note that mattered.** The first version ran a full-size distance transform on
> 400³–512³ volumes (a 512³ float64 transform is 1 GB) and allocated a 134 MB boolean array per
> draw, needing tens of minutes for a single pair. Changed to crop to the surface bounding box
> expanded by the tolerance plus two voxels — the transform stays exact because the window already
> covers every point that could fall within tolerance.

## 3. Results

| Metric | Observed | Null |
|---|---|---|
| Largest patch share | median **0.096** (0.018–0.325) | median **0.001** (0.000–0.012) |
| Observed / null | median **52.0×**, minimum 9.5× | — |
| Significant pairs | **35 / 35** (p = 0.010, the floor of 100 draws) | — |

**Contact is far more clustered than random placement.** Scattering the same number of voxels over
the same surface puts 0.1% of them in the largest patch; the data has 9.6%, a factor of 52.

**But contact is not localised to a single site.** The largest patch is typically under a tenth of
the total, so contact is made of many small-to-medium patches rather than one or two large zones.
Examples from one volume (Golgi):

| Class pair | Contact voxels | % of surface | Patches | Largest share | Null |
|---|---|---|---|---|---|
| golgi:er | 3 162 | 0.28% | 524 | 0.052 | 0.001 |
| golgi:endo | 2 046 | 0.18% | 256 | 0.046 | 0.001 |
| golgi:ves | 8 214 | 0.72% | 1 166 | 0.024 | 0.000 |
| golgi:lyso | 103 | 0.01% | 44 | **0.165** | 0.010 |
| er:endo | 18 903 | 1.14% | 2 088 | 0.018 | 0.000 |
| er:lyso | 20 694 | 1.25% | 964 | 0.085 | 0.000 |

Note `golgi:lyso` and `ves:lyso` behave differently: very few contact voxels (103 and 276), yet a
high largest share (0.165 and 0.221). **The less contact there is, the more easily it concentrates
in a few patches**; abundant contact (er:endo with 18 903 voxels) is spread over thousands of
small patches.

### 3.1 Robustness across tolerances

The main result uses one tolerance. If the conclusion held only at a particular threshold it could
be an artefact of that choice, so the whole analysis was rerun at 16 nm (stricter) and 32 nm
(looser):

| Tolerance | Pairs | Significant | Largest share median | Null median | Ratio median |
|---|---|---|---|---|---|
| 16 nm | 31 | **31 / 31** | 0.112 | 0.003 | 42.5× |
| **24 nm** | **35** | **35 / 35** | **0.096** | **0.001** | **52.0×** |
| 32 nm | 36 | **36 / 36** | 0.066 | 0.001 | 98.7× |

**All three tolerances are 100% significant**, and both trends are explicable. A looser tolerance
enlarges the contact area (median contact voxels rising from hundreds to thousands between 16 and
32 nm), which spreads contact over more patches, so the **largest share falls**
(0.112 → 0.096 → 0.066); meanwhile the null falls faster, so the **ratio rises**
(42.5 → 52.0 → 98.7×).

The conclusion does not depend on the threshold.

## 4. Conclusion: complementary to the enrichment analysis, not contradictory

| Analysis | Question | Answer |
|---|---|---|
| Toroidal null (`CONTACT-MULTICLASS-OO.md`) | Is contact **amount** above random? | **No**, 55/55 not significant |
| Distance profile | At what distance do contacts appear? | Steep rise over 16–40 nm, matching the MCS definition |
| **Polarity (this document)** | How is existing contact **arranged**? | **Significantly clustered** (52×), but spread over many patches |

Together these give a coherent picture:

**Organelles occupy distinct territories, so heterotypic contact falls below a shape-preserving
null; yet wherever contact does occur, it is aggregated rather than incidental.**

That matches the morphological expectation for membrane contact sites — local structures held by
protein complexes, not random encounters. That aggregation appears as "many small patches" rather
than "one large patch" is also consistent: a Golgi is made of several cisternae and an ER of a
tubular network, each contributing its own small interface.

**What this analysis cannot claim**: it does not identify specific contact sites and involves no
protein localisation. It shows only that contact is spatially non-random.

## 5. Limitations

- **crop48 contributes 7 pairs.** The other three fail the `--min-contact 30` threshold because
  crop48's Golgi annotation is nearly empty — not a program fault.
- **The p-value sits at its floor.** With 100 draws the smallest possible p is 1/101 ≈ 0.010, and
  every pair lands there, i.e. no null draw ever reached the observed value. Finer resolution
  needs more draws, but a median ratio of 52× (minimum 9.5×) already settles the direction.
- **Volume size affects fragmentation.** The volumes differ in extent (476×600×600 against 400³
  and 512³). Smaller volumes simply contain fewer separate contact zones, so the largest share
  runs higher there. **Cross-volume comparison of the share must account for this** — it should
  not be read as "jurkat contacts are more concentrated".

## 6. Reproduction

```bash
# full run (about an hour; must use tmux because server_exec's channel dies at 600 s)
bash tools/server/scripts/launch_polarity.sh

# single run with a chosen tolerance and draw count
python tools/interaction/contact_polarity.py --tol-nm 24 --null 200
```

Output goes to `logs/CONTACT_POLARITY.json`, one log line per pair so progress is visible.
