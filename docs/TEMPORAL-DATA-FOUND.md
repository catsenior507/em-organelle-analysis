# Found: live-cell multi-organelle time series in S-BIAD712

**English** ｜ [中文](TEMPORAL-DATA-FOUND.zh-CN.md)

> The earlier conclusion of `TEMPORAL-DATA-SEARCH.md` was that "multi-organelle data with a time
> axis **does not exist**, so dynamics cannot be established". **That conclusion needs
> correcting.**
>
> What does not exist is an **electron microscopy** time series (fixed preparation, so that
> conclusion stands); but a **fluorescence** multi-organelle time series does exist, and it is
> public, readable, and already through a preliminary analysis.

---

## 1. The data found

**S-BIAD712** (BioImage Archive, Cohen lab, UNC), CC BY 4.0.

| Item | Value |
|---|---|
| Content | **live** human iPSC (KOLF2.1J), 7 organelles labelled simultaneously |
| Markers | nucleus, lysosome, mitochondrion, Golgi, peroxisome, ER, lipid droplet |
| Time points | **25–100 frames** (median 100) |
| Files | 32 time series + 35 single Z-stacks, 67 in total / 32.5 GB |
| Replicates | 3 biological replicates (13 / 8 / 11 fields) |
| Size | 768 × 768, 8 channels (7 organelles + 1 residual), uint16 |
| Pixel | **79.87 nm** |
| Instrument | Zeiss 880 confocal, 63×/1.4 NA, 1.90 s/frame |

### 1.1 Why we can be sure it is a real time series

The description is not trustworthy — **another** study on the same batch of data (S-BIAD2983)
also says in its description "timelapse images were acquired", yet **0** of its 4518 files carry a
time index; what is stored is all snapshots. So four independent checks were run on S-BIAD712:

| Check | Result |
|---|---|
| CZI header dimensions | `SizeT = 25`, `SizeZ = 1`, `SizeC = 8` |
| XML flag | `TimeSeries = true` |
| Whether the first two frames are identical | **No**, mean absolute difference between first and last frame 3808 (channel 0) |
| Time series of one field, or tiles of different fields | **the same field**: nuclear IoU between adjacent frames **0.891**, between first and last 0.874 |

The last item is the key one. If these frames were different fields (the protocol does mention
"Tile series"), the nucleus's frame-to-frame IoU would be close to 0. The measured 0.891 shows it
is the same cell advancing in time.

## 2. Two others checked and ruled out

| Accession | Title looked useful | Reality |
|---|---|---|
| **S-BIAD2983** | "8 organelles in live… at day7/14/21/28", 1.52 TB | **snapshots only**. 4518 files, 0 with a time index; the naming is `_Day7_BR5_N01..N11`, where `N` is the field, not time |
| **S-BIAD3460** | the title is literally "2D **Time-Lapse** Multispectral…" | no files on the FTP, possibly unpublished |

S-BIAD2983 is still valuable (8 organelles + already segmented + CC BY), but it **cannot support
dynamics**.

## 3. Time calibration: one ambiguity that cannot be resolved

This is the point that has to be stated clearly.

The XML has `FrameTime = 1.896 s` (the time to scan one frame), while `Interval = 1 ms`. **A 1 ms
interval is physically impossible** — a single frame alone takes 1.9 seconds.

Two interpretations:

- acquisition ran **continuously** (no delay), so the real interval is ≈ 1.9 s → 25 frames is about
  **47 seconds**;
- or the interval field was set wrongly and the real interval is far larger → the total duration
  could be tens of minutes.

A CZI should have per-frame timestamps, but in this file's metadata block the `TimeStamp` tag
count is **0**, so it cannot be settled from the file itself.

**Therefore every dynamical quantity in this document is reported in units of "frames", with no
conversion to absolute speed.**
Any number involving nm/s is labelled "assuming interval = 1.90 s", and it is not directly
comparable with absolute speeds from other papers.

## 4. Preliminary dynamics analysis

On one file (`TS_PBTOhNGN2hiPSCs_BR1_N08`, 25 frames) the full pipeline was run: per-frame Otsu
segmentation → class-pair contact series → circular time-shift null model.

### 4.1 The organelles really are moving

| Organelle | Adjacent-frame IoU | First-to-last IoU |
|---|---|---|
| nucleus | 0.891 | 0.874 |
| mitochondrion | 0.668 | 0.425 |
| ER | 0.687 | 0.549 |

The mitochondrion's and ER's first-to-last IoU drop to 0.43/0.55, showing substantial
rearrangement within 25 frames; the nucleus hardly moves (0.874), which is as expected and also
confirms the field is stable.

### 4.2 For most class pairs the "contact" is static interleaving, not dynamics

Contact criterion 1 pixel (≈ 80 nm), circular-shift null model:

| Class pair | Contact frames | Probability | Formation | Resolution | Temporal CV | Enrichment |
|---|---|---|---|---|---|---|
| mitochondria:er | 25/25 | 1.00 | 1 | 1 | 0.14 | 1.01 |
| er:golgi | 25/25 | 1.00 | 1 | 1 | 0.11 | 1.01 |
| er:lysosome | 25/25 | 1.00 | 1 | 1 | 0.08 | 1.02 |
| er:lipid_droplet | 25/25 | 1.00 | 1 | 1 | 0.08 | 1.04 |
| mitochondria:golgi | 25/25 | 1.00 | 1 | 1 | 0.37 | 0.95 |
| golgi:lysosome | 23/25 | 0.92 | 3 | 3 | 0.82 | 0.95 |
| **mitochondria:lysosome** | **9/25** | **0.36** | **5** | **5** | **1.74** | **0.42** |

**What enrichment ≈ 1.0 means**: after shifting B's time series circularly as a whole, the amount
of contact is almost unchanged.
That is, these contacts are **entirely explained by static spatial interleaving**, with no extra
temporal coordination — the ER and mitochondria are already interwoven throughout the cell, so an
"adjacency" criterion alone necessarily produces persistent contact.

**The only pair with real dynamics is mitochondrion–lysosome**: contact probability 0.36 (not
permanently on), temporal CV 1.74 (large fluctuation), 5 formation + 5 resolution events (real
events), while an enrichment of 0.42 means it is in contact **less** than the time-shift null
model — the lysosome switches between contact and no contact rather than staying attached.

### 4.3 One confound that cannot be ignored

Within 25 frames the nucleus has a net displacement of 7.2 pixels (about 575 nm), the
mitochondrion 32 pixels, the Golgi 41 pixels. The frame-to-frame consistency of displacement
direction is only 0.40 (1.0 = fully aligned), and the Golgi and the mitochondrion move in
**opposite** directions, so this is not stage drift.

But the Golgi's absolute speed (assuming a 1.9 s interval) would reach ~1.1 μm/s, on the fast
side. This again shows that **absolute speed is unreliable**, and that only relative comparisons
(which moves more, which has contact events) are usable.

## 5. What this means for the paper as a whole

The original wording, "dynamical metrics are in principle not applicable to static snapshots",
**still holds** — it refers to phase-space quantities such as the Lyapunov exponent, which really
do require a time series, and EM snapshots have none.

But the sentence "**there is no data with a time axis at all**" has to be changed to:

> A public **EM** time series does not exist (a limit of fixed preparation); but a public
> **fluorescence** multi-organelle time series does exist (S-BIAD712, 7 organelles, 25–100
> frames). Dynamics are therefore computable at **optical resolution**; it is just that their
> scale (μm, seconds) does not overlap with the scale of membrane contact sites (nm, milliseconds
> to seconds).

The two are complementary rather than substitutes: EM gives structure, fluorescence gives
dynamics.

## 6. Reproduction

```bash
# take the smallest time-series file (236 MB)
curl -sSL -o data/biad712/TS_BR1_N08.czi \
  https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/712/S-BIAD712/Files/TS_PBTOhNGN2hiPSCs_BR1_N08_Unmixed.czi

# verify dimensions, channels and that the time axis is real
python tools/validate_biad712.py

# contact dynamics (contact criterion 1 px ~ 80 nm)
python tools/dynamics_biad712.py --dilate 1
```

Helper scripts: `tools/probe_czi.py` (fetch only the file header, no full download),
`tools/walk_biad.py` (walk any S-BIAD study and decide whether it contains a time series).

## 7. Limitations

1. **Only one file, one field was run**. 31 of the 32 time-series files are still unanalysed, so
   the conclusions are preliminary.
2. **Time calibration is undetermined**, so absolute speeds cannot be reported.
3. **Threshold segmentation stands in for manual annotation**, so absolute values cannot be
   compared with EM results.
4. **Optical resolution 80 nm/pixel** cannot resolve the 10–30 nm membrane spacing, so "contact"
   here is not the same quantity as a "membrane contact site" in EM.
