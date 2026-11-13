# Temporal and live-cell data: six sources searched, nothing usable

**English** ｜ [中文](TEMPORAL-DATA-SEARCH.zh-CN.md)

> Answer to "is there any time-resolved EM data of organelle dynamics we could use?"
> **Short answer: no, and the reason is structural rather than an artefact of searching badly.**

---

## 1. Why this matters to the project

The project's early version reported Lyapunov exponents, mutual information and other dynamical
quantities. Those numbers turned out to be fabricated (see `MATLAB-VERIFICATION-STATUS.md`), and
the implementation was removed. That left a question the project had to answer honestly:

> Is the dynamics missing because we implemented it wrong, or because the data cannot support it
> at all?

The first is a bug to fix. The second is a limit of the method that should be stated. This document
establishes which.

## 2. What was searched

| Source | What it holds | Result |
|---|---|---|
| **EMPIAR** | Raw EM image archives | REST API exposes only `latest_citations`; **no usable search or listing endpoint**. BioStudies covers EMPIAR in principle but EMPIAR is not among its searchable collections. |
| **BioStudies** | Multi-omics study archive | Queries work, but no time-resolved organelle EM |
| **IDR** (Image Data Resource) | Published microscopy image datasets | **147 projects.** 28 contain organelle keywords, 10 contain temporal keywords, **none is a multi-organelle time-lapse** — and all are fluorescence, not EM |
| **Cell Tracking Challenge** | Cell and nucleus tracking | Cell-level, not organelle-level |
| **BBBC** | Broad bioimage benchmark | Cell-level; no organelle time series |
| **OpenOrganelle** | Janelia COSEM volume EM | Single time point per specimen, by construction |

## 3. The two candidates that looked promising, and why they failed

**Valm et al. 2017, "Applying systems-level spectral imaging and analysis to reveal the organelle
interactome"** — this is the closest published work to our problem: six organelles imaged
simultaneously in live cells with spectral confocal microscopy, with interaction maps between all
pairs. It is exactly the kind of time-resolved multi-organelle data that would make dynamics
computable.

**No public data deposit exists.** The paper's data availability is limited to supplementary
material; there is no accession number and no repository record. It cannot be used.

**IDR's temporal subset** — 10 projects match both a temporal and an organelle keyword, which
sounds like the answer. On inspection all are fluorescence, and the "organelle" match is usually
incidental (a nuclear marker, a membrane dye). None images multiple organelles over time in a way
that would support interaction analysis.

## 4. The structural reason

**Electron microscopy requires fixation.** Fixation cross-links and immobilises the sample; after
it, there is no time axis. A volume EM stack is one instant.

Live-cell imaging that does have a time axis is fluorescence, which gives organelle identity and
position but not the membrane-level ultrastructure that a contact site requires — the 10–30 nm gap
between two membranes is below the resolution of a light microscope.

So the two properties needed — **ultrastructure** and **time** — are not available in the same
public dataset, and cannot be, because the sample preparation that gives one destroys the other.

## 5. What this means for the paper

The Lyapunov exponent is not merely implemented incorrectly. **There is no public data on which it
could have been computed.** The removal is therefore not a retreat but a correction of a category
error: a dynamical invariant was reported for a system with no dynamics to measure.

The substitute used in the paper is spatial statistics on the network (degree-distribution entropy,
clustering, percolation, spatial transition entropy). These are well defined on a static snapshot
and are described as such — they are **not** dynamical quantities, and the paper says so explicitly
in section 2.7.

## 6. What would be needed

1. **Collect it ourselves**: live-cell fluorescence with multiple organelle markers to supply the
   dynamics, plus fixed single-time-point EM of the same cells to supply the morphology. The
   registration between the two is the hard part.
2. **Or find a correlative dataset** that already pairs live fluorescence with subsequent EM of the
   same specimen. None was found in the six sources above.

Either is beyond the scope of an undergraduate project, which is itself worth stating in the
report rather than leaving the reader to wonder why the dynamics section is missing.
