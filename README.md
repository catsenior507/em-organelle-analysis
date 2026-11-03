# EM Organelle Analysis

**English** ｜ [中文](README.zh-CN.md)

Electron-microscopy organelle segmentation, and — the part that motivated the
project — reconstruction of **inter-organelle interaction** from segmentation.

Two-dimensional EM slices are segmented into organelle masks, masks are turned
into instance regions, and regions become a contact graph. All numbers below are
reproduced from the artifacts under `logs/`; the document that justifies each one
is named beside it.

---

## What is here

| Area | Code | Result |
|---|---|---|
| Mitochondria segmentation (own U-Net) | `src/train_seg.py` | **Dice 0.7839** on 18 held-out images |
| Transfer from a generalist EM model (MitoNet) | `src/finetune_mitonet.py` | 0.7518 zero-shot → **0.7847** fine-tuned |
| Golgi segmentation | `src/train_seg.py`, `tools/predict_golgi.py` | 0.4849 initially; see *Known limitation* |
| **Mito–ER contact analysis** | `tools/network/multiclass_contacts.py` | **4.60%** of mitochondrial surface within 25 nm of ER |
| Same-organelle adjacency network | `tools/network/run_analysis.py` | 491 annotated vs 458 predicted edges |
| Morphology transfer | `tools/compare_morpho.py` | area −10.5%, axes −2…−5% |
| Chaos / entropy analysis (MATLAB) | `code/*.m` | see `docs/` |
| DETR-based region detection | `tools/detr/` | exploratory, see `docs/DETR-TRAINING-READINESS.md` |

The interesting negative results are kept, not tidied away:

* **Cross-domain transfer fails.** A model trained on our TEM panels scores 0.0779
  and 0.2242 on public FIB-SEM volumes, against 0.2636 in-domain. Organelle
  appearance is domain-specific. (`docs/GOLGI-IMPROVEMENTS.md` §11)
* **Adding external data did not improve in-domain accuracy.** The estimate went
  +0.027 → +0.025 → +0.009 → −0.014 as the experimental design tightened.
  (`docs/GOLGI-IMPROVEMENTS.md` §5)
* **ER was abandoned.** It failed detection and segmentation both in-domain and
  cross-organ; intensity AUC 0.365 against a chance level near its prevalence.
  (`docs/OTHER-ORGANELLES-FEASIBILITY.md`)

---

## Quick start

```bash
pip install -r requirements.txt

# inspect the declared datasets, then build tiles from the declarations
python tools/unified.py list
python tools/unified.py probe ourdata_mito
python tools/unified.py build ourdata_mito

# train
python src/train_seg.py --train-root data/ourdata_mito_tiles \
    --val-root split --classes mito --size 256 --epochs 50 --out runs/seg_mito

# the whole chain, eight ordered and idempotent steps
python tools/run_pipeline.py --status
python tools/run_pipeline.py --steps all
```

The pipeline is `env → tiles → train → predict → filter → regions → network → report`.
Steps skip themselves when their outputs exist, `--force` re-runs, and a failure
stops the chain rather than propagating a half-built artifact.
`docs/PIPELINE.md` documents the per-step input/output contract.

Run the unit tests with:

```bash
python src/test_metrics.py      # metric implementations against known cases
python src/test_data_paths.py   # dataset path resolution
python src/test_ignore_mask.py  # ignored pixels receive exactly zero gradient
```

### Adding a dataset

Datasets are **declared, not coded** (`datasets.yaml`). The reader already handles
palette TIFF, RGB, greyscale and 16-bit input, three annotation styles (separate
green mask, overlaid strokes, binary mask), annotation/image size mismatch,
per-image pixel size, and two negative-sampling policies:

```yaml
ourdata_mito:
  class: mito
  images: data/原图片
  annotations: data/分离的标注信息
  annotation_kind: green_separated
  annotation_pattern: "green_{id}.tif"
  scale_json: data/annotations_v2/dataset.json
  target_nm: 4.0
```

---

## Reference numbers, and what they may not be compared with

Only figures reproducible from `logs/` are quoted. Four comparisons are invalid
and are flagged here because they are easy to make by accident:

| Do not compare | Why |
|---|---|
| 0.4849 with 0.3879 / 0.3605 | Different validation splits — 317 tiles from a random 20% versus 281 tiles from every-fifth-image; 41 training images versus 38 |
| tile-level Dice with full-image pixel Dice | Different quantities; tile Dice is roughly 0.06 higher |
| `best-of-epochs` with `last-5-epoch mean` | A systematic 0.03–0.05 apart. `best` also carries selection bias over 60 epochs and is noisier across seeds |
| contact statistics on public FIB-SEM with our own samples | Different cells (BSC-1), different preparation (chemical fixation) |

A verifier enforces the first three mechanically where it can:
`python tools/verify_report_numbers.py` and `python tools/verify_golgi_numbers.py`.

---

## Known limitation: annotation coverage

This is the honest explanation for the low Golgi number, and the one thing that
would most improve the project.

Ground truth is partial. On 17 of 48 Golgi images, annotation covers under 3% of
the field while the model predicts roughly ten times that area — the model finds
real organelle that nobody outlined, and every such detection is scored as a false
positive. Dice tracks annotation coverage almost monotonically:

| Annotation coverage | Images | Dice | predicted / GT |
|---|---|---|---|
| < 3% | 17 | **0.0900** | **10.8×** |
| 3–8% | 18 | 0.4370 | 1.77× |
| 8–20% | 15 | **0.6046** | 0.91× |
| ≥ 20% | 3 | **0.7139** | — |

Coverage versus Dice: **r = +0.730**. On adequately annotated images the same
model reaches 0.60–0.71, and the predicted contours sit on the cisternal stacks.
The 0.30 headline is produced by the under-annotated tail.

Mitochondria show the same shape (0.2969 / 0.7837 / 0.7987 / 0.8504 across the
same bands), found independently on a different organelle, which is why this is
treated as a data property rather than a modelling artefact.

**Planned work.** Complete the annotations for the low-coverage images. By the
stratified numbers, lifting those 17 to 8% would move the average above 0.5
without touching the model. Every model-side lever has already been tried and
failed — learning rate and schedule, Dice weight, external data, label
defragmentation, dropping sparse images (which made it *worse*), boundary
tolerance.

An `ignore` region mechanism is implemented and unit-tested
(`src/test_ignore_mask.py`) so that partially annotated fields can be excluded
from the loss once annotators record what they skipped: place an `ignore.png`
beside a tile and those pixels contribute exactly zero gradient. It is not
enabled anywhere yet, because deriving ignore regions from the model's own
predictions would merely bake in its current mistakes.

---

## A note on the numbers being traceable

Every figure in `docs/` is regenerated from artifacts rather than retyped.
`tools/golgi_report.py` assembles `logs/GOLGI_STATUS.md` by reading logs and never
recomputing, and it explicitly flags superseded conclusions quoted from older logs
instead of editing history. `tools/verify_golgi_numbers.py` cross-checks all 90
decimal figures in the Golgi document against the logs — it caught a real
transcription error where a run's mid-training value had been recorded as final.

---

## Repository layout

```
src/          trainer, dataset readers, unit tests
tools/        pipeline, unified data layer, network analysis, DETR, morphology
code/         MATLAB chaos/entropy analysis, OCR of scale bars, feature extraction
docs/         per-topic result documents — English, with the Chinese original
              alongside as *.zh-CN.md
logs/         result artifacts the documents cite
datasets.yaml dataset declarations
```

Documents are bilingual. Each one opens with an English/中文 switch; the `.md`
file is English and its `.zh-CN.md` sibling is the Chinese original the English
was translated from. Numbers, tables and code blocks are identical in both — a
script checks that the numeric tokens in the two versions are the same set, so a
translation cannot quietly drift from the evidence.

## Data

Raw EM images, annotations, model checkpoints and training runs are **not** in
this repository — they are large and partly third-party. The code expects them
under `data/`, `models/` and `runs/`; `datasets.yaml` describes the expected
layout, and `docs/PIPELINE.md` §7 gives the directory conventions.

Public data used: [CellMap / OpenOrganelle](https://openorganelle.janelia.org/)
and [ASEM](https://registry.opendata.aws/asem-project/) (Kirchhausen lab,
CC BY-SA 4.0). `docs/GOLGI-DATA-SOURCES.md` records what was pulled and the
licence terms.

## Licence

Code is MIT (see `LICENSE`). Third-party datasets keep their own licences; ASEM
data is CC BY-SA 4.0 and any redistribution must respect that.
