# System overview: from raw images to network analysis

**English** ｜ [中文](PIPELINE.zh-CN.md)

> This document describes the **structure, data contracts, and verified conclusions** of the whole pipeline;
> the goal is that anyone (including us a few months from now) can run it end to end and understand the results without reading the code.

---

## 1. At a glance

```
data/原图片/*.tif            88 grayscale EM images (1.004 / 2.667 / 5.291 nm/px)
data/分离的标注信息/green_*.tif  green outline = mitochondria (red = scale bar, yellow = text)
        │
        │  ① tiles          tools/build_ourdata_tiles.py
        ▼
data/ourdata_tiles/         2657 256×256 tiles, unified to 4 nm/px
        │                     grouped by image, with pos/neg and coverage annotations
        │  ② train          tools/server/scripts/train_seg.py
        ▼                      or finetune_mitonet.py (fine-tune MitoNet)
runs/seg_ourdata*/          U-Net weights (2.29 M parameters)
        │
        │  ③ predict        tools/predict_all_ourdata.py
        ▼
data/masks_ourdata/         93 binary masks at original resolution + index.csv
        │
        │  ④ filter         tools/filter_masks.py (physical area 0.05 µm²)
        ▼
data/masks_ourdata_clean/   filtered masks, 6.3 mitochondria per image on average
        │
        │  ⑤ regions        tools/seg_to_regions.py
        ▼
data/annotations_seg/       regions JSON **isomorphic** to annotations_v2
        │
        │  ⑥ network        tools/network/run_analysis.py --method radius
        ▼
data/network_analysis_seg_radius/   network metrics and report
        │
        │  ⑦ report         tools/pipeline_report.py
        ▼
logs/PIPELINE_REPORT.md
```

Single entry point:

```bash
python tools/run_pipeline.py --status        # check whether each step is done
python tools/run_pipeline.py --steps all     # full pipeline (completed steps are skipped)
python tools/run_pipeline.py --from predict --force   # rerun the later steps after changing the model
```

---

## 2. The unified layer: `tools/unified.py` + `datasets.yaml`

### 3.1 What it solves

We have **three mutually incompatible input forms** at the same time:

| Form | Image | Annotation | Previously handled by |
|---|---|---|---|
| `green_separated` | palette TIFF (grayscale stored as `mode=P`) | separate green outline image `green_<id>.tif` | `build_ourdata_tiles.py` |
| `green_overlay` | RGB JPEG (actually grayscale) | green outline overlaid on a copy of the image | `build_golgi_tiles.py` |
| `prebuilt_tiles` | already `<crop>/z####/{em.png,<class>.png}` | same as left | `train_seg.py` |

The result was **a new script for every new batch of data**, each handling the details of "convert to grayscale / fill holes / split"
slightly differently, which produced results that could not be compared.

It has now converged to **one declaration + one reader**: adding a dataset means adding a YAML block, not writing code.

```bash
python tools/unified.py list              # list all datasets and the available counts
python tools/unified.py probe all          # check only: pairing / sizes / GT coverage / skip reasons
python tools/unified.py build all          # generate tiles
```

### 3.2 Compatibility already implemented

| Dimension | Support |
|---|---|
| Image format | palette TIFF, RGB, grayscale, 8/16-bit → unified uint8 grayscale |
| Annotation form | separate green mask / overlaid green outline / ready-made binary mask |
| Naming | pairing by identical name, pairing by `green_<id>` prefix |
| Size mismatch | rescale the annotation to the image size and record it (**no more dropped images**) |
| Scale | `scale_json` looked up per image + unified resampling to `target_nm` |
| Negative-sample policy | `all_if_trusted` (for partial annotations) or `sampled` |
| Split | `val_every` (deterministic) or `val_images` (external list) |

### 3.3 Compatibility is regression-testable

`tools/test_unified_compat.py` turns "consistent with the existing scripts" into assertions:

```
[2] Golgi: must reproduce the old script's tile counts exactly
  PASS  positive tiles = 976      PASS  negative tiles = 820      PASS  images = 51
[3] Mitochondria: compared with the old script image by image (12_0002 is a known improvement)
  PASS  no image lost out of thin air
  PASS  the only extra image is 12_0002
  PASS  pos/neg identical for shared images   [88 images]
  PASS  total tile difference = contribution of 12_0002 only   [old 2657 new 2778 diff 121]
```

**`12_0002` is a known and confirmed improvement, not a regression**: its annotation image is transposed
(2048×2115 vs 2115×2048), and the old script's `if o.shape != img.shape: continue` threw this annotated
image away entirely; the unified layer rescales instead, hence 1 extra image / 121 extra tiles.

### 3.4 A pitfall we actually hit

The unified layer originally wrote `<out>/<name>` when `--out` was given and `data/<name>_tiles` when it was not.
Those two conventions meant that once the pipeline passed `--out`, the products landed in `data/ourdata_mito`,
while the status check looked for `data/ourdata_mito_tiles`, so it **always showed "pending"**.
It is now unified to `<out>/<name>_tiles` (`--out` is only the parent directory).

### 3.5 A pipeline with several datasets coexisting

```bash
python tools/run_pipeline.py --dataset ourdata_mito  --status
python tools/run_pipeline.py --dataset ourdata_golgi --status
python tools/run_pipeline.py --dataset ourdata_golgi --from predict --force
```

Product paths are derived from the dataset, and the two kinds of data never overwrite each other:

| Step | `ourdata_mito` | `ourdata_golgi` |
|---|---|---|
| tiles | `data/ourdata_mito_tiles/` | `data/ourdata_golgi_tiles/` |
| masks | `data/masks_ourdata_mito*` | `data/masks_ourdata_golgi*` |
| regions | `data/annotations_ourdata_mito/` | `data/annotations_ourdata_golgi/` |
| network | `data/network_ourdata_mito/` | `data/network_ourdata_golgi/` |

> The old `data/ourdata_tiles/`, `data/annotations_seg/`, and `data/golgi_tiles/`
> are kept for historical comparison, to reproduce the Dice numbers published earlier.

---

## 3. Data contracts (input/output format of each step)

| Step | Input | Output | Key convention |
|---|---|---|---|
| ① tiles | `原图片/*.tif` (palette grayscale), `green_*.tif` (outlines) | `<out>/ourdata/<image>/z####/{em.png,mito.png,meta.json}` | `<image>` must be a crop-level directory, otherwise the per-crop split breaks |
| ② train | tile directory | `runs/<name>/{best,last}.pt`, `history.json` | `--val-images` fixes the validation set; changing the seed only changes the training side |
| ③ predict | weights + raw images | `<out>/<image>.png` + `index.csv` | masks are at **original resolution**; the `split` column in `index.csv` separates train and held-out |
| ④ filter | masks + `index.csv` | filtered masks + a new `index.csv` | the threshold uses **physical area**, not pixel count |
| ⑤ regions | filtered masks | `regions/<image>.json` + `dataset.json` | fields are isomorphic to `annotations_v2`; features reuse `tools/morpho/features.py` |
| ⑥ network | regions directory | `summary.json` etc. | **must use `--method radius`** |
| ⑦ report | products from all steps | `logs/PIPELINE_REPORT.md` | reads only, never recomputes |

---

## 4. Verified conclusions (and where each number comes from)

| Conclusion | Value | Source file |
|---|---|---|
| CellMap single-modality model fails zero-shot | Dice 0.1516 (0/6) | `logs/fullimage_ourdata.json`, `docs/EMPANADA-TRANSFER-RESULTS.md` |
| **MitoNet general model is usable zero-shot** | Dice **0.7518** (17/18) | `logs/EMPANADA_HEAD2HEAD.txt` |
| **Own U-Net trained from scratch** | Dice **0.7839** (18/18) | `logs/fullimage_ourdata.json` |
| Excluding the 1 image with annotation coverage <3% | 0.8125 (17 images) | `docs/OURDATA-RESULTS.md` §7 |
| Multi-seed stability | **0.7890 ± 0.0016** | `docs/OURDATA-RESULTS.md` §8 |
| Morphology: size usable, boundary not | area −10.5%, solidity 0.996→0.882 | `logs/MORPHO_COMPARISON.txt` |
| Network: the distance-proximity criterion is required | hard contact 140→4; radius 491→458 | `docs/OURDATA-RESULTS.md` §6 |
| **Golgi segmentation (own annotations)** | Dice **0.4849**, trivial baseline 0.2027 | `logs/GOLGI_RESULT.txt` |
| Unified layer matches the output of the old scripts | Golgi 976/820 reproduced exactly; mitochondria differ only by 12_0002 | `tools/test_unified_compat.py` |

**When reporting accuracy, cite only the numbers whose sources are listed in the table above.** In particular:
**the released model (trained on all 88 images) has no held-out set, so no Dice can be quoted for it.**

---

## 5. Pitfalls hit (all fixed in the code now, but worth remembering)

These are mistakes that actually cost time; they are kept here so they are not repeated:

| Pitfall | Symptom | Fix |
|---|---|---|
| `ls <dir> >/dev/null` returns 255 | a directory that exists is judged missing | use `test -d` |
| zips produced by `Compress-Archive` use backslashes | Linux `unzip` extracts empty directories | use `tar` instead |
| Model scale mismatch | MitoNet produces **zero output** on 1 nm/px images | resample to 8 nm/px before inference |
| Hard-contact criterion | predicted network edge count −97% | switch to `--method radius` |
| Partial annotation | Dice is depressed; low-scoring images are actually under-annotated | report strata and baselines, do not look only at the mean |
| Validation set changes with the seed | differences between seeds are confounded | fix it with `--val-images` |
| `python ... \| tail -N` | the log does not grow for the whole training run, so it cannot be monitored | drop the pipe, stream the output |
| Train from scratch vs fine-tune | training from scratch on the target domain beats fine-tuning from the source domain | see `docs/EMPANADA-TRANSFER-RESULTS.md` |

---

## 6. Environment

Two environments, for different purposes:

**Local `.venv` (Python 3.11)** — inference, mask export, morphology, network analysis
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install empanada-napari==1.2.4 --no-deps
pip install pyyaml requests zarr scikit-image numba opencv-python-headless tqdm cztile imagecodecs
```

> The `__init__.py` of `empanada-napari` imports Qt components and fails in a GUI-less environment.
> It has been replaced with a minimal version (the original file is backed up as `__init__.py.orig`).

**Server `dachuang` (Python 3.10, 2× RTX 3080)** — training and fine-tuning
- `empanada-napari==1.2.4` (the wheel bundles the empanada core package)
- `numba` must install from a prebuilt wheel; attempting to build `llvmlite` from source fails,
  so force it with `--only-binary=:all:`
- weights are downloaded from zenodo (reachable from the server) and land in `~/.empanada/`

---

## 7. Directory conventions

| Path | Contents | Reproducible |
|---|---|---|
| `data/原图片/`, `data/分离的标注信息/` | raw data | **not reproducible, do not delete** |
| `data/ourdata_tiles/` | tile dataset | reproducible (①) |
| `data/masks_*/` | predicted masks | reproducible (③④) |
| `data/annotations_seg/` | regions JSON | reproducible (⑤) |
| `runs/*/best.pt`, `last.pt` | model weights | reproducible (②), but training is time-consuming |
| `runs/*/epoch*.pt` | intermediate snapshots | deletable (already cleaned up once, saved 4.86 GB) |
| `logs/*.txt`, `docs/*.md` | **conclusions and evidence** | **not reproducible, do not delete** |

---

## 8. How to extend

**Adding a new organelle**:
1. Confirm it is actually visible in the images (do not repeat the "recommend lipid droplets that are not in the images" mistake)
2. Annotate 20–30 instances as `<root>/<image>/z####/{em.png,<class>.png}`
3. `train_seg.py --classes mito,<class>` (when training both classes together you **must** add
   `--class-balance`, otherwise the minority class is suppressed below the trivial baseline)
4. the later steps are unchanged: ④→⑦

**Adding a new metric**: just read the corresponding product in `tools/pipeline_report.py`;
do not recompute it inside the report — the numbers in the report and in the products must have one source.

**Swapping the model**: as long as it produces `data/masks_<name>/` (binary masks at original resolution) + `index.csv`,
run `python tools/run_pipeline.py --from filter --force`, and every later step applies unchanged.

---

## 9. Toolchain for Golgi and external data

Beyond mitochondria, the system has been extended to a **second organelle** and a **second data source**.
The tools below are not part of the eight main pipeline steps, but belong to the same system; they are listed in order of use:

| Tool | Purpose |
|---|---|
| `tools/extract_golgi.py` | unpack the Golgi archive provided by the user (handles GBK filenames and `__MACOSX`) |
| `tools/quantify_text_contamination.py` | quantify how much burned-in caption text in the figures contaminates the GT (conclusion: about 2%, not worth masking) |
| `tools/detect_text_artifacts.py` | implementations of five text-detection criteria (all failed; kept as a negative record) |
| `tools/golgi_scale_analysis.py` | judge scale consistency by structural equivalent diameter (the middle 50% differ by only 1.8×) |
| `tools/clean_golgi_labels.py` | drop annotation fragments <500 px |
| `tools/asem_inspect.py` | check the alignment offset of each ASEM volume (subtract the offsets and divide by the resolution) |
| `tools/asem_fetch.py` | pull the needed volume subregions from the public S3 bucket, chunk by chunk |
| `tools/asem_to_tiles.py` | 3D volume -> 2D training slices (including padding when smaller than a tile) |
| `tools/predict_golgi.py` | full-image inference + **tolerance Dice** + overlay images |
| `tools/eval_crossdomain.py` | cross-dataset evaluation (answer: almost no transfer between domains) |
| `tools/golgi_dice_vs_coverage.py` | Dice stratified by annotation coverage (r = +0.73) |
| `tools/cv_partial.py` | partial cross-validation results (only the folds that finished) |
| `tools/golgi_report.py` | generate `logs/GOLGI_STATUS.md` automatically from the products |

**Server-side entry points** (all in `tools/server/`):
`upload.py` (upload, handles `~` and relative paths), `download.py` (fetch files over SFTP,
bypassing the shell so they are not BOM-contaminated), `server_exec.py` (run remote commands),
`push_run.py` (upload + run).

> These three small tools were forced out by one class of problem: invoking remote commands inline
> with `python -c "..."` was repeatedly broken by PowerShell variable expansion, quote swallowing,
> CRLF conversion, and BOM contamination. Anything "that has to be passed to a shell" should be
> written to a file rather than inlined as a string.

The full experimental records and conclusions are in `docs/GOLGI-IMPROVEMENTS.md` and `docs/MULTICLASS-CONTACTS.md`.

### 9.1 Golgi can also go through the main pipeline

```bash
# build tiles (via the unified layer, from the declaration in datasets.yaml)
python tools/run_pipeline.py --dataset ourdata_golgi --only tiles

# inference (automatically switches to tools/predict_golgi.py, not the mitochondria path)
python tools/run_pipeline.py --dataset ourdata_golgi --from predict
```

**The inference step is dispatched by dataset**, which is necessary rather than lazy:

| | Mitochondria | Golgi |
|---|---|---|
| Scale | reliable nm/px available (`dataset.json`), uniformly resampled to 4 nm | **no calibration**, must infer at original resolution |
| Weights | `models/ourdata_release/best.pt` | `runs/golgi_v3/best.pt` etc. |

Early on, `step_predict` hard-coded the mitochondria script, so `--dataset ourdata_golgi` also "ran",
but with the mitochondria resampling logic — **an error that raises no exception**. Dispatch is now
per dataset, and the weights are derived automatically too (so a mitochondria model is not used to infer Golgi).

> Note: Golgi training was done on the server; to run inference locally you must first run
> `python tools/server/download.py "~/dachuang_organelle/runs/golgi_v3/best.pt" "runs/golgi_v3/best.pt"`.
> All 53 images are slow on CPU; running `tools/predict_golgi.py` on the server GPU is recommended.

