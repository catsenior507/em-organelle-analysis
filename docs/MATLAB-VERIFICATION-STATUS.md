# MATLAB Verification Status

**English** ｜ [中文](MATLAB-VERIFICATION-STATUS.zh-CN.md)

> Conclusion: **done**. All 8 (previously patched) files under `code/` pass
> **`mtree` syntax parsing**, **number of syntax failures = 0**; at the same time the runtime
> resolution of `project_paths()` and the end-to-end run of `analyze_emergent_entropy.m` have both been verified in practice.

---

## 0. Prerequisite: MATLAB used to fail to start (fixed)

During the audit, MATLAB on this machine could not start at all, reporting
`Unable to communicate with the required MathWorks services (error 5201)`. This is **unrelated to the project code**;
it is permission corruption of the lock directory of the MathWorks Service Host (two cascading faults).
The root cause and the repair process are in [`../tools/README-MATLAB-5201.md`](../tools/README-MATLAB-5201.md).

After the fix (normal privileges, non-elevated token):

```
User: CAT_SISTER\user   IsAdmin: False
matlab -batch "disp(version)"
  → 25.2.0.3177638 (R2025b) Update 5
```

---

## 1. Syntax verification results

`checkcode(file,'-m0')` **still returns style/performance advisories on R2025b**,
so "message count is 0" is **not** a criterion for syntactic validity. The real syntax criterion is
`mtree(file)` — building the syntax tree **parses** the file, and a syntax error throws an exception directly.

`tools/check_all.m` reports both and keeps them explicitly separate:

```
==== syntax validity (mtree parse) + advisory message counts ====
project_paths                syntax=OK     advisory_messages=0
project_assert_cwd           syntax=OK     advisory_messages=0
analyze_emergent_entropy     syntax=OK     advisory_messages=38
find_scale_bar               syntax=OK     advisory_messages=0
normalize_features_um        syntax=OK     advisory_messages=0
simple_region_features       syntax=OK     advisory_messages=3
extract_region_features      syntax=OK     advisory_messages=1
analyze_annotated_regions    syntax=OK     advisory_messages=5
------------------------------------------------------------------
SYNTAX FAILURES = 0
```

**8/8 syntactically valid, 0 failures.**

### What those 38/3/1/5 "advisory" messages are

All of them are **style and performance suggestions** that do not affect execution, and most are exactly the problems
the audit report already pointed out — they now have official confirmation from MATLAB itself:

| Type | Example | Corresponding audit conclusion |
|---|---|---|
| Unused parameters/return values | `analyze_emergent_entropy.m` L477, L523 (`estimate_lyapunov_discrete` is never called), L579, L1034 | §2.3 B5 / B8 dead code |
| `nanmean` suggested to be replaced by `mean` | L435, L495 | — |
| Array growing inside a loop | L914, L933-935; `simple_region_features.m` L165/L226/L273 | performance issue |
| Deprecated function | L850 `caxis`→`clim`; L1041 `datestr`/`now`→`datetime` | compatibility |
| Indexing style | `analyze_annotated_regions.m` L197 | — |

**I deliberately did not clean up these advisories one by one**: that script has been superseded by `tools/network/`,
and its core problems (a fabricated Lyapunov exponent, log₂(sample count) passed off as mutual information, a hard-coded α→λ lookup table)
cannot be solved by silencing warnings — the method itself has to be rewritten. Cleaning up the warnings would only make
a piece of code that should no longer be used look "healthier".

---

## 2. Runtime verification (not just static checking)

`tools/check_all.m` also verifies the resolution result of `project_paths()` at **actual runtime**:

```
==== project_paths() runtime resolution ====
root                : .
annotations_v2      : 1 (exists)
regions_v2          : 1 (exists)
dataset_json        : 1 (exists)
features_json       : 94 json
normalized_features : 88 csv
scale_mat           : 94 mat
python (OCR-capable): .venv\Scripts\python.exe
python probe        : status=0 out=ocr deps ok
```

### End-to-end verification: after the path fix the old numbers really are reproducible

```bash
matlab -batch "addpath('tools'); probe_emergence"
```

`analyze_emergent_entropy.m` **ran successfully for the first time** (it generated 4 figures and wrote them to
`data/emerg_inf_advanced/2026-10-15_20-06-21`), and the numbers **agree item by item** with the old report of 2026-05-18:

| Metric | Old report `2026-05-18_01-55-53` | This reproduction |
|---|---|---|
| s_11 entropy / λ / confidence | 3.5368 / −0.0079 / 0.43 | **exactly the same** |
| s_12 entropy / λ / confidence | 3.6284 / −0.0056 / 1.00 | exactly the same |
| s_13 entropy / α / inferred λ | 3.7502 / 0.8274 / **0.5000** | exactly the same |
| s_18 entropy / λ / confidence | 3.4461 / −0.0074 / 0.67 | exactly the same |
| s_21 entropy / α / inferred λ | 3.5264 / 1.0657 / **0.4803** | exactly the same |

This also **corrects an erroneous judgment in the first version of the audit**: it previously said "the old outputs cannot be reproduced
from the existing source code", while the actual cause was that the script hard-coded the no-longer-existing `d:\dachuang` and therefore
**could not run at all** — not "it ran but the artifacts did not agree". See [`AUDIT-REPORT.md`](AUDIT-REPORT.md) §2.3, last subsection.

The reproduction also confirmed the core problem once more: the λ of s_13 / s_21 comes from the **hard-coded lookup table**
of the branch that the script itself prints as `Insufficient time points for Lyapunov, using power-law...`,
while the three λ values that are actually computed are all **negative** (which by the script's own standard means "stable").

---

## 3. Structural checks (the substitute when MATLAB is unavailable, kept for cross-validation)

```bash
python tools/check_matlab_syntax.py
```

Block-keyword pairing / cross-line bracket balance / quote-parsing status of the 22 `.m` files under `code/`:

```
19/22 structural checks passed; 3 undeterminable; 0 failed
```

The 3 "undeterminable" ones are `analyze_annotated_regions.m`, `extract_region_features.m`,
`separate_annotation_colors.m` — pure text parsing cannot tell whether a MATLAB single quote is a
**transpose operator** or the **start of a string** (e.g. `fieldnames(colors)'`).
These 3 were equally "undeterminable" in their original versions **before the change**, so this has nothing to do with the current modification;
the `mtree` parsing above now gives a definite conclusion (syntax=OK).

---

## 4. Reproducible verification commands

```bash
# all at once: syntax check + runtime resolution + OCR dependency probe
matlab -batch "addpath('tools'); check_all"

# end-to-end: actually run analyze_emergent_entropy.m
# (writes a new timestamped output into data/emerg_inf_advanced/)
matlab -batch "addpath('tools'); probe_emergence"
```

Expected: `SYNTAX FAILURES = 0`, and `python probe : status=0 out=ocr deps ok`.

---

## 5. Changes to `code/` in this round (all have passed verification)

| File | Change |
|---|---|
| `project_paths.m` | 🆕 derives all absolute paths from `mfilename('fullpath')`, replacing the hard-coded `D:\dachuang` and pwd-relative paths; includes an **actual import probe** of the Python interpreter (see below) |
| `project_assert_cwd.m` | 🆕 raises an explicit error when the working directory does not match, no longer silently reading the wrong directory |
| `analyze_emergent_entropy.m` | paths changed to `project_paths()` (the original hard-coded `d:\dachuang` meant the script **could not run at all**) |
| `find_scale_bar.m` | removed the silent fallback to `0.1 µm/px` after an OCR failure → `NaN` + `scale_ok=false` + warning; fixed the global replacement in `strrep(filename,'red','')` |
| `normalize_features_um.m` | per-image min-max → **whole-dataset** statistics; the CSV keeps the original µm values |
| `simple_region_features.m` | added a "data source deprecated" guard at the top; `errordlg` (a modal dialog that hangs under batch processing) → `warning` |
| `extract_region_features.m` | same as above |
| `analyze_annotated_regions.m` | added a deprecation note at the top; the Python path now uses `project_paths()` |
| `detr_region_feature.py` | ⛔ marked as deprecated (three "guaranteed to crash"-level defects), pointing to `tools/detr/` |

### About the Python probe in `project_paths.m`

This project has **two** Pythons with different capabilities:

| Interpreter | Has | Missing |
|---|---|---|
| `.venv\Scripts\python.exe` | numpy / cv2 / PIL / pytesseract | **scipy, torch, transformers** |
| System Python (PATH) | numpy / cv2 / PIL / pytesseract / **scipy** / torch / transformers | — |

`P.python` is used only for `code\ocr.py` and `code\ocr_scale_bar.py`, so the probe condition is
"can `import PIL, pytesseract`", and it **actually executes** that command instead of merely checking that the files exist —
otherwise you get the hard-to-diagnose failure "the script was found but the import fails".
For scripts under `tools\` (`extract_regions.py` needs scipy), please use the command-line `python` directly.

---

## 6. Two encoding traps when writing .m files (hit this time)

1. **`.m` files must be encoded with the system code page (GBK on this machine), or contain ASCII only.**
   A debug script I created stored Chinese comments as UTF-8, and MATLAB reported
   `Invalid text character. Check for unsupported symbol, invisible character, or pasting of non-ASCII characters`
   and could not load it. The project's existing `.m` files are encoded correctly, so they are unaffected —
   the newly added `tools\check_all.m` and `tools\probe_emergence.m` were therefore deliberately written in **pure English**.
2. **The same applies to `.bat` files, and it is more dangerous**: cmd.exe will **execute the garbled bytes as a command**.
   See [`../tools/README-MATLAB-5201.md`](../tools/README-MATLAB-5201.md) §3.

---

## 7. Still to be noted

Syntax verification only proves that the files **can be parsed by MATLAB**; it does not prove that these scripts **give correct results**.
The problems of the following files in `code/` are at the method level and have been superseded by the implementations under `tools/`:

- `analyze_annotated_regions.m`: the segmentation takes the **background** (`~green_mask`), and the coordinates `[row,col]` are read as `(x,y)`;
- `simple_region_features.m` / `extract_region_features.m`: they parse **invalid JSON** (91/94 files contain the literal `...`);
- `representation_learning.m` / `neural_network_training.m`: data leakage + an ill-posed task;
- `analyze_emergent_entropy.m`: no network reconstruction, and the Lyapunov exponent is fabricated (now reproducible, see §2).

Do not use these scripts to regenerate training data — the only correct data source is
`data/annotations_v2/` (generated by `tools/morpho/extract_regions.py`).
