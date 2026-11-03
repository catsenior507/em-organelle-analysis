# Research Integrity Issues

**English** ｜ [中文](INTEGRITY-NOTES.zh-CN.md)

> This document discusses only **issues outside the scope of "code bugs"**: matters that
> would make the conclusions untenable even if the code were written entirely correctly.
> The complete list of code issues is in [`AUDIT-REPORT.md`](AUDIT-REPORT.md).

---

## 1. Simulated experimental metadata was labeled "for publication"

### The facts

`code/simulate_structure_data.py` generates `data/structure/structure_metadata.json`.
The script's own comments and field names are as follows:

```python
def generate_realistic_time_series(sample_prefix, num_regions, num_time_points):
    """Generate realistic time series attributes with actual timestamps"""
    start_date = datetime(2024, 1, 1, 9, 0, 0)
    time_interval = timedelta(minutes=random.randint(5, 30))
```

The `metadata` field in the artifact:

```json
{
  "description": "High-quality simulated structure data with comprehensive attributes",
  "generation_method": "realistic_simulation_with_statistics",
  "purpose": "Scientific research and publication",
  "version": "2.0"
}
```

That is: **the script declares this to be a simulation, and at the same time marks `purpose` as for publication.**

### Which fields are not measured values

| Field | How it is generated | Measured? |
|---|---|---|
| `actual_date` | `datetime(2024,1,1,9,0,0) + i * timedelta(minutes=random.randint(5,30))` | ❌ |
| `timestamp` | `i + 1` (file index) | ❌ |
| `perturbation_level` | `random` / linear interpolation to 1–7 then `round()` | ❌ |
| `experiment_conditions_detail` | `25 + level*5 + random.gauss(0,1)` (temperature), etc. | ❌ |
| `sequence_length` | file count | ❌ |
| `region_info.sampling_rate` | the same random time interval as above | ❌ |

### How these fields enter the analysis

`code/analyze_emergent_entropy.m` reads this file (`:28-37`), extracts
`timestamps` and `perturbation_levels`, and then:

- `:387-388` sorts the "time series" by `timestamps`;
- `:483-484` sorts the "perturbation response" by `perturbation_levels`;
- the confidence formula at `:410`, `min(1, num_time_points/30)`, is determined directly by the file count.

Therefore **synthetic timestamps and perturbation levels take part in the computation of the "Lyapunov exponent" and the "perturbation sensitivity"**,
and appear in `data/emerg_inf_advanced/*/comprehensive_report.txt`.

### The authors' own statement in the progress document

`进度4.pdf`, page 3:

> Because the sample size is limited, it is necessary to reasonably arrange, **with simulated data**, the different
> representations under the time-series and classification modes, and then run the scripts to check whether they meet expectations.

The author knew this was a simulation. The problem is not "using simulated data to validate a method" — that is entirely legitimate methodologically —
but that **the boundary between simulated artifacts and real artifacts disappeared from the filesystem**:
the same directory, the same set of filenames, metadata saying the purpose is publication,
while downstream scripts cannot distinguish which number came from an experiment and which came from a random number generator.

### Suggested handling (three options — please pick one explicitly)

**Option A — delete it and keep only real data (recommended, if this batch of images really is not a time series)**

```bash
python tools/data_integrity/quarantine_simulated_metadata.py --apply
```
The files are moved into `data/_QUARANTINE_simulated_metadata/` with a `DO_NOT_USE.md`.
Then abandon the "time series / chaos" framework and switch to cross-sample statistics (see Section 2).

**Option B — keep it, but rename globally and add explicit markers**

If simulated data really is to be used to validate the method (for example, to show that "the Lyapunov exponent is not
estimable when the sample size is insufficient"), then you must:
1. prefix every filename with `SIMULATED_`;
2. add `_simulated` to every output directory;
3. annotate every figure title with "simulated data";
4. state explicitly in the paper/records: "this figure is based on synthetic data, is used for method validation,
   and does not represent experimental results".

**Option C — replace it with real metadata**

Read the real acquisition times and conditions from the TIFF `DateTime` tag or from an acquisition log exported by the instrument.
This workspace has **no** such files; they need to be requested from the colleagues who ran the experiments.

---

## 2. The "time series / chaos" framework lacks support from real data

### The facts

`data/normalized_features/` contains 88 CSVs. `analyze_emergent_entropy.m:53-65`
groups them by filename prefix (the number before `_`), giving:

| Group | Number of CSVs (= number of "time points") |
|---|---|
| `11` | 13 |
| `12` | 49 |
| `13` | **2** |
| `18` | 20 |
| `21` | **4** |

Apart from 11 and 12, each group has only 2–4 points. Yet the script's own confidence formula
(`:410`) says `min(1, num_time_points/30)` — that is, it considers 30 points the threshold for sufficiency.
**The two groups are insufficient for any dynamical analysis**, yet they still rank first in the final ranking
(because their λ went through the lookup-table path of `infer_lyapunov_from_powerlaw`, which gives inflated values).

The grouping is based on filenames:

```
normalized_region_analysis__11_0002.csv
                          ^^ ^^^^
                          batch  index
```

`进度4.pdf` page 3 says "the same numeric prefix before '_' represents the same batch, so this can be used
to **assist in adding perturbation patterns**" — that is, the very concept of a "time point" was constructed from filenames,
not from an experimental time-series design.

### Impact

- conclusions such as "convergence of the Lyapunov exponent over time" have no subject;
- the entropy computation merges the rows of the 13–49 images of the same batch into one matrix (`:157-160`),
  so the difference in sample size directly determines the magnitude of the entropy;
- the reported ranking is decided mainly by **file counts** and the **lookup function**, not by biological differences.

### Suggested handling

**One question must be answered first**: were these 88 images acquired in the same experiment, in temporal order?

- **If yes**: provide the true acquisition time (or perturbation condition) for each file,
  and confirm the number of time points; groups with only 2–4 points must be excluded from the dynamical analysis.
- **If no** (which looks more likely): delete the time-series framework and switch to
  **cross-sectional statistics across samples** — for example, comparing the distributions of network topology metrics
  across batches/conditions. This needs no Lyapunov exponent; `tools/network/` is already implemented in this direction.

---

## 3. Cited numbers cannot be reproduced from the existing source code

`data/emerg_inf_advanced/` contains 8 runs, of which:

- the reports of `2026-05-18_01-37-35` and `2026-05-18_01-39-03` contain
  `SECTION 4: POWER-LAW ANALYSIS`, `SECTION 6: INTERPRETATION`
  and strings such as `Method: insufficient_levels`,
  **which the existing `analyze_emergent_entropy.m` cannot generate**;
- `2026-05-18_01-51-18` groups by the full prefix and obtains **88 "samples"** (one per image),
  which is not comparable with the other runs in the same directory (5 samples);
- the report headers of `data/emerg_inf/` (2026-04-15, 7 runs) together with a single `unknown` sample —
  **no .m/.py in the whole workspace produces it**.

In other words, the same analysis has left **three mutually incompatible generations of results** on disk,
and their ranking conclusions contradict each other (s_13/s_21 rank first in both 01-52-49 and 01-55-53,
yet these two groups have only 2 and 4 time points).

### Suggested handling

Before citing any emergence-analysis result in the final report/paper:
1. decide explicitly which run is used, and write a `PROVENANCE.md` into that directory
   (source commit, parameters, input file list);
2. rerun once with the current source code and confirm the numbers are reproducible;
3. move irreproducible old outputs into `archive/` and remove them from the conclusions.

---

## 4. Other matters that need to be stated in the records

1. **OCR has never once successfully recognized annotation text**. Statistics of the historical results: 504 empty strings,
   237 `"M"`, 36 `"S"`, 20 `"co"`.
   The cause is that `code/ocr.py` does not do the polarity inversion for white text on a black background.
   This audit has confirmed that the yellow annotations are **all the glyph "M" (mitochondria)** —
   that is, **this batch of data has only a single organelle type**, so any statement about "interaction of multiple types" lacks support.
2. **91/94 files in `data/features_json` are invalid JSON** (containing the literal `...`),
   and **all** 805 regions are truncated to 1000 coordinate points.
   If the paper describes a feature computation pipeline that differs from the actual code, it must be corrected.
3. **In the scale-bar calibration, 4 images carry untrustworthy fallback values** (0.1 µm/px, 20–100× larger than the correct value),
   and one more (`12_0002`) treats some number as a scale bar.
   Any conclusion expressed in µm requires manual verification of these 5 images first.
4. **The scheme described in `进度3.pdf` does not match the code**: the document says "Hungarian algorithm + GIoU + weighted cross-entropy",
   while the code uses `torch.topk` greedy matching + MSE. If the records describe DETR implementation details, they need to be aligned with the code.

---

## 5. Suggested order of handling

1. **First answer "is it a time series?"** (Section 2) — this decides whether stage three is torn down or amended;
2. **then decide the fate of the simulated metadata** (Section 1, Option A/B/C);
3. the quarantine tool is ready:
   `python tools/data_integrity/quarantine_simulated_metadata.py --apply`
4. archive `data/emerg_inf/` and the three-generation `emerg_inf_advanced/` (keep them but move them out of the main directory);
5. rerun a **reproducible** pipeline with `tools/morpho/` + `tools/network/`,
   and save `config.json` together with the git commit hash in the output directory.
