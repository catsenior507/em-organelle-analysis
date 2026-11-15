# The Lyapunov exponent: from "inapplicable in principle" to "computable, and there is no chaos"

**English** ｜ [中文](LYAPUNOV-RESULTS.zh-CN.md)

> This indicator was once fabricated in this project. Now that a real time axis exists,
> the question can be answered directly. **The answer: it can be computed, and what it
> measures is zero.**

---

## 1. How this was handled before

The paper used to say:

> The Lyapunov exponent characterises the exponential divergence of neighbouring
> trajectories in phase space, and its definition requires trajectories evolving in time.
> The data used here are static snapshots with no time axis, so the indicator is
> **inapplicable in principle**.

That judgement was correct. The numbers the early version reported were a power-law fit
exponent mapped through a lookup table into a "Lyapunov exponent", with neither phase-space
reconstruction nor a divergence curve, some of them hard-coded. They were withdrawn.

**But S-BIAD712 has a real time axis (25 to 100 frames)**, so the indicator finally has the
precondition for computation. Is "inapplicable in principle" still enough? No -- because it
now applies. So the work was done.

## 2. The method has precedent in this field

**Pham T D. The butterfly effect in ER dynamics and ER-mitochondrial contacts.
Chaos, Solitons & Fractals, 2014, 65: 5-19.**

That paper uses exactly the pipeline implemented here: Rosenstein's largest-exponent
estimator, Fraser's auto-mutual-information delay, Kennel's false-nearest-neighbour
embedding dimension, Takens' phase-space reconstruction. Its subject is **endoplasmic
reticulum dynamics and ER-mitochondrial contacts in living cells** -- almost overlapping
with this project's data and question.

**So what was missing was never the method, it was the time axis.** This citation turns
"we deleted the indicator" into "we implemented it by the field's standard procedure and
applied it".

## 3. Calibrate the estimator before drawing conclusions

An estimator that cannot recover a known chaos gives an uninformative negative result. So
it was first calibrated on three series with known answers (`tools/lyapunov_validate.py`):

| Series | Theoretical lambda | Estimated |
|---|---|---|
| logistic map r=4 (N=1000) | 0.693 | **0.694** |
| sine wave (N=1000) | 0 | 0.004 |
| white noise (N=1000) | -- | 0.151 |

Calibration caught two real bugs, both recorded in the code comments:

1. **Wrong subscript in the divergence formula**: written as
   `||X_i - X_{nn+j}||`, the correct form is `||X_{i+j} - X_{nn+j}||`. The wrong form
   reaches the attractor diameter at j=1, so the divergence curve saturates immediately and
   **a known chaos was estimated at 0.002**.
2. **Fitting range**: fitting the whole divergence curve pulls the slope toward zero once
   it saturates.

After the fix the error is 0.1%. **Skipping calibration would have left the "no chaos"
conclusion with no foundation at all.**

## 4. Series length is the binding constraint

Calibration also gives something more important -- **the same estimator at different
lengths**:

| Length N | chaos measured (true 0.693) | underestimate | sine | white noise |
|---|---|---|---|---|
| 1000 | 0.694 | 0% | 0.004 | 0.162 |
| 300 | 0.679 | -2% | 0.001 | 0.151 |
| 100 | 0.587 | **-15%** | 0.000 | 0.124 |
| 60 | 0.439 | **-37%** | 0.000 | 0.126 |
| 39 | 0.410 | **-41%** | 0.000 | 0.055 |
| 25 | 0.252 | **-64%** | 0.208 | 0.003 |

Two conclusions:

* **Shorter series systematically underestimate chaos.** At N=25 only 36% of the true value
  survives.
* **Finite length carries a positive bias.** White noise still yields 0.124 at N=100, and
  it has no lambda at all.

So **S-BIAD712's 25 to 100 frames sit in the regime of "15-64% underestimate, with a
0.12-0.17 positive bias"**. In that regime the absolute value of lambda is uninterpretable;
**only comparison against a same-length calibration means anything**.

## 5. Result: 32 fields, all below the noise floor

Across 32 time-series files and 11 observables (7 organelle areas plus 4 class-pair contact
areas), compared against same-length calibration:

| Observable | Series | median lambda | same-length chaos | same-length noise | below noise | above chaos |
|---|---|---|---|---|---|---|
| nucleus | 32 | -0.005 | 0.587 | 0.125 | **100%** | **0%** |
| lysosome | 32 | 0.012 | 0.587 | 0.125 | **100%** | 0% |
| lipid_droplet | 32 | 0.008 | 0.587 | 0.125 | **100%** | 0% |
| golgi:lyso | 32 | -0.006 | 0.587 | 0.125 | **100%** | 0% |
| golgi | 31 | 0.006 | 0.587 | 0.125 | **100%** | 0% |
| peroxisome | 31 | 0.014 | 0.587 | 0.125 | **100%** | 0% |
| er | 31 | -0.003 | 0.587 | 0.125 | **100%** | 0% |
| golgi:mito | 31 | 0.028 | 0.587 | 0.125 | **100%** | 0% |
| mito:lyso | 30 | 0.006 | 0.587 | 0.125 | **100%** | 0% |
| mitochondria | 29 | 0.011 | 0.587 | 0.125 | 97% | 0% |
| mito:er | 27 | 0.049 | 0.587 | 0.125 | **100%** | 0% |

**Three readings:**

1. **Every median lambda lies between -0.006 and +0.049**, that is, zero.
2. **100% of series fall below the same-length white-noise bias floor** (0.125-0.172).
   They do not even reach the false-positive level of noise.
3. **0% exceed the same-length chaos calibration** (0.587).

The phase-randomised surrogate (preserving the power spectrum, destroying nonlinear
structure) agrees: of 338 series only **9 exceed z=2 (3%)**, against a random expectation of
5%; median z = -0.41.

**Conclusion: there is no detectable deterministic chaos in this data.** The temporal
evolution of organelle morphology is smoother than white noise but does not support the
strong claim of chaos.

## 6. What this negative result means

### 6.1 It changes "inapplicable" into "does not hold"

These differ:
* "inapplicable" is a **methodological** avoidance -- there is no time axis, so it is not
  computed;
* "does not hold" is an **empirical** conclusion -- there is a time axis, it was computed,
  and it measured zero.

The latter is stronger and more honest.

### 6.2 It is consistent with the project's earlier position

Section 2.7 of the paper already separated "emergence" from "dynamics": emergence
indicators are computed on the static spatial network (valid), dynamics indicators are not
computed without a time series. That is now filled in -- **when a time series exists it was
computed, and the result is no chaos in the dynamical sense**.

### 6.3 But it does not deny organelle dynamics itself

lambda ~ 0 says **no deterministic chaos**, not "organelles do not move". The same period's
analysis shows mitochondria moving 32.4 px net and Golgi 41.1 px, with direction coherence
0.40 -- **the structures really do move**, they simply do not show deterministic exponential
divergence.

## 7. Limitations

1. **Short series are a hard constraint.** 25-100 frames is far below what reliable
   estimation needs (calibration shows N>=300 to bring the underestimate within 2%). So
   this work can report "not detected", **not "does not exist"**.
2. **Morphological observables only.** A true phase space would include molecular state; a
   fluorescence image gives only first-order proxies of shape and position.
3. **Time calibration unresolved.** The inter-frame interval is assumed to be 1.90 s; lambda
   is per frame and not converted to per second.
4. **3D information is flattened.** The observables are per-channel areas and contact areas,
   not volumes.

## 8. Reproduction

```bash
# 1) calibrate the estimator first (without this, the negative result is not trustworthy)
python tools/lyapunov_validate.py

# 2) per-file computation (32 fields, about 5 minutes)
python tools/lyapunov_analysis.py --paths "data/biad712/*.czi"

# 3) same-length comparison and summary
python tools/lyapunov_summary.py
```
