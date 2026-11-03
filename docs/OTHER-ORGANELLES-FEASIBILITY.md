# Other organelles: feasibility assessment

**English** ｜ [中文](OTHER-ORGANELLES-FEASIBILITY.zh-CN.md)

> Question: besides mitochondria, can other organelles be identified?
>
> **Conclusion: it cannot be done with the existing data, but both the reasons and the remedy are clear.**

> ## ⚠ This document has been partially corrected — read `docs/EMPANADA-TRANSFER-RESULTS.md` first
>
> Later testing found empanada's generalist EM models (MitoNet / DropNet / NucleoNet),
> and the results overturned two judgements:
>
> 1. **"Transfer is infeasible" is wrong.** That held only for the single-modality CellMap model;
>    zero-shot MitoNet achieves Dice **0.72–0.80** on this data (9/9 above the trivial baseline).
> 2. **"Start with lipid droplets" is wrong.** DropNet produces **no output at all** on this data
>    (and none after inversion either), which means there are simply no lipid droplets in these fields of view.
>
> In Section 3 below, "Route A: start with lipid droplets" has been refuted — **do not follow it**.
> The remaining factual statements — "the annotations contain only mitochondria" and "CellMap has 73 classes" — are still valid.

---

## 1. Established facts

### 1.1 Your annotations contain only mitochondria

Colour separation was performed on the annotation images and each colour verified individually (taking `11_0004` as the example):

| Colour | Pixel count | Connected components | Location | Verdict |
|---|---|---|---|---|
| **Green** `(0,180,75)` | 36550 | 15, largest 4496 px | Distributed across the whole image | **Mitochondrial contours** |
| **Red** `(237,28,36)` | 4749 | 4 | **100% in the bottom 60 rows** | **Scale bar** (bar + ticks + text) |
| **Yellow** `(255,242,0)` | 1350 | 15, about 90 px each | Distributed across the whole image | **Text letters** (for OCR) |

The distribution of the `organelle_type` field is **543 "M" + 27 empty**, with no second class.

**So there is no supervision signal available for "identifying other organelles".**

### 1.2 CellMap has 73 classes that could be trained

`tools/data_acquire/catalog.json` records that the CellMap Segmentation Challenge provides
73 annotated classes, including:

`mito / mito_mem / mito_lum / er / er_mem / er_lum / lyso / lyso_mem /
ld (lipid droplet) / perox / golgi / ves / endo / nuc / np / cell` and so on.

**Technically these classes can be trained.** But —

### 1.3 Transfer has been shown to fail

Mitochondria are the class in CellMap **most similar** to our data, and its zero-shot transfer result is:

| Approach | Dice |
|---|---|
| CellMap model applied directly | **0.1516** (0/6 above the trivial baseline) |
| Applied after inversion | 0.0486 |

Applying the same transfer to even less similar targets (lysosomes, Golgi, nucleus) **can only be worse**.
Until new evidence appears, "train other CellMap classes and then transfer them" should not be treated as a viable route.

---

## 2. Are there any other distinguishable structures in the images at all?

Before spending hours on training, answer this more basic question first.
Using `tools/explore_other_structures.py` in the **regions left after excluding mitochondria**,
candidates were sought by appearance (locally significantly brighter/darker than the median grey level, area limited to 0.05–2 µm²):

| Candidate type | Mean per image | Median | Maximum |
|---|---|---|---|
| Bright and uniform (possible lipid droplets / vacuoles) | 8.1 | 4 | 45 |
| Dark and enclosed (possible membrane-bound compartments) | 4.1 | 2 | 38 |

**But this number must not be read as "8 other organelles found".**
After assembling the candidate crops into a contact sheet (`logs/other_candidates_sheet.png`) and inspecting it by eye:

* **Most are cytoplasmic texture** (granular, with no clear boundary)
* Some are **membranous edges** (dark curves belonging to the membrane system rather than to discrete organelles)
* A few are diffuse bright regions with irregular shapes, unlike discrete organelles

**Reading: in this image, apart from mitochondria, there are no organelles that are morphologically clearly distinguishable and can be counted independently.**

This is related to the signal-to-noise ratio of the images themselves: `11_0004` is 2.667 nm/px with a field of view of 5.5×5.6 µm,
and when a crop 1.9 µm wide is magnified, the granular noise is already pronounced.

---

## 3. Three viable routes (ordered by cost)

### Route A: annotate our own data (most reliable)

For one specific target (for example lipid droplets — usually "bright and uniform", hence the easiest to annotate),
manually circling 20–30 instances is enough.

Basis: this project has already shown that **training from scratch in the target domain far outperforms transfer**
(detection: 0.5288 in the target domain vs 0.3075 for source-domain fine-tuning; segmentation: Dice 0.79 when trained on the target domain).
So a small amount of target-domain annotation is highly cost-effective.

**Recommendation to start with lipid droplets**: its appearance (bright, uniform, no internal texture) differs clearly
from that of mitochondria (bright, with cristae texture), so annotation consistency is easy to guarantee.

### Route B: improve imaging quality before discussing multiple classes

Current image noise limits distinguishability. If higher-magnification / lower-noise images of the same samples can be obtained,
structures such as lipid droplets and vesicles will be clearer. This is a decision on the experimental side.

### Route C: unsupervised / self-supervised segmentation (exploratory; not recommended as the main line)

Texture-based partitioning could be attempted (e.g. superpixel clustering + manual naming), but:
* there is no ground truth, so "whether the identification is right" cannot be quantified
* the results depend on hyperparameters in a way that is not reproducible
* the help it gives to the project's conclusions is limited

---

## 4. One useful thing that can be done right now

`tools/explore_other_structures.py` can already export candidate positions (`data/other_candidates/candidates.csv`)
and a crop summary image. **It can serve as a starting point for "pre-annotation"**: first filter candidates by appearance,
and a human only has to confirm "is / is not a lipid droplet", which is much faster than drawing from scratch.

But to be clear: what it provides is a **location proposal**, not class identification.
The script itself makes no claim about which organelle any candidate is.

---

## 5. Summary in one sentence

> Your annotations contain only mitochondria (verified: green = mitochondria, red = scale bar, yellow = text),
> CellMap does have 73 classes but transfer has been shown to fail,
> and in your own images, apart from mitochondria, **there are no other morphologically clearly distinguishable organelles**.
> "Identifying other organelles" is therefore currently infeasible;
> the cheapest path is **manual annotation of 20–30 instances of a single target (lipid droplets suggested), then training on our own data**.
