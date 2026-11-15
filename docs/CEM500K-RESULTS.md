# CEM500K encoders: a larger backbone helps, its pretrained weights do not

**English** ｜ [中文](CEM500K-RESULTS.zh-CN.md)

> The fifth attempt at importing outside knowledge, and the first with a positive component —
> but the positive part is the architecture, not the pretraining.

---

## 1. Why try again

The previous four routes (mixed training, few-shot fine-tuning, lower learning rate, multi-class
pretraining) were all negative, and the negative effect grew monotonically with how strongly the
model was anchored to OpenOrganelle weights. The mechanism we inferred was that OpenOrganelle is
a single FIB-SEM modality, so the discriminative features it learns do not transfer to our TEM
sections.

That mechanism left a premise untested: **is the incompatibility about that particular data
source, or about pretraining as such?**

CEM500K is a clean discriminator. It is not any one microscope's output — it is 496,544
micrographs selected from 2.6 million, spanning many cell types and acquisition settings, trained
with MoCo v2 self-supervision specifically to learn general EM texture. If the bottleneck really
is a lack of general features, it should help.

## 2. What we obtained

CEM500K lives at **EMPIAR-10592** under CC0, directly downloadable:

| File | Size | Use |
|---|---|---|
| `pretrained_models/cem500k_mocov2_resnet50_200ep_pth.tar` | 370 MB | MoCo v2 ResNet-50 encoder |
| `metadata/cem500k_image_metadata.csv` | 43 MB | Image provenance and acquisition conditions |
| `data/cem500k.zip` | not fetched | The dataset itself (we use the weights, not the images) |

**Load verification**: 312 of 318 tensors match, **98.1%**. The six unmatched are classifier-head
keys; the backbone loads completely. This step is not optional — a silent 3% load would make
every downstream number meaningless.

## 3. Three-arm design

| Arm | Encoder | Parameters | Initialisation | Epochs |
|---|---|---|---|---|
| A | ResNet-50 | 48.94 M | **CEM500K MoCo v2** | 30 |
| B | ResNet-50 | 48.94 M | random | 30 |
| C | small U-Net | 2.29 M | random | 30 |

**Arm B is not optional.** Comparing A with C alone cannot separate "the pretraining helped" from
"the model got 21× bigger". Only A against B attributes the difference to the weights.

## 4. Results

Same validation set (our Golgi, 281 tiles), 30 epochs, batch 8, seeds 0/1/2:

| Arm | n | best mean ± sd | last5 mean ± sd |
|---|---|---|---|
| **A** ResNet-50 + CEM500K | 3 | 0.3559 ± 0.0248 | 0.3060 ± 0.0143 |
| **B** ResNet-50 from scratch | 3 | 0.3811 ± 0.0148 | **0.3643 ± 0.0059** |
| **C** small U-Net from scratch | 3 | 0.3317 ± 0.0089 | 0.3108 ± 0.0157 |

| Comparison | Difference | Pooled sd | Effect / sd |
|---|---|---|---|
| A − B (pretraining) | **−0.0583** | 0.0089 | −6.5 |
| B − C (architecture) | **+0.0535** | 0.0097 | **+5.5** |

> The first run had a single seed (A 0.3821 / B 0.3710 / C 0.3280) and A's best was slightly
> *above* B's, which muddied the direction. With three seeds the direction is unambiguous: A is
> lower than B on both best and last5. This is why single-seed comparisons are not worth
> reporting — one seed would have turned a −0.058 effect into an apparent +0.011 gain.

### 4.1 Pretraining does not help (fifth negative result)

**A − B = −0.0583**, 6.5 times the pooled standard deviation. A also has a larger spread
(0.0248 against 0.0148), so the pretrained weights introduce instability as well as bias.

The shape matches earlier rounds: A's best-to-last5 gap (0.050) exceeds B's (0.017), i.e. A
degrades more in the later epochs. That is consistent with pretrained weights anchoring the
optimiser at a point it has to leave.

### 4.2 A larger backbone does help — and the explanation I first gave for it was wrong

**B − C = +0.0535**, 5.5 times the pooled standard deviation. Both arms are randomly initialised;
the only difference is encoder capacity (48.94 M against 2.29 M), so this is a clean architectural
control.

My initial explanation was **insufficient capacity** — 1515 training tiles cannot support a
2.29 M model. That explanation has a falsifiable consequence:

> If capacity is the binding constraint, then on mitochondria, where the small U-Net already
> reaches 0.7849, a larger encoder should add little.

So we ran the same three-arm structure on mitochondria (fixed split `data/mito_split`,
2053 train / 604 val tiles, two seeds per arm, 40 epochs):

| Arm | n | best | last5 |
|---|---|---|---|
| B ResNet-50 (48.94 M) | 2 | 0.8371 ± 0.0006 | **0.8351 ± 0.0009** |
| C small U-Net (2.29 M) | 2 | 0.7896 ± 0.0006 | 0.7849 ± 0.0006 |
| R reference: original 120-epoch baseline | 1 | 0.7911 | 0.7666 |

**B − C = +0.0503** (pooled sd 0.0008, effect/sd = 65.6).

Against Golgi's +0.0535 this is the same within noise. **So the capacity explanation is wrong.**

What survives is simpler and more general: **a larger encoder is worth roughly +0.05 on both
tasks, regardless of difficulty.** That is a weaker mechanistic claim but a stronger empirical
one — it is not the repair of a specific bottleneck, it is a stable general return.

A useful side observation: R (the 120-epoch, batch-16 small U-Net) reaches 0.7911, essentially
the same as C's 0.7896 at 40 epochs. Mitochondria is saturated by 40 epochs and extra epochs do
not help — unlike Golgi, where longer training still gains.

### 4.3 Limitations we have to state

- **Long training still catches up on Golgi.** C reached best 0.4242 at 60 epochs with three
  seeds, above B's 0.3811 at 30. So on Golgi the claim is only "better at equal budget".
  Mitochondria has no such caveat: 40 epochs is saturated and the 120-epoch baseline is lower.
- **Two tasks, two organelles, one dataset.** Results rest on 89 of our own images.
- **Cost is roughly 3× wall-clock.** ResNet-50 runs about 28 s per epoch against 18 s for the
  small U-Net.

## 5. What this means for the project

**Do**: make the ResNet-50 encoder the default. +0.0535 at equal budget, at 3× the time and memory.

**Do not**: keep investing in pretrained weights from external data. This is the fifth negative
result and every one of them reduces to the same thing: the discriminative features of another
domain do not transfer, and it does not matter how heterogeneous that domain is.

**For the report**: this is the only one of the five routes with a positive outcome, and it
happens to support the paper's overall thesis — the bottleneck is model capacity and annotation
volume, not whether external data should be imported.

## 6. Reproduction

```bash
# 1) fetch CEM500K weights (370 MB, CC0)
bash tools/server/scripts/fetch_cem_wget.sh

# 2) verify the encoder loads (should report 98.1%)
python tools/resnet_unet.py --weights models/cem500k/cem500k_mocov2_resnet50_200ep_pth.tar

# 3) three-arm training on Golgi
bash tools/server/scripts/train_cem500k_golgi.sh

# 4) the same on mitochondria (fixed split)
bash tools/server/scripts/build_mito_split.sh
bash tools/server/scripts/mito_arch.sh
```

## 7. Two bugs hit along the way

1. **`ModuleNotFoundError: No module named 'resnet_unet'`** — the module lives in
   `<root>/tools/`, the trainer in `<root>/scripts/`, and my sys.path edit covered neither. Arms A
   and B therefore died two seconds after starting, while the driver carried on to arm C, so the
   log showed only `rc=1` with no cause.
2. **The download failed silently three times** — `curl -sSL` swallows errors,
   `--retry-all-errors` does not exist in curl 7.29 (rc=2), and `wget --show-progress` does not
   exist in the older wget either. The only tell was the byte count failing to match the
   server's `Content-Length`.


---

## 7. Follow-up: a width ablation falsifies the capacity explanation

The sections above place the encoder gain alongside CEM500K's failure, but never tested whether the
gain is simply more parameters. That ablation was run, and it falsifies the capacity explanation
too.

Doubling every channel of the same small U-Net (--base 32 -> 64) raises parameters from 2.29M to
9.16M, four times, under an otherwise identical budget: 30 epochs, batch 8, the same fixed split's
300 validation tiles, three seeds.

| Model | Parameters | best Dice | last5 Dice |
|---|---|---|---|
| small U-Net base=32 | 2.29 M | 0.3317 +- 0.0089 | **0.3108 +- 0.0157** |
| small U-Net base=64 | 9.16 M | 0.3134 +- 0.0512 | **0.2975 +- 0.0538** |
| small U-Net base=128 | 36 M | 0.2746 +- 0.0098 | **0.2633 +- 0.0104** |
| ResNet-50 | 48.94 M | 0.3861 +- 0.0170 | **0.3677 +- 0.0018** |

Stepping the width up, last5 Dice falls monotonically: 0.3108, 0.2975, 0.2633 across a 16-fold
increase in parameters. ResNet-50 at a comparable parameter count, 48.94M against 36M, reaches
0.3677 -- 0.104 above the same-size plain U-Net.

So spending the same parameters differently is what matters: swapping the encoder gains 0.10,
merely widening loses 0.048.

What survives is therefore: the encoder swap is worth about +0.05 over the original small U-Net and
+0.10 over a same-size plain U-Net, independent of task difficulty and not a function of parameter
count. What does the work is the architecture itself -- residuals and batch normalisation, not
available capacity.

This also corrects the paper's earlier claim that what should change is encoder capacity.

Reproduction:

`ash
bash tools/server/scripts/ablation_width.sh

# or a single arm locally
python scripts/train_seg.py --train-root data/golgi_split/train \
    --val-root data/golgi_split/val --classes golgi --size 256 \
    --batch 8 --epochs 30 --lr 3e-4 --base 64 --seed 0 \
    --out runs/abl_base64_s0
`
