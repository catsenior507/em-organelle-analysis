# Using OpenOrganelle: coordinates, alignment, costs and six traps

**English** ｜ [中文](PUBLIC-DATA-OPENORGANELLE.zh-CN.md)

> Practical notes for pulling ground-truth organelle labels out of the Janelia COSEM public data.
> Written because every step of this cost us a debugging session, and the traps are not documented
> anywhere else.

---

## 1. What the data is

`OpenOrganelle` publishes 105 volume-EM datasets at **`s3://janelia-cosem-datasets`** — public, no
credentials, no requester-pays. Licence is CC0.

Each dataset `<ds>` contains:

| Path | Format | Usable? |
|---|---|---|
| `<ds>/<ds>.n5/labels/` | N5 with a z5 custom header | Not directly — the header is not raw gzip |
| `<ds>/<ds>.zarr/recon-1/labels/groundtruth/<crop>/<channel>/s0/` | **standard zarr v2 + zstd** | **This is the one to use** |

The zarr path has `dimension_separator: "/"`, 64³ chunks, zstd compression. Reads cleanly with
`zarr` + `numcodecs`.

## 2. Coordinates: where in the volume is a crop?

The annotations are 3D crops, not whole specimens. Their location is **not** in the metadata you
would expect. It is in the Neuroglancer view states:

```
<ds>/neuroglancer/viewer_states/neuroglancer-<ds>-<crop>.json
```

Inside, the `position` field holds (x, y, z) in **nanometres**, and it is the **centre** of the
crop, not its origin. Getting this backwards shifts every extraction by half a crop.

`tools/oo_locate.py` parses these into EM voxel ranges and writes `logs/oo_locate.json`.

## 3. Alignment: prove it, do not assume it

Before trusting any of it we rendered the EM with the mask overlaid. `tools/verify_oo_alignment.py`
does this, handling the crop-local versus EM-global coordinate conversion via `crop_origin()`.

**What correct alignment looks like**: the printed result (`temp/oo_align3/align.png`, Golgi at
crop4 z=3905) shows the mask covering the parallel cisternal bands exactly, with no offset.

**Always render one slice before extracting gigabytes.** A half-voxel or half-crop offset still
produces plausible-looking masks and a completely wrong dataset.

## 4. Costs, measured

For the four crops we use:

| Item | Cost |
|---|---|
| Listing all 105 datasets under `.n5/labels/` | ~6 min |
| Inventorying one crop's channels (real chunks, not just directories) | ~2 min |
| Fetching four crops × ~6 channels | ~40 min |
| Fetching the matching EM volumes (dense, no key listing) | ~10 min/crop |

`tools/oo_scan.py` does the first, `tools/oo_inventory.py` the second, `tools/oo_fetch_crops.py` the
third and fourth.

## 5. Six traps

**1. The N5 labels are not readable as plain N5.** The header is a z5 extension. Use the zarr mirror.

**2. `list_keys` needs a key prefix, not a URL.** Passing a full URL returns zero keys, and the
downloader then writes a correctly-shaped array of all zeros while reporting success at every step.
The tell is foreground fraction 0.00 %. We lost 1.7 GB to this.

**3. Listing a dense EM volume is a waste of time.** Paginating 127 033 keys took 495–721 s per
volume. The fetcher now takes a `dense=True` path that walks the chunk grid by arithmetic instead.
Listing the sparse label channels, by contrast, is worth it — most chunks are empty.

**4. Directory existence does not imply data.** Several channels have the directory structure and
zero real chunks. `oo_inventory.py` checks for actual chunk presence; `crop48` turned out to have a
Golgi channel that is 0.05 % foreground, effectively empty.

**5. Golgi is only usable in four crops.** Of 105 datasets, real Golgi annotation with usable
coverage exists in three crops (`jrc_hela-2/crop4` 6.12 %, `jrc_jurkat-1/crop37` 29.17 %,
`jrc_macrophage-2/crop39` 25.63 %); `crop48` is 0.05 % and effectively empty.

**6. `find` does not follow symlinks.** After building the label tree with symlinks, `find` counted
zero files and we concluded the tree was empty. Use `find -L`. This mistake cost us time twice.

## 6. What we got out of it

| Crop | Extent | Classes with real annotation |
|---|---|---|
| `jrc_hela-2/crop4` | 476×600×600 | golgi, er, endo, ves, lyso *(no mito)* |
| `jrc_jurkat-1/crop37` | 400³ | golgi, er, endo, ves, lyso *(mito 0.20 %)* |
| `jrc_macrophage-2/crop39` | 400³ | golgi, er, endo, ves, lyso |
| `jrc_hela-2/crop48` | 512³ | Golgi effectively empty |

All at 4 nm isotropic. These four volumes are what the multi-class contact analysis
(`CONTACT-MULTICLASS-OO.md`) and the polarity analysis (`CONTACT-POLARITY.md`) are built on.

## 7. Licence and citation

CC0, so no restrictions on reuse. The originating publication is Heinrich et al., *Nature* 2021,
599, 141–146.
