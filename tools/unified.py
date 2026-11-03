#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unified.py —— 多数据集统一入口：读取、校验、切片、汇总

## 它解决什么

到目前我们手上同时存在**三种互不兼容的输入形态**，每种的读取方式都不一样：

| 形态 | 图像 | 标注 | 曾被哪些脚本处理 |
|---|---|---|---|
| `green_separated` | 调色板 TIFF（灰度存成 `mode=P`） | 单独的绿色描边图 `green_<id>.tif` | `build_ourdata_tiles.py` |
| `green_overlay` | RGB JPEG，但 R=G=B（实为灰度） | 在同名图的副本上叠加绿色描边 | `build_golgi_tiles.py` |
| `prebuilt_tiles` | 已经是 `<crop>/z####/{em.png,<类>.png}` | 同左 | `train_seg.py` 直接读 |

结果是**每来一批新数据就要写一个新脚本**，而且每个脚本都自己处理
"调色板怎么转灰度""绿描边怎么填洞""按什么划分验证集"这些同样的细节，
实现稍有出入就产生不可比的结果。

本工具把这些收敛成**一份数据声明（`datasets.yaml`）+ 一个读取器**，
新增数据集只需加一段配置，不必再写脚本。

## 兼容性清单（均已在我们真实数据上验证）

* **图像**：调色板 TIFF（PIL `mode=P`，调色板是灰阶斜坡）、RGB、灰度、8/16 位
  —— 统一转 `uint8` 灰度
* **标注**：独立绿色掩码图、叠加式绿色描边、现成二值掩码
* **命名**：同名配对、`green_<id>` 前缀配对
* **尺寸**：图像与标注尺寸不一致时按图像尺寸缩放标注（并记录）
* **尺度**：可选 `target_nm`，把不同 nm/px 的数据统一重采样后再切片
* **划分**：`val_every`（确定性隔 N 取一）或 `val_images`（外部名单）

## 命令

    python tools/unified.py list                 # 列出所有数据集与状态
    python tools/unified.py probe <name|all>     # 只检查不写盘：配对/尺寸/GT覆盖率
    python tools/unified.py build <name|all>     # 生成 tile 数据集

## 输出结构（与既有脚本一致，保证向后兼容）

    <out>/<class>/<id>/z####/{em.png,<class>.png}
    <out>/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import yaml

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SPEC = os.path.join(ROOT, "datasets.yaml")

FACE8 = np.ones((3, 3), bool)


# ==========================================================================
# 一、图像读取（兼容层）
# ==========================================================================
def read_gray(path: str) -> np.ndarray:
    """读任意图像为 uint8 灰度。

    踩过的坑：
    * 我们自己的 EM 是**调色板 TIFF**（PIL `mode='P'`），调色板为灰阶斜坡
      （R=G=B），必须先 `convert('L')` 才是真正的灰度，否则拿到的是索引值
    * 高尔基体数据是 RGB JPEG，但 R=G=B，convert('L') 无损
    * 16 位图若直接 astype(uint8) 会截断高位，这里先按最大值归一化
    """
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I", "F"):
        arr = np.asarray(im, dtype=np.float64)
        mx = arr.max()
        if mx > 255:
            arr = arr / mx * 255.0
        return arr.astype(np.uint8)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def green_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """从 RGB 图里提取绿色描边。

    标注绿约 (132,175,96)，与灰阶背景差异明显。
    判据：G 明显高于 R 与 B，且 G 有足够亮度。
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > 110) & (g > r + 20) & (g > b + 30)


def fill_outline(o: np.ndarray, max_frac: float = 0.6) -> np.ndarray:
    """把闭合描边填成实心。断裂处先做闭运算。

    `max_frac` 是自检：若填出来超过整图 60%，说明描边不闭合导致填成了整片，
    此时退回闭运算结果，避免产生"满图都是目标"的错误 GT。
    """
    if not o.any():
        return o
    c = ndi.binary_closing(o, structure=np.ones((3, 3)), iterations=2)
    f = ndi.binary_fill_holes(c)
    return c if f.mean() > max_frac else f


# ==========================================================================
# 二、标注提取（按 kind 分派）
# ==========================================================================
def extract_gt(cfg: Dict, img_id: str, img_shape: Tuple[int, int]
               ) -> Tuple[Optional[np.ndarray], str]:
    """返回 (GT 布尔图, 说明)。失败时返回 (None, 原因)。"""
    kind = cfg.get("annotation_kind", "green_separated")
    adir = os.path.join(ROOT, cfg["annotations"])
    pattern = cfg.get("annotation_pattern")

    if kind == "prebuilt_tiles":
        return None, "prebuilt_tiles 无需提取"

    # 定位标注文件
    if pattern:
        path = os.path.join(adir, pattern.format(id=img_id))
    else:
        ext = cfg.get("extension", "")
        path = os.path.join(adir, img_id + ext)
    if not os.path.exists(path):
        return None, f"缺标注文件 {os.path.basename(path)}"

    if kind == "green_separated":
        # 独立的绿色描边图：可能存成 RGB 或索引图，统一按 RGB 判绿
        rgb = np.asarray(Image.open(path).convert("RGB"))
        o = green_from_rgb(rgb)
    elif kind == "green_overlay":
        # 叠加式：标注图上同时有原图内容，取绿色即可
        rgb = np.asarray(Image.open(path).convert("RGB"))
        o = green_from_rgb(rgb)
    elif kind == "binary_mask":
        o = np.asarray(Image.open(path).convert("L")) > 127
    else:
        return None, f"未知的 annotation_kind: {kind}"

    if o.shape[:2] != img_shape:
        # 尺寸不一致时把标注缩放到图像尺寸（记录，不静默）
        note = f"尺寸不一致 {o.shape[:2]}->{img_shape}，已缩放"
        o = np.asarray(Image.fromarray((o * 255).astype(np.uint8)).resize(
            (img_shape[1], img_shape[0]), Image.NEAREST)) > 127
        return fill_outline(o), note

    return fill_outline(o), ""


def discover_ids(cfg: Dict) -> List[str]:
    """找出图像与标注都能配上的 id 列表。"""
    idir = os.path.join(ROOT, cfg["images"])
    if not os.path.isdir(idir):
        return []
    ext = cfg.get("extension", "")
    if not ext:
        # 未指定扩展名时按常见后缀探测
        for cand in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
            if glob.glob(os.path.join(idir, "*" + cand)):
                ext = cand
                break
    files = [f for f in os.listdir(idir) if f.lower().endswith(ext)
             and not f.startswith(".")]
    return sorted(os.path.splitext(f)[0] for f in files)


# ==========================================================================
# 三、切片
# ==========================================================================
def tile_starts(total: int, tile: int, stride: int) -> List[int]:
    if total <= tile:
        return [0]
    s = list(range(0, total - tile + 1, max(1, stride)))
    if s[-1] != total - tile:
        s.append(total - tile)
    return s


def resample_pair(img: np.ndarray, gt: np.ndarray, nm: float,
                  target: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    if not target or nm <= 0 or abs(nm - target) < 1e-6:
        return img, gt, 1.0
    f = nm / target
    h, w = img.shape
    nw, nh = max(16, int(round(w * f))), max(16, int(round(h * f)))
    i2 = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    g2 = np.asarray(Image.fromarray((gt * 255).astype(np.uint8))
                    .resize((nw, nh), Image.BILINEAR)) > 127
    return i2, g2, f


def load_scale_map(cfg: Dict) -> Dict[str, float]:
    """按图读取像素尺寸（单位 nm）。

    ## 为什么需要这个

    我们自有 EM 数据有三种尺度（1.004 / 2.667 / 5.291 nm/px），是**逐图不同**的。
    而 `build_ourdata_tiles.py` 当时靠 `annotations_v2/dataset.json` 逐图查表，
    再把所有图统一重采样到 4 nm/px；如果统一层只支持一个标量 `nm_per_px`，
    就无法复现那次的切片结果（tile 数会不同）。

    这里支持 `scale_json`：一个含 `images[].um_per_px` 的 JSON
    （结构与 `annotations_v2/dataset.json` 一致），返回 id -> nm 的映射。
    """
    p = cfg.get("scale_json")
    if not p:
        return {}
    path = os.path.join(ROOT, p)
    if not os.path.exists(path):
        print(f"    [警告] scale_json 不存在 {p}，按无标定处理")
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"    [警告] 读 scale_json 失败 {type(exc).__name__}")
        return {}
    ents = d.get("images")
    if isinstance(ents, dict):
        ents = list(ents.values())
    out: Dict[str, float] = {}
    for e in ents or []:
        um = e.get("um_per_px")
        if um:
            out[str(e.get("image"))] = float(um) * 1000.0
    return out


# ==========================================================================
# 四、probe / build
# ==========================================================================
def probe_one(name: str, cfg: Dict, verbose: bool = True) -> Dict:
    ids = discover_ids(cfg)
    cls = cfg.get("class", "organelle")
    info = {"name": name, "class": cls, "n_ids": len(ids),
            "rows": [], "skipped": []}
    if verbose:
        print(f"--- {name}  (class={cls}, kind={cfg.get('annotation_kind')})")
        print(f"    图像 {cfg['images']}  标注 {cfg['annotations']}")
    for i in ids:
        ip = os.path.join(ROOT, cfg["images"], i + cfg.get("extension", ""))
        try:
            img = read_gray(ip)
        except Exception as exc:  # noqa: BLE001
            info["skipped"].append((i, f"读图失败 {type(exc).__name__}"))
            continue
        gt, note = extract_gt(cfg, i, img.shape)
        if gt is None:
            info["skipped"].append((i, note))
            continue
        if gt.sum() < cfg.get("min_gt_px", 200):
            info["skipped"].append((i, f"GT 过小 {int(gt.sum())} px"))
            continue
        info["rows"].append({"id": i, "shape": list(img.shape),
                             "gt_frac": float(gt.mean()), "note": note})
    if verbose:
        fracs = [r["gt_frac"] for r in info["rows"]]
        print(f"    可用 {len(info['rows'])} / {len(ids)} 组；跳过 {len(info['skipped'])}")
        if fracs:
            print(f"    GT 覆盖率 中位 {100*np.median(fracs):.2f}%  "
                  f"范围 [{100*min(fracs):.2f}%, {100*max(fracs):.2f}%]")
            shapes = {tuple(r["shape"]) for r in info["rows"]}
            print(f"    图像尺寸 {len(shapes)} 种，"
                  f"{'示例 ' + str(sorted(shapes)[0]) if shapes else ''}")
        for i, why in info["skipped"][:5]:
            print(f"    [跳过] {i}: {why}")
    return info


def build_one(name: str, cfg: Dict, out_dir: str, args) -> Dict:
    cls = cfg.get("class", "organelle")
    ids = discover_ids(cfg)
    val_every = cfg.get("val_every", args.val_every)
    val_ids: set = set()
    vm = cfg.get("val_images")
    if vm:
        p = os.path.join(ROOT, vm)
        if os.path.exists(p):
            val_ids = {l.strip() for l in open(p, encoding="utf-8") if l.strip()}
    elif val_every and val_every > 1:
        val_ids = {i for k, i in enumerate(ids) if k % val_every == 0}

    rng = np.random.RandomState(args.seed)
    T = cfg.get("tile", args.tile)
    S = cfg.get("stride", args.stride)
    min_pos = cfg.get("min_pos_frac", args.min_pos_frac)
    neg_ratio = cfg.get("neg_ratio", args.neg_ratio)
    target_nm = cfg.get("target_nm", args.target_nm)
    scale_map = load_scale_map(cfg)
    if scale_map:
        print(f"    按图查表得到像素尺寸 {len(scale_map)} 条；"
              f"目标尺度 {target_nm}")

    # 输出路径约定：**始终** <out>/<name>_tiles，`--out` 只作为父目录（缺省 data）。
    #
    # 早期版本在给了 --out 时写成 <out>/<name>、没给时写成 data/<name>_tiles。
    # 两套约定导致 run_pipeline 传了 --out 后产物落到 data/ourdata_mito，
    # 而状态检查找的是 data/ourdata_mito_tiles，于是永远显示"待运行"。
    out_root = os.path.join(ROOT, out_dir or "data", f"{name}_tiles")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)

    counts = {"pos": 0, "neg": 0, "imgs": 0, "skipped": 0}
    manifest: List[Dict] = []
    print(f"--- 构建 {name}: class={cls} tile={T} stride={S} "
          f"val={len(val_ids)} 张")

    for i in ids:
        sub = "val" if i in val_ids else "train"
        ip = os.path.join(ROOT, cfg["images"], i + cfg.get("extension", ""))
        try:
            img = read_gray(ip)
        except Exception:  # noqa: BLE001
            counts["skipped"] += 1
            continue
        gt, _ = extract_gt(cfg, i, img.shape)
        if gt is None or gt.sum() < cfg.get("min_gt_px", 200):
            counts["skipped"] += 1
            continue

        # 像素尺寸：优先按图查表（scale_json），否则用数据集标量
        nm = scale_map.get(i) or cfg.get("nm_per_px") or 0.0
        img, gt, factor = resample_pair(img, gt, nm, target_nm)

        h, w = img.shape
        pos, neg = [], []
        for y0 in tile_starts(h, T, S):
            for x0 in tile_starts(w, T, S):
                g = gt[y0:y0 + T, x0:x0 + T]
                if g.shape != (T, T):
                    continue
                (pos if g.mean() >= min_pos else neg).append((y0, x0))
        k = int(round(len(pos) * neg_ratio))
        # 负样本策略。
        #
        # `all_if_trusted`：标注覆盖率 ≥ trust_coverage 的图视为"标注较完整"，
        # 其全部背景 tile 都保留；覆盖率低的图（可能是部分标注）只按 neg_ratio 抽样。
        #
        # 这条策略来自 `build_ourdata_tiles.py`，当时是为应对部分标注：
        # 覆盖率高的图里"没被标注的线粒体"少，背景可信；覆盖率低的图里
        # 大量真实目标未标注，全留会把它们当负样本教坏模型。
        # 不实现这条就无法复现既有的 2657 个 tile，故做成可声明的策略。
        policy = cfg.get("neg_policy", args.neg_policy)
        trust = float(cfg.get("trust_coverage", args.trust_coverage))
        if policy == "all_if_trusted" and float(gt.mean()) >= trust:
            keep_neg = neg
        else:
            keep_neg = ([neg[j] for j in rng.choice(len(neg), min(k, len(neg)),
                                                    replace=False)]
                        if neg and k > 0 else [])

        crop_dir = os.path.join(out_root, cls, i)
        idx = 0
        posset = set(pos)
        # 若源目录里存在 ignore.png（与图像同尺寸），随切片一并带出。
        # 用途：标注不完整/不可信的区域不计入损失（见 train_seg 的 ignore 机制）。
        ign_full = None
        ign_src = os.path.join(ROOT, cfg["annotations"], f"{i}_ignore.png")
        if not os.path.exists(ign_src):
            alt = os.path.join(ROOT, cfg["annotations"], "ignore", f"{i}.png")
            ign_src = alt if os.path.exists(alt) else ign_src
        if os.path.exists(ign_src):
            im = Image.open(ign_src).convert("L")
            if im.size == (w, h):
                ign_full = np.asarray(im) > 127
            else:
                ign_full = np.asarray(im.resize((w, h), Image.NEAREST)) > 127
        for y0, x0 in pos + keep_neg:
            is_pos = (y0, x0) in posset
            d = os.path.join(crop_dir, f"z{idx:04d}")
            os.makedirs(d, exist_ok=True)
            Image.fromarray(img[y0:y0 + T, x0:x0 + T]).save(
                os.path.join(d, "em.png"))
            Image.fromarray((gt[y0:y0 + T, x0:x0 + T].astype(np.uint8) * 255)).save(
                os.path.join(d, f"{cls}.png"))
            if ign_full is not None:
                Image.fromarray(
                    (ign_full[y0:y0 + T, x0:x0 + T].astype(np.uint8) * 255)
                ).save(os.path.join(d, "ignore.png"))
            idx += 1
            counts["pos" if is_pos else "neg"] += 1
        counts["imgs"] += 1
        manifest.append({"dataset": name, "class": cls, "id": i, "split": sub,
                         "shape": [int(h), int(w)], "factor": round(float(factor), 4),
                         "gt_frac": round(float(gt.mean()), 5),
                         "n_pos": len(pos), "n_neg": len(keep_neg)})
        print(f"    {i:<10} {sub:<5} {w}x{h:<6} GT {100*gt.mean():5.2f}%  "
              f"pos {len(pos):>3}  neg {len(keep_neg):>3}")

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"dataset": name, "class": cls, "tile": T, "stride": S,
                   "target_nm": target_nm, "counts": counts,
                   "images": manifest}, fh, ensure_ascii=False, indent=1)

    print(f"    合计 正 {counts['pos']} / 负 {counts['neg']}，"
          f"图像 {counts['imgs']}，跳过 {counts['skipped']}")
    print(f"    输出 {os.path.relpath(out_root, ROOT)}")
    return {"name": name, "out": os.path.relpath(out_root, ROOT), **counts}


# ==========================================================================
# 五、CLI
# ==========================================================================
def load_spec(path: str) -> Dict:
    if not os.path.exists(path):
        print(f"[错误] 找不到数据集声明 {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    return (spec or {}).get("datasets") or {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="多数据集统一入口：校验与切片")
    ap.add_argument("cmd", choices=["list", "probe", "build"],
                    help="list=列出；probe=只检查；build=生成 tile")
    ap.add_argument("name", nargs="?", default="all",
                    help="数据集名，或 all")
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--out", default="", help="输出根目录（默认 data/<name>_tiles）")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--stride", type=int, default=192)
    ap.add_argument("--min-pos-frac", type=float, default=0.01)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--neg-policy", default="all_if_trusted",
                    choices=["all_if_trusted", "sampled"],
                    help="负样本策略：all_if_trusted=覆盖率高的图保留全部背景 tile"
                         "（为应对部分标注）；sampled=统一按 neg_ratio 抽样")
    ap.add_argument("--trust-coverage", type=float, default=0.05,
                    help="neg_policy=all_if_trusted 时的覆盖率阈值")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--target-nm", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if ndi is None:
        print("[错误] 需要 scipy", file=sys.stderr)
        return 2

    spec = load_spec(args.spec)
    if not spec:
        print("[错误] 声明里没有 datasets", file=sys.stderr)
        return 2

    names = list(spec) if args.name == "all" else [args.name]
    for n in names:
        if n not in spec:
            print(f"[错误] 未知数据集 {n}；可选 {list(spec)}", file=sys.stderr)
            return 2

    if args.cmd == "list":
        print(f"{'数据集':<16} {'类别':<8} {'标注方式':<18} {'图像数':>6} {'可用':>5}")
        print("-" * 62)
        for n in names:
            cfg = spec[n]
            ids = discover_ids(cfg)
            info = probe_one(n, cfg, verbose=False)
            print(f"{n:<16} {cfg.get('class','?'):<8} "
                  f"{cfg.get('annotation_kind','?'):<18} "
                  f"{len(ids):>6} {len(info['rows']):>5}")
        return 0

    if args.cmd == "probe":
        for n in names:
            probe_one(n, spec[n], verbose=True)
            print()
        return 0

    results = []
    for n in names:
        results.append(build_one(n, spec[n], args.out, args))
        print()
    print("=" * 62)
    for r in results:
        print(f"  {r['name']:<16} 正 {r['pos']:>4} 负 {r['neg']:>4} "
              f"图 {r['imgs']:>3} 跳过 {r['skipped']:>2}  -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
