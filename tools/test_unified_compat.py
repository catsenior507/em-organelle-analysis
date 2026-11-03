#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_unified_compat.py —— 统一层的兼容性回归测试

## 为什么必须有这个测试

`tools/unified.py` 的目标是"用一份声明取代每批数据一个新脚本"。
但只有当它**产出的结果与既有脚本一致**时，这个替代才是安全的；
否则所有基于旧切片训练出来的模型、Dice 数字都会变得不可复现。

这里把"一致"写成可执行的断言：

1. **高尔基体**：统一层必须精确复现 `build_golgi_tiles.py` 的
   976 正 / 820 负 / 51 图（同一份 spec 参数下）
2. **线粒体**：逐图比对 `data/ourdata_tiles/manifest.json`（旧脚本产物），
   除 `12_0002` 外每个图的 pos/neg 计数必须完全相同

`12_0002` 是**已知且经过确认的改进**，不是回归：
它的标注图是转置的（2048×2115 对 2115×2048），
旧脚本 `if o.shape != img.shape: continue` 把这张有标注的图整张丢弃，
统一层改为把标注缩放到图像尺寸，因此多出 1 张图、121 个 tile。

## 运行

    python tools/test_unified_compat.py

需要先跑过一次 `python tools/unified.py build all --out temp/unified_check`
（本脚本会自己调用）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(ROOT, "temp", "unified_check")
VENV = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
PY = VENV if os.path.exists(VENV) else sys.executable

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def build(name: str) -> None:
    r = subprocess.run([PY, os.path.join(ROOT, "tools", "unified.py"),
                        "build", name, "--out", "temp/unified_check"],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"  [错误] 构建 {name} 失败: {r.stderr[-500:]}")
        raise SystemExit(1)


def load_manifest(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    print("=" * 74)
    print("统一层兼容性测试")
    print("=" * 74)

    print("\n[1] 构建两个数据集")
    for n in ("ourdata_golgi", "ourdata_mito"):
        build(n)
        print(f"  {n} 构建完成")

    # ---------------- 高尔基体 ----------------
    print("\n[2] 高尔基体：必须精确复现旧脚本的 tile 计数")
    g = load_manifest(os.path.join(TMP, "ourdata_golgi_tiles", "manifest.json"))
    c = g["counts"]
    check("正 tile = 976", c["pos"] == 976, f"got {c['pos']}")
    check("负 tile = 820", c["neg"] == 820, f"got {c['neg']}")
    check("图像 = 51", c["imgs"] == 51, f"got {c['imgs']}")

    # ---------------- 线粒体 ----------------
    print("\n[3] 线粒体：逐图与旧脚本比对（12_0002 为已知改进）")
    old_p = os.path.join(ROOT, "data", "ourdata_tiles", "manifest.json")
    if not os.path.exists(old_p):
        print(f"  SKIP  找不到旧 manifest {old_p}（旧 tile 数据集可能已清理）")
    else:
        old = load_manifest(old_p)
        new = load_manifest(os.path.join(TMP, "ourdata_mito_tiles", "manifest.json"))

        def agg_from_tiles(m):
            d = {}
            for t in m.get("tiles", []):
                i = t["image"]
                d.setdefault(i, [0, 0])
                d[i][0 if t["is_positive"] else 1] += 1
            return d

        def agg_from_images(m):
            return {im["id"]: [im["n_pos"], im["n_neg"]] for im in m["images"]}

        o = agg_from_tiles(old)
        n = agg_from_images(new)
        only_new = sorted(set(n) - set(o))
        only_old = sorted(set(o) - set(n))
        check("没有凭空丢失的图像", not only_old, f"只旧有 {only_old}")
        check("多出的图像只有 12_0002", only_new == ["12_0002"], f"只新有 {only_new}")

        mism = {i: (o[i], n[i]) for i in sorted(set(o) & set(n)) if o[i] != n[i]}
        check("共有图像的 pos/neg 完全一致", not mism,
              f"不一致 {list(mism)[:5]}" if mism else f"{len(set(o)&set(n))} 张")
        if mism:
            for i, (a, b) in list(mism.items())[:5]:
                print(f"      {i}: 旧 {a} vs 新 {b}")

        tot_o = sum(a + b for a, b in o.values())
        tot_n = sum(a + b for a, b in n.values())
        check("总 tile 数差异 = 仅 12_0002 的贡献", tot_n - tot_o == 121,
              f"旧 {tot_o} 新 {tot_n} 差 {tot_n - tot_o}")

    # ---------------- 结构 ----------------
    print("\n[4] 输出结构符合 train_seg.py 的期望")
    for n, cls in (("ourdata_golgi", "golgi"), ("ourdata_mito", "mito")):
        m = load_manifest(os.path.join(TMP, n, "manifest.json"))
        i0 = m["images"][0]["id"]
        d0 = os.path.join(TMP, n + "_tiles", cls, i0, "z0000")
        check(f"{n}: 存在 <class>/<id>/z0000/em.png",
              os.path.exists(os.path.join(d0, "em.png")))
        check(f"{n}: 存在 <class>/<id>/z0000/{cls}.png",
              os.path.exists(os.path.join(d0, f"{cls}.png")))

    print()
    print("=" * 74)
    if FAILS:
        print(f"失败 {len(FAILS)} 项：{FAILS}")
        return 1
    print("全部通过——统一层与既有脚本的产出相符")
    print("（唯一差异 12_0002 是修复了旧脚本丢弃合法图像的问题）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
