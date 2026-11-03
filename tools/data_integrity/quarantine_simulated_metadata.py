#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quarantine_simulated_metadata.py —— 隔离 `simulate_structure_data.py` 生成的伪造实验元数据

## 问题

`code/simulate_structure_data.py` 会生成一个"看起来像实验记录"的
`data/structure/structure_metadata.json`，其中包含**从未被测量过**的字段：

- `actual_date`：例如 `"2024-01-01T10:36:00"`（由 `start_date = datetime(2024,1,1,9,0,0)`
  加随机时间间隔生成）
- `perturbation_level`：伪造的扰动等级 1–7
- `experiment_conditions_detail`：形如
  `"chemical: 0.384, concentration: 0.668, temperature: 44.305"` 的伪造实验条件
  （由 `25 + level*5 + random.gauss(0,1)` 之类的公式合成）

而该文件的 `metadata` 里写着::

    "description": "High-quality simulated structure data with comprehensive attributes",
    "generation_method": "realistic_simulation_with_statistics",
    "purpose": "Scientific research and publication"

也就是说，**脚本自己承认是 simulation，但产物被标注为用于发表**。
这些字段随后会被 `analyze_emergent_entropy.m` 当作真实的时间戳与扰动等级
读入（`structure_metadata.json` → `timestamps` / `perturbation_levels`），
进而参与"时间序列/混沌"分析。

这不是代码 bug，而是**科研诚信问题**：任何进入竞赛记录、论文或答辩的
图表，都不能建立在合成的时间戳和实验条件之上。

## 本脚本做什么

它**不删除任何东西**（原始文件保留在 git 首次提交里），而是：

1. 把文件移到 `data/_QUARANTINE_simulated_metadata/`；
2. 在旁边写一份 `DO_NOT_USE.md`，说明为什么不能用、哪些字段是伪造的、
   要恢复真实分析需要什么；
3. 生成一份 `provenance.json`，记录原始路径、哈希、字段清单，
   以便将来需要时逐字段溯源。

## 用法

    python tools/data_integrity/quarantine_simulated_metadata.py          # 预演
    python tools/data_integrity/quarantine_simulated_metadata.py --apply  # 实际执行
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

#: 已知由 simulate_structure_data.py 生成、且含伪造字段的元数据文件
SIMULATED_ARTIFACTS: List[str] = [
    "data/structure/structure_metadata.json",
]

QUARANTINE_DIR = "data/_QUARANTINE_simulated_metadata"

#: 明确列举哪些字段不是实测值
FABRICATED_FIELDS = {
    "actual_date": "由 datetime(2024,1,1,9,0,0) + 随机间隔生成，非采集时间",
    "timestamp": "由序号推导，非采集时间",
    "perturbation_level": "由 random / 线性公式合成的扰动等级，非实验设定值",
    "experiment_conditions_detail": (
        "温度/浓度/化学等条件由 25+level*5+gauss(0,1) 之类公式合成，非实测"
    ),
    "sequence_length": "由文件计数推导",
    "index_in_sequence": "由文件排序推导",
}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    info: Dict[str, Any] = {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "size_bytes": os.path.getsize(path),
        "top_level_keys": sorted(data.keys()),
    }
    meta = data.get("metadata", {})
    info["declared_metadata"] = meta
    info["declared_purpose"] = meta.get("purpose")
    info["declared_generation_method"] = meta.get("generation_method")

    samples = data.get("samples", {})
    info["num_samples"] = len(samples)
    if samples:
        first_key = next(iter(samples))
        info["sample_keys"] = sorted(samples[first_key].keys())
        info["sample_example"] = samples[first_key]
    info["fabricated_fields_present"] = sorted(
        k for k in FABRICATED_FIELDS if k in (info.get("sample_keys") or [])
    )
    return info


def write_readme(dest_dir: str, infos: List[Dict[str, Any]]) -> str:
    lines = [
        "# ⛔ 请勿使用本目录中的数据",
        "",
        "本目录中的文件是 `code/simulate_structure_data.py` **模拟生成**的，",
        "其中包含并非实测的实验条件与时间戳。",
        "",
        "## 为什么被隔离",
        "",
        "该项目是「基于深度学习和混沌理论的细胞器互作网络重建与涌现特征分析」，",
        "所有进入竞赛记录、论文与答辩的图表都必须建立在真实测量之上。",
        "而这些文件里的以下字段来自随机数生成器，不是任何一次真实实验的记录：",
        "",
    ]
    for field, why in FABRICATED_FIELDS.items():
        lines.append(f"- `{field}` —— {why}")
    lines += [
        "",
        "文件自身的 `metadata` 写着 `\"purpose\": \"Scientific research and publication\"`，",
        "而 `\"generation_method\": \"realistic_simulation_with_statistics\"` —— ",
        "即脚本自己承认是模拟，却把产物标注为用于发表。",
        "",
        "## 这些字段是怎么被用上的",
        "",
        "`code/analyze_emergent_entropy.m` 会读取 `structure_metadata.json`，",
        "把 `timestamps` 当作时间序列的时间轴、把 `perturbation_levels` 当作",
        "扰动等级，进而参与所谓\"李雅普诺夫指数\"与\"扰动敏感性\"的计算。",
        "换句话说，**合成的实验条件直接进入了声称具有物理意义的分析**。",
        "",
        "## 原始来源与哈希",
        "",
    ]
    for info in infos:
        lines += [
            f"- 原路径：`{info['path']}`",
            f"  - sha256：`{info['sha256']}`",
            f"  - 样本数：{info['num_samples']}",
            f"  - 声明的 purpose：`{info.get('declared_purpose')}`",
            f"  - 实际存在于样本中的伪造字段：{info['fabricated_fields_present']}",
        ]
    lines += [
        "",
        "## 要恢复真实的分析，需要什么",
        "",
        "1. **真实的采集时间**：从显微镜元数据（TIFF 的 `DateTime` 标签、",
        "   OME-XML、或仪器导出的 acquisition log）读取，而不是从文件名序号推导。",
        "2. **真实的扰动条件**：实验记录本上的分组/浓度/时间点，人工录入并签名。",
        "3. **确认这批图到底是不是时间序列**：目前 `data/normalized_features` 的 88 个 CSV",
        "   按前缀分组后只有 `11→13、12→49、13→2、18→20、21→4` 个时间点，",
        "   其中三组只有 2–4 个点，不足以做任何动力学分析。",
        "4. **重跑 `analyze_emergent_entropy.m`** 前，先把 §2.3 里列出的问题",
        "   （伪造的李雅普诺夫指数、log₂(样本数) 冒充互信息、硬编码查表）一并修掉，",
        "   否则换成真数据也得不到有意义的结论。",
        "",
        "## 恢复方式",
        "",
        "文件未被删除。需要查看原文时，可从 git 首次提交 `aeef301` 取回：",
        "",
        "```bash",
        "git show aeef301:data/structure/structure_metadata.json | head -40",
        "```",
        "",
        f"隔离时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    path = os.path.join(dest_dir, "DO_NOT_USE.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="隔离 simulate_structure_data.py 生成的伪造实验元数据"
    )
    parser.add_argument("--apply", action="store_true",
                        help="实际执行移动；默认只预演")
    parser.add_argument("--root", default=".", help="项目根目录")
    args = parser.parse_args(argv)

    os.chdir(args.root)

    infos: List[Dict[str, Any]] = []
    for rel in SIMULATED_ARTIFACTS:
        info = inspect(rel)
        if info is None:
            print(f"  [跳过] 不存在：{rel}")
            continue
        infos.append(info)
        print(f"  发现：{rel}")
        print(f"        样本数 {info['num_samples']}，"
              f"声明的 purpose = {info.get('declared_purpose')!r}")
        print(f"        伪造字段：{info['fabricated_fields_present']}")

    if not infos:
        print("没有发现需要隔离的文件。")
        return 0

    if not args.apply:
        print("\n以上为预演。加 --apply 实际执行隔离。")
        return 0

    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    moved: List[str] = []
    for info in infos:
        src = info["path"]
        dst = os.path.join(QUARANTINE_DIR, os.path.basename(src))
        shutil.move(src, dst)
        moved.append(dst)
        print(f"  已移动：{src} -> {dst}")

    readme = write_readme(QUARANTINE_DIR, infos)
    provenance = os.path.join(QUARANTINE_DIR, "provenance.json")
    with open(provenance, "w", encoding="utf-8") as fh:
        json.dump({
            "quarantined_at": datetime.now().isoformat(timespec="seconds"),
            "reason": "simulated experimental metadata (timestamps, perturbation "
                      "levels, experiment conditions) must not enter published analysis",
            "generator_script": "code/simulate_structure_data.py",
            "files": infos,
        }, fh, ensure_ascii=False, indent=1)

    print(f"\n已写入说明：{readme}")
    print(f"已写入溯源：{provenance}")
    print(f"共隔离 {len(moved)} 个文件（原始内容仍在 git 提交 aeef301 中）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
