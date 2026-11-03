#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch.py —— 把公开细胞器数据集抓到外接盘（可断点续传、带空间预检、写清单）

## 为什么需要它

本项目的训练数据只有 89 张图、570 个区域，而且**所有区域都是同一个类别 "M"**
（线粒体）。这一条硬约束使得"细胞器互作网络"里的"多类互作"完全没有数据基础：
没有第二类细胞器，就谈不上哪两类之间有接触/共定位。

公开数据能补上这一块，最直接的是 CellMap Segmentation Challenge：
73 个类别里同时含 `mito` 与 `er`，正好支撑 **mito–ER 接触位点**统计。

## 设计要点

1. **空间预检**：下载前检查目标盘剩余空间，不够就直接拒绝，不会写到一半磁盘满。
2. **断点续传**：用 HTTP `Range` + `.part` 文件。S3 与 EMPIAR FTP-over-HTTPS
   都返回 `206 Partial Content`（已实测），所以断网/中断后可继续。
3. **尺寸校验**：下完比对字节数；与 catalog 里核实过的 `size_bytes` 不符则报错，
   不留下"看起来下完了但其实是截断"的文件。
4. **清单留痕**：每个数据集目录写 `dataset.json`，根目录写 `manifest.json`，
   记录 url、大小、sha256、抓取时间、以及本文件版本——便于论文里说明数据来源。

## 用法

    # 看目录（含已核实的体积与优先级）
    python tools/data_acquire/fetch.py --list

    # 只探测可达性与真实体积，不下载
    python tools/data_acquire/fetch.py --target W:\\dachuang_public_data --dry-run

    # 下载第一优先级数据（约 42 GB）
    python tools/data_acquire/fetch.py --target W:\\dachuang_public_data --only cellmap_challenge_matched

    # 下载全部 tier<=2
    python tools/data_acquire/fetch.py --target W:\\dachuang_public_data --max-tier 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "catalog.json")
USER_AGENT = "dachuang-data-acquire/1.0 (academic use)"

CHUNK = 1 << 20          # 1 MiB 读写块
PROGRESS_EVERY = 8       # 每 N 秒打印一次进度


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def human(n: Optional[float]) -> str:
    if n is None:
        return "未知"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def load_catalog() -> Dict[str, Any]:
    with open(CATALOG, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get(url: str, *, headers: Optional[Dict[str, str]] = None,
        timeout: int = 60) -> urllib.request.addinfourl:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)


def remote_size(url: str) -> Optional[int]:
    """用 Range 请求探测真实字节数（同时验证服务器支持断点续传）。"""
    try:
        with get(url, headers={"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range") or ""
            # 形如 bytes 0-0/42020987070
            if "/" in cr:
                total = cr.rsplit("/", 1)[1].strip()
                if total.isdigit():
                    return int(total)
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            print(f"    [!] 不可访问 {exc.code}: {url}")
            return None
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"    [!] 探测失败 {type(exc).__name__}: {exc}")
        return None


def free_space(path: str) -> int:
    """目标路径所在卷的剩余字节数。路径不存在时向上找到最近的已存在祖先。"""
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free


def sha256_of(path: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 目录列举（EMPIAR 的 Apache 索引 / BioStudies 的 MCZ 清单）
# --------------------------------------------------------------------------

_APACHE_ROW = re.compile(
    r'<tr>.*?<a href="(?P<href>[^"]+)".*?</a></td>'
    r'.*?<td align="right">(?P<date>[^<]*)</td>'
    r'\s*<td align="right">\s*(?P<size>[^<]*?)\s*</td>',
    re.S,
)

_SIZE_UNITS = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_apache_index(html: str) -> List[Dict[str, Any]]:
    """解析 Apache/nginx 风格的目录索引。

    返回 ``[{name, href, is_dir, size_str, size_bytes, date}]``。
    注意：``size_str`` 是列表页显示的**近似值**（如 ``32M``），
    精确值需对单文件发 HEAD（见 ``--exact``）。
    """
    out: List[Dict[str, Any]] = []
    for m in _APACHE_ROW.finditer(html):
        href = m.group("href")
        if href.startswith("?") or href.startswith("/") or href in ("..", ""):
            continue
        is_dir = href.endswith("/")
        size_str = (m.group("size") or "").strip()
        size = None
        if size_str and size_str != "-":
            mm = re.match(r"([\d.]+)\s*([KMGT]?)", size_str)
            if mm:
                val = float(mm.group(1))
                unit = mm.group(2)
                size = int(val * _SIZE_UNITS.get(unit, 1)) if unit else int(val)
        out.append({
            "name": urllib.parse.unquote(href).rstrip("/"),
            "href": href,
            "is_dir": is_dir,
            "size_str": size_str,
            "size_bytes": size,
            "date": (m.group("date") or "").strip(),
        })
    return out


def walk_http_tree(base_url: str, max_depth: int = 6) -> List[Dict[str, Any]]:
    """递归列举一个 HTTP 目录树，返回所有文件（含相对路径与大小）。"""
    files: List[Dict[str, Any]] = []
    queue: List[Tuple[str, int]] = [(base_url.rstrip("/") + "/", 0)]
    seen = set()
    while queue:
        url, depth = queue.pop(0)
        if url in seen or depth > max_depth:
            continue
        seen.add(url)
        try:
            with get(url, timeout=60) as r:
                html = r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            print(f"    [!] 列举失败 {url}: {type(exc).__name__}: {exc}")
            continue
        for ent in parse_apache_index(html):
            child = urllib.parse.urljoin(url, ent["href"])
            if ent["is_dir"]:
                queue.append((child, depth + 1))
            else:
                ent["url"] = child
                ent["rel"] = child[len(base_url.rstrip("/")) + 1:]
                files.append(ent)
    return files


def head_size(url: str) -> Optional[int]:
    """精确字节数（用 Range 探测，比 HEAD 更可靠）。"""
    try:
        with get(url, headers={"Range": "bytes=0-0"}) as r:
            cr = r.headers.get("Content-Range") or ""
            if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                return int(cr.rsplit("/", 1)[1])
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def list_tree_https_tree(ds: Dict[str, Any], *, exact: bool) -> Dict[str, Any]:
    base = ds["access"].get("base_url") or ds["access"].get("file_base_url")
    print(f"  -> {ds['id']}  (https_tree)")
    print(f"     {base}")
    files = walk_http_tree(base)
    print(f"     找到 {len(files)} 个文件")
    if exact:
        for f in files:
            real = head_size(f["url"])
            if real is not None:
                f["size_bytes"] = real
        print("     体积：逐文件 Range 探测（精确）")
    else:
        print("     体积：目录列表显示值（近似，加 --exact 取精确）")
    total = sum(f["size_bytes"] or 0 for f in files)
    print(f"     合计：{human(total)}")
    for f in files[:40]:
        print(f"       {human(f['size_bytes']):>10}  {f['rel']}")
    if len(files) > 40:
        print(f"       ... 另有 {len(files) - 40} 个文件")
    return {"id": ds["id"], "status": "listed", "base": base,
            "num_files": len(files), "total_bytes": total,
            "exact": exact, "files": files}


def list_tree_biostudies(ds: Dict[str, Any], *, group: Optional[str] = None
                         ) -> Dict[str, Any]:
    """BioStudies 的文件清单是 MCZ JSON（逐文件 path + size）。"""
    url = ds["access"]["manifest_url"]
    print(f"  -> {ds['id']}  (biostudies manifest)")
    print(f"     {url}")
    with get(url, timeout=180) as r:
        entries = json.loads(r.read().decode("utf-8"))
    files = [e for e in entries if e.get("type", "file") == "file"]

    groups: Dict[str, Dict[str, Any]] = {}
    for e in files:
        parts = e["path"].split("/")
        g = parts[1] if len(parts) > 1 else "(root)"
        rec = groups.setdefault(g, {"files": 0, "bytes": 0})
        rec["files"] += 1
        rec["bytes"] += int(e.get("size") or 0)

    total = sum(int(e.get("size") or 0) for e in files)
    print(f"     共 {len(files)} 个文件，合计 {human(total)}")
    print("     按分组：")
    for g, rec in sorted(groups.items()):
        print(f"       {g:<42} {rec['files']:5d} 文件  {human(rec['bytes']):>10}")

    if group:
        sel = [e for e in files if group.lower() in e["path"].lower()]
        print(f"\n     匹配 {group!r}：{len(sel)} 个，"
              f"{human(sum(int(e.get('size') or 0) for e in sel))}")
        for e in sel[:20]:
            print(f"       {human(int(e.get('size') or 0)):>10}  {e['path']}")
        if len(sel) > 20:
            print(f"       ... 另有 {len(sel) - 20} 个")

    return {"id": ds["id"], "status": "listed", "manifest": url,
            "num_files": len(files), "groups": groups, "total_bytes": total}


def cmd_list_tree(cat: Dict[str, Any], ds_id: str, *, exact: bool,
                  group: Optional[str]) -> int:
    ds = next((d for d in cat["datasets"] if d["id"] == ds_id), None)
    if ds is None:
        print(f"[错误] 目录里没有 {ds_id}", file=sys.stderr)
        return 2
    method = ds["access"]["method"]
    if method == "biostudies":
        rec = list_tree_biostudies(ds, group=group)
    elif method in ("https_tree", "https_tree_or_aspera"):
        rec = list_tree_https_tree(ds, exact=exact)
    else:
        print(f"[错误] {ds_id} 的方法 {method} 不支持目录列举", file=sys.stderr)
        return 2
    out = os.path.join(HERE, f"tree_{ds_id}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    print(f"\n清单已写入：{out}")
    return 0


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------

def download_resumable(url: str, dest: str, expected: Optional[int]) -> Tuple[bool, str]:
    """下载到 ``dest``，支持断点续传。返回 (成功, 说明)。"""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    have = os.path.getsize(part) if os.path.exists(part) else 0
    if expected is not None and have > expected:
        print(f"    [!] .part 比预期大（{have} > {expected}），删除重下")
        os.remove(part)
        have = 0
    if expected is not None and have == expected and have > 0:
        print(f"    [=] .part 已是完整大小，直接收尾")
        os.replace(part, dest)
        return True, "already-complete"

    headers: Dict[str, str] = {}
    mode = "wb"
    if have > 0:
        headers["Range"] = f"bytes={have}-"
        mode = "ab"
        print(f"    [~] 续传：已 {human(have)}"
              f"{' / ' + human(expected) if expected else ''}")

    t0 = time.time()
    last = t0
    done = have
    try:
        with get(url, headers=headers, timeout=120) as r, open(part, mode) as fh:
            total = expected
            if total is None:
                cr = r.headers.get("Content-Range") or ""
                if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                    total = int(cr.rsplit("/", 1)[1])
                elif r.headers.get("Content-Length"):
                    total = done + int(r.headers["Content-Length"])
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= PROGRESS_EVERY:
                    last = now
                    speed = done / max(now - t0, 1e-6)
                    pct = f"{100.0 * done / total:.1f}%" if total else "?"
                    eta = ((total - done) / speed) if (total and speed > 0) else None
                    print(f"      {human(done)} / {human(total)}  ({pct})  "
                          f"{human(speed)}/s"
                          + (f"  剩余 {int(eta)}s" if eta else ""), flush=True)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            # Range 超出：说明 .part 其实已完整
            if expected is not None and os.path.getsize(part) == expected:
                os.replace(part, dest)
                return True, "already-complete"
        return False, f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}（可重跑续传）"

    got = os.path.getsize(part)
    if expected is not None and got != expected:
        return False, f"大小不符：得到 {got}，预期 {expected}（可重跑续传）"
    os.replace(part, dest)
    return True, f"ok ({human(got)})"


# --------------------------------------------------------------------------
# 单个数据集
# --------------------------------------------------------------------------

def handle_zip_dataset(ds: Dict[str, Any], target: str, *, dry_run: bool,
                       allow_large: bool) -> Dict[str, Any]:
    url = ds["access"]["url"]
    name = url.rsplit("/", 1)[-1]
    outdir = os.path.join(target, ds["id"])
    dest = os.path.join(outdir, name)

    print(f"  -> {ds['id']}")
    print(f"     {url}")

    declared = ds.get("size_bytes")
    real = remote_size(url)
    size = real if real is not None else declared
    print(f"     体积：catalog={human(declared)}  实际={human(real)}"
          + ("  [不一致!]" if (real and declared and real != declared) else ""))

    if dry_run:
        target_free = free_space(target)
        print(f"     目标盘剩余：{human(target_free)}"
              + ("" if target_free >= (size or 0) else "  ⚠ 空间不足"))
        return {"id": ds["id"], "status": "dry-run", "url": url, "size": size}

    # 空间预检：留 5% 余量，避免盘满导致系统异常
    need = int((size or 0) * 1.05)
    target_free = free_space(target)
    if size is not None and target_free < need:
        print(f"     [x] 空间不足：需要 {human(need)}，剩余 {human(target_free)}")
        return {"id": ds["id"], "status": "no-space",
                "url": url, "size": size, "free": target_free}
    if size and size > 100e9 and not allow_large:
        print(f"     [x] 体积 {human(size)} 超过 100 GB，"
              f"确需下载请加 --allow-large")
        return {"id": ds["id"], "status": "refused-large", "url": url, "size": size}

    ok, msg = download_resumable(url, dest, size)
    print(f"     {'完成' if ok else '失败'}：{msg}")
    rec: Dict[str, Any] = {
        "id": ds["id"], "status": "ok" if ok else "failed",
        "url": url, "path": dest, "size": size, "message": msg,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if ok:
        print("     计算 sha256（大文件需几分钟）...", flush=True)
        rec["sha256"] = sha256_of(dest)
        print(f"     sha256={rec['sha256']}")
        with open(os.path.join(outdir, "dataset.json"), "w", encoding="utf-8") as fh:
            json.dump({**ds, "fetched": rec}, fh, ensure_ascii=False, indent=1)
    return rec


def handle_tree_dataset(ds: Dict[str, Any], target: str, *, dry_run: bool,
                        allow_large: bool) -> Dict[str, Any]:
    """https_tree：先探测可达性，再按需抓取。默认只做探测。"""
    base = ds["access"]["base_url"]
    print(f"  -> {ds['id']}  (https_tree)")
    print(f"     {base}")
    reachable = False
    try:
        with get(base, timeout=60) as r:
            reachable = r.status in (200, 301, 302)
    except Exception as exc:  # noqa: BLE001
        print(f"     [!] 探测失败：{type(exc).__name__}: {exc}")
    print(f"     可达：{reachable}")
    print(f"     候选子目录：{ds['access'].get('directories')}")
    print(f"     提示：该数据集需先列目录再逐个抓取；"
          f"用 --only {ds['id']} --list-tree 查看文件清单")
    return {"id": ds["id"], "status": "probe-only" if not dry_run else "dry-run",
            "url": base, "reachable": reachable}


def handle_biostudies(ds: Dict[str, Any], target: str, *, dry_run: bool,
                      allow_large: bool) -> Dict[str, Any]:
    print(f"  -> {ds['id']}  (biostudies)")
    api = ds["access"]["api_url"]
    try:
        with get(api, timeout=60) as r:
            blob = json.loads(r.read().decode("utf-8"))
        title = None
        def walk(n):
            nonlocal title
            if isinstance(n, dict):
                if n.get("name") == "Title" and n.get("value") and not title:
                    title = n["value"]
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)
        walk(blob)
        print(f"     标题：{title}")
        return {"id": ds["id"], "status": "probe-only", "title": title,
                "api": api}
    except Exception as exc:  # noqa: BLE001
        print(f"     [!] 失败：{type(exc).__name__}: {exc}")
        return {"id": ds["id"], "status": "failed", "error": str(exc)}


HANDLERS = {
    "https_zip": handle_zip_dataset,
    "https_tree": handle_tree_dataset,
    "https_tree_or_aspera": handle_tree_dataset,
    "biostudies": handle_biostudies,
}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_list(cat: Dict[str, Any]) -> None:
    print(f"数据集目录  (schema {cat['schema_version']}, 编制于 {cat['compiled']})")
    print(f"目的：{cat['purpose']}\n")
    for ds in cat["datasets"]:
        size = ds.get("size_bytes")
        print(f"[tier {ds['tier']}] {ds['id']}")
        print(f"    {ds['title']}")
        print(f"    体积 : {human(size)}"
              + ("  (estimate)" if ds.get("size_is_estimate") else "  [已核实]")
              if size else "    体积 : 未核实")
        print(f"    作用 : {ds['role']}")
        print(f"    许可 : {ds.get('license', '见条目')}")
        print()
    if cat.get("rejected"):
        print("已排除（理由）：")
        for r in cat["rejected"]:
            print(f"  - {r['id']}: {r['reason']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="抓取公开细胞器数据集到外接盘（可断点续传）")
    ap.add_argument("--list", action="store_true", help="打印数据集目录")
    ap.add_argument("--target", default=None,
                    help=r"目标目录，例如 W:\dachuang_public_data")
    ap.add_argument("--only", default=None,
                    help="只处理这些 id（逗号分隔）")
    ap.add_argument("--max-tier", type=int, default=None,
                    help="只处理 tier <= N 的数据集")
    ap.add_argument("--dry-run", action="store_true",
                    help="只探测体积/空间/可达性，不下载")
    ap.add_argument("--allow-large", action="store_true",
                    help="允许下载 >100 GB 的条目")
    ap.add_argument("--list-tree", default=None, metavar="ID",
                    help="列举某个数据集的完整文件清单（不下载）")
    ap.add_argument("--exact", action="store_true",
                    help="配合 --list-tree：逐文件精确探测体积（较慢）")
    ap.add_argument("--group", default=None,
                    help="配合 --list-tree：只看路径含该子串的文件（如 'day 21'）")
    args = ap.parse_args(argv)

    cat = load_catalog()

    if args.list:
        cmd_list(cat)
        return 0

    if args.list_tree:
        return cmd_list_tree(cat, args.list_tree,
                             exact=args.exact, group=args.group)

    if not args.target:
        print("[错误] 需要 --target（外接盘上的目标目录）", file=sys.stderr)
        print(r"       例如：--target W:\dachuang_public_data", file=sys.stderr)
        return 2

    wanted = {s.strip() for s in args.only.split(",")} if args.only else None
    selected = [
        ds for ds in cat["datasets"]
        if (wanted is None or ds["id"] in wanted)
        and (args.max_tier is None or ds["tier"] <= args.max_tier)
    ]
    if not selected:
        print("[错误] 没有匹配的数据集", file=sys.stderr)
        return 2

    os.makedirs(args.target, exist_ok=True)
    print(f"目标目录：{os.path.abspath(args.target)}")
    print(f"剩余空间：{human(free_space(args.target))}")
    print(f"将处理 {len(selected)} 个数据集\n")

    results: List[Dict[str, Any]] = []
    for ds in selected:
        handler = HANDLERS.get(ds["access"]["method"])
        if handler is None:
            print(f"  -> {ds['id']}: 未知 access.method={ds['access']['method']}")
            results.append({"id": ds["id"], "status": "unsupported"})
            continue
        print("-" * 70)
        try:
            results.append(handler(ds, args.target,
                                   dry_run=args.dry_run,
                                   allow_large=args.allow_large))
        except Exception as exc:  # noqa: BLE001
            print(f"     [x] 异常：{type(exc).__name__}: {exc}")
            results.append({"id": ds["id"], "status": "error",
                            "error": str(exc)})

    # 汇总写 manifest
    manifest_path = os.path.join(args.target, "manifest.json")
    old: Dict[str, Any] = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                old = json.load(fh)
        except Exception:  # noqa: BLE001
            old = {}
    merged = {r["id"]: r for r in old.get("datasets", [])}
    for r in results:
        merged[r["id"]] = r
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "catalog_schema": cat["schema_version"],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "target": os.path.abspath(args.target),
            "datasets": list(merged.values()),
        }, fh, ensure_ascii=False, indent=1)

    print("\n" + "=" * 70)
    for r in results:
        print(f"  {r['id']:<38} {r['status']}")
    print(f"\n清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
