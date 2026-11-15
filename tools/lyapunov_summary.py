"""summarise_lyapunov.py —— 汇总 S-BIAD712 上的李雅普诺夫结果

## 判读框架

λ 的绝对值不能直接解释。校准脚本已经证明：
同一估计器在**已知混沌**上，N=100 时低估 15%、N=39 时低估 41%、N=25 时低估 64%；
而**非混沌**的正弦/白噪声在小 N 上仍有 0~0.2 的正偏。

因此本文只做**同长度对照**：
* 每条真实序列的 λ，与「同长度的逻辑斯蒂映射」（已知混沌的上限参考）
  和「同长度的白噪声」（无结构的偏置参考）比较。
* 若真实 λ 落在两者之间，说明**没有超出噪声的可判定证据**。

## 输出

按观测量聚合：中位 λ、落在白噪声以下的百分比、超过同长度混沌校准的百分比。
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "tools")
from lyapunov_analysis import rosenstein  # noqa: E402


def logistic(n, r=4.0, x0=0.31):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def est(x, m=3, tau=1):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    v, _, _ = rosenstein(x, m, tau)
    return v


def main(path="logs/BIAD712_LYAPUNOV.json"):
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.load(open(path, encoding="utf-8"))
    print(f"文件 {len(data)} 个")

    # 同长度校准（按 N 缓存）
    cal = {}
    for rec in data:
        n = rec["T"]
        if n not in cal:
            cal[n] = (est(logistic(n)),
                      est(np.random.default_rng(7).standard_normal(n)))
    print("\n同长度校准（N: 混沌 / 白噪声）：")
    for n in sorted(cal):
        print(f"  N={n:<4} {cal[n][0]:+.3f} / {cal[n][1]:+.3f}")

    by = defaultdict(list)
    for rec in data:
        n = rec["T"]
        lc, ln = cal[n]
        for lab, v in rec["series"].items():
            by[lab].append((v["lambda"], v.get("z_phase"), lc, ln))

    print()
    hdr = (f"{'观测量':<14}{'n':>4}{'中位 λ':>9}{'同长混沌':>10}"
           f"{'同长噪声':>10}{'低于噪声':>9}{'超混沌校准':>11}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    order = sorted(by, key=lambda k: -len(by[k]))
    for lab in order:
        rows = by[lab]
        lam = np.array([r[0] for r in rows])
        lcs = np.array([r[2] for r in rows])
        lns = np.array([r[3] for r in rows])
        below = float((lam <= lns).mean())
        above = float((lam > lcs).mean())
        print(f"{lab:<14}{len(rows):>4}{np.median(lam):>9.3f}"
              f"{np.median(lcs):>10.3f}{np.median(lns):>10.3f}"
              f"{below:>8.0%}{above:>11.0%}")
    print("=" * len(hdr))

    # 相位代理显著性
    zs = [v.get("z_phase") for rec in data for v in rec["series"].values()
          if v.get("z_phase") is not None]
    zs = np.array(zs)
    print(f"\n相位随机化代理：{len(zs)} 条序列，z>2 的有 {(zs > 2).sum()} 条"
          f"（{100 * (zs > 2).mean():.0f}%），中位 z = {np.median(zs):.2f}")
    print("若 z>2 的比例接近 5%，与随机期望无异——即没有可判定的确定性结构。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
