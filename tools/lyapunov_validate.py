"""validate_lyapunov.py —— 用已知答案校准 Rosenstein 估计器

## 为什么必须先做这一步

如果估计器本身测不出已知的混沌，那么"S-BIAD712 上没有混沌"这个负结论
**毫无信息量**——可能只是工具坏了。

本脚本发现过两个真实 bug：

1. **发散公式的下标错**：写成 `||X_i - X_{nn+j}||`，正确的是
   `||X_{i+j} - X_{nn+j}||`。错的形式在 j=1 就让距离达到吸引子直径，
   发散曲线立刻饱和，已知混沌（λ=ln2）被估成 0.002。
2. **拟合区间**：对整条曲线拟合会被饱和段拉向 0。

修正后同一校准测出 0.694，误差 0.1%。

## 校准结果本身就是结论

序列越短，混沌被**系统性低估**：N=25 时真值 0.693 只测出 0.252（−64%）。
而正弦波与白噪声在有限长度上存在约 0.15 的**正偏**。

两者叠加意味着：在 S-BIAD712 的 25 至 100 帧上，λ 的绝对值不可解释，
只能与**同长度的校准值**比较。本脚本输出的就是那张对照表。
"""

import sys

import numpy as np

sys.path.insert(0, "tools")
from lyapunov_analysis import rosenstein  # noqa: E402


def logistic(n, r=4.0, x0=0.31):
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def estimate(x, m=3, tau=1):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / (x.std() + 1e-12)
    lam, _, _ = rosenstein(x, m, tau)
    return lam


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("一、估计器校准（m=3, tau=1）")
    print("=" * 72)
    print(f"{'序列':<26}{'理论 λ':>10}{'估计 λ':>10}")
    print("-" * 72)

    print(f"{'逻辑斯蒂 r=4（N=1000）':<26}{0.693:>10.3f}{estimate(logistic(1000)):>10.3f}")
    s = np.sin(2 * np.pi * np.arange(1000) / 12)
    print(f"{'正弦波（N=1000）':<26}{0.0:>10.3f}{estimate(s):>10.3f}")
    rng = np.random.default_rng(0)
    print(f"{'白噪声（N=1000）':<26}{'—':>10}{estimate(rng.standard_normal(1000)):>10.3f}")

    print()
    print("=" * 72)
    print("二、序列长度的影响——这是本文真正的结论")
    print("=" * 72)
    print(f"{'长度 N':<10}{'混沌实测':>10}{'低估幅度':>10}{'正弦实测':>10}{'白噪声':>10}")
    print("-" * 72)
    for n in (1000, 300, 100, 60, 39, 25):
        lc = estimate(logistic(n))
        ls = estimate(np.sin(2 * np.pi * np.arange(n) / 12))
        ln = estimate(np.random.default_rng(1).standard_normal(n))
        dev = (lc - 0.693) / 0.693
        print(f"{n:<10}{lc:>10.3f}{dev:>9.0%}{ls:>10.3f}{ln:>10.3f}")

    print()
    print("读法：")
    print("  * 混沌实测随 N 下降而系统性低估——N=25 时只剩真值的 36%。")
    print("  * 正弦与白噪声存在正偏，N=1000 时仍有约 0.15。")
    print("  * 因此在 25 至 100 帧上，**λ 的绝对值没有意义**，")
    print("    只能与同长度的校准值对照。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
