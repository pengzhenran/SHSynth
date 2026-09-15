# -*- coding: utf-8 -*-
"""
tests/test_engine.py
====================

**综合引擎测试**:勒让德函数、任意点/任意网格求值、分块一致性、多时次、
截断、以及不依赖往返的**解析物理校验**。

解析校验的意义:不靠"算完再算回去",而是拿已知的闭式解直接对照 ——
例如 ``C₁₀ = 1/√3`` 对应的场必须逐点等于 ``sin(φ)``,全球加权平均必须等于
``C₀₀``。这类检查能抓住"系数列错位""归一化差常数"这类最致命也最容易溜过去的
错误。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Checker, guard                                 # noqa: E402

from shsynth.coeffs import SHCoeffs, triangle_order                # noqa: E402
from shsynth.engine import (evaluate, fibonacci_points,            # noqa: E402
                            legendre_pbar, regular_grid, synthesize,
                            synthesis_grid)


def _single(n: int, m: int, val: float = 1.0, sine: bool = False,
            nmax: int = None) -> SHCoeffs:
    L = nmax if nmax is not None else n
    C = np.zeros((L + 1, L + 1))
    S = np.zeros((L + 1, L + 1))
    if sine:
        S[n, m] = val
    else:
        C[n, m] = val
    return SHCoeffs(C, S, {})


def test_analytic_degree1(check: Checker):
    """解析校验:一阶项的闭式解。"""
    check.section("解析解(1 阶)")
    rng = np.random.default_rng(5)
    lat = rng.uniform(-89, 89, 200)
    lon = rng.uniform(0, 360, 200)

    # C10: P̄10 = √3 sinφ  → 取 C10 = 1/√3 时场 = sinφ
    c = _single(1, 0, 1.0 / np.sqrt(3.0))
    v = synthesize(lat, lon, c.C, c.S)
    check.close(v, np.sin(np.deg2rad(lat)), 1e-14,
                "C₁₀ = 1/√3 → 场 = sin φ(最大偏差)")

    # C11: P̄11 = √3 cosφ  → 取 C11 = 1/√3 时场 = cosφ·cosλ
    c = _single(1, 1, 1.0 / np.sqrt(3.0))
    v = synthesize(lat, lon, c.C, c.S)
    ref = np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    check.close(v, ref, 1e-14, "C₁₁ = 1/√3 → 场 = cos φ·cos λ")

    # S11: 取 S11 = 1/√3 时场 = cosφ·sinλ
    c = _single(1, 1, 1.0 / np.sqrt(3.0), sine=True)
    v = synthesize(lat, lon, c.C, c.S)
    ref = np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    check.close(v, ref, 1e-14, "S₁₁ = 1/√3 → 场 = cos φ·sin λ")

    # C20: P̄20 = √5(3sin²φ − 1)/2 → 取 C20 = 2/√5 时场 = 3sin²φ − 1
    c = _single(2, 0, 2.0 / np.sqrt(5.0))
    v = synthesize(lat, lon, c.C, c.S)
    ref = 3.0 * np.sin(np.deg2rad(lat)) ** 2 - 1.0
    check.close(v, ref, 1e-13, "C₂₀ = 2/√5 → 场 = 3sin²φ − 1")


def test_legendre_vs_scipy(check: Checker):
    """全部 (n,m) 的 4π 归一化勒让德函数与 scipy 对照(去掉 Condon–Shortley)。"""
    check.section("勒让德函数 vs scipy")
    try:
        from math import factorial

        from scipy.special import lpmv            # 稳定的 ufunc 接口
    except ImportError:                                   # pragma: no cover
        check.ok(True, "scipy 不可用,跳过")
        return

    def pnm(m, n, x):
        """无 Condon–Shortley 相位的连带勒让德函数(整数阶)。"""
        return float(lpmv(m, n, x)) * (-1) ** m

    nmax = 12
    rng = np.random.default_rng(9)
    lat = rng.uniform(-90, 90, 17)
    P = legendre_pbar(lat, nmax)
    m_vec, n_vec = triangle_order(nmax)
    worst = 0.0
    for i, la in enumerate(lat):
        x = np.sin(np.deg2rad(la))
        for k in range(len(m_vec)):
            m, n = int(m_vec[k]), int(n_vec[k])
            val = pnm(m, n, x)
            norm = np.sqrt(2 * n + 1) if m == 0 else \
                np.sqrt(2.0 * (2 * n + 1) * factorial(n - m) / factorial(n + m))
            worst = max(worst, abs(float(P[i, k]) - val * norm))
    check.ok(worst < 1e-11, f"17 个纬度 × 91 个 (n,m) 最大偏差 {worst:.3e}")


def test_orthonormality(check: Checker):
    """4π 归一化的正交性:在 GLQ 网格上 ∫P̄²Y²dΩ = 4π(自检归一化没错)。

    换元 ``x = sinφ``(所以 ``dx = cosφ dφ``)之后,体积元就是 ``dx dλ``,
    再乘 ``cosφ`` 就重复计了一次。
    """
    check.section("4π 归一化正交性")
    ngl = 60
    x, w = np.polynomial.legendre.leggauss(ngl)
    lat = np.rad2deg(np.arcsin(x))
    lon = np.linspace(0, 360, 2 * ngl + 1)[:-1]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    nmax = 6
    worst = 0.0
    for n in range(nmax + 1):
        for m in range(n + 1):
            for sine in (False, True) if m >= 1 else (False,):
                c = _single(n, m, 1.0, sine=sine, nmax=nmax)
                v = synthesize(LA.ravel(), LO.ravel(), c.C, c.S)
                val = float(np.sum(v ** 2 * w.repeat(lon.size)
                                   * (2 * np.pi / lon.size)))
                worst = max(worst, abs(val / (4 * np.pi) - 1.0))
    check.ok(worst < 1e-10, f"所有 (n,m) 单元积分 = 4π,最大相对偏差 {worst:.3e}")


def test_global_mean_equals_c00(check: Checker):
    """物理校验:全球加权平均必须等于 C₀₀,而且误差按 O(Δφ²) 收敛。

    ``P̄00 = 1`` 且其它 (n,m) 在整个球面上积分为 1,所以
    ``(1/4π)∮f dΩ = C₀₀`` —— 这条等式**不依赖往返**,是真正的外在判据。

    采样用**球带中心**(``lat = −90 + (i+0.5)Δ``):等价于中点法则,误差是
    ``O(Δ²)``。若改用格点本身(左黎曼和),误差只有 ``O(Δ)``,看起来更大,
    但那是对"求和方式"的惩罚,与综合无关 —— 实测 4°→0.5° 误差比值为 2 倍
    而不是 4 倍,所以这里用带中心并同时验证收敛阶。
    """
    check.section("全球平均 = C₀₀")

    rng = np.random.default_rng(3)
    nmax = 10
    C = rng.normal(0, 1e-3, (nmax + 1, nmax + 1))
    S = rng.normal(0, 1e-3, (nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for m in range(nmax + 1):
            if m > n:
                C[n, m] = S[n, m] = 0.0
    C[0, 0] = 2.5
    c = SHCoeffs(C, S, {})

    def mean_at(step: float) -> float:
        nlat = int(round(180.0 / step))
        lat = -90.0 + (np.arange(nlat) + 0.5) * step
        lon = np.arange(0.0, 360.0, step)
        g = synthesis_grid(lat, lon, c)
        w = np.cos(np.deg2rad(lat))
        return float(np.sum(g * w[:, None]) / (np.sum(w) * g.shape[1]))

    errs = {}
    for step in (2.0, 1.0, 0.5):
        m = mean_at(step)
        errs[step] = abs(m - c.C[0, 0])
        check.ok(errs[step] < 1e-6,
                 f"{step:g}° 网格加权平均 {m:.10f} vs C₀₀ {c.C[0,0]:.10f}"
                 f"(偏差 {errs[step]:.2e})")
    r1 = errs[2.0] / max(errs[1.0], 1e-300)
    r2 = errs[1.0] / max(errs[0.5], 1e-300)
    check.ok(3.4 < r1 < 4.6 and 3.4 < r2 < 4.6,
             f"误差随步长加倍收敛到 O(Δφ²)(比值 {r1:.2f}、{r2:.2f},理论 4)")


def test_chunk_consistency(check: Checker):
    """分块必须只是内存策略:不同 chunk 的结果要逐位相同。"""
    check.section("分块一致性")
    rng = np.random.default_rng(21)
    nmax = 15
    C = rng.normal(0, 1.0, (nmax + 1, nmax + 1))
    S = rng.normal(0, 1.0, (nmax + 1, nmax + 1))
    c = SHCoeffs(C, S, {})
    lat = rng.uniform(-90, 90, 5000)
    lon = rng.uniform(0, 360, 5000)
    a = evaluate(lat, lon, c, chunk=1000)
    b = evaluate(lat, lon, c, chunk=10_000_000)
    check.close(a, b, 0.0, "chunk=1000 与不分块逐位相同")


def test_multitime(check: Checker):
    """多时次:整块求值必须等于逐时次求值。"""
    check.section("多时次")
    rng = np.random.default_rng(33)
    nmax = 6
    C = rng.normal(0, 1, (nmax + 1, nmax + 1, 3))
    S = rng.normal(0, 1, (nmax + 1, nmax + 1, 3))
    c = SHCoeffs(C, S, {})
    lat = rng.uniform(-90, 90, 200)
    lon = rng.uniform(0, 360, 200)
    v = evaluate(lat, lon, c)
    check.ok(v.shape == (200, 3), f"多时次输出形状 {v.shape}")
    for k in range(3):
        sub = SHCoeffs(C[:, :, k].copy(), S[:, :, k].copy(), {})
        # 多时次是一起做矩阵乘的,与逐时次单独做乘法的求和顺序不同,
        # 所以只要求到 1e-12(而不是逐位相同)。
        check.close(v[:, k], evaluate(lat, lon, sub), 1e-12,
                    f"第 {k} 个时次与单独计算一致")


def test_truncation(check: Checker):
    """按阶截断:用 nmax=L 求值 == 先把系数截到 L 再求值。"""
    check.section("阶数截断")
    rng = np.random.default_rng(44)
    nmax = 12
    C = rng.normal(0, 1, (nmax + 1, nmax + 1))
    S = rng.normal(0, 1, (nmax + 1, nmax + 1))
    c = SHCoeffs(C, S, {})
    lat = rng.uniform(-90, 90, 150)
    lon = rng.uniform(0, 360, 150)
    for L in (0, 1, 4, 7, 12):
        a = evaluate(lat, lon, c, nmax=L)
        b = evaluate(lat, lon, c.truncate(L))
        check.close(a, b, 0.0, f"nmax={L} 截断与 truncate() 等价")
    check.close(evaluate(lat, lon, c, nmax=4),
                evaluate(lat, lon, c.truncate(4)), 0.0, "nmax=4 复核")


def test_zonal_only(check: Checker):
    """只有 m=0 的场必须与经度无关;这是最容易被列错位破坏的性质。"""
    check.section("纯带谐场与经度无关")
    rng = np.random.default_rng(55)
    nmax = 8
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    C[:, 0] = rng.normal(0, 1, nmax + 1)
    c = SHCoeffs(C, S, {})
    lat = np.linspace(-90, 90, 37)
    v1 = synthesize(lat, np.zeros_like(lat), c.C, c.S)
    for lon0 in (0.0, 37.5, 123.4, 359.9):
        v2 = synthesize(lat, np.full_like(lat, lon0), c.C, c.S)
        check.close(v1, v2, 1e-14, f"经度 = {lon0}° 与 0° 结果相同")


def test_grid_api(check: Checker):
    """网格接口与散点接口必须给同一批数。"""
    check.section("网格/散点接口一致")
    rng = np.random.default_rng(66)
    nmax = 7
    C = rng.normal(0, 1, (nmax + 1, nmax + 1))
    S = rng.normal(0, 1, (nmax + 1, nmax + 1))
    c = SHCoeffs(C, S, {})
    latv = np.linspace(-90, 90, 19)
    lonv = np.linspace(0, 360, 25)
    g = synthesis_grid(latv, lonv, c)
    LA, LO = np.meshgrid(latv, lonv, indexing="ij")
    v = evaluate(LA.ravel(), LO.ravel(), c)
    check.ok(g.shape == (19, 25), f"网格形状 {g.shape}")
    check.close(g.ravel(), v, 0.0, "网格值与对应散点值逐位相同")


def test_regular_grid_and_fibonacci(check: Checker):
    """经纬网格生成与球面采样的小性质。"""
    check.section("网格/球面点生成")
    lat, lon = regular_grid(-90, 90, 0, 360, 1.0, 1.0)
    check.ok(lat.size == 181 and lon.size == 361, f"1° 全网格 {lat.size}×{lon.size}")
    check.ok(lat[0] == -90.0 and lat[-1] == 90.0, "纬度两端点都含")
    lat, lon = regular_grid(5, 55, 60, 140, 1.0, 1.0)
    check.ok(lat.size == 51 and lon.size == 81,
             f"区域网格 {lat.size}×{lon.size}(5–55N, 60–140E)")

    la, lo = fibonacci_points(5000)
    check.ok(la.size == 5000 and lo.size == 5000, "Fibonacci 点数正确")
    check.ok(abs(float(np.mean(np.sin(np.deg2rad(la))))) < 0.02,
             f"球面均匀:sinφ 均值 {float(np.mean(np.sin(np.deg2rad(la)))):.4f} ≈ 0")
    check.ok(np.all((lo >= 0) & (lo < 360)), "经度落在 [0, 360)")
    # 球面均匀 = **等面积带**里点数相同(按 sinφ 分箱),不是按纬度等分箱:
    # 按纬度等分时中纬度自然多、两极自然少(cosφ 的差别,实测 0.158 倍)。
    hist, _ = np.histogram(np.sin(np.deg2rad(la)), bins=10, range=(-1, 1))
    check.ok(hist.min() / hist.max() > 0.93,
             f"等面积带均匀(最小/最大 = {hist.min() / hist.max():.3f})")
    hb, _ = np.histogram(la, bins=10, range=(-90, 90))
    ratio = hb.min() / hb.max()
    check.ok(0.10 < ratio < 0.25,
             f"按纬度等分箱时两极应明显更少(实测比 {ratio:.3f},理论 cos81°/cos9° ≈ 0.156)")


def test_sn0_forced_zero(check: Checker):
    """S[:, 0] 必须被强制为 0(sin(0·λ) ≡ 0),否则最小二乘会秩亏。"""
    check.section("S[:,0] ≡ 0")
    C = np.ones((5, 5))
    S = np.ones((5, 5))
    c = SHCoeffs(C, S, {})
    check.ok(np.all(c.S[:, 0] == 0.0), "构造时 S[:,0] 被清零")
    C3 = np.ones((5, 5, 2))
    S3 = np.ones((5, 5, 2))
    c3 = SHCoeffs(C3, S3, {})
    check.ok(np.all(c3.S[:, 0, :] == 0.0), "3 维对象也只清 S[:,0,:],不动时间轴")
    check.ok(np.all(c3.S[:, 1, :] == 1.0), "其它次不被误清")


def test_large_point_set(check: Checker):
    """点集大到需要分块时结果仍然正确(与直接公式对照)。"""
    check.section("大点集(分块路径)")
    rng = np.random.default_rng(77)
    c = _single(2, 0, 2.0 / np.sqrt(5.0), nmax=2)
    lat = rng.uniform(-90, 90, 300_000)
    lon = rng.uniform(0, 360, 300_000)
    v = evaluate(lat, lon, c, chunk=50_000)
    ref = 3.0 * np.sin(np.deg2rad(lat)) ** 2 - 1.0
    check.close(v, ref, 1e-12, "30 万点、chunk=5 万 仍与解析解一致")


def main() -> int:
    check = Checker("test_engine —— 综合引擎与解析物理校验")
    for fn in (test_analytic_degree1, test_legendre_vs_scipy,
               test_orthonormality, test_global_mean_equals_c00,
               test_chunk_consistency, test_multitime, test_truncation,
               test_zonal_only, test_grid_api, test_regular_grid_and_fibonacci,
               test_sn0_forced_zero, test_large_point_set):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
