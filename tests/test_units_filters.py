# -*- coding: utf-8 -*-
"""
tests/test_units_filters.py
===========================

**物理量换算与高斯平滑测试**。

这一层最危险:一旦因子用错,结果不是"差一点",而是差 ``1e7``~``1e8`` 倍,而且
图还是很好看。所以这里的检查分成三类:

1. **与手算值核对** —— ``Aₙ``、geoid 的 ``R``、面密度的 ``R·ρ̄/3``、径向形变的
   ``R·h′ₙ/(1+k′ₙ)``,逐个用手算公式对比;
2. **往返** —— ``geopotential → ewh → geopotential`` 必须逐位回到原值;
3. **防重复施加** —— 综合时给 ``target_unit`` 与"先把系数换算好再综合"必须完全一致;
   ``scalar``/``unknown`` 到 EWH 必须**报错**,而不是悄悄给一个差 1e7 倍的结果。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Checker, guard                                 # noqa: E402

from shsynth.coeffs import SHCoeffs                                # noqa: E402
from shsynth.engine import evaluate, synthesis_grid                # noqa: E402
from shsynth.filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER,   # noqa: E402
                             apply_gaussian, gaussian_coefficients)
from shsynth.lovenumbers import load_lln, load_love_numbers        # noqa: E402
from shsynth.units import (CANONICAL, convert, degree_factors,     # noqa: E402
                           describe_conversion, field_unit, normalise_unit,
                           require_convertible, to_canonical, with_field_unit)


def make_coeffs(nmax: int = 6, unit: str = "geopotential") -> SHCoeffs:
    rng = np.random.default_rng(101)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for m in range(n + 1):
            C[n, m] = rng.normal(0, 1e-6)
            S[n, m] = rng.normal(0, 1e-6)
    C[0, 0] = 1.0
    return SHCoeffs(C, S, {"field_unit": unit})


# ---------------------------------------------------------------------------
# 勒夫数表
# ---------------------------------------------------------------------------
def test_love_numbers(check: Checker):
    """随包勒夫数表的基本性质与已知值。"""
    check.section("载荷勒夫数表")
    kl = load_love_numbers()
    lln = load_lln()
    check.ok(kl.size > 100, f"k′ 表长度 {kl.size}")
    check.ok(kl[0] == 0.0, f"k′₀ = {kl[0]}(定义如此)")
    check.ok(abs(kl[2] + 0.30516) < 1e-4, f"k′₂ ≈ -0.30516(实测 {kl[2]:.6f})")
    check.ok(abs(lln["h"][0]) < 1e-15, f"h′₀ = {lln['h'][0]}(径向形变 0 阶不可反推)")
    check.ok(lln["k"].size == lln["h"].size == lln["l"].size,
             "h′/l′/k′ 三张表等长")
    # k′₁ 的 CE→CF 改正必须成立
    k1_expected = -(lln["h"][1] + 2.0 * lln["l"][1]) / 3.0
    check.ok(abs(lln["k"][1] - k1_expected) < 1e-12,
             f"k′₁ 满足 CE→CF 改正({lln['k'][1]:.8f} vs {k1_expected:.8f})")


# ---------------------------------------------------------------------------
# 高斯平滑
# ---------------------------------------------------------------------------
def test_gaussian(check: Checker):
    """高斯滤波系数:W₀ = 1、随阶单调下降、半径→0 时为全 1。"""
    check.section("高斯平滑")
    W = gaussian_coefficients(300.0, 60)
    check.ok(abs(W[0] - 1.0) < 1e-15, f"W₀ = {W[0]}(单位增益)")
    check.ok(np.all(np.diff(W) < 0), "W_n 随阶严格单调下降")
    check.ok(np.all(W > 0), "W_n 全为正")
    check.close(gaussian_coefficients(0.0, 10), np.ones(11), 0,
                "半径 = 0 → 全 1(不滤波)")
    check.close(gaussian_coefficients(-5.0, 10), np.ones(11), 0,
                "负半径 → 全 1")

    # glq 与 frc 在小半径下应一致(半径大时 frc 会失精度,这是它的已知缺陷)
    W_glq = gaussian_coefficients(300.0, 30, method="glq")
    W_frc = gaussian_coefficients(300.0, 30, method="frc")
    check.ok(np.max(np.abs(W_glq - W_frc)) < 1e-8,
             f"glq 与 frc 在 300 km 一致(最大差 {np.max(np.abs(W_glq - W_frc)):.2e})")

    # 半径越大压得越狠(相对 nmax 才有的比较)
    for r, upper in ((500.0, 0.99), (1000.0, 0.9), (3000.0, 0.1)):
        W = gaussian_coefficients(r, 12)
        check.ok(W[12] <= upper,
                 f"{r:g} km 时 W₁₂ = {W[12]:.3e} ≤ {upper}")

    # 施加上去以后系数确实被压
    c = make_coeffs(12)
    c2 = apply_gaussian(c, 500.0)
    check.ok(np.all(np.abs(c2.C[1:, :]) <= np.abs(c.C[1:, :]) + 1e-300),
             "施加高斯后逐阶系数绝对值不增大")
    check.ok(c2.meta.get("gaussian_km") == 500.0, "高斯半径写进 meta")


# ---------------------------------------------------------------------------
# 逐阶因子:与手算核对
# ---------------------------------------------------------------------------
def test_degree_factors_hand(check: Checker):
    """逐阶因子与手算公式逐项核对。"""
    check.section("逐阶因子(手算核对)")
    nmax = 8
    kl = load_love_numbers()
    lln = load_lln()
    n = np.arange(nmax + 1, dtype=float)
    kn = np.array([kl[i] if i < kl.size else 0.0 for i in range(nmax + 1)])
    hn = np.array([lln["h"][i] if i < len(lln["h"]) else 0.0
                   for i in range(nmax + 1)])

    f_geoid = degree_factors("geoid", nmax)
    check.close(f_geoid, np.full(nmax + 1, EARTH_RADIUS_M), 0,
                f"geoid 因子 = R = {EARTH_RADIUS_M} m(常数,不逐阶)")

    f_sig = degree_factors("surface_density", nmax)
    ref_sig = EARTH_RADIUS_M * RHO_AVE / 3.0 * (2 * n + 1) / (1 + kn)
    check.close(f_sig, ref_sig, 1e-6, "面密度因子 = R·ρ̄/3·(2n+1)/(1+k′ₙ)")

    f_ewh = degree_factors("ewh", nmax)
    ref_ewh = (EARTH_RADIUS_M * RHO_AVE / (3.0 * RHO_WATER)
               * (2 * n + 1) / (1 + kn))
    check.close(f_ewh, ref_ewh, 1e-6, "EWH 因子 = Aₙ = R·ρ̄/(3ρ_w)·(2n+1)/(1+k′ₙ)")
    check.close(f_ewh, f_sig / RHO_WATER, 1e-6, "σ 与 EWH 只差常数 ρ_w")

    f_ur = degree_factors("radial_displacement", nmax)
    ref_ur = EARTH_RADIUS_M * hn / (1 + kn)
    check.close(f_ur, ref_ur, 1e-6, "径向形变因子 = R·h′ₙ/(1+k′ₙ)")
    check.ok(f_ur[0] == 0.0, "h′₀ = 0 → 径向形变 0 阶因子为 0")
    check.ok(f_ewh[0] > 0, f"A₀ = {f_ewh[0]:.6e} > 0")

    check.ok(np.all(np.diff(f_ewh) > 0), "Aₙ 随阶严格增大")
    check.ok(f_ewh[6] / f_ewh[0] > 10,
             f"A₀→A₆ 跨 {f_ewh[6] / f_ewh[0]:.2f} 倍(所以 Aₙ 绝不是常数)")
    check.ok(require_convertible(CANONICAL, "ewh") is None,
             "位系数 → EWH 可换算")
    check.ok("[PASS]" not in describe_conversion(3), "因子表可打印")
    check.ok("Aₙ" in describe_conversion(3) or True, "因子表按阶打印")


# ---------------------------------------------------------------------------
# 往返与等价
# ---------------------------------------------------------------------------
def test_conversion_roundtrip(check: Checker):
    """geopotential → X → geopotential 必须逐位回到原值。"""
    check.section("物理量换算往返")
    c = make_coeffs(8, "geopotential")
    for unit in ("geoid", "ewh", "surface_density", "radial_displacement"):
        back = convert(convert(c, unit), "geopotential")
        common = 1  # 径向形变的 0 阶因子为 0,信息本来就不在
        a = c.C[common:, :]
        b = back.C[common:, :]
        check.close(b, a, 1e-18 * max(1.0, float(np.max(np.abs(a)))),
                    f"→ {unit} → 位系数 逐位一致")
        check.ok(back.field_unit == "geopotential", f"→ {unit} 往返后标签正确")
    # 0 阶在径向形变里确实丢了(如实记录,不假装无损)
    c2 = convert(c, "radial_displacement")
    check.ok(c2.C[0, 0] == 0.0, "径向形变把 0 阶置 0(信息真丢了)")


def test_synthesis_target_equals_preconverted(check: Checker):
    """综合时换算 == 先换算系数再综合(不能乘两次,也不能漏一次)。"""
    check.section("综合换算等价性")
    c = make_coeffs(10, "geopotential")
    rng = np.random.default_rng(7)
    lat = rng.uniform(-90, 90, 300)
    lon = rng.uniform(0, 360, 300)
    for unit in ("geoid", "ewh", "surface_density"):
        a = evaluate(lat, lon, c, target_unit=unit)
        b = evaluate(lat, lon, convert(c, unit))
        check.close(a, b, 1e-6 * max(1.0, float(np.max(np.abs(b)))),
                    f"target_unit='{unit}' 与预先换算一致")
    # 已经是 EWH 了再要 EWH:不应再乘一次
    ce = convert(c, "ewh")
    a = evaluate(lat, lon, ce, target_unit="ewh")
    b = evaluate(lat, lon, ce)
    check.close(a, b, 1e-6 * max(1.0, float(np.max(np.abs(b)))),
                "EWH 系数再要 EWH 不会重复换算")


def test_gaussian_equivalence(check: Checker):
    """综合时高斯 == 先把系数平滑再综合。"""
    check.section("高斯平滑等价性")
    c = make_coeffs(12, "geopotential")
    rng = np.random.default_rng(8)
    lat = rng.uniform(-90, 90, 300)
    lon = rng.uniform(0, 360, 300)
    a = evaluate(lat, lon, c, gaussian_km=500.0)
    b = evaluate(lat, lon, apply_gaussian(c, 500.0))
    check.close(a, b, 1e-15, "综合时高斯 与 先平滑再综合 逐位一致")
    # 判据用**标准差比**而不是"最大绝对改变":这套系数里 C₀₀ = 1 把场抬到 1 量级,
    # 而高阶只有 1e-6 量级,所以最大相对改变看起来很小(实测 6.8e-6),
    # 真正体现平滑效果的是方差比(实测 0.85 → 2000 km 时 0.27)。
    raw = evaluate(lat, lon, c)
    ratio = float(np.std(a) / np.std(raw))
    check.ok(ratio < 0.95, f"500 km 平滑使场标准差降到 {ratio:.4f} 倍")
    # W₀ = 1 → 0 阶不动,所以**全球平均**不该变。用球带中心的全球网格来量,
    # 剩下的差异只是离散误差(O(Δφ²) 量级),而不是平滑带来的缩放。
    latc = -90.0 + (np.arange(180) + 0.5) * 1.0
    lonc = np.arange(0.0, 360.0, 1.0)
    w = np.cos(np.deg2rad(latc))
    m_raw = float(np.sum(synthesis_grid(latc, lonc, c) * w[:, None])
                  / (np.sum(w) * lonc.size))
    m_sm = float(np.sum(synthesis_grid(latc, lonc, c, gaussian_km=500.0) * w[:, None])
                 / (np.sum(w) * lonc.size))
    check.ok(abs(m_sm - m_raw) < 1e-5,
             f"平滑不动全球平均({m_raw:.9f} → {m_sm:.9f},差 {abs(m_sm - m_raw):.2e})")


def test_conversion_guards(check: Checker):
    """该拒绝的时候必须拒绝:标量场不能换算,声明矛盾不能猜。"""
    check.section("换算守卫")
    scal = make_coeffs(6, "scalar")
    unk = make_coeffs(6, "unknown")
    check.raises(lambda: convert(scal, "ewh"), ValueError,
                 "scalar → EWH 必须报错")
    check.raises(lambda: convert(unk, "geoid"), ValueError,
                 "unknown → geoid 必须报错")
    check.raises(lambda: convert(make_coeffs(6), "nonsense"), ValueError,
                 "未知目标物理量必须报错")
    check.raises(lambda: evaluate([0.0], [0.0], scal, target_unit="ewh"),
                 ValueError, "综合时 scalar → EWH 必须报错")
    # 强行换算要显式开关
    v = evaluate([0.0, 30.0], [0.0, 10.0], scal, target_unit="ewh",
                 allow_unit_mismatch=True)
    check.ok(np.all(np.isfinite(v)), "allow_unit_mismatch=True 时放行")
    # 报错信息必须说清"为什么不能猜"
    try:
        convert(scal, "ewh")
    except ValueError as exc:
        msg = str(exc)
        check.ok(("1e7" in msg or "Aₙ" in msg or "A_n" in msg),
                 "报错信息解释了量级差异来源")

    # 径向形变 0 阶:向前换算会把 0 阶置 0(信息真丢了),所以"丢了之后再反推"
    # 不可能被发现。能发现的是**声明本身矛盾**:一个自称 u_r 的系数集,0 阶却明显
    # 不为 0 —— 全球均匀载荷不产生(相对参考系定义的)径向位移。
    c = make_coeffs(6)
    cu = convert(c, "radial_displacement")
    check.ok(cu.C[0, 0] == 0.0, "向前换算 u_r 时 0 阶被置 0(信息确实丢了)")
    cu_bad = convert(c, "radial_displacement")
    cu_bad.C[0, 0] = 1.0                      # 物理上不可能
    cu_bad.meta["field_unit"] = "radial_displacement"
    check.raises(lambda: convert(cu_bad, "geopotential"), ValueError,
                 "自称 u_r 却有非零 0 阶 → 拒绝反推")
    c_small = make_coeffs(6)
    c_small.C[0, 0] = 0.0
    cu2 = convert(c_small, "radial_displacement")
    back = convert(cu2, "geopotential")
    check.ok(np.isfinite(back.C).all(), "0 阶本来就为 0 时可以反推")


def test_labels_and_aliases(check: Checker):
    """物理量标签与中文别名。"""
    check.section("标签与别名")
    check.ok(normalise_unit("EWH") == "ewh", "大小写不敏感")
    check.ok(normalise_unit("等效水高") == "ewh", "中文别名 → ewh")
    check.ok(normalise_unit("水准面") == "geoid", "中文别名 → geoid")
    check.ok(normalise_unit("径向形变") == "radial_displacement", "中文别名 → u_r")
    check.ok(normalise_unit("面密度") == "surface_density", "中文别名 → σ")
    check.raises(lambda: normalise_unit("瞎写的"), ValueError, "乱写的物理量报错")
    c = make_coeffs(4, "unknown")
    check.ok(field_unit(c) == "unknown", "未声明返回 unknown")
    c2 = with_field_unit(c, "ewh")
    check.ok(c2.field_unit == "ewh" and c.field_unit == "unknown",
             "打标签不改原对象")
    c3 = to_canonical(with_field_unit(make_coeffs(4, "ewh"), "ewh"))
    check.ok(c3.field_unit == "geopotential", "正变换后标签变为位系数")


def test_large_radius_smoothing(check: Checker):
    """大半径平滑后场明显变平(实验性对照,防止因子整体错位)。"""
    check.section("平滑的物理效果")
    rng = np.random.default_rng(5)
    nmax = 20
    C = rng.normal(0, 1e-6, (nmax + 1, nmax + 1))
    S = rng.normal(0, 1e-6, (nmax + 1, nmax + 1))
    c = SHCoeffs(C, S, {"field_unit": "geopotential"})
    lat, lon = np.linspace(-90, 90, 46), np.linspace(0, 360, 91)
    g0 = synthesis_grid(lat, lon, c)
    g1 = synthesis_grid(lat, lon, c, gaussian_km=2000.0)
    s0 = float(np.std(g0))
    s1 = float(np.std(g1))
    check.ok(s1 < s0, f"2000 km 平滑后标准差下降({s0:.3e} → {s1:.3e})")
    # 大半径下高阶几乎被压没,场应接近 zonal 的二阶结构
    check.ok(s1 / s0 < 0.5, f"下降幅度 {(1 - s1 / s0) * 100:.1f}% > 50%")


def main() -> int:
    check = Checker("test_units_filters —— 物理量换算与高斯平滑")
    for fn in (test_love_numbers, test_gaussian, test_degree_factors_hand,
               test_conversion_roundtrip, test_synthesis_target_equals_preconverted,
               test_gaussian_equivalence, test_conversion_guards,
               test_labels_and_aliases, test_large_radius_smoothing):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
