# -*- coding: utf-8 -*-
"""
shsynth.filters
===============

逐阶(各向同性)滤波:高斯平滑。

公式与参考实现 ``m2py`` / SHKit 完全相同(Jekeli/Wahr):

.. math::

    b = \\frac{\\ln 2}{1-\\cos(r/R)},\\qquad
    W(\\alpha) = \\frac{b\\,e^{-b(1-\\cos\\alpha)}}{1-e^{-2b}}

.. math::

    W_n = \\frac{1}{2}\\int_{-1}^{1} W(\\cos\\alpha)\\,P_n(\\cos\\alpha)\\,
          \\mathrm{d}(\\cos\\alpha)

用 Gauss–Legendre 求积(``method='glq'``,各阶都稳定)或经典正向递推
(``method='frc'``,快但高阶不稳)。``W_0 ≡ 1``(强制为单位增益)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .coeffs import SHCoeffs

__all__ = [
    "R_EARTH_KM",
    "EARTH_RADIUS_M",
    "RHO_AVE",
    "RHO_WATER",
    "gaussian_coefficients",
    "apply_degree_filter",
    "apply_gaussian",
]

#: 高斯滤波用的地球平均半径(km),与参考实现一致。
R_EARTH_KM = 6378.1363
#: EWH / 面密度 / 径向形变因子用的赤道半径(m)。
EARTH_RADIUS_M = 6378136.46
#: 地球平均密度(kg/m³)。
RHO_AVE = 5517.0
#: 水的密度(kg/m³)。
RHO_WATER = 1000.0


def gaussian_coefficients(radius_km: float, nmax: int, method: str = "glq",
                          ngl: Optional[int] = None) -> np.ndarray:
    """各向同性高斯滤波系数 ``W_n``(``n = 0..nmax``)。

    Parameters
    ----------
    radius_km : float
        平均半径。``<= 0`` 时返回全 1(即不滤波)。
    nmax : int
        最高阶。
    method : {'glq', 'frc'}
        ``'glq'``(默认)为 Gauss–Legendre 求积,任意阶都稳定;
        ``'frc'`` 是经典正向递推,快但在约 300 km 以上半径会失精度。
    ngl : int, optional
        ``'glq'`` 的节点数(默认 ``max(501, 2*nmax+51)``,强制奇数)。
    """
    if radius_km <= 0:
        return np.ones(nmax + 1)

    b = np.log(2.0) / (1.0 - np.cos(radius_km / R_EARTH_KM))

    if method == "frc":
        W = np.ones(nmax + 1)
        if nmax >= 1:
            e2b = np.exp(-2.0 * b)
            W[1] = (1.0 + e2b) / (1.0 - e2b) - 1.0 / b
            for n in range(1, nmax):
                W[n + 1] = -((2.0 * n + 1.0) / b) * W[n] + W[n - 1]
        return W

    if method != "glq":
        raise ValueError(f"未知的 method {method!r};可选 'glq' | 'frc'")

    if ngl is None:
        ngl = max(501, 2 * nmax + 51)
    if ngl % 2 == 0:
        ngl += 1
    x, gw = np.polynomial.legendre.leggauss(ngl)
    Wa = b * np.exp(-b * (1.0 - x)) / (1.0 - np.exp(-2.0 * b))
    P = np.zeros((nmax + 1, ngl))
    P[0] = 1.0
    if nmax >= 1:
        P[1] = x
    for n in range(1, nmax):
        P[n + 1] = ((2.0 * n + 1.0) * x * P[n] - n * P[n - 1]) / (n + 1.0)
    W = P @ (gw * Wa)
    # 核函数只定义到一个常数,用 W_0 == 1 定标;这同时消掉了小半径下
    # 尖峰被积函数的求积误差(实测 r=100 km 时 1.1e-10)。
    return W / W[0]


def apply_degree_filter(coeffs: SHCoeffs, W: np.ndarray) -> SHCoeffs:
    """把第 ``n`` 阶的所有系数乘以 ``W[n]``。"""
    W = np.asarray(W, dtype=float).ravel()
    L = coeffs.nmax
    if W.size < L + 1:
        raise ValueError(f"需要至少 {L + 1} 个滤波系数,实际 {W.size}")
    w = W[:L + 1]
    C = coeffs.C * w[:, None] if coeffs.C.ndim == 2 else \
        coeffs.C * w[:, None, None]
    S = coeffs.S * w[:, None] if coeffs.S.ndim == 2 else \
        coeffs.S * w[:, None, None]
    meta = dict(coeffs.meta)
    meta["degree_filter"] = "applied"
    return SHCoeffs(C, S, meta)


def apply_gaussian(coeffs: SHCoeffs, radius_km: float,
                   method: str = "glq") -> SHCoeffs:
    """便利函数:直接对系数做高斯平滑(返回新对象)。"""
    W = gaussian_coefficients(radius_km, coeffs.nmax, method=method)
    out = apply_degree_filter(coeffs, W)
    out.meta["gaussian_km"] = radius_km
    out.meta["gaussian_method"] = method
    return out
