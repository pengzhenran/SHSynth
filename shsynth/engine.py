# -*- coding: utf-8 -*-
"""
shsynth.engine
==============

球谐**综合**(系数 → 任意点 / 任意网格上的值)。

综合不需要积分元,只需要正确的归一化,所以对任何几何(全球/区域、规则/散乱)
都成立。

4π 归一化连带勒让德函数用**跨阶**递推(参考 ``pz_legendre.m`` 的 method 3,
与 SHKit ``shkit.basis`` 逐运算一致):

.. math::

    \\bar P_{n,m} = \\alpha_{n,m}\\bar P_{n-2,m}
                  + \\beta_{n,m}\\bar P_{n-2,m-2}
                  - \\gamma_{n,m}\\bar P_{n,m-2} \\qquad (m \\ge 2)

内存策略:朴素实现会materialize ``(nmax+1, nmax+1, npoints)`` 的三维立方体
(``nmax=100``、10 万点约 8 GB),这里只保留当前 ``m`` 列、``m-1`` 列和 ``m-2``
列,峰值内存约 ``2*(nmax+1)*npoints`` 个 double;更大点集再按 ``chunk`` 分块。
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

from .coeffs import SHCoeffs
from .filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                      gaussian_coefficients)

__all__ = [
    "legendre_columns",
    "legendre_columns_vec",
    "legendre_pbar",
    "q_at_pole_m1",
    "degree_scale",
    "synthesize",
    "synthesis_grid",
    "synthesis_grid_fft",
    "fft_path_applicable",
    "synthesize_horizontal",
    "evaluate_horizontal",
    "horizontal_grid_fft",
    "love_horizontal_factors",
    "regular_grid",
    "fibonacci_points",
    "DEFAULT_CHUNK",
    "DEFAULT_TIME_CHUNK",
    "LOVE_SOURCE",
]

#: 分块大小(点数)。每个块只保留两列勒让德函数,100 万点也不会爆内存。
DEFAULT_CHUNK = 200_000

#: FFT 经度路径内部的**时间**分块大小。临时数组是 ``(nlat, ntc, nlon)`` 复数,
#: ``ntc=32`` 时 0.5° 全球约 133 MB、1° 全球约 33 MB。与 ``chunk``(按点数)是两件事。
DEFAULT_TIME_CHUNK = 32

#: 本模块里"逐阶因子"的来源说明(报告里打印用)
LOVE_SOURCE = "随包载荷勒夫数表(PREM / Wang 2012)"


# ---------------------------------------------------------------------------
# 勒让德递推
# ---------------------------------------------------------------------------
def _legendre_column_forward(col: np.ndarray, m: int, nmax: int,
                             t: np.ndarray, start: int) -> None:
    """填充 ``m = 0``(start=1)或 ``m = 1``(start=2)这条列。"""
    for n in range(start, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
        num = (2 * n + 1) * (n + m - 1) * (n - m - 1)
        c2 = np.sqrt(num / ((2 * n - 3) * (n + m) * (n - m))) if num > 0 else 0.0
        col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]


def legendre_columns(lat_deg, nmax: int) -> Iterator[Tuple[int, np.ndarray]]:
    """逐列产出 ``(m, block)``,其中 ``block[n - m, :] == P̄_{n,m}(lat)``。

    ``n`` 从 ``m`` 走到 ``nmax``,所以 ``block`` 形状为
    ``(nmax - m + 1, npoints)``。

    Parameters
    ----------
    lat_deg : array_like
        地心纬度(度)。
    nmax : int
        最高阶。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    t = np.sin(np.deg2rad(lat))          # sin(lat)
    u = np.cos(np.deg2rad(lat))          # cos(lat)

    if nmax < 0:
        raise ValueError("nmax 必须 >= 0")

    keep: dict = {}

    # ---- m = 0 ----------------------------------------------------------
    col = np.zeros((nmax + 1, npts))
    col[0] = 1.0
    if nmax >= 1:
        _legendre_column_forward(col, 0, nmax, t, start=1)
    keep[0] = col
    yield 0, col[0:nmax + 1]

    if nmax >= 1:
        # ---- m = 1 ------------------------------------------------------
        col = np.zeros((nmax + 1, npts))
        col[1] = np.sqrt(3.0) * u
        if nmax >= 2:
            _legendre_column_forward(col, 1, nmax, t, start=2)
        keep[1] = col
        yield 1, col[1:nmax + 1]

    # ---- m >= 2 ---------------------------------------------------------
    for m in range(2, nmax + 1):
        cm2 = keep.pop(m - 2)
        col = np.zeros((nmax + 1, npts))
        for n in range(m, nmax + 1):
            a1 = np.sqrt((2 * n + 1) * (n - m) * (n - m - 1) /
                         ((2 * n - 3) * (n + m) * (n + m - 1)))
            if m == 2:
                g1 = np.sqrt(2.0) * np.sqrt((n - m + 1) * (n - m + 2) /
                                            ((n + m) * (n + m - 1)))
                b1 = np.sqrt(2.0) * np.sqrt((2 * n + 1) * (n + m - 2) *
                                            (n + m - 3) /
                                            ((2 * n - 3) * (n + m) *
                                             (n + m - 1)))
            else:
                g1 = np.sqrt((n - m + 1) * (n - m + 2) /
                             ((n + m) * (n + m - 1)))
                b1 = np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                             ((2 * n - 3) * (n + m) * (n + m - 1)))
            col[n] = a1 * col[n - 2] + b1 * cm2[n - 2] - g1 * cm2[n]
        keep[m] = col
        yield m, col[m:nmax + 1]


def legendre_pbar(lat_deg, nmax: int, chunk: Optional[int] = None) -> np.ndarray:
    """全部 ``m <= n <= nmax`` 的 ``P̄_{n,m}(sin lat)``。

    返回 ``(npoints, (nmax+1)(nmax+2)/2)``,列序为 m2py 三角顺序
    (m 外层 / n 内层)。``chunk`` 只限制中间列的内存占用,结果仍是完整的。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    npts = lat.size
    NC = (nmax + 1) * (nmax + 2) // 2
    out = np.empty((npts, NC))

    if chunk is None or chunk >= npts:
        idx = 0
        for _m, block in legendre_columns(lat, nmax):
            k = block.shape[0]
            out[:, idx:idx + k] = block.T
            idx += k
        return out

    for s in range(0, npts, chunk):
        e = min(s + chunk, npts)
        idx = 0
        for _m, block in legendre_columns(lat[s:e], nmax):
            k = block.shape[0]
            out[s:e, idx:idx + k] = block.T
            idx += k
    return out


# ---------------------------------------------------------------------------
# 逐阶缩放
# ---------------------------------------------------------------------------
def _unit_ratio(coeffs: SHCoeffs, target_unit: str,
                love_numbers, love_numbers_h, radius_m, rho_ave, rho_water,
                allow_unit_mismatch: bool) -> np.ndarray:
    """把 ``coeffs`` 声明的物理量换成 ``target_unit`` 的逐阶因子 ``f_t/f_u``。"""
    from . import units

    src = units.field_unit(coeffs)
    if units.is_no_conversion(target_unit):
        return np.ones(coeffs.nmax + 1)          # 「不换算」
    tgt = units.normalise_unit(target_unit)
    if src == tgt:
        return np.ones(coeffs.nmax + 1)

    problem = units.require_convertible(src, tgt)
    if problem is not None and not allow_unit_mismatch:
        raise ValueError(
            f"target_unit={target_unit!r} 的单位不一致:\n" + problem +
            "\n\n若确实要强行换算,传 allow_unit_mismatch=True;\n"
            "若本意就是「什么都不换算」,请把 target_unit 留空(传 None)。")

    kw = dict(love_numbers=love_numbers, love_numbers_h=love_numbers_h,
              radius_m=radius_m, rho_ave=rho_ave, rho_water=rho_water)
    f_dst = units.degree_factors(tgt, coeffs.nmax, **kw)
    if src in ("scalar", "unknown"):
        # 只有 allow_unit_mismatch=True 才可能走到这里:假定数字本身就是位系数
        return f_dst
    f_src = units.degree_factors(src, coeffs.nmax, **kw)
    return f_dst / f_src


def degree_scale(coeffs: SHCoeffs,
                 gaussian_km: float = 0.0,
                 love_numbers=None,
                 gaussian_method: str = "glq",
                 extra: Optional[np.ndarray] = None) -> np.ndarray:
    """综合时施加的逐阶乘性因子(不改动存下来的系数)。"""
    L = coeffs.nmax
    s = np.ones(L + 1)
    if gaussian_km and gaussian_km > 0:
        s = s * gaussian_coefficients(gaussian_km, L, method=gaussian_method)
    if extra is not None:
        e = np.asarray(extra, dtype=float).ravel()
        s = s * e[:L + 1]
    return s


# ---------------------------------------------------------------------------
# 综合
# ---------------------------------------------------------------------------
def synthesize(lat_deg, lon_deg, C, S, nmax: Optional[int] = None,
               weights_scale=1.0) -> np.ndarray:
    """在任意点上求值(核心例程,不做分块、不做单位换算)。

    ``weights_scale`` 为逐阶(或全局)乘子,用于把高斯滤波、物理量换算折进来
    而不改动系数。

    Returns
    -------
    ndarray
        ``(npoints,)`` 或 ``(npoints, ntime)``。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)

    if C.ndim == 2:
        C3, S3 = C[:, :, None], S[:, :, None]
    else:
        C3, S3 = C, S
    ntime = C3.shape[2]

    if nmax is None:
        nmax = C3.shape[0] - 1
    nmax = min(int(nmax), C3.shape[0] - 1)

    if np.ndim(weights_scale) == 0:
        ws = np.full(nmax + 1, float(weights_scale))
    else:
        ws = np.asarray(weights_scale, dtype=float)[:nmax + 1]

    lam = np.deg2rad(lon)
    out = np.zeros((lat.size, ntime))
    for m, block in legendre_columns(lat, nmax):
        n_idx = np.arange(m, nmax + 1)
        pb = block * ws[n_idx][:, None]                       # (n-m+1, npts)
        c = C3[m:nmax + 1, m, :]                              # (n-m+1, ntime)
        s = S3[m:nmax + 1, m, :]
        acc = pb.T @ c                                        # (npts, ntime)
        if m >= 1:
            cm = np.cos(m * lam)[:, None]
            sm = np.sin(m * lam)[:, None]
            acc = acc * cm
            acc = acc + (pb.T @ s) * sm
        out += acc
    return out[:, 0] if ntime == 1 else out


def evaluate(lat_deg, lon_deg, coeffs: SHCoeffs,
             nmax: Optional[int] = None,
             gaussian_km: float = 0.0,
             gaussian_method: str = "glq",
             target_unit: Optional[str] = None,
             love_numbers=None,
             love_numbers_h=None,
             chunk: int = DEFAULT_CHUNK,
             allow_unit_mismatch: bool = False,
             progress=None) -> np.ndarray:
    """在一组点上综合(自动分块 + 高斯平滑 + 物理量换算)。

    Parameters
    ----------
    lat_deg, lon_deg : array_like
        目标坐标(度)。
    coeffs : SHCoeffs
        系数(可多时次)。
    nmax : int, optional
        截断到该阶(默认用系数自身的 ``nmax``)。
    gaussian_km : float
        各向同性高斯平滑半径(km),在综合时施加,**不改动系数**。
    target_unit : str, optional
        目标物理量(:data:`shsynth.units.FIELD_UNITS` 之一,也接受中文别名)。
        这是**换算**,不是显示选项:若系数的 ``field_unit`` 未声明或不可换算,
        会**报错**,而不是悄悄给出差 ``1e7`` 倍的数字。
    chunk : int
        每块点数(内存控制)。
    progress : callable, optional
        ``progress(done, total)`` 进度回调(界面用)。

    Returns
    -------
    ndarray
        ``(npoints,)`` 或 ``(npoints, ntime)``。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if lat.size != lon.size:
        raise ValueError("lat 与 lon 长度必须一致")

    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    if L < 0:
        raise ValueError("nmax 必须 >= 0")

    ws = degree_scale(coeffs, gaussian_km, gaussian_method=gaussian_method)[:L + 1]
    if target_unit is not None:
        ratio = _unit_ratio(coeffs, target_unit, love_numbers, love_numbers_h,
                            EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                            allow_unit_mismatch)
        ws = ws * ratio[:L + 1]

    n = lat.size
    if n <= chunk:
        return synthesize(lat, lon, coeffs.C, coeffs.S, L, weights_scale=ws)

    ntime = coeffs.ntime
    out = np.empty((n, ntime))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = np.atleast_2d(
            synthesize(lat[s:e], lon[s:e], coeffs.C, coeffs.S, L,
                       weights_scale=ws).reshape(e - s, -1))
        if progress is not None:
            progress(e, n)
    return out[:, 0] if ntime == 1 else out


def regular_grid(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                 lat_step: float, lon_step: float):
    """构造规则经纬采样(两端点都含);返回 ``(lat_vec, lon_vec)``,lat 递增。"""
    if lat_step <= 0 or lon_step <= 0:
        raise ValueError("网格步长必须为正")
    nlat = int(round((lat_max - lat_min) / lat_step)) + 1
    nlon = int(round((lon_max - lon_min) / lon_step)) + 1
    if nlat < 1 or nlon < 1:
        raise ValueError("网格范围与步长不匹配(算出的格点数为 0)")
    lat = np.linspace(lat_min, lat_max, nlat)
    lon = np.linspace(lon_min, lon_max, nlon)
    return lat, lon


def synthesis_grid(lat_vec, lon_vec, coeffs: SHCoeffs,
                   nmax: Optional[int] = None,
                   gaussian_km: float = 0.0,
                   gaussian_method: str = "glq",
                   target_unit: Optional[str] = None,
                   love_numbers=None,
                   love_numbers_h=None,
                   chunk: int = DEFAULT_CHUNK,
                   allow_unit_mismatch: bool = False,
                   progress=None) -> np.ndarray:
    """在规则经纬网格上综合,返回 ``(nlat, nlon)`` 或 ``(nlat, nlon, ntime)``。"""
    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    vals = evaluate(LA.ravel(), LO.ravel(), coeffs, nmax=nmax,
                    gaussian_km=gaussian_km, gaussian_method=gaussian_method,
                    target_unit=target_unit, love_numbers=love_numbers,
                    love_numbers_h=love_numbers_h, chunk=chunk,
                    allow_unit_mismatch=allow_unit_mismatch, progress=progress)
    if vals.ndim == 1:
        return vals.reshape(lat.size, lon.size)
    return vals.reshape(lat.size, lon.size, vals.shape[1])


# ---------------------------------------------------------------------------
# 水平形变:球面梯度算子(方案 §4.9)
# ---------------------------------------------------------------------------
def q_at_pole_m1(nmax: int, t_pole: float) -> np.ndarray:
    """``m=1`` 时 ``Q_n1 = P̄_n1/sinθ`` 在极点的**精确**值。

    ``Q_n1`` 是 ``cosθ`` 的 ``n-1`` 次多项式,所以在极点直接跑同一条 ``m=1``
    递推(种子 ``Q_11 = √3``)并取 ``t = ±1`` 即得极限 —— 实测与"极近纬度外推"
    逐位相同。**不能写成 0 或 NaN**(方案 §4.9.2)。
    """
    q = np.zeros(max(nmax, 0) + 1)
    if nmax < 1:
        return q
    q[1] = np.sqrt(3.0)
    for n in range(2, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - 1) * (n + 1)))
        num = (2 * n + 1) * (n + 1 - 1) * (n - 1 - 1)
        c2 = np.sqrt(num / ((2 * n - 3) * (n + 1) * (n - 1))) if num > 0 else 0.0
        q[n] = c1 * t_pole * q[n - 1] - c2 * q[n - 2]
    return q


def legendre_columns_vec(lat_deg, nmax: int):
    """逐列产出 ``(m, P, dP, Q)`` —— 水平形变要用的三套量。

    * ``P[n-m, :]  = P̄_nm(cosθ)``
    * ``dP[n-m, :] = dP̄_nm/dθ`` —— 把 :func:`legendre_columns` 的**每条递推两边
      对 θ 求导**(``t = cosθ``,``dt/dθ = −sinθ``),**全程不出现 1/sinθ**,
      所以极点天然安全;
    * ``Q[n-m, :]  = P̄_nm/sinθ``(``m≥1``;``m=0`` 时为 ``None``)—— 由同列 ``P``
      直接除 ``sinθ``,只在**极点行**用解析极限覆盖(``m=1`` 用
      :func:`q_at_pole_m1`,`m≥2` 恰为 0)。

    内存策略与 :func:`legendre_columns` 相同:只保留 ``m``/``m-1``/``m-2`` 三列。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    t = np.sin(np.deg2rad(lat))          # = cos(theta)
    u = np.cos(np.deg2rad(lat))          # = sin(theta)
    at_pole = np.abs(np.abs(lat) - 90.0) < 1e-9
    is_north = at_pole & (lat > 0)
    is_south = at_pole & (lat < 0)
    safe = np.where(np.abs(u) < 1e-12, 1.0, u)
    qp_n = q_at_pole_m1(nmax, 1.0)
    qp_s = q_at_pole_m1(nmax, -1.0)

    def qcol(col, m):
        if m == 0:
            return None
        q = col / safe
        if at_pole.any():
            q = q.copy()
            if m == 1:
                q[:, is_north] = qp_n[:, None]
                q[:, is_south] = qp_s[:, None]
            else:
                q[:, at_pole] = 0.0
        return q

    if nmax < 0:
        raise ValueError("nmax 必须 >= 0")
    keep: dict = {}

    # ---- m = 0 ----------------------------------------------------------
    col = np.zeros((nmax + 1, lat.size))
    dcol = np.zeros((nmax + 1, lat.size))
    col[0] = 1.0
    for n in range(1, nmax + 1):
        c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / (n * n))
        num = (2 * n + 1) * (n - 1) * (n - 1)
        c2 = np.sqrt(num / ((2 * n - 3) * n * n)) if num > 0 else 0.0
        col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]
        dcol[n] = c1 * (-u) * col[n - 1] + c1 * t * dcol[n - 1] - c2 * dcol[n - 2]
    keep[0] = (col, dcol)
    yield 0, col[0:nmax + 1], dcol[0:nmax + 1], None

    if nmax >= 1:
        # ---- m = 1:P̄_11 = √3 u ⇒ dP̄_11/dθ = √3 t ------------------------
        col = np.zeros((nmax + 1, lat.size))
        dcol = np.zeros((nmax + 1, lat.size))
        col[1] = np.sqrt(3.0) * u
        dcol[1] = np.sqrt(3.0) * t
        m = 1
        for n in range(2, nmax + 1):
            c1 = np.sqrt((2 * n - 1) * (2 * n + 1) / ((n - m) * (n + m)))
            num = (2 * n + 1) * (n + m - 1) * (n - m - 1)
            c2 = np.sqrt(num / ((2 * n - 3) * (n + m) * (n - m))) if num > 0 else 0.0
            col[n] = c1 * t * col[n - 1] - c2 * col[n - 2]
            dcol[n] = c1 * (-u) * col[n - 1] + c1 * t * dcol[n - 1] - c2 * dcol[n - 2]
        keep[1] = (col, dcol)
        yield 1, col[1:nmax + 1], dcol[1:nmax + 1], qcol(col, 1)[1:nmax + 1]

    # ---- m >= 2:col[n] = a1 col[n-2] + b1 cm2[n-2] - g1 cm2[n](无显式 θ)
    for m in range(2, nmax + 1):
        cm2, dcm2 = keep.pop(m - 2)
        col = np.zeros((nmax + 1, lat.size))
        dcol = np.zeros((nmax + 1, lat.size))
        for n in range(m, nmax + 1):
            a1 = np.sqrt((2 * n + 1) * (n - m) * (n - m - 1) /
                         ((2 * n - 3) * (n + m) * (n + m - 1)))
            if m == 2:
                g1 = np.sqrt(2.0) * np.sqrt((n - m + 1) * (n - m + 2) /
                                            ((n + m) * (n + m - 1)))
                b1 = np.sqrt(2.0) * np.sqrt((2 * n + 1) * (n + m - 2) *
                                            (n + m - 3) /
                                            ((2 * n - 3) * (n + m) * (n + m - 1)))
            else:
                g1 = np.sqrt((n - m + 1) * (n - m + 2) /
                             ((n + m) * (n + m - 1)))
                b1 = np.sqrt((2 * n + 1) * (n + m - 2) * (n + m - 3) /
                             ((2 * n - 3) * (n + m) * (n + m - 1)))
            col[n] = a1 * col[n - 2] + b1 * cm2[n - 2] - g1 * cm2[n]
            dcol[n] = a1 * dcol[n - 2] + b1 * dcm2[n - 2] - g1 * dcm2[n]
        keep[m] = (col, dcol)
        yield m, col[m:nmax + 1], dcol[m:nmax + 1], qcol(col, m)[m:nmax + 1]


def love_horizontal_factors(nmax: int, love_l=None, love_k=None,
                            radius_m: float = EARTH_RADIUS_M) -> np.ndarray:
    """水平形变的逐阶因子 ``F_n = R·l′_n/(1+k′_n)``(米)。

    ⚠️ **不要**再乘 ``(2n+1)/3`` 或 ``ρ̄/ρ_w`` —— 那是 EWH/面密度的因子。
    实测水平与径向同为 mm 量级;乘错会立刻差 1e7 倍(方案 §4.9.1)。
    """
    from .lovenumbers import load_lln
    d = load_lln()
    l_ = d["l"] if love_l is None else np.asarray(love_l, dtype=float)
    k_ = d["k"] if love_k is None else np.asarray(love_k, dtype=float)
    n = min(len(l_), len(k_), nmax + 1)
    out = np.zeros(nmax + 1)
    out[:n] = radius_m * l_[:n] / (1.0 + k_[:n])
    if n < nmax + 1:
        raise ValueError(
            f"载荷勒夫数表只有 {min(len(l_), len(k_))} 阶,而要算到 {nmax} 阶;"
            "缺失的阶按 0 处理等于漏掉这部分水平形变")
    return out


def synthesize_horizontal(lat_deg, lon_deg, C, S, nmax: Optional[int] = None,
                          weights_scale=1.0, want=("north", "east"),
                          with_magnitude: bool = True) -> dict:
    """在任意点上求**水平形变** ``u_N`` / ``u_E``(核心例程)。

    ``u_N = −Σ_n F_n ∂S_n/∂θ``,``u_E = +Σ_n F_n (1/sinθ) ∂S_n/∂λ``。

    返回 ``dict``;每个值是 ``(npoints,)`` 或 ``(npoints, ntime)``。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float))
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)
    if C.ndim == 2:
        C3, S3 = C[:, :, None], S[:, :, None]
    else:
        C3, S3 = C, S
    ntime = C3.shape[2]
    nmax = C3.shape[0] - 1 if nmax is None else min(int(nmax), C3.shape[0] - 1)
    ws = np.asarray(weights_scale, dtype=float)
    if ws.ndim == 0:
        ws = np.full(nmax + 1, float(ws))

    lam = np.deg2rad(lon)
    uN = np.zeros((lat.size, ntime))
    uE = np.zeros((lat.size, ntime))
    for m, pb, dpb, qb in legendre_columns_vec(lat, nmax):
        n_idx = np.arange(m, nmax + 1)
        f = ws[n_idx][:, None]                       # (K,1)
        c = C3[m:nmax + 1, m, :]
        s = S3[m:nmax + 1, m, :]
        cm = np.cos(m * lam)[None, :]
        sm = np.sin(m * lam)[None, :]
        # u_N = −Σ F_n (∂P̄/∂θ)[C cos mλ + S sin mλ](m=0 也有贡献!)
        wN = f * dpb
        uN += -((wN * cm).T @ c + (wN * sm).T @ s)
        if m >= 1:
            # u_E = +Σ F_n (m/sinθ) P̄ [S cos mλ − C sin mλ]
            wE = f * m * qb
            uE += (wE * cm).T @ s - (wE * sm).T @ c
    out = {}
    if ntime == 1:
        uN, uE = uN[:, 0], uE[:, 0]
    if "north" in want:
        out["north"] = uN
    if "east" in want:
        out["east"] = uE
    if with_magnitude:
        out["magnitude"] = np.hypot(uN, uE)
        out["azimuth"] = np.rad2deg(np.arctan2(uE, uN)) % 360.0
    return out


def evaluate_horizontal(lat_deg, lon_deg, coeffs: "SHCoeffs",
                        nmax: Optional[int] = None,
                        gaussian_km: float = 0.0,
                        gaussian_method: str = "glq",
                        chunk: int = DEFAULT_CHUNK,
                        love_l=None, love_k=None,
                        want=("north", "east")) -> dict:
    """水平形变的分块版本(自动按点数分块 + 高斯平滑 + 载荷勒夫数)。

    平均面形变因子 ``R·l′_n/(1+k′_n)`` 已折进逐阶乘子,所以"换算只做一次"。
    """
    lat = np.atleast_1d(np.asarray(lat_deg, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_deg, dtype=float)).ravel()
    if lat.size != lon.size:
        raise ValueError("lat 与 lon 长度必须一致")
    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    f = love_horizontal_factors(L, love_l, love_k)
    ws = f * degree_scale(coeffs, gaussian_km, gaussian_method=gaussian_method)[:L + 1]
    n = lat.size
    if n <= chunk:
        return synthesize_horizontal(lat, lon, coeffs.C, coeffs.S, L,
                                     weights_scale=ws, want=want)
    acc: dict = {}
    for s0 in range(0, n, chunk):
        e0 = min(s0 + chunk, n)
        part = synthesize_horizontal(lat[s0:e0], lon[s0:e0], coeffs.C, coeffs.S,
                                     L, weights_scale=ws, want=want)
        for k, v in part.items():
            acc.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def horizontal_grid_fft(lat_vec, lon_vec, coeffs: "SHCoeffs",
                        nmax: Optional[int] = None, gaussian_km: float = 0.0,
                        gaussian_method: str = "glq",
                        time_chunk: int = DEFAULT_TIME_CHUNK,
                        love_l=None, love_k=None, workers: int = -1) -> dict:
    """水平形变的 **FFT 经度路径**(两个复谱 + 两次 IFFT)。

    ``F_N,m = −Σ F_n (∂P̄/∂θ)(C − iS)``、``F_E,m = +i·m·Σ F_n (P̄/sinθ)(C − iS)``,
    再各做一次 ``Re(nlon·ifft(·))``。要求与标量 FFT 路径相同(整圈均匀经度、
    ``nlon > 2·nmax``)。
    """
    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    C = np.asarray(coeffs.C, dtype=float)
    S = np.asarray(coeffs.S, dtype=float)
    if C.ndim == 2:
        C, S = C[:, :, None], S[:, :, None]
    ntime = C.shape[2]
    nlat, nlon = lat.size, lon.size
    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    ok, why = fft_path_applicable(lon, L)
    if not ok:
        raise ValueError(f"horizontal_grid_fft 不适用: {why}")
    F = love_horizontal_factors(L, love_l, love_k) * \
        degree_scale(coeffs, gaussian_km, gaussian_method=gaussian_method)[:L + 1]

    uN = np.empty((nlat, nlon, ntime))
    uE = np.empty((nlat, nlon, ntime))
    for t0 in range(0, ntime, time_chunk):
        t1 = min(t0 + time_chunk, ntime)
        ntc = t1 - t0
        sN = np.zeros((nlat, ntc, nlon), dtype=np.complex128)
        sE = np.zeros((nlat, ntc, nlon), dtype=np.complex128)
        for m, pb, dpb, qb in legendre_columns_vec(lat, L):
            n_idx = np.arange(m, L + 1)
            f = F[n_idx][:, None]
            cc = C[m:L + 1, m, t0:t1] - 1j * S[m:L + 1, m, t0:t1]
            sN[:, :, m] = -((f * dpb).T @ cc)        # m=0 也要(∂P̄_n0/∂θ ≠ 0)
            if m >= 1:
                sE[:, :, m] = 1j * m * ((f * qb).T @ cc)
        uN[:, :, t0:t1] = (_ifft_along_lon(sN, workers).real
                           * nlon).transpose(0, 2, 1)
        uE[:, :, t0:t1] = (_ifft_along_lon(sE, workers).real
                           * nlon).transpose(0, 2, 1)
    if ntime == 1:
        uN, uE = uN[:, :, 0], uE[:, :, 0]
    return {"north": uN, "east": uE, "magnitude": np.hypot(uN, uE),
            "azimuth": np.rad2deg(np.arctan2(uE, uN)) % 360.0}


def fibonacci_points(n: int, seed: int = 0):
    """确定性全球准均匀 Fibonacci 球面采样,返回 ``(lat, lon)``(度)。

    用于「不给点位、只要一张全球示意网格/球面点」的快速出图场景。
    """
    n = int(n)
    if n < 1:
        raise ValueError("点数必须 >= 1")
    i = np.arange(n, dtype=float) + 0.5
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    z = 1.0 - 2.0 * i / n
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.rad2deg(phi) % 360.0
    if seed:
        rng = np.random.default_rng(int(seed))
        lon = (lon + rng.uniform(0, 360.0)) % 360.0
    return lat, lon


# ---------------------------------------------------------------------------
# FFT 经度快路径(方案 §6.5)
# ---------------------------------------------------------------------------
def fft_path_applicable(lon_vec, nmax: int) -> tuple:
    """判断目标经度网格能不能走 FFT 路径。

    条件(方案 §6.5.5 / §6.5.7):

    1. 经度是**整圈、均匀**网格(区域经度范围、散点都不行);
    2. ``nlon > 2·nmax``(否则 ``m ≥ nlon/2`` 会折叠到低频,Nyquist 混叠 ——
       这不是"慢一点",而是**会算错**)。

    **纬度怎么取不影响判定**:FFT 只作用在经度方向,纬向区域照样能用。

    Returns
    -------
    (ok, reason)
        ``ok=False`` 时 ``reason`` 是中文原因(可直接打印给用户)。
    """
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    n = lon.size
    if n < 2:
        return False, "经度点数 < 2"
    if n <= 2 * int(nmax):
        return False, (f"nlon={n} ≤ 2·nmax={2 * int(nmax)}:"
                       "高阶 m 会超过 Nyquist 混叠(用直接法)")
    step = 360.0 / n
    ref = np.arange(n) * step
    if not np.allclose(np.mod(lon, 360.0), ref, atol=1e-9):
        return False, "经度不是 360*j/nlon 的整圈均匀网格(区域经度/散点请用直接法)"
    return True, ""


def _ifft_along_lon(spec, workers=-1):
    """沿最后一维做逆 FFT(优先 scipy,可用多线程)。"""
    try:
        from scipy.fft import ifft
        return ifft(spec, axis=-1, workers=workers)
    except Exception:                                        # noqa: BLE001
        return np.fft.ifft(spec, axis=-1)


def synthesis_grid_fft(lat_vec, lon_vec, coeffs: "SHCoeffs",
                       nmax: Optional[int] = None,
                       gaussian_km: float = 0.0,
                       gaussian_method: str = "glq",
                       target_unit: Optional[str] = None,
                       love_numbers=None,
                       love_numbers_h=None,
                       allow_unit_mismatch: bool = False,
                       time_chunk: int = DEFAULT_TIME_CHUNK,
                       workers: int = -1,
                       progress=None) -> np.ndarray:
    """**FFT 经度路径**的规则网格综合(方案 §6.5)。

    与 :func:`synthesis_grid` **逐点一致**(实测相对场量级 ~1e-14),
    但在"整圈均匀经度 + ``nlon > 2·nmax``"时快 **30~130 倍**:把经度方向的
    显式 ``cos mλ/sin mλ`` 展开换成一次批量 IFFT。

    ``lon_vec`` 必须整圈均匀;不满足时请用 :func:`synthesis_grid`(调用方应先用
    :func:`fft_path_applicable` 判定并**报告**走了哪条路)。

    Returns
    -------
    ndarray
        ``(nlat, nlon)`` 或 ``(nlat, nlon, ntime)``。
    """
    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    C = np.asarray(coeffs.C, dtype=float)
    S = np.asarray(coeffs.S, dtype=float)
    if C.ndim == 2:
        C, S = C[:, :, None], S[:, :, None]
    ntime = C.shape[2]
    nlat, nlon = lat.size, lon.size

    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    ok, why = fft_path_applicable(lon, L)
    if not ok:
        raise ValueError(f"synthesis_grid_fft 不适用: {why}")

    ws = degree_scale(coeffs, gaussian_km, gaussian_method=gaussian_method)
    if target_unit is not None:
        ratio = _unit_ratio(coeffs, target_unit, love_numbers, love_numbers_h,
                            EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                            allow_unit_mismatch)
        ws = ws * ratio
    ws = np.asarray(ws, dtype=float)[:L + 1]

    out = np.empty((nlat, nlon, ntime))
    total = max(1, (ntime + time_chunk - 1) // max(1, time_chunk))
    done = 0
    for t0 in range(0, ntime, time_chunk):
        t1 = min(t0 + time_chunk, ntime)
        ntc = t1 - t0
        spec = np.zeros((nlat, ntc, nlon), dtype=np.complex128)
        for m, pb in legendre_columns(lat, L):
            n_idx = np.arange(m, L + 1)
            p = pb * ws[n_idx][:, None]                  # (K, nlat)
            c = C[m:L + 1, m, t0:t1]                     # (K, ntc)
            s = S[m:L + 1, m, t0:t1]
            # f(θ,λ) = Re Σ_m F_m(θ) e^{imλ},F_m = Σ_n (C_nm - i S_nm) P̄_nm
            spec[:, :, m] = (p.T @ c) - 1j * (p.T @ s)
        rec = _ifft_along_lon(spec, workers=workers)      # (nlat, ntc, nlon)
        out[:, :, t0:t1] = (rec.real * nlon).transpose(0, 2, 1)
        done += 1
        if progress is not None:
            progress(done, total)
    return out[:, :, 0] if ntime == 1 else out
