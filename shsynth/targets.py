# -*- coding: utf-8 -*-
"""
shsynth.targets
===============

"在哪里求值"这件事的解析:规则网格(给范围 + 步长)、已有网格文件(借用它的
格点)、散点文件、全球准均匀球面点。

命令行与界面共用这里的逻辑,所以两边对"目标几何"的理解一定一致。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import fieldio
from .engine import fibonacci_points, regular_grid

__all__ = [
    "GLOBAL_GRID_PRESETS",
    "grid_from_spec",
    "grid_from_file",
    "points_from_file",
    "global_grid",
    "scatter_from_file",
    "analyse_target",
]

#: 界面上的一键全球网格(名称 → 步长,度)。
GLOBAL_GRID_PRESETS = (
    ("0.5° (361×720, 26 万点)", 0.5),
    ("1° (181×360, 6.5 万点)", 1.0),
    ("2° (91×180)", 2.0),
    ("2.5° (73×144)", 2.5),
    ("5° (37×72)", 5.0),
)


def grid_from_spec(lat_min: float = -90.0, lat_max: float = 90.0,
                   lon_min: float = 0.0, lon_max: float = 360.0,
                   lat_step: float = 1.0, lon_step: Optional[float] = None):
    """由"范围 + 步长"生成规则网格,返回 ``(lat_vec, lon_vec, meta)``。

    经度上界给 360 时按"不含端点"处理(``0, 1, ..., 359`` 而不是重复 0 与 360);
    其它情况两端都含。纬度必须落在 ``[-90, 90]``。
    """
    if lon_step is None:
        lon_step = lat_step
    if not -90.0 <= lat_min < lat_max <= 90.0:
        raise ValueError(f"纬度范围非法: {lat_min} .. {lat_max}(应在 [-90, 90] 内且递增)")
    if lat_step <= 0 or lon_step <= 0:
        raise ValueError("步长必须为正")
    if lon_max - lon_min <= 0:
        raise ValueError(f"经度范围非法: {lon_min} .. {lon_max}")
    if abs((lon_max - lon_min) - 360.0) < 1e-9:
        lon_max_eff = lon_max - lon_step          # 不重复 360 ≡ 0
    else:
        lon_max_eff = lon_max

    lat, lon = regular_grid(lat_min, lat_max, lon_min, lon_max_eff,
                            lat_step, lon_step)
    meta = {
        "kind": "grid", "source": "range+step",
        "nlat": int(lat.size), "nlon": int(lon.size),
        "n_points": int(lat.size * lon.size),
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "lat_step_deg": float(lat_step) if lat.size > 1 else None,
        "lon_step_deg": float(lon_step) if lon.size > 1 else None,
        "warnings": [],
    }
    return lat, lon, meta


def global_grid(step_deg: float = 1.0):
    """全球网格(经度 0..360 不含端点)。返回 ``(lat_vec, lon_vec, meta)``。"""
    return grid_from_spec(-90.0, 90.0, 0.0, 360.0, step_deg, step_deg)


def grid_from_file(path, var: Optional[str] = None):
    """借用已有网格文件的**格点**,返回 ``(lat_vec, lon_vec, ref_grid, meta)``。

    ``ref_grid`` 是该文件里的场(可能是 ``None`` 之外的参考值),可用于差值图。
    """
    lat, lon, grid, meta = fieldio.read_grid(path, var=var)
    meta = dict(meta)
    meta["kind"] = "grid"
    meta["source"] = "grid_file"
    meta["n_points"] = int(lat.size * lon.size)
    return lat, lon, grid, meta


def points_from_file(path, lat_col=None, lon_col=None):
    """读散点**位置**(不要求有数值列)。返回 ``(lat, lon, meta)``。"""
    lat, lon, meta = fieldio.read_positions(path, lat_col=lat_col,
                                            lon_col=lon_col)
    meta = dict(meta)
    meta["kind"] = "points"
    meta["source"] = "points_file"
    return lat, lon, meta


def scatter_from_file(path, lat_col=None, lon_col=None):
    """读散点(位置 + 已有数值),返回 ``(lat, lon, values, meta)``。"""
    lat, lon, values, meta = fieldio.read_points(
        path, lat_col=lat_col, lon_col=lon_col)
    meta = dict(meta)
    meta["kind"] = "points"
    meta["source"] = "points_file"
    return lat, lon, values, meta


def spherical_points(n: int = 20000, seed: int = 0):
    """全球准均匀 Fibonacci 球面点,返回 ``(lat, lon, meta)``。"""
    lat, lon = fibonacci_points(n, seed=seed)
    meta = {
        "kind": "points", "source": "fibonacci", "n_points": int(lat.size),
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "warnings": [],
    }
    return lat, lon, meta


def analyse_target(lat, lon, meta: Optional[dict] = None) -> dict:
    """给目标几何做个体检(点数、覆盖范围、经度跨度是否连续)。

    区域网格/区域散点会被明确标出来 —— 球谐综合在任何位置都成立,但"看起来
    是全球"的图如果只有一块区域,容易被误读。
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    m = dict(meta or {})
    out = dict(m)
    out.setdefault("kind", "points" if lat.size == lon.size else "grid")

    lon_span = float(np.nanmax(lon) - np.nanmin(lon)) if lon.size else 0.0
    lat_span = float(np.nanmax(lat) - np.nanmin(lat)) if lat.size else 0.0
    out["lat_span_deg"] = lat_span
    out["lon_span_deg"] = lon_span
    global_like = (lat_span >= 170.0) and (lon_span >= 350.0)
    out["is_global_like"] = bool(global_like)
    return out
