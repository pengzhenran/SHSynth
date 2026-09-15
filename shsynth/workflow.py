# -*- coding: utf-8 -*-
"""
shsynth.workflow
================

一站式流程:**读系数 → 解析目标几何 → 综合 → 落盘 → 出图 → 汇总报告**。

命令行与图形界面都只调用这里的 :func:`run`,所以两条路径的行为一定一致
(数值、警告、输出格式、图的种类全部相同)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from . import fieldio, plotting, targets
from .coeffio import coeff_info, read_coeffs, write_coeffs
from .engine import DEFAULT_CHUNK, evaluate, synthesis_grid
from .units import (FIELD_UNIT_LABELS, field_unit, is_no_conversion,
                    normalise_unit)

__all__ = ["SynthRequest", "SynthResult", "run", "plan_text", "select_time"]

#: 图的种类(命令行 ``--figure-kind`` / 界面下拉共用)。
FIGURE_KINDS = ("report", "map", "spectrum", "hist", "none")


def select_time(coeffs, t: Optional[int]):
    """取单个时次的系数视图;``t=None`` 时:单时次原样返回,多时次取第 0 个。"""
    if coeffs.ntime == 1 or t is None:
        if coeffs.ntime > 1 and t is None:
            sub = type(coeffs)(coeffs.C[:, :, 0].copy(), coeffs.S[:, :, 0].copy(),
                               dict(coeffs.meta))
            sub.meta["time_index"] = 0
            sub.meta["time_selected"] = f"多时次(共 {coeffs.ntime})默认取第 0 个"
            return sub
        return coeffs
    t = int(t)
    if not 0 <= t < coeffs.ntime:
        raise ValueError(f"time={t} 超出范围(ntime={coeffs.ntime})")
    sub = type(coeffs)(coeffs.C[:, :, t].copy(), coeffs.S[:, :, t].copy(),
                       dict(coeffs.meta))
    sub.meta["time_index"] = t
    return sub


@dataclass
class SynthRequest:
    """一次综合任务的全部参数。"""

    # ---- 系数 ----
    coeffs_path: str = ""
    coeffs_layout: str = "auto"
    coeffs_nmax: Optional[int] = None
    time: Optional[int] = None

    # ---- 目标几何 ----
    mode: str = "grid"                 # 'grid' | 'points'
    grid_source: str = "range"         # 'range' | 'file' | 'global'
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = 0.0
    lon_max: float = 360.0
    lat_step: float = 1.0
    lon_step: Optional[float] = None
    grid_file: str = ""
    grid_var: Optional[str] = None
    points_file: str = ""
    lat_col: Any = None
    lon_col: Any = None
    n_sphere_points: int = 20000

    # ---- 综合选项 ----
    truncate_nmax: Optional[int] = None
    gaussian_km: float = 0.0
    gaussian_method: str = "glq"
    #: ``None`` = **不换算**(输出与系数声明的物理量相同);给具体键才做逐阶换算。
    target_unit: Optional[str] = None
    allow_unit_mismatch: bool = False
    chunk: int = DEFAULT_CHUNK

    # ---- 输出 ----
    out_path: str = ""
    out_var: str = "value"
    out_units: str = ""
    out_comment: str = ""
    out_coeffs_path: str = ""
    out_coeffs_layout: str = "auto"

    # ---- 出图 ----
    figure_path: str = ""
    figure_kind: str = "report"
    figure_dpi: int = 160
    cmap: str = "RdBu_r"
    contour: bool = False
    symmetric: bool = True
    coast: bool = True
    #: 地图聚焦:``'auto'`` 区域数据自动缩放到结果范围;``'global'`` 画全球
    focus: str = "auto"

    # ---- 其它 ----
    extra: dict = field(default_factory=dict)


@dataclass
class SynthResult:
    """一次综合任务的结果。"""

    lat: Any
    lon: Any
    values: Any
    kind: str
    coeffs: Any
    meta: dict
    spec: SynthRequest
    summary: str = ""
    out_path: Optional[str] = None
    figure: Any = None
    figure_path: Optional[str] = None
    out_coeffs_path: Optional[str] = None
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 计划预览(界面/命令行都可以先打印)
# ---------------------------------------------------------------------------
def plan_text(spec: SynthRequest) -> str:
    """把任务的参数整理成人可读的几行文字。"""
    lines = [f"系数文件  : {spec.coeffs_path or '(未指定)'}",
             f"读取布局  : {spec.coeffs_layout}",
             "综合目标  : " + ("规则网格" if spec.mode == "grid" else "散点")]
    if spec.mode == "grid":
        if spec.grid_source == "global":
            lines.append(f"  全球网格 : 步长 {spec.lat_step:g}°")
        elif spec.grid_source == "file":
            lines.append(f"  借用网格 : {spec.grid_file or '(未指定)'}")
        else:
            lines.append(
                f"  范围     : lat {spec.lat_min:g}..{spec.lat_max:g}°  "
                f"lon {spec.lon_min:g}..{spec.lon_max:g}°")
            lines.append(f"  步长     : {spec.lat_step:g}° × "
                         f"{spec.lon_step if spec.lon_step else spec.lat_step:g}°")
    else:
        if spec.points_file:
            lines.append(f"  散点文件 : {spec.points_file}")
        else:
            lines.append(f"  球面点   : Fibonacci N={spec.n_sphere_points}")
    if spec.truncate_nmax is not None:
        lines.append(f"截断阶数  : {spec.truncate_nmax}")
    if spec.gaussian_km:
        lines.append(f"高斯平滑  : {spec.gaussian_km:g} km ({spec.gaussian_method})")
    lines.append(f"输出物理量: "
                 f"{_target_unit_text(spec.target_unit)}")
    lines.append(f"输出文件  : {spec.out_path or '(不写出)'}")
    if spec.figure_kind != "none":
        lines.append(f"图        : {spec.figure_kind} -> "
                     f"{spec.figure_path or '(不保存,仅界面显示)'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(spec: SynthRequest,
        progress: Optional[Callable[[str, float], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> SynthResult:
    """执行一次完整的综合任务。

    Parameters
    ----------
    spec : SynthRequest
        任务参数。
    progress : callable, optional
        ``progress(阶段文字, 0..1)``。
    cancel : callable, optional
        返回 ``True`` 时中止(界面上的"停止"按钮)。

    Returns
    -------
    SynthResult
        结果对象:数值(``lat``/``lon``/``values``)、元信息、汇总文字
        (``summary``)、落盘路径,以及绘图对象(``figure``;界面可直接嵌入,
        不必从磁盘再读回来)。
    """

    def tick(msg: str, frac: float):
        if progress is not None:
            progress(msg, float(frac))
        if cancel is not None and cancel():
            raise InterruptedError("用户中止")

    warnings: list = []

    # ---- 1. 读系数 ----
    if not spec.coeffs_path:
        raise ValueError("请先指定球谐系数文件")
    tick("读取系数…", 0.02)
    coeffs_all = read_coeffs(spec.coeffs_path, nmax=spec.coeffs_nmax,
                             layout=spec.coeffs_layout)
    warnings.extend(coeffs_all.meta.get("warnings", []) or [])
    ntime_in_file = coeffs_all.ntime
    # 综合/绘图只能处理**一个**时次(一个时次对应一张场);没指定时默认取第 0 个,
    # 并记一条警告。但**导出系数**时不能这么截:没指定时次就应该原样保留全部时次。
    coeffs = select_time(coeffs_all, spec.time)
    if coeffs.meta.get("time_selected"):
        warnings.append(str(coeffs.meta["time_selected"]))
    src_unit_label = FIELD_UNIT_LABELS.get(field_unit(coeffs), field_unit(coeffs))

    # ---- 2. 目标几何 ----
    tick("解析目标几何…", 0.10)
    kind = spec.mode
    ref_grid = None
    if kind == "grid":
        if spec.grid_source == "file":
            if not spec.grid_file:
                raise ValueError("网格来源选了『借用网格文件』,但没有给文件")
            lat_vec, lon_vec, ref_grid, gmeta = targets.grid_from_file(
                spec.grid_file, var=spec.grid_var)
            warnings.extend(gmeta.get("warnings", []) or [])
        elif spec.grid_source == "global":
            lat_vec, lon_vec, gmeta = targets.global_grid(spec.lat_step)
        else:
            lat_vec, lon_vec, gmeta = targets.grid_from_spec(
                spec.lat_min, spec.lat_max, spec.lon_min, spec.lon_max,
                spec.lat_step, spec.lon_step)
        lat, lon = lat_vec, lon_vec
        n_eval = int(lat_vec.size * lon_vec.size)
    else:
        if spec.points_file:
            lat, lon, pmeta = targets.points_from_file(
                spec.points_file, lat_col=spec.lat_col, lon_col=spec.lon_col)
            warnings.extend(pmeta.get("warnings", []) or [])
        else:
            lat, lon, pmeta = targets.spherical_points(spec.n_sphere_points)
        gmeta = pmeta
        lat_vec = lon_vec = None
        n_eval = int(np.size(lat))

    # ---- 3. 综合 ----
    tick(f"综合 {n_eval:,} 个点…", 0.20)
    nmax_use = spec.truncate_nmax
    if nmax_use is not None and nmax_use > coeffs.nmax:
        warnings.append(f"请求截断到 {nmax_use} 阶,但系数只有 {coeffs.nmax} 阶;"
                        "按系数自身的阶数计算")
        nmax_use = None

    if is_no_conversion(spec.target_unit):
        target_unit = field_unit(coeffs)          # 不换算
    else:
        target_unit = normalise_unit(spec.target_unit)
        if target_unit == field_unit(coeffs):
            target_unit = field_unit(coeffs)
    if target_unit != field_unit(coeffs):
        warnings.append(
            f"物理量换算: {src_unit_label} → "
            f"{FIELD_UNIT_LABELS.get(target_unit, target_unit)}")

    def _prog(done, total):
        tick(f"综合 {done:,}/{total:,} 个点…", 0.20 + 0.55 * done / max(total, 1))

    if kind == "grid":
        values = synthesis_grid(lat_vec, lon_vec, coeffs, nmax=nmax_use,
                                gaussian_km=spec.gaussian_km,
                                gaussian_method=spec.gaussian_method,
                                target_unit=target_unit,
                                chunk=spec.chunk, progress=_prog,
                                allow_unit_mismatch=spec.allow_unit_mismatch)
    else:
        vals = evaluate(lat, lon, coeffs, nmax=nmax_use,
                        gaussian_km=spec.gaussian_km,
                        gaussian_method=spec.gaussian_method,
                        target_unit=target_unit, chunk=spec.chunk,
                        progress=_prog,
                        allow_unit_mismatch=spec.allow_unit_mismatch)
        values = vals

    flat = np.asarray(values, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    stats = {
        "n_points": int(flat.size),
        "n_finite": int(finite.size),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "rms": float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
    }

    # ---- 4. 写出结果 ----
    tick("写出结果…", 0.80)
    out_path = None
    if spec.out_path:
        meta_out = {
            "source_coeffs": os.path.basename(spec.coeffs_path),
            "source_coeffs_path": str(spec.coeffs_path),
            "nmax": int(nmax_use if nmax_use is not None else coeffs.nmax),
            "coeff_nmax": int(coeffs.nmax),
            "ntime": int(coeffs.ntime),
            "time_index": coeffs.meta.get("time_index", 0),
            "field_unit_in": field_unit(coeffs),
            "field_unit_out": target_unit,
            "gaussian_km": float(spec.gaussian_km),
            "gaussian_method": spec.gaussian_method if spec.gaussian_km else None,
            "units": spec.out_units or None,
            "producer": "SHSynth",
        }
        comment = spec.out_comment or (
            f"SHSynth 球谐综合结果  系数={os.path.basename(spec.coeffs_path)}  "
            f"nmax={meta_out['nmax']}  高斯平滑={spec.gaussian_km:g} km  "
            f"物理量={target_unit}")
        if kind == "grid":
            out_path = str(fieldio.write_grid(
                spec.out_path, lat_vec, lon_vec, values, var=spec.out_var,
                meta=meta_out, units=spec.out_units or None,
                long_name="SHSynth spherical harmonic synthesis",
                comment=comment))
        else:
            out_path = str(fieldio.write_points(
                spec.out_path, lat, lon, values, comment=comment))

    out_coeffs_path = None
    if spec.out_coeffs_path:
        # 单时次布局(.gfc)装不下多时次:源文件有多个时次而用户又没指定 time 时,
        # 明确要求先选一个,而不是默默只导第 0 个 —— 与"宁可拒绝也不猜"一致。
        if ntime_in_file > 1 and spec.time is None and \
                _is_single_epoch_format(spec.out_coeffs_path,
                                        spec.out_coeffs_layout):
            raise ValueError(
                f"要导出的系数文件格式只装得下**一个时次**,但源文件里有 "
                f"{ntime_in_file} 个时次。请二选一:\n"
                f"  (a) 指定时次:time=<0..{ntime_in_file - 1}>(命令行 --time K);\n"
                "  (b) 改用能装多时次的布局:triangle 或 npz。")
        # 没指定时次 → 原样导出全部时次;指定了 → 只导出那一个
        c_out = coeffs_all if spec.time is None else coeffs
        if nmax_use is not None and nmax_use < c_out.nmax:
            c_out = c_out.truncate(nmax_use)
        if spec.gaussian_km:
            from .filters import apply_gaussian
            c_out = apply_gaussian(c_out, spec.gaussian_km,
                                   method=spec.gaussian_method)
        meta_extra = {"producer": "SHSynth", "nmax": int(c_out.nmax),
                      "ntime": int(c_out.ntime),
                      "gaussian_km": float(spec.gaussian_km)}
        if c_out.ntime > 1:
            meta_extra["time_note"] = ("源文件多时次且未指定 time → 全部保留"
                                       "(综合/绘图用的仍是第 0 个)")
        out_coeffs_path = str(write_coeffs(
            c_out, spec.out_coeffs_path, layout=spec.out_coeffs_layout,
            comment=spec.out_comment or "SHSynth 导出的系数", meta_extra=meta_extra))

    # ---- 5. 出图 ----
    figure_path = None
    fig = None
    if spec.figure_kind != "none":
        tick("绘图…", 0.90)
        cb_label = spec.out_units or _unit_label(target_unit)
        title = (f"{os.path.basename(spec.coeffs_path)}  nmax="
                 f"{nmax_use if nmax_use is not None else coeffs.nmax}"
                 + (f"  高斯 {spec.gaussian_km:g} km" if spec.gaussian_km else "")
                 + (f"  [{cb_label}]" if cb_label else ""))
        curves = {"系数逐阶 RMS": coeffs.degree_rms()}
        styles = {"系数逐阶 RMS": {"color": "0.45", "linestyle": "--"}}
        if spec.figure_kind == "report":
            fig = plotting.make_report_figure(
                lat, lon, values, coeffs=coeffs, curves=curves, title=title,
                cmap=spec.cmap, cb_label=cb_label, contour=spec.contour,
                lat_vec=lat_vec, lon_vec=lon_vec, focus=spec.focus,
                annotation=[f"n={stats['n_points']:,}",
                            f"RMS={stats['rms']:.4g}",
                            f"min={stats['min']:.4g}",
                            f"max={stats['max']:.4g}"])
        elif spec.figure_kind == "spectrum":
            fig = plotting.make_spectrum_figure(curves, styles=styles)
        elif spec.figure_kind == "hist":
            fig = plotting.make_hist_figure(flat, title=title, xlabel=cb_label)
        else:                                        # 'map' 或 'points'
            fig = plotting.make_map_figure(
                lat, lon, values, title=title, cmap=spec.cmap,
                symmetric=spec.symmetric, cb_label=cb_label,
                contour=spec.contour, coast=spec.coast, focus=spec.focus)
        if spec.figure_path:
            figure_path = plotting.save_figure(fig, spec.figure_path,
                                               dpi=spec.figure_dpi)

    # ---- 6. 汇总 ----
    tick("总结…", 0.98)
    info = coeff_info(coeffs)
    lines = [
        "=" * 68,
        "SHSynth 球谐系数解算(综合)",
        "=" * 68,
        f"系数文件      : {spec.coeffs_path}",
        f"  读取布局    : {info['layout']}   (自动识别,适配 SHKit 全部输出格式)",
        f"  最高阶      : nmax = {coeffs.nmax}   独立系数 {coeffs.ncoef}",
        f"  时次数      : ntime = {coeffs.ntime}"
        + (f"   使用第 {coeffs.meta.get('time_index', 0)} 个"
           if coeffs.ntime > 1 else ""),
        f"  物理量声明  : {src_unit_label}",
        f"本次使用阶数  : {nmax_use if nmax_use is not None else coeffs.nmax}",
        "高斯平滑      : "
        + (f"{spec.gaussian_km:g} km ({spec.gaussian_method})"
           if spec.gaussian_km else "未施加"),
        f"输出物理量    : {FIELD_UNIT_LABELS.get(target_unit, target_unit)}",
        "-" * 68,
        _target_line(spec, lat, lon, kind, stats["n_points"]),
        "-" * 68,
        f"求值点数      : {stats['n_points']:,}",
        f"值 范围       : [{stats['min']:.6g}, {stats['max']:.6g}]",
        f"值 均值 / RMS : {stats['mean']:.6g} / {stats['rms']:.6g}",
        f"值 标准差     : {stats['std']:.6g}",
    ]
    warnings = list(dict.fromkeys(str(w) for w in warnings if w))
    if warnings:
        lines.append("-" * 68)
        lines.append(f"警告 ({len(warnings)} 条):")
        lines.extend(f"  ! {w}" for w in warnings)
    if out_path:
        lines.append("-" * 68)
        lines.append(f"已写出结果    : {out_path}")
    if out_coeffs_path:
        lines.append(f"已写出系数    : {out_coeffs_path}")
    if figure_path:
        lines.append(f"已保存图      : {figure_path}")
    summary = "\n".join(lines)

    tick("完成", 1.0)
    return SynthResult(lat=lat, lon=lon, values=values, kind=kind, coeffs=coeffs,
                       meta=dict(gmeta or {}), spec=spec, summary=summary,
                       out_path=out_path, figure=fig, figure_path=figure_path,
                       out_coeffs_path=out_coeffs_path, warnings=warnings,
                       stats=stats)


def _unit_label(unit: str) -> str:
    """物理量 → 图注里用的单位提示(只作提示,不改数值)。"""
    return {
        "geopotential": "无量纲",
        "geoid": "m",
        "ewh": "m (EWH)",
        "surface_density": "kg/m²",
        "radial_displacement": "m",
        "scalar": "",
        "unknown": "",
    }.get(unit, "")


def _is_single_epoch_format(path: str, layout: str = "auto") -> bool:
    """该系数输出布局是否只装得下一个时次(目前只有 ICGEM ``.gfc``)。"""
    lay = (layout or "auto").lower()
    if lay != "auto":
        return lay == "gfc"
    low = str(path).lower()
    return low.endswith(".gfc") or low.endswith(".gfc.gz")


def _target_unit_text(unit: Optional[str]) -> str:
    """``target_unit`` 的人可读说明(``None`` = 不换算)。"""
    if unit is None:
        return "不换算(与系数声明的物理量相同)"
    return FIELD_UNIT_LABELS.get(normalise_unit(unit), str(unit))


def _target_line(spec: SynthRequest, lat, lon, kind: str, n_points: int) -> str:
    """目标几何的一行描述(汇总文字用)。"""
    la = np.asarray(lat, dtype=float).ravel()
    lo = np.asarray(lon, dtype=float).ravel()
    if kind == "grid":
        span = (f"纬度 {la.min():g}..{la.max():g}°  "
                f"经度 {lo.min():g}..{lo.max():g}°")
        src = {"range": "范围+步长", "file": f"借用网格 {spec.grid_file}",
               "global": f"全球 {spec.lat_step:g}°"}.get(spec.grid_source, "?")
        return (f"目标几何      : 规则网格 {la.size} × {lo.size} = "
                f"{n_points:,} 点  ({src};{span})")
    src = spec.points_file or f"Fibonacci N={spec.n_sphere_points}"
    span = (f"纬度 {la.min():g}..{la.max():g}°  "
            f"经度 {lo.min():g}..{lo.max():g}°")
    return (f"目标几何      : 散点 {n_points:,} 个  ({src};{span})")


#: ``run`` 的别名(返回 :class:`SynthResult`,其中的 ``.figure`` 即绘图对象)。
run_and_figure = run
