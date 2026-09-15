# -*- coding: utf-8 -*-
"""
shsynth.series
==============

**批量综合与序列产品**:一条系数序列 → 一整条场序列 / 点序列 / 趋势场。

设计要点(与 ``docs/多时间数据批量处理方案.md`` §4.4 一致):

* **一次调用把 ``ntime`` 全部算完** —— 实测比逐历元循环快 11.3 倍且逐位相同;
  规则整圈经度网格上再走 FFT 经度路径,整体可达 **30~130 倍**(§6.5);
* 目标几何只解析**一次**,所有历元共用;
* 逐历元诊断表(含缺测/重复/`unused_days`)只**报告**,不自动剔除、不自动插补;
* 趋势/周年在**系数域**拟合(线性算子与空间基可交换 → 与逐点拟合机器精度一致,
  代价从 ``O(npoints·ntime)`` 降到 ``O(ncoef·ntime)``)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .coeffs import SHCoeffs
from .engine import (DEFAULT_CHUNK, DEFAULT_TIME_CHUNK, evaluate,
                     evaluate_horizontal, fft_path_applicable,
                     horizontal_grid_fft, synthesis_grid, synthesis_grid_fft)
from .timeaxis import TimeAxis

__all__ = [
    "SeriesResult",
    "synth_series",
    "series_at_points",
    "basin_average",
    "FitResult",
    "fit_trend_seasonal",
    "estimate_memory",
    "remove_reference",
    "remove_mean",
    "remove_mean_window",
    "mean_window",
    "MEAN_MODES",
    "GRACE_MEAN_FROM",
    "GRACE_MEAN_TO",
    "drop_degree0",
    "check_degree0_trap",
    "check_static_dominance",
]


# ---------------------------------------------------------------------------
@dataclass
class SeriesResult:
    """一条场序列 + 诊断。

    Attributes
    ----------
    times : TimeAxis
        时间轴(可能 ``kind='index'``,表示无真实日期)。
    values : ndarray
        ``(nlat, nlon, ntime)``(网格)或 ``(npoints, ntime)``(散点)。
        多分量物理量(水平形变)时这里是**派生的大小** ``hypot(u_N,u_E)``。
    components : dict
        多分量时的原始分量,例如 ``{"north": ..., "east": ...}``;标量场为空。
    """

    times: TimeAxis
    values: np.ndarray
    kind: str = "grid"
    lat: object = None
    lon: object = None
    components: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    per_epoch: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: str = ""

    # ---------------------------------------------------------------- 诊断
    def outliers(self, k: float = 3.0) -> list:
        """MAD 稳健离群历元(**只报告,不剔除**)。"""
        v = np.asarray(self.stats.get("rms_by_epoch", []), dtype=float)
        if v.size < 4:
            return []
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad <= 0:
            return []
        z = 0.6745 * (v - med) / mad
        return [int(i) for i in np.nonzero(np.abs(z) > k)[0]]

    def to_csv(self, path) -> str:
        """逐历元诊断表 → csv。"""
        cols = list(self.stats.keys())
        p = os.fspath(path)
        if str(os.path.dirname(p)) not in ("", "."):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8-sig") as f:
            f.write(",".join(cols) + "\n")
            n = len(self.times)
            for i in range(n):
                row = []
                for c in cols:
                    v = self.stats[c][i] if isinstance(self.stats[c], (list, tuple)) \
                        else self.stats[c]
                    if isinstance(v, float):
                        row.append(f"{v:.10g}")
                    else:
                        row.append(str(v).replace(",", ";"))
                f.write(",".join(row) + "\n")
        return p


# ---------------------------------------------------------------------------
def estimate_memory(nlat: int, nlon: int, ntime: int, nmax: int,
                    *, dtype_bytes: int = 8) -> dict:
    """给用户一个"这次要多少内存"的估算(方案 §6.3)。"""
    npts = int(nlat) * int(nlon)
    out = npts * int(ntime) * dtype_bytes
    coeffs = (int(nmax) + 1) ** 2 * int(ntime) * 8 * 2
    fft_tmp = int(nlat) * min(int(ntime), DEFAULT_TIME_CHUNK) * int(nlon) * 16
    return {"output_bytes": out, "coeff_bytes": coeffs, "fft_tmp_bytes": fft_tmp,
            "npoints": npts}


def check_degree0_trap(coeffs: SHCoeffs, target_unit, *, warn_above: float = 1.0
                       ) -> Optional[str]:
    """**零阶项陷阱**:从 GRACE ``GSM`` 换算 EWH/面密度时,``C₀₀`` 通常是 1。

    ``A₀ = R·ρ̄/(3ρ_w)/(1+k′₀) ≈ 1.17e7``,于是 ``A₀·C₀₀`` 会把整张场抬到
    **1.17e7 m(≈11 700 km)** 量级 —— 那不是"水高",那是地球总质量。

    实测:用户目录里的 ``GSM-2_*_0600.gfc`` 的 ``C₀₀`` 恰好是 1.0,
    直接 ``target_unit='ewh'`` 得到的点序列是 ``1.17e7`` 米。

    返回中文警告(或 ``None``);**只警告,不擅自改数**(方案 §3 原则 8)。
    """
    if target_unit is None:
        return None
    tgt = str(target_unit).lower()
    if tgt not in ("ewh", "surface_density"):
        return None
    if coeffs.nmax < 0 or coeffs.C.shape[0] < 1:
        return None
    c00 = float(coeffs.C[0, 0]) if coeffs.C.ndim == 2 else float(coeffs.C[0, 0, 0])
    if abs(c00) < 1e-12:
        return None
    from .units import degree_factors
    a0 = float(degree_factors(tgt, 0)[0])
    if abs(a0 * c00) < warn_above:
        return None
    return (
        f"零阶项 C₀₀ = {c00:.6g} 会让 {tgt} 出现 {a0 * c00:.4g} 的**常量偏移**"
        f"(换算因子 A₀ = {a0:.4g})。\n"
        "  GRACE GSM 文件的 C₀₀ 通常是 1(地球总质量),不是水/质量异常。\n"
        "  做异常场/时间序列之前请先去掉零阶项(以及按惯例的一阶项),"
        "例如 coeffs - coeffs.time_slice(参考历元) 或先做 remove_mean。\n"
        "  本软件**不会**替你悄悄去掉它。")


def remove_reference(coeffs: SHCoeffs, *, when=None, kind: str = "mean"
                     ) -> SHCoeffs:
    """减去参考历元 / 时间去均值 —— **得到异常场**(产品必需项,不是通用时间滤波)。

    为什么需要这个函数:``coeffs - coeffs.time_slice(0)`` 会被
    :meth:`SHCoeffs._require_same_times` 拦下(时间轴长度不一致)—— 那条守卫是
    刻意的(绝不静默广播)。要做"整段减一个参考",就走这里,意图明确。

    Parameters
    ----------
    when : str | int | None
        参考历元:日期字符串(如 ``'2004-01-01'``)、下标,或 ``None``;
        ``None`` 且 ``kind='mean'`` 时用**时间平均**。
    kind : {'mean', 'median', 'first'}
        ``when`` 为空时的参考量。

    Returns
    -------
    SHCoeffs
        与输入同 ``times``;``meta`` 记下参考口径。
    """
    C = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    if when is not None:
        if isinstance(when, str):
            if coeffs.times is None:
                raise ValueError("按日期选参考历元需要时间轴;请改用下标")
            idx = coeffs.times.nearest(when, tol_days=None)
        else:
            idx = int(when)
        refC = C[:, :, idx][:, :, None]
        refS = S[:, :, idx][:, :, None]
        label = (f"epoch {idx}"
                 + (f" ({str(coeffs.times.values[idx])[:10]})"
                    if coeffs.times is not None else ""))
    elif kind == "mean":
        refC, refS, label = C.mean(axis=2, keepdims=True), \
            S.mean(axis=2, keepdims=True), "时间平均"
    elif kind == "median":
        refC, refS, label = np.median(C, axis=2, keepdims=True), \
            np.median(S, axis=2, keepdims=True), "时间中位"
    elif kind == "first":
        refC, refS, label = C[:, :, :1], S[:, :, :1], "第一个历元"
    else:
        raise ValueError("kind 必须是 'mean'|'median'|'first'")
    meta = dict(coeffs.meta)
    meta["reference_removed"] = label
    meta.pop("op", None)
    out = SHCoeffs(C - refC, S - refS, meta, coeffs.times)
    return out


def remove_mean(coeffs: SHCoeffs) -> SHCoeffs:
    """``remove_reference(kind='mean')`` 的别名(**全时段**去均值 → 异常场)。

    只是"全时段"这一个口径;要挑时段请用 :func:`remove_mean_window`。
    """
    return remove_reference(coeffs, kind="mean")


# ---------------------------------------------------------------------------
# 去均值的三种口径(GRACE 惯例 / 全时段 / 自定义)
# ---------------------------------------------------------------------------
#: GRACE 惯例的去均值时窗。
#:
#: 依据是用户自己的 legacy 脚本 ``3_processed/read_GRACE_SH_preprocess_postprocess_SH60.m``
#: 第 38-45 行(以及 GRACE-FO 版的第 49-56 行),两处都写着::
#:
#:     %% remove the mean of 2004-2010
#:     replace_deMean_SHC = replace_SHC - mean(replace_SHC(:,dur_mascon),2,'omitnan');
#:
#: ⚠️ 但两个文件里的 ``dur_mascon`` **不一样**(``19:90`` 与 ``90:150``),
#: 说明那是**各文件里的位置**而不是口径本身。实测(``docs/probes/_probe_demean.py``,
#: CSR RL06 203 历元):
#:
#: * ``19:90``  → 2004-01-07 .. 2009-12-16(72 个)← 与 "2004-2010" 大致相符
#: * ``90:150`` → 2009-12-16 .. 2016-01-16(61 个)← **与注释差了 6 年**
#:
#: 所以本实现**按日期**定口径(注释的字面意思 2004-01-01..2010-12-31),
#: 并把实际用到的历元数与日期范围报出来。要精确复刻 ``19:90``(即 2004–2009),
#: 用 ``mode='custom', from_date='2004-01-01', to_date='2009-12-31'``。
GRACE_MEAN_FROM = "2004-01-01"
GRACE_MEAN_TO = "2010-12-31"

#: 去均值口径
MEAN_MODES = ("grace", "all", "custom")

_MEAN_MODE_LABELS = {
    "grace": "GRACE 惯例",
    "all": "全时段",
    "custom": "自定义时段",
}


def _as_date(v, what: str, role: str = "from"):
    """把用户给的日期写法统一成 ``datetime64[D]``。

    接受的写法(都不要求写全,这是刻意的 —— "从 2005 年开始"太常见了):

    ==============  ====================  ====================
    写法            ``role='from'``       ``role='to'``
    ==============  ====================  ====================
    ``2005``        2005-01-01            **2005-12-31**
    ``2005-03``     2005-03-01            **2005-03-31**
    ``2005-03-15``  2005-03-15            2005-03-15
    ``2005.5``      十进制年 → 该日       十进制年 → 该日
    ==============  ====================  ====================

    终点补到**月末/年末**很关键:``--mean-to 2009`` 若补成 2009-01-01,
    窗口会少掉整整 11 个月,而且不报错 —— 这类"静默少算"最难发现。
    """
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, np.datetime64):
        return np.datetime64(v, "D")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        # 十进制年 → 该年的某一天(小数部分按 365.25 天/年折算)
        if not (1000.0 <= float(v) <= 3000.0):
            raise ValueError(f"{what} 的十进制年 {v!r} 不在 1000..3000 之间")
        y = int(np.floor(float(v)))
        day = int(round((float(v) - y) * 365.25))
        return np.datetime64(f"{y:04d}-01-01", "D") + np.timedelta64(day, "D")
    s = str(v).strip().replace("/", "-").replace(".", "-")
    parts = [p for p in s.split("-") if p != ""]
    if not parts or not parts[0].isdigit() or len(parts[0]) != 4:
        raise ValueError(
            f"{what} 解析不了:{v!r}。请用 YYYY-MM-DD(也接受 YYYY-MM、YYYY、"
            "或十进制年 2005.5)")
    y = parts[0]
    if len(parts) == 1:                       # 只给年
        return np.datetime64(f"{y}-12-31" if role == "to" else f"{y}-01-01", "D")
    if len(parts) == 2:                       # 只给年月
        m = parts[1].zfill(2)
        if role == "to":
            # 该月最后一天 = 下月 1 日 − 1 天(用 numpy 自己算,不引日历库)
            nxt = (np.datetime64(f"{y}-{m}-01", "D")
                   + np.timedelta64(1, "M") - np.timedelta64(1, "D"))
            return nxt
        return np.datetime64(f"{y}-{m}-01", "D")
    try:
        return np.datetime64("-".join([y, parts[1].zfill(2), parts[2].zfill(2)]), "D")
    except Exception as exc:                          # noqa: BLE001
        raise ValueError(
            f"{what} 解析不了:{v!r}。请用 YYYY-MM-DD(也接受 YYYY-MM、YYYY、"
            "或十进制年 2005.5)") from exc


def mean_window(times: TimeAxis, *, mode: str = "grace",
                from_date=None, to_date=None) -> dict:
    """把去均值的三种口径解析成**具体历元窗口**。

    ============  ==========================================================
    ``mode``      含义
    ============  ==========================================================
    ``'grace'``   **GRACE 惯例**:减 2004-01-01 .. 2010-12-31 的平均
                  (见 :data:`GRACE_MEAN_FROM`;用户 legacy 脚本的口径)
    ``'all'``     **全时段**:减所有历元的平均
    ``'custom'``  **自定义**:减 ``from_date .. to_date`` 的平均(两个都要给)
    ============  ==========================================================

    Returns
    -------
    dict
        ``mode, label, from_date, to_date, idx, n_epochs, n_total,
        first_date, last_date, coverage, note, warnings, ref_label``
    """
    if mode not in MEAN_MODES:
        raise ValueError(f"mode 必须是 {' | '.join(MEAN_MODES)},收到 {mode!r}")
    idx = np.arange(len(times), dtype=int)
    if mode == "all":
        sel = idx
        note = "减**全时段**平均"
        warnings: list = []
    else:
        if mode == "grace":
            d0, d1 = _as_date(GRACE_MEAN_FROM, "GRACE 窗口起点", "from"), \
                _as_date(GRACE_MEAN_TO, "GRACE 窗口终点", "to")
            what = f"GRACE 惯例窗口 {GRACE_MEAN_FROM} .. {GRACE_MEAN_TO}"
        else:
            d0 = _as_date(from_date, "自定义起点", "from")
            d1 = _as_date(to_date, "自定义终点", "to")
            if d0 is None or d1 is None:
                raise ValueError(
                    "自定义时段必须**同时**给起点和终点(界面里填两个日期,"
                    "命令行给 --mean-from/--mean-to)")
            what = f"自定义窗口 {d0} .. {d1}"
        if d0 > d1:
            raise ValueError(f"{what}:起点晚于终点,窗口为空")
        if not times.has_dates:
            raise ValueError(
                f"{what} 需要真实日期,但这条系数序列没有时间轴"
                "(kind='index')。请先用 series-read 从 gfc 目录读入带日期的序列,"
                "或改用 mode='all' 去全时段平均。")
        vals = times.values.astype("datetime64[D]")
        sel = np.nonzero((vals >= d0) & (vals <= d1))[0]
        note = f"减 {d0} .. {d1} 的平均"
        warnings = []
        if sel.size == 0:
            raise ValueError(
                f"{what} 里**一个历元都没有** —— 序列实际跨度是 "
                f"{str(times.values[0])[:10]} .. {str(times.values[-1])[:10]}"
                f"({len(times)} 个历元)。\n"
                "  不会退化成全时段平均(那会悄悄换掉静态场基准)。"
                "请改窗口,或显式用全时段(mode='all')。")
    if sel.size == len(times) and mode != "all":
        warnings.append(
            f"{_MEAN_MODE_LABELS[mode]}的窗口覆盖了**全部** {len(times)} 个历元 "
            "—— 这个口径实际退化为『全时段平均』,两者结果相同。")
    return {
        "mode": mode,
        "label": _MEAN_MODE_LABELS[mode],
        "from_date": str(times.values[sel[0]])[:10] if sel.size else None,
        "to_date": str(times.values[sel[-1]])[:10] if sel.size else None,
        "idx": sel,
        "n_epochs": int(sel.size),
        "n_total": int(len(times)),
        "coverage": float(sel.size) / max(len(times), 1),
        "note": note,
        "warnings": warnings,
        "ref_label": (f"{_MEAN_MODE_LABELS[mode]}"
                      + (f"({str(times.values[sel[0]])[:10]}.."
                         f"{str(times.values[sel[-1]])[:10]},"
                         f"{sel.size} 个历元)" if sel.size else "")),
    }


def remove_mean_window(coeffs: SHCoeffs, *, mode: str = "grace",
                       from_date=None, to_date=None
                       ) -> tuple[SHCoeffs, dict]:
    """**按口径**去时间平均 → 异常场。返回 ``(结果, 报告)``。

    三种口径见 :func:`mean_window`。这是界面/命令行**唯一**的去均值入口 ——
    去均值会决定整个异常场的静态基准,必须只走一条路。

    口径细节(与 legacy MATLAB 对齐):

    * 窗口内用 ``nanmean``(对应 MATLAB 的 ``'omitnan'``);
    * 窗口内的历元**原样保留**(不去掉、不重采样),减的是同一组参考系数;
    * 参考口径写进 ``meta['reference_removed']``,窗口写进 ``meta['mean_window']``,
      所以落盘后的文件自己说明得清是拿哪一段做的基准。
    """
    times = coeffs.times
    if times is None:
        raise ValueError(
            "去均值需要时间轴。这条系数只有单时次 —— "
            "单独一个历元减自己的平均恒为 0,没有意义。")
    rep = mean_window(times, mode=mode, from_date=from_date, to_date=to_date)
    C = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    sel = rep["idx"]
    n_t = C.shape[2]

    # ⚠️ 两个容易踩坏"逐位兼容"的细节,这里都躲开了:
    #
    # 1. **不要无条件用花式索引** ``C[:, :, sel]``。它产生的是**副本**,
    #    内存布局与原来不同,NumPy 的成对求和分块也就跟着变 ——
    #    实测 ``C[:, :, np.arange(n)].mean(2)`` 与 ``C.mean(2)`` 能差 1e-19。
    #    窗口覆盖全部历元时直接用手上的数组,``mode='all'`` 才与旧的
    #    ``remove_mean`` **逐位一致**(CLI 的 ``--remove-mean`` 依赖这一点)。
    # 2. 没有非有限值时用 ``.mean()`` 而不是 ``nanmean()``:语义与 MATLAB 的
    #    ``'omitnan'`` 一致(没有 NaN 可省),但少一层掩膜机制,更不容易出现
    #    "换个 NumPy 版本末位就变"的意外。
    sub_c = C if sel.size == n_t else C[:, :, sel]
    sub_s = S if sel.size == n_t else S[:, :, sel]
    bad_c = int(np.count_nonzero(~np.isfinite(sub_c)))
    bad_s = int(np.count_nonzero(~np.isfinite(sub_s)))
    rep["n_nonfinite"] = bad_c + bad_s
    with np.errstate(invalid="ignore"):
        if bad_c:
            refC = np.nanmean(sub_c, axis=2, keepdims=True)
        else:
            refC = sub_c.mean(axis=2, keepdims=True)
        if bad_s:
            refS = np.nanmean(sub_s, axis=2, keepdims=True)
        else:
            refS = sub_s.mean(axis=2, keepdims=True)
    if rep["n_nonfinite"]:
        rep["warnings"].append(
            f"窗口内有 {rep['n_nonfinite']} 个非有限系数,求平均时按"
            "'忽略缺测'处理(MATLAB 的 omitnan 口径)。")
    meta = dict(coeffs.meta)
    meta["reference_removed"] = rep["ref_label"]
    meta["mean_window"] = {
        "mode": rep["mode"], "from": rep["from_date"], "to": rep["to_date"],
        "n_epochs": rep["n_epochs"], "n_total": rep["n_total"],
    }
    meta.pop("op", None)
    out = SHCoeffs(C - refC, S - refS, meta, times)
    rep["summary"] = (
        f"去均值口径 = {rep['label']}:{rep['note']};"
        f"实用 {rep['n_epochs']}/{rep['n_total']} 个历元"
        f"({rep['from_date']} .. {rep['to_date']},"
        f"占 {rep['coverage'] * 100:.1f}%)")
    return out, rep


def drop_degree0(coeffs: SHCoeffs, *, nmax_drop: int = 0) -> SHCoeffs:
    """把 ``n ≤ nmax_drop`` 的系数置零(默认只去 ``C₀₀``;去一阶传 1)。

    GRACE ``GSM`` 的 ``C₀₀`` 是地球总质量(通常为 1),不置零会让 EWH
    出现 ``1.17e7`` 的常量偏移(见 :func:`check_degree0_trap`)。
    """
    L = coeffs.nmax
    k = min(int(nmax_drop), L) + 1
    C = (coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C).copy()
    S = (coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S).copy()
    C[:k, :k, :] = 0.0
    S[:k, :k, :] = 0.0
    meta = dict(coeffs.meta)
    meta["degree_dropped"] = int(k - 1)
    return SHCoeffs(C, S, meta, coeffs.times)


def check_static_dominance(values: np.ndarray, times: TimeAxis, target_unit,
                           *, ratio_warn: float = 10.0) -> Optional[str]:
    """**静态场压过时间变化**的诊断。

    ``GSM`` 文件装的是**完整静态重力场**(``C₂₀ ≈ -4.8e-4`` ⇒ 换成 EWH 约 4 万米),
    不是质量异常。实测:203 个真实 CSR 历元直接换成 EWH 后,逐历元 RMS ≈ 53 828 m,
    而且**每个历元几乎一样** —— 这说明看到的是静态场,不是水。

    判据:``|时间平均| / 时间标准差 > ratio_warn`` 时报出来。
    """
    if target_unit is None:
        return None
    tgt = str(target_unit).lower()
    if tgt not in ("ewh", "surface_density", "geoid", "radial_displacement",
                   "horizontal_displacement"):
        return None
    v = np.asarray(values, dtype=float)
    if v.ndim < 2 or v.shape[-1] < 3:
        return None
    flat = v.reshape(-1, v.shape[-1])
    finite = np.isfinite(flat)
    if not finite.any():
        return None
    mean = np.nanmean(flat, axis=1)
    std = np.nanstd(flat, axis=1)
    big = np.abs(mean) > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(big, np.abs(mean) / np.maximum(std, 1e-300), 0.0)
    if not np.any(ratio > ratio_warn):
        return None
    return (
        "场的**时间平均远大于时间变化**(|均值|/标准差 中位 "
        f"{np.nanmedian(ratio[big]):.3g},最大 {np.nanmax(ratio):.3g})。\n"
        "  GSM 文件装的是**完整静态重力场**(C₂₀ ≈ -4.8e-4),不是质量异常;\n"
        "  直换算出来的大数主要是静态场,不是水。做异常场请先去掉参考历元或\n"
        "  时间平均(series-drop 或 --remove-mean),再解释为质量变化。")


def _per_epoch_stats(values: np.ndarray, times: TimeAxis,
                     coeffs: SHCoeffs, warnings: list,
                     components: Optional[dict] = None) -> dict:
    """逐历元诊断列(列式,直接能变 csv)。"""
    v = np.asarray(values, dtype=float)
    if v.ndim == 2 and v.shape[0] != len(times) and v.shape[1] == len(times):
        v = v.T
    flat = v.reshape(-1, v.shape[-1]) if v.ndim > 1 else v[:, None]
    finite = np.isfinite(flat)
    nt = flat.shape[1]
    cols: dict = {
        "epoch": list(range(nt)),
        "time": [str(t)[:19] for t in times.values],
        "decimal_year": [float(x) for x in times.decimal_years],
        "n_points": [int(flat.shape[0])] * nt,
        "n_finite": [int(finite[:, k].sum()) for k in range(nt)],
        "finite_frac": [float(finite[:, k].mean()) for k in range(nt)],
    }
    with np.errstate(all="ignore"):
        cols["min"] = [float(np.nanmin(np.where(finite[:, k], flat[:, k], np.nan)))
                       for k in range(nt)]
        cols["max"] = [float(np.nanmax(np.where(finite[:, k], flat[:, k], np.nan)))
                       for k in range(nt)]
        cols["mean"] = [float(np.nanmean(np.where(finite[:, k], flat[:, k], np.nan)))
                        for k in range(nt)]
        rms = []
        for k in range(nt):
            x = flat[finite[:, k], k]
            rms.append(float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan"))
        cols["rms"] = rms
    cols["rms_by_epoch"] = cols["rms"]
    if components:
        for name, arr in components.items():
            a = np.asarray(arr, dtype=float).reshape(-1, arr.shape[-1])
            cols[f"max_abs_{name}"] = [float(np.nanmax(np.abs(a[:, k])))
                                       for k in range(nt)]
    if times.sources:
        cols["source_file"] = [os.path.basename(s) if s else "" for s in times.sources]
    miss = times.missing()
    if miss:
        gap = {i: g for a, b, g in miss for i in (b,)}
        cols["gap_days_before"] = [float(gap.get(k, 0.0)) for k in range(nt)]
    cols["n_warnings"] = [len(warnings)] * nt
    return cols


def _summarise_components(h: dict, lat_vec, lon_vec, nlat, nlon, ntime):
    """把 ``evaluate_horizontal`` / ``horizontal_grid_fft`` 的输出整成网格形状。"""
    out = {}
    for k in ("north", "east", "azimuth"):
        if k in h:
            out[k] = np.asarray(h[k]).reshape(nlat, nlon, ntime)
    return out


def synth_series(coeffs: SHCoeffs, *, lat_vec=None, lon_vec=None,
                 points=None, nmax: Optional[int] = None,
                 gaussian_km: float = 0.0, gaussian_method: str = "glq",
                 target_unit: Optional[str] = None,
                 component: str = "scalar",
                 chunk: int = DEFAULT_CHUNK,
                 time_chunk: int = DEFAULT_TIME_CHUNK,
                 use_fft: str = "auto",
                 progress: Optional[Callable[[str, float], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None) -> SeriesResult:
    """把系数序列一次综合成场序列。

    Parameters
    ----------
    coeffs : SHCoeffs
        ``ntime = N`` 的系数序列(建议带 ``times``)。
    lat_vec, lon_vec : array_like, optional
        规则网格(``points`` 为空时用)。
    points : (lat, lon) tuple, optional
        散点;给了它就走直接法(FFT 对散点不适用,方案 §6.5.7)。
    component : {'scalar','radial','north','east','horizontal'}
        ``'scalar'``(默认):标量场,``values`` 就是场本身。
        ``'north'`` / ``'east'``:水平形变的单个分量。
        ``'horizontal'``:水平形变,**一次算完北 + 东**;``values`` 取
        ``magnitude``(旋转不变量,处处有定义),原始分量放在
        :attr:`SeriesResult.components`(设计原则 11)。
    use_fft : {'auto','yes','no'}
        ``'auto'``:规则整圈经度且 ``nlon > 2·nmax`` 时走 FFT,否则直接法,
        并在汇总里**写明走了哪条路**。

    Returns
    -------
    SeriesResult
    """
    def tick(msg: str, frac: float):
        if progress is not None:
            progress(msg, float(frac))
        if cancel is not None and cancel():
            raise InterruptedError("用户中止")

    warnings = list(coeffs.meta.get("warnings", []) or [])
    vector = component in ("north", "east", "horizontal")
    if vector and target_unit is not None and \
            str(target_unit).lower() not in ("horizontal_displacement", "uh", "u_h",
                                             "水平形变", "horizontal", "不换算", "none", "无"):
        raise ValueError(
            f"component={component!r} 是水平形变,但 target_unit={target_unit!r} "
            "不是 horizontal_displacement。\n"
            "水平形变的逐阶因子是 R·l′ₙ/(1+k′ₙ),与 EWH/geoid 的因子不同;"
            "两者混用会差好几个数量级。请去掉 --target-unit(或显式写 "
            "horizontal_displacement)。")
    if vector:
        target_unit = "horizontal_displacement"
    d0 = check_degree0_trap(coeffs, target_unit)
    if d0:
        warnings.append(d0)
    times = coeffs.times
    if times is None:
        times = TimeAxis.from_index(coeffs.ntime, label="系数序号")
        if coeffs.ntime > 1:
            warnings.append(
                f"系数没有时间轴:已按序号当时间({coeffs.ntime} 个历元)。"
                "要真实日期请用 read_coeffs_series() 从 gfc 目录/清单读入")
    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)

    tick("综合全部历元…", 0.05)
    n_eval = 0
    used = "直接法"
    components: dict = {}
    if points is not None:
        la, lo = points
        la = np.atleast_1d(np.asarray(la, dtype=float)).ravel()
        lo = np.atleast_1d(np.asarray(lo, dtype=float)).ravel()
        n_eval = int(la.size)
        if vector:
            h = evaluate_horizontal(la, lo, coeffs, nmax=L,
                                    gaussian_km=gaussian_km,
                                    gaussian_method=gaussian_method, chunk=chunk)
            components = {k: h[k] for k in ("north", "east") if k in h}
            if component == "north":
                values = h["north"]
            elif component == "east":
                values = h["east"]
            else:
                values = np.hypot(h["north"], h["east"])
            used = "直接法(水平形变球面梯度;散点不能用 FFT)"
        else:
            values = evaluate(la, lo, coeffs, nmax=L, gaussian_km=gaussian_km,
                              gaussian_method=gaussian_method,
                              target_unit=target_unit, chunk=chunk)
        kind = "points"
        lat_vec = lon_vec = None
    else:
        if lat_vec is None or lon_vec is None:
            raise ValueError("synth_series: 要么给 lat_vec/lon_vec,要么给 points")
        lat_vec = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
        lon_vec = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
        n_eval = int(lat_vec.size * lon_vec.size)
        ok, why = fft_path_applicable(lon_vec, L)
        want = (use_fft == "yes") or (use_fft == "auto" and ok)
        if want and not ok:
            warnings.append(f"要求走 FFT 但条件不满足({why});已回退直接法")
            want = False
        if want:
            used = (f"FFT 经度路径(nlon={lon_vec.size} > 2·nmax={2 * L})"
                    if use_fft == "auto" else "FFT 经度路径(用户指定)")
            if vector:
                used += " + 水平形变"

            def _p(done, total):
                tick(f"FFT 综合 {done}/{total} 个时间块…",
                     0.05 + 0.85 * done / max(total, 1))
            if vector:
                h = horizontal_grid_fft(lat_vec, lon_vec, coeffs, nmax=L,
                                        gaussian_km=gaussian_km,
                                        gaussian_method=gaussian_method,
                                        time_chunk=time_chunk)
                components = {k: h[k] for k in ("north", "east")}
                values = h["north"] if component == "north" else (
                    h["east"] if component == "east" else h["magnitude"])
            else:
                values = synthesis_grid_fft(
                    lat_vec, lon_vec, coeffs, nmax=L, gaussian_km=gaussian_km,
                    gaussian_method=gaussian_method, target_unit=target_unit,
                    time_chunk=time_chunk, progress=_p)
        else:
            if use_fft == "auto" and why:
                warnings.append(f"走直接法:{why}")
            used = "直接法(显式经度展开)" + (" + 水平形变" if vector else "")
            if vector:
                LA, LO = np.meshgrid(lat_vec, lon_vec, indexing="ij")
                h = evaluate_horizontal(LA.ravel(), LO.ravel(), coeffs, nmax=L,
                                        gaussian_km=gaussian_km,
                                        gaussian_method=gaussian_method,
                                        chunk=chunk)
                nlat, nlon = lat_vec.size, lon_vec.size
                nt = coeffs.ntime
                components = {k: np.asarray(h[k]).reshape(nlat, nlon, nt)
                              for k in ("north", "east")}
                if component == "north":
                    values = components["north"]
                elif component == "east":
                    values = components["east"]
                else:
                    values = np.hypot(components["north"], components["east"])
            else:
                values = synthesis_grid(
                    lat_vec, lon_vec, coeffs, nmax=L, gaussian_km=gaussian_km,
                    gaussian_method=gaussian_method, target_unit=target_unit,
                    chunk=chunk)
        kind = "grid"

    tick("逐历元诊断…", 0.95)
    st = check_static_dominance(values, times, target_unit)
    if st:
        warnings.append(st)
    stats = _per_epoch_stats(values, times, coeffs, warnings,
                             components or None)
    stats["eval_points"] = n_eval
    stats["method"] = used
    stats["component"] = component
    res = SeriesResult(times=times, values=values, kind=kind,
                       lat=lat_vec, lon=lon_vec, components=components,
                       stats=stats, warnings=list(dict.fromkeys(
                           str(w) for w in warnings if w)))
    res.summary = _summary(res, coeffs, L, gaussian_km, target_unit, used, n_eval)
    tick("完成", 1.0)
    return res


def _summary(res: SeriesResult, coeffs: SHCoeffs, L: int, gaussian_km: float,
             target_unit, used: str, n_eval: int) -> str:
    comp = res.stats.get("component", "scalar")
    comp_txt = {"scalar": "标量场", "radial": "径向形变",
                "north": "水平形变 · 北分量", "east": "水平形变 · 东分量",
                "horizontal": "水平形变 · 北+东(values 存 magnitude)"}.get(comp, comp)
    lines = [
        "=" * 68,
        "SHSynth 批量综合(时间序列)",
        "=" * 68,
        f"系数来源      : {coeffs.meta.get('source_file', '(未知)')}",
        f"  最高阶      : nmax = {L}",
        f"  物理量      : {coeffs.meta.get('field_unit', '未声明')}"
        + (f"  →  {target_unit}" if target_unit else ""),
        f"输出分量      : {comp_txt}",
        "高斯平滑      : " + (f"{gaussian_km:g} km" if gaussian_km else "未施加"),
        "综合路径      : " + used,
        f"求值点数      : {n_eval:,}  ×  {len(res.times)} 个历元"
        f"  =  {n_eval * len(res.times):,} 个值",
        "-" * 68,
    ]
    lines.extend("  " + ln for ln in res.times.summary().splitlines())
    rms = np.asarray(res.stats.get("rms_by_epoch", []), dtype=float)
    if rms.size:
        lines.append("-" * 68)
        lines.append(f"逐历元 RMS    : min {np.nanmin(rms):.6g} / 中位 "
                     f"{np.nanmedian(rms):.6g} / max {np.nanmax(rms):.6g}")
        out = res.outliers()
        if out:
            lines.append(
                f"可疑历元      : {len(out)} 个(MAD 稳健离群,**只报告不剔除**):"
                + ", ".join(f"#{i}({str(res.times.values[i])[:10]})" for i in out[:8]))
    if res.warnings:
        lines.append("-" * 68)
        lines.append(f"警告 ({len(res.warnings)} 条):")
        lines.extend(f"  ! {w}" for w in res.warnings)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def series_at_points(lat, lon, coeffs: SHCoeffs, *,
                     nmax: Optional[int] = None,
                     gaussian_km: float = 0.0,
                     target_unit: Optional[str] = None,
                     chunk: int = DEFAULT_CHUNK):
    """点上的 ``(ntime,)`` 或 ``(npoints, ntime)`` 时间序列。

    Returns
    -------
    (TimeAxis, ndarray)
    """
    times = coeffs.times or TimeAxis.from_index(coeffs.ntime)
    v = evaluate(lat, lon, coeffs, nmax=nmax, gaussian_km=gaussian_km,
                 target_unit=target_unit, chunk=chunk)
    return times, v


def _mask_setup(mask_or_polygon, lat_vec=None, lon_vec=None):
    """把掩膜整成 ``(lat_vec, lon_vec, mask2d, cell_area)``。

    掩膜可以是:

    * **二维数组** —— 默认按**全球等经纬网格的格心**理解
      (``lat = -90 + (i+0.5)Δ``);给了 ``lat_vec``/``lon_vec`` 就按给的走;
    * **网格文件路径**(``.nc``/``.grd``/``.npy``/三列文本)—— 用
      :func:`shsynth.fieldio.read_grid` 读进经纬度与掩膜值(这与"借用网格文件"
      是同一套识别逻辑,平面米制坐标会被明确拒绝)。

    面积用**格心等经纬网格的精确球面元** ``ΔΩ = Δλ·(sin φ₂ - sin φ₁)``,
    比 ``cos φ·ΔφΔλ`` 更准(后者在极点附近有 O(Δ²) 偏差)。
    """
    from . import fieldio
    if isinstance(mask_or_polygon, (str, os.PathLike)):
        la, lo, m, _meta = fieldio.read_grid(os.fspath(mask_or_polygon))
        m = np.asarray(m, dtype=float)
        if m.ndim == 3:
            raise ValueError(
                f"这个网格有 {m.shape[2]} 个时间层,不是一张静态掩膜 —— "
                "区域平均需要一个静态的 0/1(或权重)掩膜;"
                "多时次的值请走 series-synth 那条路")
        la = np.asarray(la, dtype=float)
        lo = np.asarray(lo, dtype=float)
    else:
        m = np.asarray(mask_or_polygon, dtype=float)
        if m.ndim != 2:
            raise ValueError(
                "掩膜必须是二维数组(等经纬网格),或一个网格文件路径;"
                "平面/投影坐标的掩膜请先用 Surfer/GMT 重采样成等经纬网格")
        nlat, nlon = m.shape
        la = (np.asarray(lat_vec, dtype=float) if lat_vec is not None
              else -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat))
        lo = (np.asarray(lon_vec, dtype=float) if lon_vec is not None
              else 360.0 * np.arange(nlon) / nlon)
    nlat, nlon = len(la), len(lo)
    if m.shape != (nlat, nlon):
        raise ValueError(f"掩膜形状 {m.shape} 与坐标轴 ({nlat}, {nlon}) 不一致")
    dlat = (la[1] - la[0]) if nlat > 1 else 180.0
    dlon = (lo[1] - lo[0]) if nlon > 1 else 360.0
    edge = np.deg2rad(np.concatenate([[la[0] - dlat / 2],
                                      la + dlat / 2]))
    dsin = np.abs(np.diff(np.sin(np.clip(edge, -90, 90))))
    area = np.deg2rad(abs(dlon)) * dsin[:, None]      # (nlat,1),单位 sr
    if nlat == 1:
        area = np.full((1, 1), 4 * np.pi)
    return la, lo, m, np.broadcast_to(area, (nlat, nlon)).copy()


def _project_mask_to_sh(mask, la, lon_vec, nmax: int) -> np.ndarray:
    """把掩膜**投影**成球谐系数 ``(nmax+1, nmax+1)``(只用 (0,0) 之外的阶次做核)。

    ⚠️ 这不是给用户用的"球谐分析"功能(SHSynth 只做综合,分析是 SHKit 的事)——
    它只是**区域平均核**的内部实现:把掩膜写成 ``g = Σ g_nm Y_nm``,
    之后区域平均就能用系数内积算(方案 §4.4 的 ``method='coeff'``)。

    用等经纬网格上的面积加权投影 ``g_nm = (1/4π) Σ g·Y_nm·ΔΩ``(4π 归一化)。
    """
    from .engine import legendre_columns
    lam = np.deg2rad(lon_vec)
    _, _, _m, area = _mask_setup(mask, la, lon_vec)
    g = np.nan_to_num(np.asarray(mask, dtype=float), nan=0.0)
    if g.shape != area.shape:
        g = np.broadcast_to(g, area.shape)
    P = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for m, block in legendre_columns(la, nmax):          # block: (n-m+1, nlat)
        c = np.cos(m * lam)[None, :]
        s = np.sin(m * lam)[None, :]
        # Σ_lon g·cos(mλ)·Δλ  → (nlat,)
        gc = (g * c).sum(axis=1) * area[:, 0]            # ΔΩ 的经度部分已含在 area
        gs = (g * s).sum(axis=1) * area[:, 0]
        P[m:nmax + 1, m] = (block * gc[None, :]).sum(axis=1) / (4 * np.pi)
        if m >= 1:
            S[m:nmax + 1, m] = (block * gs[None, :]).sum(axis=1) / (4 * np.pi)
    return P, S


def basin_average(coeffs: SHCoeffs, mask_or_polygon, *, nmax: Optional[int] = None,
                  method: str = "both", lat_vec=None, lon_vec=None,
                  mask_kernel=None):
    """区域平均时间序列 —— **两种方法都算,并把差异明确报出来**。

    ``method='spatial'`` —— 逐历元在网格上按**面积加权**平均;
    ``method='coeff'``   —— 把掩膜展开成球谐(**区域平均核**),再与序列做系数内积;
    ``method='both'``(默认)—— 两种都算,返回 ``(times, spatial, coeff, diff)``。

    ⚠️ **实测结论(必须知道,否则会误以为两个口径在互相比对)**:当 ``coeff``
    用的掩膜网格与 ``spatial`` 用的综合格点**是同一套点**、``nmax`` 也相同时,
    两者**代数上恒等**,只差浮点舍入(→ 探针 ``_probe_basin.py`` 实测相对差
    ~1e-16)。原因:

        Σ_{n≤L,m} f_nm·g_nm  =  Σ_i w_i·f^L(x_i)      (逐点综合的定义)

    也就是说 coeff 口径**不是**另一个数值答案,而是一次**等价改写**;它的价值
    是**算得快** —— 核只建一次(O(nSHCS·N_mask)),之后每个历元只要
    O(nSHCS) 的内积,不用在网格上逐点综合。

    两者真正会分岔的地方只有一处:``mask_kernel`` 给的掩膜比综合用的掩膜**更细**
    (现实中很常见:流域边界是高分数字化的,而场只在粗网格上算)。这时
    ``coeff`` 用的是细网格上的核,``spatial`` 用的是粗网格上的加权,粗网格那一侧
    就带着掩膜边界的离散化误差。

    Parameters
    ----------
    mask_or_polygon : ndarray | str | Path
        二维掩膜(默认按全球格心网格),或一个网格文件路径
        (``.nc``/``.grd``/``.npy``/三列文本;平面米制坐标会被明确拒绝)。
    lat_vec, lon_vec : array_like, optional
        掩膜网格的坐标轴(不给就按全球等经纬格心推断)。
    mask_kernel : (lat, lon, mask) | str | Path, optional
        **只用来建球谐核**的高分辨率掩膜。给了它,``coeff`` 口径按它算,
        ``spatial`` 仍按 ``mask_or_polygon`` 算 —— 这才是两个口径该比的东西。
        **不给就与掩膜本身一致,此时两者恒等**(见上)。

    Returns
    -------
    (times, out[, out2, diff])
        ``method='spatial'``/``'coeff'`` 时返回 ``(times, 序列)``;
        ``'both'`` 时返回 ``(times, spatial, coeff, diff)``,其中 ``diff`` 是
        ``coeff - spatial``(同单位,便于直接看两个口径差多少)。
    """
    if method not in ("coeff", "spatial", "both"):
        raise ValueError("method 必须是 'coeff' | 'spatial' | 'both'")
    times = coeffs.times or TimeAxis.from_index(coeffs.ntime)
    L = coeffs.nmax if nmax is None else min(int(nmax), coeffs.nmax)
    la, lo, mask, area = _mask_setup(mask_or_polygon, lat_vec, lon_vec)
    g = np.nan_to_num(mask, nan=0.0)
    atot = float(np.sum(np.abs(g) * area))
    if atot <= 0:
        raise ValueError("掩膜全为 0(或全 NaN),无法做区域平均")

    results = {}
    if method in ("spatial", "both"):
        # 只算权重大于 0 的格点:流域掩膜通常只覆盖球面的一小部分,
        # 全网格综合是纯浪费(而且这些点乘 0 之后对结果毫无贡献)。
        ii, jj = np.nonzero(g != 0.0)
        if ii.size == 0:
            raise ValueError("掩膜没有非零格点,无法做区域平均")
        wts = (g[ii, jj] * area[ii, jj])
        # 经度可能被掩膜切成好几段;按"纬度分组"没法直接喂 evaluate,
        # 就直接把这些点的 (lat, lon) 成对喂进去(它本来就支持散点)。
        field = evaluate(la[ii], lo[jj], coeffs, nmax=L)
        if field.ndim == 1:
            field = field[:, None]
        results["spatial"] = (wts @ field) / atot
    if method in ("coeff", "both"):
        if mask_kernel is not None:
            if isinstance(mask_kernel, (str, os.PathLike)):
                kla, klo, km, _ = _mask_setup(mask_kernel)
            else:
                kla, klo, km = mask_kernel
                km = np.asarray(km, dtype=float)
            gp, gs = _project_mask_to_sh(np.nan_to_num(km, nan=0.0), kla, klo, L)
        else:
            gp, gs = _project_mask_to_sh(g, la, lo, L)
        g00 = float(gp[0, 0])
        if abs(g00) < 1e-12:
            raise ValueError(
                "掩膜展开后的零阶项接近 0:这说明掩膜在球面上的积分被正负抵消了"
                "(掩膜应当是非负的 0/1 权重)。请检查掩膜取值。")
        fC = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
        fS = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
        tri = np.tril(np.ones((L + 1, L + 1), dtype=bool))    # 有效 (n,m):m<=n
        out = np.empty(coeffs.ntime)
        for t in range(coeffs.ntime):
            C = fC[:L + 1, :L + 1, t]
            S = fS[:L + 1, :L + 1, t]
            # 4π 归一化下 ∫f·g dΩ = 4π Σ_{n,m}(C^f C^g + S^f S^g),而
            # g00 = (1/4π)∫g dΩ ⇒ ⟨f⟩_R = Σ / g00。
            acc = float(np.sum((C * gp[:L + 1, :L + 1])[tri]
                               + (S * gs[:L + 1, :L + 1])[tri]))
            out[t] = acc / g00
        results["coeff"] = out

    if method == "spatial":
        return times, results["spatial"]
    if method == "coeff":
        return times, results["coeff"]
    diff = results["coeff"] - results["spatial"]
    return times, results["spatial"], results["coeff"], diff


def basin_compare(coeffs: SHCoeffs, mask_or_polygon, *, nmax: Optional[int] = None,
                  lat_vec=None, lon_vec=None, mask_kernel=None) -> dict:
    """区域平均的**两种口径对比表**(方案 §4.4 要求:不许默不作声地选一个)。

    Returns
    -------
    dict
        ``times, spatial, coeff, diff, rel`` 以及一行中文 ``summary``。
    """
    times, sp, co, diff = basin_average(coeffs, mask_or_polygon, nmax=nmax,
                                       method="both", lat_vec=lat_vec,
                                       lon_vec=lon_vec, mask_kernel=mask_kernel)
    scale = float(np.nanmax(np.abs(sp))) or 1.0
    rel = diff / scale
    d = {
        "times": times, "spatial": sp, "coeff": co, "diff": diff,
        "rel": rel,
        "max_abs_diff": float(np.nanmax(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.nanmean(diff ** 2))),
        "max_rel_diff": float(np.nanmax(np.abs(rel))),
        "field_scale": scale,
    }
    d["summary"] = (
        f"区域平均两种口径(nmax={coeffs.nmax if nmax is None else nmax}):"
        f"最大差 {d['max_abs_diff']:.4g},RMS 差 {d['rms_diff']:.4g},"
        f"相对场量级最大 {d['max_rel_diff'] * 100:.2f}%\n"
        "  spatial = 网格面积加权;coeff = 掩膜球谐核内积。\n"
        + ("  ⚠️ 这次没有给 mask_kernel,两个口径用的是同一套格点 —— "
           "此时它们**代数上恒等**,差异应当只是浮点舍入(~1e-16 相对)。\n"
           "     若差异明显大于舍入,说明实现有问题,而不是'两种方法本就不同'。\n"
           "  coeff 的价值是**快**(核建一次,之后每历元只要 O(nSHCS) 的内积)。\n"
           if mask_kernel is None else
           "  这次给了 mask_kernel(更细的掩膜建核),两个口径用的是**不同的**"
           "掩膜离散化 ——\n"
           "     差异 = 粗网格掩膜边界的离散化误差,这才是该看的对比。\n"))
    return d


# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """系数域的时间拟合结果(趋势 + 固定周期周年项)。

    ``coef[0]`` 是常数项,``coef[1]``(``trend='linear'`` 时)是趋势(单位:1/年),
    之后每两个参数是一个周期的 ``cos``/``sin``。
    """

    periods: tuple
    trend: str
    coef: np.ndarray            # (npar, L+1, L+1, 2) → [参数, n, m, (C,S)]
    sigma: Optional[np.ndarray] = None
    times: Optional[TimeAxis] = None
    ntime: int = 0
    design: Optional[np.ndarray] = None
    rank: int = 0
    period_offset: int = 1
    param_names: list = field(default_factory=list)
    table: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    residual_rms: float = float("nan")
    source_meta: dict = field(default_factory=dict)

    # ---- 便捷量 --------------------------------------------------------
    def slope_coeffs(self) -> SHCoeffs:
        """趋势项系数(单位:系数/年);``trend='none'`` 时报错。"""
        if self.trend == "none":
            raise ValueError("这次拟合没有趋势项(trend='none')")
        return self._as_coeffs(1, "trend")

    def intercept_coeffs(self) -> SHCoeffs:
        """常数项(参考历元处的系数值)。"""
        return self._as_coeffs(0, "intercept")

    def _as_coeffs(self, i: int, label: str) -> SHCoeffs:
        # ⚠️ 必须带上源系数的物理量标签:否则 slope_coeffs() 再拿去综合时
        # 单位语义丢了,会出现"系数域趋势 ≡ 场域趋势"对不上的假象。
        # 拟合参数是**单张 (n,m) 图**(不是序列),所以 times=None。
        meta = {k: v for k, v in self.source_meta.items()
                if k in ("field_unit", "norm", "gaussian_km", "tide_system",
                         "center", "mission", "product", "rl")}
        meta.update({"fit": label, "trend": self.trend,
                     "periods": list(self.periods)})
        if self.times is not None and self.ntime:
            meta["fit_span"] = (f"{str(self.times.values[0])[:10]}.."
                                f"{str(self.times.values[-1])[:10]}")
            meta["fit_ntime"] = int(self.ntime)
        return SHCoeffs(self.coef[i][:, :, 0].copy(),
                        self.coef[i][:, :, 1].copy(), meta, None)

    def field(self, i: int, **kw):
        """第 ``i`` 个参数对应的**场**(把该参数当系数综合出去)。"""
        return self._as_coeffs(i, self.param_names[i] if self.param_names
                               else f"param{i}")

    def amplitude(self, period: float) -> np.ndarray:
        """周期项的**振幅场** ``(L+1, L+1)`` = ``hypot(cos, sin)``(逐 (n,m) 独立)。"""
        i = self._period_index(period)
        return np.hypot(self.coef[i][:, :, 0], self.coef[i + 1][:, :, 0])

    def phase(self, period: float) -> np.ndarray:
        """周期项的**相位场**(度,``atan2(sin, cos)`` 取 ``[0,360)``,跨 0 不断裂)。"""
        i = self._period_index(period)
        return np.rad2deg(np.arctan2(self.coef[i + 1][:, :, 0],
                                     self.coef[i][:, :, 0])) % 360.0

    def amplitude_sigma(self, period: float):
        """振幅的 1σ(只有 ``sigma=True`` 时可用;按误差传播合成)。"""
        if self.sigma is None:
            raise ValueError("这次拟合没有估误差(sigma=False)")
        i = self._period_index(period)
        c, s = self.coef[i][:, :, 0], self.coef[i + 1][:, :, 0]
        sc, ss = self.sigma[i][:, :, 0], self.sigma[i + 1][:, :, 0]
        amp = np.hypot(c, s)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt((c * sc) ** 2 + (s * ss) ** 2) / np.maximum(amp, 1e-300)

    def _period_index(self, period: float) -> int:
        for k, p in enumerate(self.periods):
            if abs(p - float(period)) < 1e-9:
                return self.period_offset + 2 * k
        raise ValueError(f"拟合里没有周期 {period};可用 {self.periods}")

    def summary(self) -> str:
        lines = [f"系数域时间拟合: trend={self.trend}  periods={self.periods}",
                 f"  历元数 {self.ntime}  设计矩阵秩 {self.rank}/"
                 f"{0 if self.design is None else self.design.shape[1]}",
                 f"  参数: {', '.join(self.param_names)}"]
        for p in self.periods:
            a = self.amplitude(p)
            lines.append(f"  周期 {p:g} 年 振幅 max = {np.nanmax(a):.6g}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def fit_trend_seasonal(coeffs: SHCoeffs, *, periods=(1.0, 0.5),
                       trend: str = "linear", reference: Optional[str] = None,
                       sigma: bool = False) -> FitResult:
    """在**系数域**拟合趋势 + 周年/半年项(与逐点拟合机器精度一致)。

    ``y(t) = a0 + a1·t + Σ_k [b_k cos(2πt/P_k) + c_k sin(2πt/P_k)]``,``t`` 以年为单位
    (用真实 ``datetime64`` 差,不做等间隔假设)。
    """
    if coeffs.times is None:
        raise ValueError(
            "fit_trend_seasonal 需要时间轴:系数没有日期就无法拟合趋势。"
            "请用 read_coeffs_series() 读入带日期的序列")
    ax = coeffs.times
    nt = coeffs.ntime
    if nt < 4:
        raise ValueError(f"只有 {nt} 个历元,拟合趋势没有意义(至少 4 个)")
    t = ax.decimal_years.astype(float)
    if reference == "mean":
        t = t - t.mean()
    elif reference is not None:
        t0 = float(ax.decimal_years[ax.nearest(reference)])
        t = t - t0

    cols = []
    names = []
    offset = 0
    if trend == "linear":
        cols.append(np.ones(nt)); names.append("intercept")
        cols.append(t.copy()); names.append("trend_per_year")
        offset = 1
    elif trend == "quadratic":
        cols.append(np.ones(nt)); names.append("intercept")
        cols.append(t.copy()); names.append("trend_per_year")
        cols.append(t ** 2); names.append("trend_per_year2")
        offset = 1
    elif trend == "none":
        cols.append(np.ones(nt)); names.append("mean")
        offset = 0
    else:
        raise ValueError("trend 必须是 'none'|'linear'|'quadratic'")
    period_offset = len(cols)
    for p in periods:
        cols.append(np.cos(2 * np.pi * t / float(p)))
        names.append(f"cos_{p}y")
        cols.append(np.sin(2 * np.pi * t / float(p)))
        names.append(f"sin_{p}y")
    A = np.column_stack(cols)                     # (ntime, npar)

    # sol 形状 (npar, ncol);ncol = 2*ncoef(前 ncoef 列是 C,后 ncoef 列是 S)
    C3 = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S3 = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    npar = A.shape[1]
    L = coeffs.nmax
    ncoef = (L + 1) * (L + 1)
    Y = np.concatenate([C3.reshape(-1, nt).T, S3.reshape(-1, nt).T], axis=1)
    sol, _res, rank, _sv = np.linalg.lstsq(A, Y, rcond=None)
    out = np.zeros((npar, L + 1, L + 1, 2))
    out[:, :, :, 0] = sol[:, :ncoef].reshape(npar, L + 1, L + 1)
    out[:, :, :, 1] = sol[:, ncoef:].reshape(npar, L + 1, L + 1)

    sig = None
    if sigma and rank == npar and nt > npar:
        resid = Y - A @ sol
        dof = nt - npar
        s2 = np.sum(resid ** 2, axis=0) / dof
        cov = np.linalg.inv(A.T @ A)
        se = np.sqrt(np.outer(np.diag(cov), s2))          # (npar, ncol)
        sig = np.zeros_like(out)
        sig[:, :, :, 0] = se[:, :ncoef].reshape(npar, L + 1, L + 1)
        sig[:, :, :, 1] = se[:, ncoef:].reshape(npar, L + 1, L + 1)

    res = FitResult(periods=tuple(float(p) for p in periods), trend=trend,
                    coef=out, sigma=sig, times=ax, ntime=nt, design=A,
                    rank=int(rank), period_offset=period_offset,
                    param_names=names, source_meta=dict(coeffs.meta))
    resid = Y - A @ sol
    res.table = {"param": names, "rank": int(rank), "ntime": nt}
    res.residual_rms = float(np.sqrt(np.mean(resid ** 2)))
    if rank < npar:
        res.warnings.append(
            f"设计矩阵秩亏(rank={rank} < {npar}):周期与时间跨度不匹配"
            "(例如不足一年的数据拟合周年项),结果不可靠")
    res.warnings.append(
        "残差里可能还有未建模的相关噪声(本版不做 AR(1)/Mann-Kendall,见方案 §8)")
    return res
