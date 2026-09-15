# -*- coding: utf-8 -*-
"""
shsynth.plotting
================

用 matplotlib 出图:等经纬地图(带离线海岸线)、散点图、逐阶谱、直方图、
差值图,以及一张四联"报告图"。

* 地图**不需要 cartopy**:海岸线用随包的 Natural Earth 110m 提取物
  (``data/coastline_110m.npz``,公有领域),离线可用,不会在第一次画图时去下载;
* 画图统一用 **matplotlib**(BSD 风格许可),不碰任何 GPL-only 的 Qt Charts;
* 中文标签:自动挑选本机已装的中文字体,并把 ``axes.unicode_minus`` 关掉
  (U+2212 在很多中文字体里没有字形,会出现方框);
* 经度统一画到 ``[-180, 180]``(与 SHKit 界面一致),并在换日线处**断开**
  折线(``prepare_polyline``),否则会出现"飞线"。

所有 ``make_*_figure`` 函数都返回 :class:`matplotlib.figure.Figure`,所以命令行
(保存到文件)和界面(嵌进 Qt 画布)用的是同一套绘图代码。
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "setup_fonts",
    "load_coastlines",
    "wrap_longitude",
    "prepare_polyline",
    "data_extent",
    "make_map_figure",
    "make_spectrum_figure",
    "make_hist_figure",
    "make_diff_figure",
    "make_report_figure",
    "make_timeseries_figure",
    "make_vector_map_figure",
    "epoch_label",
    "make_series_frame",
    "save_animation",
    "save_figure",
    "figure_to_array",
]

_DATA_DIR = os.environ.get(
    "SHSYNTH_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

_COAST_CACHE: dict = {}

_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                   "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB",
                   "WenQuanYi Zen Hei", "Arial Unicode MS", "MS Gothic")


def setup_fonts(verbose: bool = False) -> list:
    """让 matplotlib 能画中文(幂等)。返回实际选中的字体名列表。"""
    import matplotlib
    from matplotlib import font_manager

    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:                                    # pragma: no cover
        available = set()
    chosen = [n for n in _CJK_CANDIDATES if n in available]
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    if verbose:
        print("matplotlib 中文字体:", chosen or "未找到,中文可能显示为方框")
    return chosen


setup_fonts()


# ---------------------------------------------------------------------------
# 海岸线
# ---------------------------------------------------------------------------
def load_coastlines(which: str = "coastline"):
    """返回带 NaN 分隔的 ``(lon, lat)`` 折线数组;读不到返回 ``None``。"""
    if which in _COAST_CACHE:
        return _COAST_CACHE[which]
    path = os.path.join(_DATA_DIR, f"{which}_110m.npz")
    try:
        with np.load(path, allow_pickle=False) as z:
            val = (np.asarray(z["lon"], dtype=float),
                   np.asarray(z["lat"], dtype=float))
    except Exception:                                    # noqa: BLE001
        val = None
    _COAST_CACHE[which] = val
    return val


def wrap_longitude(lon) -> np.ndarray:
    """把经度映射到 ``[-180, 180)``。"""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def wrap_into(lon, lo: float = -180.0) -> np.ndarray:
    """把经度映射到 ``[lo, lo+360)``(跨换日线的窗口需要用 ``lo=0`` 画)。"""
    return (np.asarray(lon, dtype=float) - lo) % 360.0 + lo


def prepare_polyline(lon, lat, lon_min: float = -180.0):
    """环绕并**在换日线处断开**折线(逐点环绕是不够的)。

    逐点环绕时,+180 的顶点会变成 −180,本来无害的 1° 步长就变成 359° 跳变,
    matplotlib 会画一条横穿全图的线 —— 就是那个经典的"飞线"。这里在所有
    环绕后跳变超过 180° 的位置插入 NaN,让渲染器断线。
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    bad = np.isnan(lon) | np.isnan(lat)

    w = (np.where(bad, 0.0, lon) + 180.0) % 360.0 - 180.0
    if lon_min not in (-180.0, 180.0):
        w = np.where(bad, 0.0, w % 360.0)

    jump = np.zeros(w.size, dtype=bool)
    if w.size > 1:
        jump[1:] = np.abs(np.diff(w)) > 180.0
    broken = bad | jump

    n_add = int(broken.sum())
    out_lon = np.full(w.size + n_add, np.nan)
    out_lat = np.full(w.size + n_add, np.nan)
    if w.size:
        shifts = np.cumsum(broken)
        pos = np.arange(w.size) + shifts
        good = ~bad
        out_lon[pos[good]] = w[good]
        out_lat[pos[good]] = lat[good]
    return out_lon, out_lat


def _new_fig(fig=None, figsize=(7.4, 4.3), dpi: int = 110):
    """取一张要画的图:``fig=None`` 新建;否则清空复用。

    复用同一个 :class:`~matplotlib.figure.Figure` 让**命令行保存**与**界面内嵌**
    走完全相同的绘图代码 —— 界面把自带的 Figure 传进来,画完以后既显示也能存盘。
    """
    import matplotlib.pyplot as plt

    if fig is None:
        return plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
    fig.clear()
    try:
        fig.set_layout_engine("constrained")
    except Exception:                                    # pragma: no cover
        pass
    return fig


def _symmetric_range(v, symmetric: bool = True, pct: float = 98.0,
                     robust: bool = True):
    """配色范围;``symmetric`` 时关于 0 对称(异常场看得更清楚)。"""
    v = np.asarray(v, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return -1.0, 1.0
    if robust:
        lo, hi = np.percentile(v[finite], [100.0 - pct, pct])
    else:
        lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    if symmetric:
        a = max(abs(lo), abs(hi))
        return -a, a
    return float(lo), float(hi)


def _decorate_map(ax, title: str = "", grid_lines: bool = True,
                  coast: bool = True, extent=None):
    """统一地图外观:等经纬框、经纬网、**只画海岸线**。

    ⚠️ 这里**刻意不画国界**。国界数据(即使在 Natural Earth 这类公开数据里)也
    带有主权争议的划线方式,画在地图上容易引起不必要的政治争议;海岸线是自然
    地理要素,没有这个问题。需要国界的场合请用户自己叠加自己的数据。

    ``extent = (lon_min, lon_max, lat_min, lat_max)`` 时只显示该窗口(聚焦)。
    """
    if extent is None:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    try:
        ax.set_box_aspect(((extent[3] - extent[2]) / (extent[1] - extent[0]))
                          if extent else 0.5)
    except Exception:                                    # pragma: no cover
        pass
        ax.set_box_aspect(0.5)
    ax.set_xlabel("经度 (°)")
    ax.set_ylabel("纬度 (°)")
    if title:
        ax.set_title(title, fontsize=10)
    if grid_lines:
        if extent is None:
            ax.set_xticks(np.arange(-180, 181, 60))
            ax.set_yticks(np.arange(-90, 91, 30))
        else:
            xs = _nice_ticks(extent[0], extent[1], 6)
            ys = _nice_ticks(extent[2], extent[3], 5)
            ax.set_xticks(xs)
            ax.set_yticks(ys)
            ax.tick_params(labelsize=8)
        ax.grid(True, color="0.88", linewidth=0.6, zorder=0)
    if coast:
        # **只画海岸线,不画国界**(见上面的说明)
        c = load_coastlines("coastline")
        if c is not None:
            px, py = prepare_polyline(c[0], c[1],
                                      lon_min=ax.get_xlim()[0])
            ax.plot(px, py, color="0.25", linewidth=0.6, zorder=3)


def _nice_ticks(lo: float, hi: float, target: int = 6) -> np.ndarray:
    """在 ``[lo, hi]`` 里取一组"好看"的刻度(聚焦小窗口时用)。"""
    span = float(hi) - float(lo)
    if span <= 0:
        return np.array([lo])
    raw = span / max(int(target), 1)
    mag = 10.0 ** np.floor(np.log10(abs(raw))) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if raw <= step:
            break
    else:                                                # pragma: no cover
        step = raw
    start = np.ceil(lo / step) * step
    ticks = np.arange(start, hi + step * 1e-9, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


def data_extent(lat, lon, values=None, pad_frac: float = 0.06,
                global_frac: float = 0.95):
    """由数据算出要聚焦的经纬窗口 ``(lon_min, lon_max, lat_min, lat_max)``。

    * ``values`` 给网格(二维)时,``lat``/``lon`` 是坐标轴;给散点(一维)时是
      坐标数组。
    * **覆盖接近全球时返回 ``None``**(没有可聚焦的东西,就画全球)。
    * 跨换日线的经度会自动按"最窄跨度"处理,再留一点边距。

    Returns
    -------
    tuple or None
    """
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    if values is not None:
        v = np.asarray(values, dtype=float)
        if v.ndim == 3:
            v = v[:, :, 0]
        if v.ndim == 2 and v.shape[0] == lat.size and v.shape[1] == lon.size:
            # 只按**有值**的格点定窗口(全 NaN 的行/列不要撑大范围)
            rows = np.isfinite(v).any(axis=1)
            cols = np.isfinite(v).any(axis=0)
            if rows.any() and cols.any():
                lat = lat[rows]
                lon = lon[cols]
    if lat.size == 0 or lon.size == 0:
        return None
    lat_lo, lat_hi = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_lo, lon_hi = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_span = lat_hi - lat_lo
    lon_span_raw = lon_hi - lon_lo

    # 覆盖接近全球 → 不聚焦
    if lat_span >= 170.0 * global_frac and lon_span_raw >= 350.0 * global_frac:
        return None
    # 纬度覆盖全球但经度只占一部分(例如一条纬向剖面):纬度仍给全球
    if lat_span >= 170.0 * global_frac:
        lat_lo, lat_hi = -90.0, 90.0

    # 经度:若有 0/360 两侧的点,按"最窄跨度"重排(避免横跨全球)
    if lon_span_raw > 180.0:
        lons = np.sort(np.mod(lon[np.isfinite(lon)], 360.0))
        gaps = np.diff(np.concatenate([lons, [lons[0] + 360.0]]))
        k = int(np.argmax(gaps))
        lon_lo = float(lons[k + 1]) if k + 1 < lons.size else float(lons[0])
        lon_hi = float(lons[k]) if k < lons.size else float(lons[-1])
        if lon_hi < lon_lo:
            lon_hi += 360.0
        lon_span = lon_hi - lon_lo
    else:
        lon_span = lon_span_raw

    pad_lon = max(lon_span * pad_frac, 0.5)
    pad_lat = max(lat_span * pad_frac, 0.5)
    lon_lo, lon_hi = lon_lo - pad_lon, lon_hi + pad_lon
    lat_lo = max(-90.0, lat_lo - pad_lat)
    lat_hi = min(90.0, lat_hi + pad_lat)
    if lon_hi - lon_lo > 360.0:
        lon_lo, lon_hi = -180.0, 180.0
    return (float(lon_lo), float(lon_hi), float(lat_lo), float(lat_hi))


def _add_colorbar(fig, mappable, ax, label: str = ""):
    cb = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
    if label:
        cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return cb


# ---------------------------------------------------------------------------
# 地图
# ---------------------------------------------------------------------------
def make_map_figure(lat, lon, values,
                    title: str = "", cmap: str = "RdBu_r",
                    symmetric: bool = True, cb_label: str = "",
                    coast: bool = True, grid_lines: bool = True,
                    contour: bool = False, n_contour: int = 12,
                    point_size: float = 8.0, robust: bool = True,
                    vmin: Optional[float] = None, vmax: Optional[float] = None,
                    figsize=(7.4, 4.3), dpi: int = 110,
                    annotate: Optional[str] = None, fig=None,
                    extent=None, focus: str = "auto"):
    """画一张地图。

    * ``values`` 是二维 ``(nlat, nlon)`` 时按**规则网格**画
      (``lat``/``lon`` 为坐标轴向量);
    * ``values`` 是一维 ``(N,)`` 时按**散点**画
      (``lat``/``lon`` 为长度相同的坐标数组)。

    Parameters
    ----------
    values : array_like
        网格 ``(nlat, nlon)``(或 ``(nlat, nlon, ntime)``,只画第一个时次)
        或散点 ``(N,)``。
    contour : bool
        网格数据额外叠加等值线。
    symmetric : bool
        配色关于 0 对称(异常场默认)。
    extent : tuple, optional
        显式窗口 ``(lon_min, lon_max, lat_min, lat_max)``。
    focus : {'auto', 'global', 'none'}
        ``'auto'``(默认)在数据明显只覆盖一块区域时**自动聚焦到结果范围**,
        覆盖接近全球时照旧画全球;``'global'`` 强制全球视图(不聚焦);
        ``'none'`` 与 auto 同义(便于界面开关)。
    fig : Figure, optional
        传进来就画在这张图上(界面内嵌用),否则新建一张。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()

    if extent is None and focus in ("auto", "none"):
        extent = data_extent(lat, lon, values)

    fig = _new_fig(fig, figsize, dpi)
    ax = fig.add_subplot(111)

    is_grid = values.ndim == 2 and lat.size == values.shape[0] and \
        lon.size == values.shape[1]
    # 聚焦窗口若跨过换日线(lon_max > 180),就把数据画到 0..360 那一侧,
    # 否则窗口与数据不在同一个经度域里,图会是空的。
    wrap_lo = 0.0 if (extent is not None and extent[1] > 180.0) else -180.0
    if is_grid:
        lo = wrap_into(lon, wrap_lo)
        order = np.argsort(lo)
        lo_s = lo[order]
        g = values[:, order]
        if vmin is None or vmax is None:
            a, b = _symmetric_range(g, symmetric, robust=robust)
            vmin = a if vmin is None else vmin
            vmax = b if vmax is None else vmax
        mappable = ax.pcolormesh(lo_s, lat, g, cmap=cmap, vmin=vmin,
                                 vmax=vmax, shading="auto", zorder=2)
        if contour and g.shape[0] > 2 and g.shape[1] > 2:
            try:
                lv = np.linspace(vmin, vmax, int(n_contour) + 1)[1:-1]
                ax.contour(lo_s, lat, g, levels=lv, colors="0.25",
                           linewidths=0.45, zorder=2.5)
            except Exception:                            # pragma: no cover
                pass
    else:
        if values.ndim == 2:
            raise ValueError(
                f"values {values.shape} 与坐标轴(lat {lat.size}, lon {lon.size})"
                "不匹配:网格需要 shape == (nlat, nlon)")
        lon_w = wrap_into(lon, wrap_lo)
        if vmin is None or vmax is None:
            a, b = _symmetric_range(values, symmetric, robust=robust)
            vmin = a if vmin is None else vmin
            vmax = b if vmax is None else vmax
        mappable = ax.scatter(lon_w, lat, c=values, cmap=cmap, s=point_size,
                              vmin=vmin, vmax=vmax, linewidths=0, zorder=2)

    _decorate_map(ax, title, grid_lines=grid_lines, coast=coast, extent=extent)
    _add_colorbar(fig, mappable, ax, cb_label)
    if annotate:
        ax.text(0.995, 0.02, annotate, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.5, color="0.35", zorder=5)
    return fig


def make_diff_figure(lat_vec, lon_vec, ref, vals, title: str = "差值图",
                     cmap: str = "RdBu_r", cb_label: str = "",
                     contour: bool = False, extent=None, focus: str = "auto",
                     **kw):
    """``vals - ref`` 的差值地图(两个网格同形)。"""
    ref = np.asarray(ref, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if ref.ndim == 3:
        ref = ref[:, :, 0]
    if vals.ndim == 3:
        vals = vals[:, :, 0]
    if ref.shape != vals.shape:
        raise ValueError(f"差值图要求同形,实际 {ref.shape} vs {vals.shape}")
    return make_map_figure(lat_vec, lon_vec, vals - ref, title=title, cmap=cmap,
                           symmetric=True, cb_label=cb_label,
                           contour=contour, extent=extent, focus=focus, **kw)


# ---------------------------------------------------------------------------
# 曲线 / 统计
# ---------------------------------------------------------------------------
def make_spectrum_figure(curves, title: str = "逐阶振幅 (degree RMS)",
                         ylabel: str = "RMS", log: bool = True,
                         xlabel: str = "球谐阶 n", figsize=(7.0, 4.0),
                         dpi: int = 110, styles=None, fig=None):
    """逐阶曲线(可多条),默认对数纵轴。"""
    fig = _new_fig(fig, figsize, dpi)
    ax = fig.add_subplot(111)
    for label, y in dict(curves).items():
        y = np.asarray(y, dtype=float).ravel()
        if y.size == 0:
            continue
        x = np.arange(y.size)
        style = (styles or {}).get(label, {})
        ax.plot(x, y, marker="o", markersize=2.6, linewidth=1.3, label=label,
                **style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if log:
        finite = [np.asarray(y, float) for y in curves.values()
                  if np.asarray(y, float).size and
                  np.any(np.asarray(y, float) > 0)]
        if finite:
            ax.set_yscale("log")
    ax.grid(True, which="both", color="0.9", linewidth=0.6)
    if curves:
        ax.legend(fontsize=8, frameon=False)
    return fig


def make_timeseries_figure(times, curves, title: str = "时间序列",
                           ylabel: str = "", xlabel: str = "十进制年",
                           figsize=(7.4, 4.4), dpi: int = 110,
                           styles=None, fig=None, trend=None,
                           seasonal=None, marker: bool = True,
                           ylim=None, reference_lines=True):
    """时间序列曲线(v2.0)。

    Parameters
    ----------
    times : TimeAxis | array_like
        :class:`~shsynth.timeaxis.TimeAxis`(横轴用它的 ``decimal_years``),
        或直接给等长的横轴数值。
    curves : mapping
        ``{标签: (ntime,) 数组}``;也可以直接给 ``(npoints, ntime)`` 的 ndarray
        (会自动按点数命名)。
    trend, seasonal : mapping, optional
        ``{标签: (ntime,) 数组}`` 的叠加线(趋势用虚线、周年用细实线),
        画在同一张图上但**不参与图例计数**之外的处理,便于肉眼判断。
    """
    from .timeaxis import TimeAxis
    fig = _new_fig(fig, figsize, dpi)
    ax = fig.add_subplot(111)
    if isinstance(times, TimeAxis):
        if times.kind == "index" or not times.has_dates:
            x = np.arange(len(times), dtype=float)
            xlabel = "历元序号(无日期)"
        else:
            x = np.asarray(times.decimal_years, dtype=float)
    else:
        x = np.atleast_1d(np.asarray(times, dtype=float)).ravel()
    if not isinstance(curves, dict):
        arr = np.asarray(curves, dtype=float)
        arr = arr[None, :] if arr.ndim == 1 else arr
        curves = {f"点 {i + 1}": arr[i] for i in range(arr.shape[0])}
    styles = dict(styles or {})
    for label, y in dict(curves).items():
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        if y.size != x.size:
            y = y[:x.size] if y.size > x.size else np.pad(
                y, (0, x.size - y.size), constant_values=np.nan)
        kw = dict(styles.get(label, {}))
        if marker:
            kw.setdefault("marker", "o")
            kw.setdefault("markersize", 2.8)
        ax.plot(x, y, linewidth=1.3, label=label, **kw)
    for label, y in dict(trend or {}).items():
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()[:x.size]
        ax.plot(x, y, color="0.25", linewidth=1.4, linestyle="--",
                label=f"{label}(趋势)")
    for label, y in dict(seasonal or {}).items():
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()[:x.size]
        ax.plot(x, y, color="0.55", linewidth=1.0, label=f"{label}(周年)")
    if reference_lines:
        ax.axhline(0.0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="0.9", linewidth=0.6)
    if curves or trend or seasonal:
        ax.legend(fontsize=8, frameon=False, ncol=1)
    return fig


def make_hist_figure(values, title: str = "数值分布", bins: int = 60,
                     xlabel: str = "值", log_y: bool = True,
                     figsize=(6.4, 3.6), dpi: int = 110, fig=None):
    """直方图(带均值/标准差标注)。"""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    fig = _new_fig(fig, figsize, dpi)
    ax = fig.add_subplot(111)
    if v.size:
        ax.hist(v, bins=int(bins), color="#4a7fb5", edgecolor="white",
                linewidth=0.4)
        mu, sd = float(np.mean(v)), float(np.std(v))
        ax.axvline(mu, color="#c0392b", linewidth=1.2,
                   label=f"均值 {mu:.4g}")
        ax.axvline(mu - sd, color="#c0392b", linewidth=0.8, linestyle="--",
                   label=f"±1σ ({sd:.4g})")
        ax.axvline(mu + sd, color="#c0392b", linewidth=0.8, linestyle="--")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "没有有限值", ha="center", va="center",
                transform=ax.transAxes, color="0.6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("频数")
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="0.9", linewidth=0.6)
    return fig


def make_report_figure(lat, lon, values, coeffs=None, curves=None,
                       title: str = "", cmap: str = "RdBu_r",
                       cb_label: str = "", contour: bool = False,
                       annotation: Optional[Sequence[str]] = None,
                       lat_vec=None, lon_vec=None, figsize=(12.0, 7.0),
                       dpi: int = 110, fig=None, extent=None,
                       focus: str = "auto"):
    """四联报告图:地图 + 逐阶谱 + 直方图 + 摘要标注。

    ``focus`` 与 :func:`make_map_figure` 相同:默认在区域数据上**自动聚焦**,
    覆盖接近全球时画全球。
    """
    values = np.asarray(values, dtype=float)
    flat = values[:, :, 0] if values.ndim == 3 else values
    if extent is None and focus in ("auto", "none"):
        extent = data_extent(lat, lon, values)
    fig = _new_fig(fig, figsize, dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=(1.45, 1.0))

    # --- 左上:地图 ---
    ax_map = fig.add_subplot(gs[:, 0])
    wrap_lo = 0.0 if (extent is not None and extent[1] > 180.0) else -180.0
    if lat_vec is not None and lon_vec is not None and flat.ndim == 2:
        lo = wrap_into(np.asarray(lon_vec, float).ravel(), wrap_lo)
        order = np.argsort(lo)
        a, b = _symmetric_range(flat, True)
        m = ax_map.pcolormesh(lo[order], np.asarray(lat_vec, float).ravel(),
                              flat[:, order], cmap=cmap, vmin=a, vmax=b,
                              shading="auto", zorder=2)
        if contour:
            try:
                ax_map.contour(lo[order], np.asarray(lat_vec, float).ravel(),
                               flat[:, order],
                               levels=np.linspace(a, b, 13)[1:-1],
                               colors="0.25", linewidths=0.4, zorder=2.5)
            except Exception:                            # pragma: no cover
                pass
        _decorate_map(ax_map, title or "球谐综合结果", extent=extent)
        _add_colorbar(fig, m, ax_map, cb_label)
    else:
        lo = wrap_into(np.asarray(lon, float).ravel(), wrap_lo)
        a, b = _symmetric_range(flat, True)
        m = ax_map.scatter(lo, np.asarray(lat, float).ravel(), c=flat.ravel(),
                           cmap=cmap, s=6.0, vmin=a, vmax=b, linewidths=0,
                           zorder=2)
        _decorate_map(ax_map, title or "球谐综合结果(散点)", extent=extent)
        _add_colorbar(fig, m, ax_map, cb_label)

    # --- 右上:逐阶谱 ---
    ax_sp = fig.add_subplot(gs[0, 1])
    drawn = False
    for label, y in dict(curves or {}).items():
        y = np.asarray(y, dtype=float).ravel()
        if y.size:
            ax_sp.semilogy(np.arange(y.size), np.maximum(y, 1e-300),
                           marker="o", markersize=2.4, linewidth=1.2,
                           label=label)
            drawn = True
    if not drawn and coeffs is not None:
        rms = coeffs.degree_rms(time="mean")      # 多时次时用时间平均(单条曲线)
        ax_sp.semilogy(np.arange(rms.size), np.maximum(rms, 1e-300),
                       marker="o", markersize=2.4, linewidth=1.2,
                       label="系数逐阶 RMS")
        drawn = True
    if drawn:
        ax_sp.legend(fontsize=8, frameon=False)
    ax_sp.set_xlabel("球谐阶 n")
    ax_sp.set_ylabel("RMS")
    ax_sp.set_title("逐阶振幅", fontsize=10)
    ax_sp.grid(True, which="both", color="0.9", linewidth=0.6)

    # --- 右下:直方图 + 摘要 ---
    ax_hi = fig.add_subplot(gs[1, 1])
    v = flat[np.isfinite(flat)]
    if v.size:
        ax_hi.hist(v, bins=50, color="#4a7fb5", edgecolor="white", linewidth=0.4)
        ax_hi.axvline(float(np.mean(v)), color="#c0392b", linewidth=1.1)
        ax_hi.set_yscale("log")
        ax_hi.set_title(f"数值分布 (RMS={np.sqrt(np.mean(v ** 2)):.4g})",
                        fontsize=10)
    else:
        ax_hi.text(0.5, 0.5, "没有有限值", ha="center", va="center",
                   transform=ax_hi.transAxes, color="0.6")
    ax_hi.set_xlabel(cb_label or "值")
    ax_hi.grid(True, color="0.9", linewidth=0.6)
    if annotation:
        ax_hi.text(0.99, 0.96, "\n".join(annotation), transform=ax_hi.transAxes,
                   ha="right", va="top", fontsize=7.0, color="0.3")
    return fig


def make_vector_map_figure(lat, lon, north, east, *, magnitude=None,
                           title: str = "水平形变", stride: int = 8,
                           cmap: str = "viridis", cb_label: str = "m",
                           coast: bool = True, focus: str = "auto",
                           vec_scale: float = 1.0, figsize=(8.6, 4.6),
                           dpi: int = 110, fig=None, max_arrows: int = 900):
    """水平形变矢量图:``magnitude`` 打底 + ``quiver`` 箭头(v2.0)。

    显示约定(方案 §4.6):

    * 底图用**大小**(非负、无零点对称性)⇒ 单色系 ``viridis``,不用 ``RdBu_r``;
    * 箭头按 ``stride`` 抽稀,箭头数控制在 ``max_arrows`` 量级;
    * **极点不画箭头**(那里"北/东"方向本身没有定义),但底图照样有值;
    * 箭头长度按 ``magnitude`` 归一化后会再乘 ``vec_scale``,便于目视。
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    n = np.asarray(north, dtype=float)
    e = np.asarray(east, dtype=float)
    if n.ndim > 2:
        n, e = n[:, :, 0], e[:, :, 0]
        if magnitude is not None:
            magnitude = np.asarray(magnitude)[:, :, 0]
    mag = np.hypot(n, e) if magnitude is None else np.asarray(magnitude, float)
    if mag.ndim > 2:
        mag = mag[:, :, 0]
    LA, LO = np.meshgrid(lat, lon, indexing="ij") if n.shape == (lat.size, lon.size) \
        else (None, None)

    fig = _new_fig(fig, figsize, dpi)
    ax = fig.add_subplot(111)
    if LA is not None:
        finite = mag[np.isfinite(mag)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        im = ax.pcolormesh(lon, lat, mag, cmap=cmap, shading="auto",
                           vmin=0.0, vmax=vmax or 1.0)
        _add_colorbar(fig, im, ax, cb_label)
    else:
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        finite = mag[np.isfinite(mag)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        im = ax.scatter(LO.ravel(), LA.ravel(), c=mag.ravel(), cmap=cmap,
                        s=6, vmin=0.0, vmax=vmax or 1.0)
        _add_colorbar(fig, im, ax, cb_label)

    # 箭头:抽稀 + 跳过极点行
    st = max(int(stride), 1)
    sel_lat = np.arange(0, lat.size, st)
    sel_lon = np.arange(0, lon.size, st)
    if len(sel_lat) * len(sel_lon) > max_arrows:
        k = int(np.ceil(np.sqrt(len(sel_lat) * len(sel_lon) / max_arrows)))
        sel_lat = np.arange(0, lat.size, st * k)
        sel_lon = np.arange(0, lon.size, st * k)
    keep = np.array([i for i in sel_lat if abs(abs(lat[i]) - 90.0) > 1e-9])
    if keep.size and n.shape == (lat.size, lon.size):
        U = e[np.ix_(keep, sel_lon)]
        V = n[np.ix_(keep, sel_lon)]
        X, Y = np.meshgrid(lon[sel_lon], lat[keep])
        m = np.hypot(U, V)
        with np.errstate(invalid="ignore", divide="ignore"):
            U = np.where(m > 0, U / np.maximum(m, 1e-300), 0.0) * vec_scale
            V = np.where(m > 0, V / np.maximum(m, 1e-300), 0.0) * vec_scale
        ax.quiver(X, Y, U, V, color="0.15", pivot="mid", width=0.0025,
                  scale=25.0, zorder=5)
    if coast:
        try:
            for ln_lon, ln_lat in prepare_polyline(*load_coastlines("coastline")):
                ax.plot(ln_lon, ln_lat, color="0.35", linewidth=0.5, zorder=4)
        except Exception:                                    # noqa: BLE001
            pass
    ext = None if focus == "global" else data_extent(lat, lon, mag)
    if ext is not None:
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    _decorate_map(ax, title, grid_lines=True)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    return fig


def epoch_label(times, k: int) -> str:
    """第 ``k`` 个历元的人类可读标签(动画角标/单帧标题共用)。

    ``TimeAxis`` 有日期就用**日期**(``2002-04-18``)—— 源文件名(GSM-2_2002095-
    2002120_...gfc)虽然也"含日期",但塞进角标又长又难认;出处信息在逐历元诊断表
    里,不靠角标承载。无日期时退化成 ``#k``。
    """
    try:
        from .timeaxis import TimeAxis
    except Exception:                                        # noqa: BLE001
        TimeAxis = ()                                        # type: ignore
    if isinstance(times, TimeAxis):
        if not times.has_dates:
            return f"#{k}"
        return str(times.values[k])[:10]
    if times is None:
        return f"#{k}"
    arr = np.atleast_1d(np.asarray(times, dtype=float)).ravel()
    return f"{arr[k]:.4f}" if k < arr.size else f"#{k}"


def make_series_frame(times, lat, lon, values, k: int, *,
                      title: str = "", cb_label: str = "",
                      annotate_time: bool = True, focus: str = "auto",
                      figsize=(7.4, 4.3), dpi: int = 110, **kw):
    """场序列 ``(nlat, nlon, ntime)`` 的第 ``k`` 帧地图(**动画用**)。

    直接复用 :func:`make_map_figure`,所以配色/投影/聚焦行为与单张图完全一致 ——
    动画里每一帧看起来就该和"单独画这一历元"一模一样。

    ``kw`` 原样转给 :func:`make_map_figure`(``cmap``/``symmetric``/``vmin``/
    ``vmax``/``contour``/``extent`` …)。**配色范围(vmin/vmax)要由调用方给定**,
    否则每帧各自定标、动画会"闪";这是动画最容易被忽略的坑,见 :func:`save_animation`。
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"场序列需要 (nlat, nlon, ntime),实际 {values.shape}")
    if not 0 <= int(k) < values.shape[2]:
        raise IndexError(f"帧号 {k} 超出范围(共 {values.shape[2]} 帧)")
    k = int(k)
    lab = epoch_label(times, k)
    ttl = f"{title}  {lab}".strip() if title else lab
    fig = make_map_figure(lat, lon, values[:, :, k], title=ttl,
                          cb_label=cb_label, focus=focus,
                          figsize=figsize, dpi=dpi, **kw)
    if annotate_time:
        try:
            fig.axes[0].text(0.005, 0.02, lab, transform=fig.axes[0].transAxes,
                             ha="left", va="bottom", fontsize=8.5, color="0.2",
                             bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                       ec="0.75", alpha=0.8), zorder=6)
        except Exception:                                    # pragma: no cover
            pass
    return fig


def save_animation(path, frames, *, fps: float = 5.0, loop: int = 0,
                   dpi: int = 100, progress=None, cancel=None,
                   n_frames: Optional[int] = None, info: Optional[dict] = None,
                   close_figures: bool = True) -> str:
    """把一串帧写成动图(**主格式 GIF**,纯 Pillow 编码)。

    Parameters
    ----------
    path : str | Path
        ``.gif`` 用 Pillow 编码(依赖最轻,推荐)。``.mp4``/``.webm``/``.mkv``
        需要 ``imageio`` + ``imageio-ffmpeg``;**没装就明确报错**并告诉你改用
        ``.gif``,绝不静默写出一张静止图冒充动画。
    frames : iterable | callable
        每个元素可以是 ``(H, W, 3|4)`` 的 uint8 数组(已渲染好的帧),或一个
        matplotlib ``Figure``(会就地渲染;渲染完默认关掉以免泄漏)。
        也可以给 callable ``fn(k) -> 上面两种之一``,此时**必须**同时给
        ``n_frames``。
    fps : float
        帧率(帧/秒)。GIF 的延时只能取 10 ms 的整数倍,所以实际帧率会被量化 ——
        真值通过 ``info['fps_effective']`` 回报,**不假装就是你要的那个数**。
    progress : callable, optional
        ``progress(done, total)``,每帧回调一次(界面进度条用)。
    cancel : callable, optional
        返回真值时**中止并抛** :class:`KeyboardInterrupt`(不写出半个文件)。
    info : dict, optional
        传进来就会被填上 ``frames / fps_requested / fps_effective / delay_ms
        / bytes / format``。
    close_figures : bool
        帧是 Figure 时,渲染完是否 ``plt.close``(默认关)。

    Returns
    -------
    str
        写出的文件路径。
    """
    p = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)
    ext = os.path.splitext(p)[1].lower()

    # 帧源:list 或惰性 callable(界面导出时不必先把 ntime 张图全建出来)
    if callable(frames):
        if n_frames is None:
            raise TypeError("frames 是 callable 时必须给 n_frames=帧数,"
                            "否则无法知道要渲染多少帧")
        total = int(n_frames)
        src = (frames(k) for k in range(total))
    else:
        src = frames
        total = len(src) if hasattr(src, "__len__") else None
    if total is not None and total <= 0:
        raise ValueError("一帧都没有,无法写动画")

    pil_frames = []
    for i, fr in enumerate(src):
        if cancel is not None and cancel():
            raise KeyboardInterrupt("动画导出已取消")
        if hasattr(fr, "canvas"):                            # matplotlib Figure
            arr = figure_to_array(fr, dpi=dpi)
            if close_figures:
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fr)
                except Exception:                            # pragma: no cover
                    pass
        else:
            arr = np.asarray(fr)
        if arr.dtype != np.uint8:
            a = np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0)
            lo, hi = float(np.min(a)), float(np.max(a))
            a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
            arr = (a * 255.0 + 0.5).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"第 {i} 帧形状 {arr.shape} 不是 (H, W, 3|4)")
        pil_frames.append(arr)
        if progress is not None:
            progress(i + 1, total if total is not None else i + 1)
    if not pil_frames:
        raise ValueError("一帧都没有,无法写动画")

    def _fill(**kw):
        if info is not None:
            info.clear()
            info.update(frames=len(pil_frames), **kw)

    if ext in (".gif", ""):
        try:
            from PIL import Image
        except Exception as exc:                             # noqa: BLE001
            raise RuntimeError(
                "写 GIF 需要 Pillow(没装)。装法:pip install Pillow;"
                "或者改成写 .mp4(需要 imageio + imageio-ffmpeg)。") from exc
        delay_ms = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
        imgs = [Image.fromarray(f if f.shape[2] == 3 else f[:, :, :3], "RGB")
                for f in pil_frames]
        imgs[0].save(p, format="GIF", save_all=True, append_images=imgs[1:],
                     duration=delay_ms, loop=int(loop), optimize=False,
                     disposal=2)
        _fill(format="GIF", fps_requested=float(fps),
              fps_effective=1000.0 / delay_ms, delay_ms=delay_ms,
              bytes=os.path.getsize(p))
        return p

    # --- 视频:需要 imageio,缺了就说清楚,不降级 ------------------------
    fmt = ext.lstrip(".") or "mp4"
    try:
        import imageio.v2 as imageio
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(
            f"写 {ext or '.mp4'} 需要 imageio(+ imageio-ffmpeg),当前环境没有:"
            f"{exc}\n  两个选择:① pip install imageio imageio-ffmpeg;"
            "② 把输出改成 .gif(只需 Pillow,本机已可用)。\n"
            "  SHSynth 不会把视频悄悄降级成一张静止图。") from exc
    try:
        with imageio.get_writer(p, fps=float(fps), format="FFMPEG") as w:
            for f in pil_frames:
                w.append_data(f[:, :, :3])
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(
            f"写 {ext} 失败({type(exc).__name__}: {exc})。"
            "多半是缺 ffmpeg 可执行文件:装 imageio-ffmpeg,或改用 .gif。") from exc
    _fill(format=fmt, fps_requested=float(fps), fps_effective=float(fps),
          delay_ms=None, bytes=os.path.getsize(p))
    return p


def save_figure(fig, path, dpi: int = 160) -> str:
    """保存图(扩展名决定格式:png/pdf/svg/jpg/tif)。"""
    p = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches=None, facecolor="white")
    return p


def figure_to_array(fig, dpi: int = 100) -> np.ndarray:
    """把图渲染成 ``(H, W, 3)`` uint8 数组(离屏冒烟/测试用)。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])
