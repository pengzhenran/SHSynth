# -*- coding: utf-8 -*-
"""
shsynth.gui.canvases
====================

嵌在 Qt 里的 matplotlib 画布。

**刻意只用 matplotlib** 画图,不用 Qt Charts / Qt Data Visualization —— 那两个是
GPL-only 的 Qt 模块,一旦用上,闭源分发就不可能了;matplotlib 是 BSD 风格许可,
没有这个问题。

每个画布只持有一张自己的 :class:`~matplotlib.figure.Figure`,绘图时把它交给
:mod:`shsynth.plotting` 的 ``make_*_figure(fig=...)``。这样**命令行保存**与
**界面显示**用的是同一段绘图代码、同一张图,不会出现"图上看得到、存下来不一样"。
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .. import plotting

__all__ = ["BaseCanvas", "MapCanvas", "SpectrumCanvas", "HistCanvas",
           "ReportCanvas", "SeriesCanvas", "VectorCanvas"]


class BaseCanvas(QWidget):
    """一张 matplotlib 图 + 标准导航工具条。"""

    def __init__(self, parent=None, figsize=(7.4, 4.4), dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi, layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

    # ------------------------------------------------------------------ 基本
    def clear(self):
        self.figure.clear()
        self.canvas.draw_idle()

    def refresh(self):
        self.canvas.draw_idle()

    def show_placeholder(self, text: str = "载入系数并点击『开始解算』后在此显示"):
        """画一条居中提示,而不是一张空的(且会报警告的)图。"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, text, ha="center", va="center", color="0.62",
                fontsize=12, transform=ax.transAxes)
        self.refresh()


class MapCanvas(BaseCanvas):
    """经纬度地图(网格 pcolormesh 或散点)。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.6, 4.4))
        self.show_placeholder("载入系数 → 选择位置 → 开始解算,结果显示在这里")

    def show_field(self, lat, lon, values, title: str = "", cmap: str = "RdBu_r",
                   cb_label: str = "", contour: bool = False, coast: bool = True,
                   symmetric: bool = True, point_size: float = 8.0,
                   focus: str = "auto", extent=None):
        """重画地图(整图重建,避免残留上一次的 colorbar)。

        ``focus='auto'`` 时数据只覆盖一块区域就自动缩放到该范围;
        ``focus='global'`` 强制全球视图;``extent`` 可直接给窗口
        ``(lon_min, lon_max, lat_min, lat_max)``。
        """
        plotting.make_map_figure(
            lat, lon, values, title=title, cmap=cmap, cb_label=cb_label,
            contour=contour, coast=coast, symmetric=symmetric,
            point_size=point_size, fig=self.figure, focus=focus, extent=extent)
        self.refresh()


class SpectrumCanvas(BaseCanvas):
    """逐阶振幅曲线。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.4, 4.4))
        self.show_placeholder("逐阶振幅会显示在这里")

    def show_curves(self, curves: dict, title: str = "逐阶振幅 (degree RMS)",
                    ylabel: str = "RMS", log: bool = True, styles=None):
        plotting.make_spectrum_figure(curves, title=title, ylabel=ylabel,
                                      log=log, styles=styles, fig=self.figure)
        self.refresh()


class HistCanvas(BaseCanvas):
    """数值分布直方图。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.4, 4.4))
        self.show_placeholder("数值分布会显示在这里")

    def show_values(self, values, title: str = "数值分布", xlabel: str = "值",
                    bins: int = 60):
        plotting.make_hist_figure(values, title=title, xlabel=xlabel, bins=bins,
                                  fig=self.figure)
        self.refresh()


class ReportCanvas(BaseCanvas):
    """四联报告图(地图 + 逐阶谱 + 直方图 + 摘要)。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(12.0, 7.0))
        self.show_placeholder("报告图会显示在这里")

    def show_report(self, lat, lon, values, coeffs=None, curves=None,
                    title: str = "", cmap: str = "RdBu_r", cb_label: str = "",
                    contour: bool = False, annotation=None, lat_vec=None,
                    lon_vec=None, focus: str = "auto", extent=None):
        plotting.make_report_figure(
            lat, lon, values, coeffs=coeffs, curves=curves, title=title,
            cmap=cmap, cb_label=cb_label, contour=contour,
            annotation=annotation, lat_vec=lat_vec, lon_vec=lon_vec,
            figsize=(12.0, 7.0), fig=self.figure, focus=focus, extent=extent)
        self.refresh()


class SeriesCanvas(BaseCanvas):
    """时间序列曲线(v2.0):横轴十进制年,可叠加趋势线/周年拟合。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7.8, 4.6))
        self.show_placeholder("载入带日期的系数序列后,这里显示点/区域的时间序列")

    def show_series(self, times, curves, title: str = "时间序列",
                    ylabel: str = "", trend=None, seasonal=None):
        plotting.make_timeseries_figure(
            times, curves, title=title, ylabel=ylabel, trend=trend,
            seasonal=seasonal, figsize=(7.8, 4.6), fig=self.figure)
        self.refresh()


class VectorCanvas(BaseCanvas):
    """水平形变矢量图(v2.0):大小打底 + 箭头叠加。"""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(8.8, 4.8))
        self.show_placeholder("水平形变:这里显示大小底图 + 位移箭头")

    def show_vectors(self, lat, lon, north, east, magnitude=None,
                     title: str = "水平形变", stride: int = 8,
                     cb_label: str = "m", focus: str = "auto"):
        plotting.make_vector_map_figure(
            lat, lon, north, east, magnitude=magnitude, title=title,
            stride=stride, cb_label=cb_label, focus=focus,
            figsize=(8.8, 4.8), fig=self.figure)
        self.refresh()
