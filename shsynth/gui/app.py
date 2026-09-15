# -*- coding: utf-8 -*-
"""
shsynth.gui.app
===============

PySide6 桌面界面:左边参数面板,右边地图 / 报告图 / 逐阶谱 / 数值分布 / 日志。

界面只负责**收集参数**与**显示结果**:参数打包成
:class:`shsynth.workflow.SynthRequest` 交给后台线程计算,回来后用
:mod:`shsynth.plotting` 在主线程画到画布自带的那张 Figure 上 —— 所以界面上看到的
图与"保存图片"存下来的图是同一张。
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

import numpy as np

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import (QAction, QColor, QDesktopServices, QFont,
                           QKeySequence, QPixmap)
from PySide6.QtWidgets import (
    QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QStackedWidget, QStatusBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import __version__, author, fieldio, plotting
from ..coeffio import COEFF_EXTENSIONS, coeff_info
from ..fieldio import GRID_EXTENSIONS, POINT_EXTENSIONS
from ..series import GRACE_MEAN_FROM, GRACE_MEAN_TO
from ..units import FIELD_UNIT_LABELS, FIELD_UNITS
from ..workflow import SynthRequest, plan_text
from .canvases import (HistCanvas, MapCanvas, ReportCanvas, SeriesCanvas,
                       SpectrumCanvas, VectorCanvas)
from .workers import (CoeffInfoWorker, SeriesReadWorker, SeriesSynthWorker,
                      SynthWorker, make_thread)

__all__ = ["MainWindow", "run_gui"]

_LAYOUTS = (("auto", "auto（自动识别,推荐）"), ("triangle", "triangle（[C; S] 堆叠）"),
            ("gmfcsv", "gmfcsv（n,m,C,S 逐行）"), ("gfc", "gfc（ICGEM/GFZ）"),
            ("npy", "npy（numpy 数组）"), ("npz", "npz（numpy 压缩包）"),
            ("matrix", "matrix（稠密方阵文本）"))

_GLOBAL_PRESETS = (("0.5°  (361×720)", 0.5), ("1°  (181×360)", 1.0),
                   ("2°  (91×180)", 2.0), ("2.5°  (73×144)", 2.5),
                   ("5°  (37×72)", 5.0))

_CMAPS = ("RdBu_r", "viridis", "BrBG", "Spectral_r", "coolwarm", "PiYG",
          "YlGnBu", "magma", "turbo", "Greys")

_FIGKIND_LABELS = (("report", "四联报告图（地图+谱+分布）"), ("map", "地图"),
                   ("spectrum", "逐阶谱"), ("hist", "数值分布"),
                   ("none", "不出图"))

#: 主窗口初始尺寸 / 左侧参数面板宽度(v2.0:面板字段多,窄了会显示不全)
#: 620 不是拍脑袋 —— 实测最宽的一张表单(「3. 综合选项」)sizeHint 就要 594px,
#: 给它留出余量;测试 test_panel_is_wide_enough 会盯着这条关系。
_WIN_W, _WIN_H = 1560, 960
_PANEL_W, _PANEL_MIN_W = 640, 620

_INPUT_MODES = (("global", "全球网格"), ("range", "范围网格"),
                ("gridfile", "借用网格文件"), ("points", "散点文件"),
                ("sphere", "球面散点"))


class MainWindow(QMainWindow):
    """SHSynth 主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"SHSynth {__version__} — 球谐系数解算")
        # 左侧参数面板字段多且路径很长,窗口窄了会显示不全(实测 1380 宽时
        # 参数区的行标签要被挤掉)。给足宽度,并把最小宽度也定住。
        self.resize(_WIN_W, _WIN_H)
        self.setMinimumSize(1120, 720)
        self.setAcceptDrops(True)

        self._thread: Optional[QThread] = None
        self._worker = None
        self._info_thread: Optional[QThread] = None
        self._info_worker = None
        # v2.0:批量序列 / 读序列 的后台线程
        self._series_thread: Optional[QThread] = None
        self._series_worker = None
        self._series_read_thread: Optional[QThread] = None
        self._series_read_worker = None
        self._last_result = None
        self._coeffs = None
        self._last_progress_msg = ""

        self._build_ui()
        self._wire()
        # 取消"滚轮滑过输入框就改数值"(见 _tame_wheel);必须在 _build_ui 之后
        self._wheel_tamed = self._tame_wheel(self)
        self._log(f"SHSynth {__version__} —— 球谐系数解算(综合)")
        self._log("输入:球谐系数 + 网格/散点位置   输出:网格/散点值 + 图")
        self._log("系数支持 SHKit 的全部输出格式:triangle / gmfcsv / gfc / npy / npz"
                  "(含 .gz、中文表头、文件头元数据)。")
        self._log("提示:点击【系数文件 …】右侧『浏览…』或把文件直接拖进窗口即可。")

    # =====================================================================
    # 界面搭建
    # =====================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---------------------------------------------------------- 左:参数
        self.panel = self._build_panel()
        splitter.addWidget(self.panel)

        # ---------------------------------------------------------- 右:结果
        self.tabs = QTabWidget()
        self.map_canvas = MapCanvas()
        self.report_canvas = ReportCanvas()
        self.spectrum_canvas = SpectrumCanvas()
        self.hist_canvas = HistCanvas()
        # ---- v2.0 新页签 -------------------------------------------------
        self.series_canvas = SeriesCanvas()
        self.vector_canvas = VectorCanvas()
        # 动画页:一块地图画布 + 播放控制条
        self.anim_canvas = MapCanvas()
        self._anim_res = None
        self._anim_values = None
        self._anim_idx = []
        self._anim_pos = 0
        self._anim_vmin = None
        self._anim_vmax = None
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_next)
        self._anim_export_thread = None
        self._anim_export_worker = None
        anim_page = QWidget()
        av = QVBoxLayout(anim_page)
        av.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.anim_play_btn = QPushButton("▶ 播放")
        self.anim_play_btn.setCheckable(True)
        self.anim_play_btn.setEnabled(False)
        self.anim_play_btn.setToolTip("在当前页签里播放场序列(GIF/视频请用『导出 GIF』)")
        self.anim_prev_btn = QPushButton("◀")
        self.anim_next_btn = QPushButton("▶")
        for b in (self.anim_prev_btn, self.anim_next_btn):
            b.setEnabled(False)
            b.setMaximumWidth(46)
        self.anim_slider = QSlider(Qt.Horizontal)
        self.anim_slider.setEnabled(False)
        self.anim_slider.setToolTip("拖动定位到某一历元(与『逐历元诊断』页点击行等效)")
        self.anim_fps = QDoubleSpinBox()
        self.anim_fps.setRange(0.2, 60.0)
        self.anim_fps.setSingleStep(0.5)
        self.anim_fps.setValue(5.0)
        self.anim_fps.setSuffix(" fps")
        self.anim_fps.setMaximumWidth(96)
        self.chk_anim_loop = QCheckBox("循环")
        self.chk_anim_loop.setChecked(True)
        self.anim_label = QLabel("—")
        self.anim_label.setMinimumWidth(96)
        self.anim_gif_btn = QPushButton("导出 GIF…")
        self.anim_gif_btn.setEnabled(False)
        self.anim_gif_btn.setToolTip(
            "把整条序列写成动图(.gif 只需 Pillow)。\n"
            "配色范围取**全序列**的稳健对称范围 —— 不逐帧定标,否则动画会随\n"
            "每历元的极值闪烁,看到的『变化』其实是配色在变。")
        bar.addWidget(self.anim_play_btn)
        bar.addWidget(self.anim_prev_btn)
        bar.addWidget(self.anim_slider, 1)
        bar.addWidget(self.anim_next_btn)
        bar.addWidget(QLabel("帧率"))
        bar.addWidget(self.anim_fps)
        bar.addWidget(self.chk_anim_loop)
        bar.addWidget(self.anim_label)
        bar.addWidget(self.anim_gif_btn)
        av.addLayout(bar)
        av.addWidget(self.anim_canvas, 1)
        self.anim_play_btn.toggled.connect(self._on_anim_play)
        self.anim_prev_btn.clicked.connect(lambda: self._anim_step(-1))
        self.anim_next_btn.clicked.connect(lambda: self._anim_step(+1))
        self.anim_slider.valueChanged.connect(self._on_anim_slider)
        self.anim_fps.valueChanged.connect(self._on_anim_fps)
        self.anim_gif_btn.clicked.connect(self.export_animation)
        self.diag_table = QTableWidget(0, 0)
        self.diag_table.setAlternatingRowColors(True)
        self.diag_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.diag_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.diag_table.horizontalHeader().setStretchLastSection(True)
        self.diag_table.setToolTip("逐历元诊断:点某一行可跳到该历元的地图")
        self.diag_table.cellClicked.connect(self._on_diag_row)
        self._series_result = None
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.setMaximumBlockCount(20000)
        self.tabs.addTab(self.map_canvas, "地图")
        self.tabs.addTab(self.report_canvas, "报告图")
        self.tabs.addTab(self.spectrum_canvas, "逐阶谱")
        self.tabs.addTab(self.hist_canvas, "数值分布")
        self.tabs.addTab(self.series_canvas, "时间序列")
        self.tabs.addTab(self.vector_canvas, "水平形变")
        self.tabs.addTab(anim_page, "动画")
        self.tabs.addTab(self.diag_table, "逐历元诊断")
        self.tabs.addTab(self.log_view, "日志 / 系数信息")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_PANEL_W, _WIN_W - _PANEL_W])
        # 参数面板不许被挤窄(拖分隔条也只能到 _PANEL_MIN_W)
        self.panel.setMinimumWidth(_PANEL_MIN_W)

        # ---------------------------------------------------------- 状态栏
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        sb.addPermanentWidget(self.progress)
        self.status_label = QLabel("就绪")
        sb.addWidget(self.status_label)

        # 菜单:一键加载 SHKit 样例
        m = self.menuBar().addMenu("文件(&F)")
        act = QAction("打开系数文件…", self)
        act.triggered.connect(self.browse_coeffs)
        m.addAction(act)
        m.addSeparator()
        quit_act = QAction("退出", self)
        quit_act.triggered.connect(self.close)
        m.addAction(quit_act)

        hm = self.menuBar().addMenu("帮助(&H)")
        guide = QAction("使用说明（F1）", self)
        guide.setShortcut(QKeySequence("F1"))
        guide.triggered.connect(self.show_guide)
        hm.addAction(guide)
        hm.addSeparator()
        about = QAction("关于 / 作者信息", self)
        about.triggered.connect(self._about)
        hm.addAction(about)
        fmt = QAction("支持的格式…", self)
        fmt.triggered.connect(self._show_formats)
        hm.addAction(fmt)

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(2, 2, 2, 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 表单里的路径很长,标签+输入框容易挤在一起;把标签换行策略定死为不换行,
        # 并把水平滚动条按需打开 —— 面板宽了以后正常不会出现。
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.panel_scroll = scroll
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setSpacing(8)

        # ---------------- 1. 系数文件 ----------------
        g1 = QGroupBox("1. 球谐系数")
        f1 = QFormLayout(g1)
        row = QHBoxLayout()
        self.coeffs_edit = QLineEdit()
        self.coeffs_edit.setPlaceholderText("例:…\\SHKit\\shkit_coeffs.sh（可拖文件进来）")
        btn = QPushButton("浏览…")
        btn.clicked.connect(self.browse_coeffs)
        row.addWidget(self.coeffs_edit, 1)
        row.addWidget(btn)
        f1.addRow("系数文件", row)

        self.layout_combo = QComboBox()
        for key, label in _LAYOUTS:
            self.layout_combo.addItem(label, key)
        f1.addRow("读取布局", self.layout_combo)

        self.coeff_nmax_spin = QSpinBox()
        self.coeff_nmax_spin.setRange(0, 2190)
        self.coeff_nmax_spin.setSpecialValueText("自动（文件自身）")
        self.coeff_nmax_spin.setToolTip("读取时就截断,0 表示用文件里的最高阶")
        f1.addRow("读取阶数", self.coeff_nmax_spin)

        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 9999)
        self.time_spin.setEnabled(False)
        self.time_spin.setToolTip("系数文件含多个时次时选择用第几个")
        f1.addRow("时次", self.time_spin)

        # ---- v2.0:时间轴控件 ----------------------------------------
        self.epoch_combo = QComboBox()
        self.epoch_combo.addItem("只用当前时次(单历元,同 v1.0)", "current")
        self.epoch_combo.addItem("整条序列(批量综合)", "all")
        self.epoch_combo.addItem("按起止时次批量", "range")
        self.epoch_combo.setEnabled(False)
        self.epoch_combo.setToolTip(
            "多时次文件怎么用:\n"
            "  只用当前时次 —— 与 v1.0 行为完全一致;\n"
            "  整条序列 —— 批量综合全部历元(203 个历元在 1° 全球上约 0.3 秒);\n"
            "  按起止时次 —— 只算第 a..b 个历元(1 基)")
        f1.addRow("历元范围", self.epoch_combo)

        self.epoch_from = QSpinBox()
        self.epoch_to = QSpinBox()
        for s in (self.epoch_from, self.epoch_to):
            s.setRange(1, 1)
            s.setEnabled(False)
        self.epoch_from.setPrefix("第 ")
        self.epoch_from.setSuffix(" 个")
        self.epoch_to.setPrefix("到第 ")
        self.epoch_to.setSuffix(" 个")
        f1.addRow("起止", self._pair(self.epoch_from, self.epoch_to))
        self.epoch_combo.currentIndexChanged.connect(self._on_epoch_mode)

        self.time_info_label = QLabel("(载入系数后显示时间轴摘要)")
        self.time_info_label.setWordWrap(True)
        self.time_info_label.setStyleSheet("color:#555;")
        f1.addRow(self.time_info_label)

        self.series_read_btn = QPushButton("读序列(目录/清单 → 带日期的序列)")
        self.series_read_btn.setToolTip(
            "选一个**目录**(里面是 GSM-*.gfc)或 .txt 清单,一次读成\n"
            "带真实日期的系数序列;日期取自 gfc 头,头里没有再退回文件名。\n"
            "读进来之后『历元范围』就能选整条序列做批量综合。")
        f1.addRow(self.series_read_btn)

        self.info_btn = QPushButton("读取系数信息")
        f1.addRow(self.info_btn)
        v.addWidget(g1)

        # ---------------- 2. 位置 ----------------
        g2 = QGroupBox("2. 求值位置")
        v2 = QVBoxLayout(g2)
        self.mode_radios = {}
        mrow = QHBoxLayout()
        for key, label in _INPUT_MODES:
            rb = QRadioButton(label)
            rb.setProperty("mode", key)
            mrow.addWidget(rb)
            self.mode_radios[key] = rb
        self.mode_radios["global"].setChecked(True)
        v2.addLayout(mrow)

        self.mode_stack = QStackedWidget()

        # (a) 全球网格
        w_global = QWidget()
        fg = QFormLayout(w_global)
        self.preset_combo = QComboBox()
        for label, step in _GLOBAL_PRESETS:
            self.preset_combo.addItem(label, step)
        self.preset_combo.setCurrentIndex(1)
        fg.addRow("预设步长", self.preset_combo)
        self.global_step = QDoubleSpinBox()
        self.global_step.setRange(0.05, 45.0)
        self.global_step.setDecimals(3)
        self.global_step.setValue(1.0)
        self.global_step.setSuffix(" °")
        fg.addRow("步长", self.global_step)
        self.mode_stack.addWidget(w_global)

        # (b) 范围网格
        w_range = QWidget()
        fr = QFormLayout(w_range)
        self.lat_min = self._dspin(-90.0); self.lat_max = self._dspin(90.0)
        self.lon_min = self._dspin(0.0); self.lon_max = self._dspin(360.0)
        self.lat_step = self._dspin(1.0); self.lon_step = self._dspin(1.0)
        fr.addRow("纬度 起/止", self._pair(self.lat_min, self.lat_max))
        fr.addRow("经度 起/止", self._pair(self.lon_min, self.lon_max))
        fr.addRow("步长 纬/经", self._pair(self.lat_step, self.lon_step))
        self.mode_stack.addWidget(w_range)

        # (c) 网格文件
        w_gf = QWidget()
        fgf = QFormLayout(w_gf)
        self.gridfile_edit = QLineEdit()
        b = QPushButton("浏览…")
        b.clicked.connect(lambda: self._browse_into(
            self.gridfile_edit, "选择网格文件",
            "网格文件 (*.nc *.nc4 *.cdf *.grd *.npy *.csv *.txt *.dat);;所有文件 (*)"))
        fgf.addRow("网格文件", self._pair_edit(self.gridfile_edit, b))
        self.gridvar_edit = QLineEdit()
        self.gridvar_edit.setPlaceholderText("netCDF 变量名（可留空）")
        fgf.addRow("变量名", self.gridvar_edit)
        self.mode_stack.addWidget(w_gf)

        # (d) 散点文件
        w_pf = QWidget()
        fpf = QFormLayout(w_pf)
        self.pointsfile_edit = QLineEdit()
        b2 = QPushButton("浏览…")
        b2.clicked.connect(lambda: self._browse_into(
            self.pointsfile_edit, "选择散点文件",
            "散点文件 (*.csv *.txt *.dat *.tsv *.npy *.xlsx);;所有文件 (*)"))
        fpf.addRow("散点文件", self._pair_edit(self.pointsfile_edit, b2))
        # 纬度列 / 经度列:选了文件后自动读表头填充成下拉框
        self.latcol_combo = QComboBox()
        self.loncol_combo = QComboBox()
        for cb, tip in ((self.latcol_combo, "纬度列"),
                        (self.loncol_combo, "经度列")):
            cb.addItem("自动(留空)", None)
            cb.setToolTip(f"{tip}:默认『自动(留空)』——软件按表头或通行约定"
                          "自己认;认不准时可以在这里手动指定")
        self.latcol_combo.currentIndexChanged.connect(self._on_point_cols_changed)
        self.loncol_combo.currentIndexChanged.connect(self._on_point_cols_changed)
        fpf.addRow("纬度列", self.latcol_combo)
        fpf.addRow("经度列", self.loncol_combo)
        self.points_col_hint = QLabel("")
        self.points_col_hint.setWordWrap(True)
        self.points_col_hint.setStyleSheet("color:#666; font-size:11px;")
        fpf.addRow(self.points_col_hint)
        self.mode_stack.addWidget(w_pf)

        # (e) 球面散点
        w_sp = QWidget()
        fsp = QFormLayout(w_sp)
        self.sphere_n = QSpinBox()
        self.sphere_n.setRange(100, 5_000_000)
        self.sphere_n.setSingleStep(1000)
        self.sphere_n.setValue(20000)
        fsp.addRow("点数 N", self.sphere_n)
        self.mode_stack.addWidget(w_sp)

        v2.addWidget(self.mode_stack)
        v.addWidget(g2)

        # ---------------- 3. 综合选项 ----------------
        g3 = QGroupBox("3. 综合选项")
        f3 = QFormLayout(g3)
        self.truncate_spin = QSpinBox()
        self.truncate_spin.setRange(0, 2190)
        self.truncate_spin.setSpecialValueText("不截断（用系数自身阶数）")
        f3.addRow("截断阶数", self.truncate_spin)

        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.0, 20000.0)
        self.gauss_spin.setDecimals(1)
        self.gauss_spin.setSingleStep(50.0)
        self.gauss_spin.setSuffix(" km")
        self.gauss_spin.setToolTip("在综合时逐阶乘 W_n,不改动系数文件本身")
        f3.addRow("高斯平滑", self.gauss_spin)

        self.gauss_method = QComboBox()
        self.gauss_method.addItem("glq（Gauss–Legendre,稳定,推荐）", "glq")
        self.gauss_method.addItem("frc（经典递推,快,高阶不稳）", "frc")
        f3.addRow("平滑算法", self.gauss_method)

        self.unit_combo = QComboBox()
        self.unit_combo.addItem("不换算（与系数声明的物理量相同）", None)
        for key in FIELD_UNITS:
            if key in ("unknown", "geopotential"):
                continue
            self.unit_combo.addItem(f"{FIELD_UNIT_LABELS[key]}  [{key}]", key)
        self.unit_combo.setToolTip(
            "换算因子是**逐阶**的(不是常数):位系数→EWH 要乘 A_n,"
            "n=0 与 n=6 之间就差 14 倍")
        f3.addRow("输出物理量", self.unit_combo)

        # ---- v2.0:分量 + 平滑算法 ------------------------------------
        self.component_combo = QComboBox()
        for key, label in (("scalar", "标量场(默认)"),
                           ("north", "水平形变 · 北分量 u_N"),
                           ("east", "水平形变 · 东分量 u_E"),
                           ("horizontal", "水平形变 · 北+东(矢量)")):
            self.component_combo.addItem(label, key)
        self.component_combo.setToolTip(
            "水平形变的逐阶因子是 R·l′ₙ/(1+k′ₙ)(与 EWH/geoid 的不同),\n"
            "所以选它时『输出物理量』会被置为水平形变。\n"
            "矢量模式一次算完北+东,并显示大小底图 + 箭头。")
        f3.addRow("输出分量", self.component_combo)

        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(1000, 5_000_000)
        self.chunk_spin.setSingleStep(50000)
        self.chunk_spin.setValue(200000)
        self.chunk_spin.setToolTip("分块点数,内存不够时调小")
        f3.addRow("分块点数", self.chunk_spin)
        v.addWidget(g3)

        # ---------------- 4. 输出 ----------------
        g4 = QGroupBox("4. 输出结果")
        f4 = QFormLayout(g4)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("留空则不写文件。扩展名决定格式")
        b3 = QPushButton("另存为…")
        b3.clicked.connect(self._browse_out)
        f4.addRow("结果文件", self._pair_edit(self.out_edit, b3))
        self.outvar_edit = QLineEdit("value")
        f4.addRow("变量名", self.outvar_edit)
        self.outunits_edit = QLineEdit()
        self.outunits_edit.setPlaceholderText("仅作标注,例:mm / m / kg/m²")
        f4.addRow("单位说明", self.outunits_edit)

        self.outcoeffs_edit = QLineEdit()
        self.outcoeffs_edit.setPlaceholderText("可选:把(截断/平滑后的)系数另存一份")
        self.outcoeffs_btn = QPushButton("另存为…")
        self.outcoeffs_btn.clicked.connect(lambda: self._browse_save_into(
            self.outcoeffs_edit, "保存系数文件",
            "系数文件 (*.sh *.txt *.csv *.gfc *.npy *.npz);;所有文件 (*)",
            "coeffs_out.sh"))
        f4.addRow("导出系数", self._pair_edit(self.outcoeffs_edit,
                                          self.outcoeffs_btn))
        self.outcoeffs_layout = QComboBox()
        for key, label in _LAYOUTS[:-1]:
            self.outcoeffs_layout.addItem(label, key)
        f4.addRow("系数布局", self.outcoeffs_layout)
        v.addWidget(g4)

        # ---------------- 5. 出图 ----------------
        g5 = QGroupBox("5. 绘图")
        f5 = QFormLayout(g5)
        self.figkind_combo = QComboBox()
        for key, label in _FIGKIND_LABELS:
            self.figkind_combo.addItem(label, key)
        f5.addRow("图类型", self.figkind_combo)

        self.figfile_edit = QLineEdit()
        self.figfile_edit.setPlaceholderText("留空则只在界面显示(可点『保存当前图』)")
        self.figfile_btn = QPushButton("另存为…")
        self.figfile_btn.clicked.connect(lambda: self._browse_save_into(
            self.figfile_edit, "保存图片",
            "图片 (*.png *.pdf *.svg *.jpg);;所有文件 (*)", "map.png"))
        f5.addRow("图片文件", self._pair_edit(self.figfile_edit,
                                          self.figfile_btn))

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(60, 600)
        self.dpi_spin.setValue(160)
        f5.addRow("DPI", self.dpi_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(_CMAPS)
        self.cmap_combo.setToolTip("正负异常用发散色系(如 RdBu_r)更直观")
        f5.addRow("配色", self.cmap_combo)

        self.chk_contour = QCheckBox("叠加等值线")
        self.chk_coast = QCheckBox("画海岸线")
        self.chk_coast.setChecked(True)
        self.chk_coast.setToolTip("只画海岸线;**不画国界**(避免主权划线争议)")
        self.chk_symmetric = QCheckBox("配色关于 0 对称")
        self.chk_symmetric.setChecked(True)
        self.chk_focus = QCheckBox("自动聚焦到结果范围")
        self.chk_focus.setChecked(True)
        self.chk_focus.setToolTip(
            "区域结果在全球视角下太小:勾选后自动把地图缩放到数据所在范围;\n"
            "覆盖接近全球时仍画全球。也可以用下面的按钮随时切换。")
        # 运行结束后的落点:区域结果在大地图上才看得清,所以默认自动打开地图页
        # (以前要手动点「地图」页签 —— 四联图里那张小地图不够用)
        self.chk_autoshow = QCheckBox("运行后自动打开地图(聚焦结果)")
        self.chk_autoshow.setChecked(True)
        self.chk_autoshow.setToolTip(
            "解算一结束就切到「地图」页签 —— 该页已经按『自动聚焦到结果范围』\n"
            "缩放好了,不用手动点页签、也不用再手动放大。\n"
            "取消勾选则停在「图类型」里选的那张图(图仍然会画、会存盘)。")
        self.chk_raise = QCheckBox("运行结束后窗口置前")
        self.chk_raise.setChecked(False)
        self.chk_raise.setToolTip(
            "解算完把窗口弹到最前面(适合跑长任务时切去干别的)。默认关闭,"
            "免得抢走你正在用的窗口。")
        for c in (self.chk_contour, self.chk_coast, self.chk_symmetric,
                  self.chk_focus, self.chk_autoshow, self.chk_raise):
            f5.addRow(c)
        v.addWidget(g5)

        # ---------------- 6. 批量序列(v2.0) ----------------
        g6 = QGroupBox("6. 批量序列(把整条时间序列一次算完)")
        f6 = QFormLayout(g6)
        self.series_out_edit = QLineEdit()
        self.series_out_edit.setPlaceholderText(
            "留空则不写盘(只在界面显示);.nc 会带**真实时间坐标**")
        self.series_out_btn = QPushButton("另存为…")
        self.series_out_btn.clicked.connect(
            lambda: self._browse_save_into(
                self.series_out_edit, "保存场序列文件",
                "netCDF (*.nc);;所有文件 (*)", "shsynth_series.nc"))
        f6.addRow("场序列输出", self._pair_edit(self.series_out_edit,
                                            self.series_out_btn))
        self.series_diag_edit = QLineEdit()
        self.series_diag_edit.setPlaceholderText("逐历元诊断表(csv,可留空)")
        self.series_diag_btn = QPushButton("另存为…")
        self.series_diag_btn.clicked.connect(
            lambda: self._browse_save_into(
                self.series_diag_edit, "保存逐历元诊断表",
                "CSV (*.csv);;所有文件 (*)", "shsynth_series_diag.csv"))
        f6.addRow("诊断表输出", self._pair_edit(self.series_diag_edit,
                                            self.series_diag_btn))
        # 去均值口径:GRACE 惯例 / 全时段 / 自定义(v2.0)
        # 为什么不是个复选框:去均值决定了整个异常场的**静态基准**,
        # GRACE 惯例(2004-2010)与全时段算出来的参考场不是一回事,必须让用户选。
        self.mean_combo = QComboBox()
        self.mean_combo.addItem("不去均值(原始场,含静态场)", None)
        self.mean_combo.addItem("GRACE 惯例(2004-01-01 .. 2010-12-31 平均)", "grace")
        self.mean_combo.addItem("全时段平均", "all")
        self.mean_combo.addItem("自定义时段…", "custom")
        self.mean_combo.setCurrentIndex(0)
        self.mean_combo.setToolTip(
            "GRACE 的 GSM 文件装的是**完整静态重力场**(C₀₀=1、C₂₀≈−4.8e-4),\n"
            "不做去均值换成 EWH 会得到上万米的常量场而不是水(实测逐历元 RMS 53 828 m)。\n\n"
            "• GRACE 惯例:减 2004-01-01..2010-12-31 的平均 —— 与你自己的 legacy\n"
            "  脚本(3_processed/read_GRACE_SH_preprocess_postprocess_SH60.m)同一口径。\n"
            "  ⚠️ 那个脚本里的 dur_mascon=90:150 与它自己的注释对不上(实测落到\n"
            "  2009-12..2016-01);所以要复刻 19:90(2004-2009)请用『自定义时段』。\n"
            "• 全时段:减所有历元的平均。\n"
            "• 自定义:自己填起止日期。")
        f6.addRow("去均值", self.mean_combo)

        mean_row = QWidget()
        mh = QHBoxLayout(mean_row)
        mh.setContentsMargins(0, 0, 0, 0)
        self.mean_from = QLineEdit()
        self.mean_from.setPlaceholderText("起 2004-01-01")
        self.mean_to = QLineEdit()
        self.mean_to.setPlaceholderText("止 2010-12-31")
        mh.addWidget(QLabel("起"))
        mh.addWidget(self.mean_from, 1)
        mh.addWidget(QLabel("止"))
        mh.addWidget(self.mean_to, 1)
        for w in (self.mean_from, self.mean_to):
            w.setToolTip("自定义去均值窗口,写法很宽松:\n"
                         "  2004-01-01 / 2004-01 / 2004 / 2004.5 都认\n"
                         "只填一边也行(起点补 01-01,终点补 **12-31**,\n"
                         "所以『止 = 2009』就是到 2009 年底,不会少算 11 个月)")
        f6.addRow("自定义时段", mean_row)
        self.mean_note = QLabel("")
        self.mean_note.setWordWrap(True)
        self.mean_note.setStyleSheet("color:#555;")
        f6.addRow(self.mean_note)
        self.mean_combo.currentIndexChanged.connect(self._on_mean_mode)
        for w in (self.mean_from, self.mean_to):
            w.editingFinished.connect(self._update_mean_note)
        self._on_mean_mode()

        self.series_btn = QPushButton("综合整条序列")
        self.series_btn.setMinimumHeight(32)
        self.series_btn.setToolTip(
            "把系数序列一次综合成场序列:规则整圈经度网格会走 FFT 快路径\n"
            "(实测 203 个历元 × 1° 全球约 0.3 秒);散点/区域网格走直接法。")
        f6.addRow(self.series_btn)
        v.addWidget(g6)

        # ---------------- 7. 按钮 ----------------
        self.run_btn = QPushButton("开始解算")
        self.run_btn.setMinimumHeight(38)
        f = self.run_btn.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.run_btn.setFont(f)
        self.run_btn.setDefault(True)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.savefig_btn = QPushButton("保存当前图…")
        self.plan_btn = QPushButton("预览参数")
        self.focus_btn = QPushButton("聚焦结果范围 / 全球视图 切换")
        self.focus_btn.setToolTip("一键在『聚焦到结果范围』与『全球视图』之间切换"
                                  "(不用手动框选放大)")

        hb = QHBoxLayout()
        hb.addWidget(self.run_btn, 2)
        hb.addWidget(self.stop_btn, 1)
        v.addLayout(hb)
        hb2 = QHBoxLayout()
        hb2.addWidget(self.savefig_btn, 1)
        hb2.addWidget(self.plan_btn, 1)
        v.addLayout(hb2)
        v.addWidget(self.focus_btn)
        v.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return panel

    # ---------------------------------------------------------------- 小工具
    def _tame_wheel(self, root: QWidget) -> int:
        """取消"鼠标滑过就把数值改掉"的行为。

        Qt 里 ``QAbstractSpinBox`` / ``QComboBox`` 默认焦点策略是
        ``Qt.WheelFocus`` —— 鼠标停在上面滚一下就改数值。在左侧这个**可滚动**
        的参数面板里这非常难受:想滚面板,滚轮一穿过某个输入框,数值就被悄悄改了
        (而且很难发现)。

        两道保险:

        1. 焦点策略改成 ``Qt.StrongFocus``(= TabFocus|ClickFocus,**不含**
           WheelFocus)。Qt 的事件分发器在派发滚轮时会沿父链找"接受滚轮"的
           控件,于是滚轮落在外层 ``QScrollArea`` 上 —— 面板正常滚,数值不动。
        2. 装一个事件过滤器兜底(见 :meth:`eventFilter`)。因为第 1 条只管
           **分发器**那条路;如果有谁把滚轮事件**直接**送到控件上
           (父控件转发、触控板手势、样式差异),``QAbstractSpinBox`` 照样会
           改数值。过滤器把这种情况也拦住,并把滚轮**转给外层的滚动区**,
           所以"滚不动面板"这种副作用也不会出现。

        Returns
        -------
        int
            被处理的控件个数(冒烟测试拿它断言"确实都扫到了")。
        """
        n = 0
        for w in root.findChildren(QWidget):
            if isinstance(w, (QAbstractSpinBox, QComboBox)):
                w.setFocusPolicy(Qt.StrongFocus)
                w.installEventFilter(self)
                n += 1
        return n

    def eventFilter(self, obj, event):                   # noqa: N802
        """拦住"滚轮改数值",并把滚轮转交给外层的滚动区。

        只在控件**没有焦点**时拦 —— 点进去(拿到焦点)之后滚轮照旧可用,
        这是"我确实要调这个值"的明确意图。
        """
        if (event.type() == QEvent.Wheel
                and isinstance(obj, (QAbstractSpinBox, QComboBox))
                and not obj.hasFocus()):
            area = obj.parentWidget()
            while area is not None and not isinstance(area, QScrollArea):
                area = area.parentWidget()
            if area is not None:
                QApplication.sendEvent(area.viewport(), event)
            return True                                  # 控件自己不处理
        return super().eventFilter(obj, event)

    @staticmethod
    def _dspin(value: float) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(-360.0, 360.0)
        s.setDecimals(4)
        s.setValue(value)
        s.setSuffix(" °")
        return s

    @staticmethod
    def _pair(a: QWidget, b: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    @staticmethod
    def _pair_edit(edit: QLineEdit, btn: QPushButton) -> QWidget:
        return MainWindow._pair(edit, btn)

    # =====================================================================
    # 事件连接
    # =====================================================================
    def _wire(self):
        self.run_btn.clicked.connect(self.start_run)
        self.stop_btn.clicked.connect(self.stop_run)
        self.savefig_btn.clicked.connect(self.save_current_figure)
        self.plan_btn.clicked.connect(self.show_plan)
        self.info_btn.clicked.connect(self.load_coeffs_info)
        self.focus_btn.clicked.connect(self.toggle_focus)
        self.series_btn.clicked.connect(self.start_series_run)
        self.series_read_btn.clicked.connect(self.browse_series_read)
        self.component_combo.currentIndexChanged.connect(self._on_component)
        self.preset_combo.currentIndexChanged.connect(
            lambda i: self.global_step.setValue(self.preset_combo.itemData(i)))
        for key, rb in self.mode_radios.items():
            rb.toggled.connect(self._mode_changed)
        self.coeffs_edit.textChanged.connect(lambda _t: self._on_coeffs_path_changed())
        self.pointsfile_edit.textChanged.connect(
            lambda _t: self.refresh_points_columns())
        self.gridfile_edit.textChanged.connect(
            lambda _t: self.refresh_grid_hint())

    def refresh_points_columns(self):
        """读一眼散点文件的列,把『纬度列 / 经度列』下拉框填好。

        * 有表头 → 列名(自动认出来的那一项会被选上,但仍保留『自动(留空)』);
        * 无表头 → 「第1列 / 第2列 / …」;
        * 什么都不选(=『自动(留空)』)→ 交给软件按表头或通行约定自己认。
        """
        path = self.pointsfile_edit.text().strip()
        for cb in (self.latcol_combo, self.loncol_combo):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("自动(留空)", None)
            cb.blockSignals(False)
        self.points_col_hint.setText("")
        if not path or not os.path.exists(path):
            return
        try:
            info = fieldio.peek_columns(path)
        except Exception as exc:                          # noqa: BLE001
            self.points_col_hint.setText(f"读列失败:{type(exc).__name__}: "
                                         f"{str(exc)[:80]}")
            return
        cols = info.get("columns") or []
        for i, name in enumerate(cols):
            label = f"{name}(第 {i + 1} 列)" if info.get("has_header") \
                else str(name)
            self.latcol_combo.addItem(label, i)
            self.loncol_combo.addItem(label, i)
        # 自动认出来的那一列先选中,方便用户直接看到软件的判断;
        # 想回到"完全自动"就选第一项『自动(留空)』。
        if info.get("suggested_lat") is not None:
            self.latcol_combo.setCurrentIndex(int(info["suggested_lat"]) + 1)
        if info.get("suggested_lon") is not None:
            self.loncol_combo.setCurrentIndex(int(info["suggested_lon"]) + 1)
        bits = [f"{len(cols)} 列",
                "有表头" if info.get("has_header") else "无表头(按序号)"]
        if info.get("n_rows"):
            bits.append(f"{info['n_rows']:,} 行")
        if info.get("error"):
            bits.append(f"错:{info['error']}")
        hint = "、".join(bits)
        hint += ("　→ 已自动选中判断出的列;选『自动(留空)』可恢复全自动"
                 if info.get("has_header") else "")
        if info.get("n_columns") and info["n_columns"] < 2:
            hint += "　⚠️ 少于两列,无法作为散点位置"
        self.points_col_hint.setText(hint)

    def refresh_grid_hint(self):
        """选择网格文件后给一行提示(格式/格点数/是否能当经纬度用)。"""
        path = self.gridfile_edit.text().strip()
        if not path or not os.path.exists(path):
            return
        try:
            la, lo, g, meta = fieldio.read_grid(path)
            self._log(f"网格文件 {os.path.basename(path)}:"
                      f"{la.size}×{lo.size}  {meta.get('format')}"
                      f"{'  变量=' + str(meta.get('variable')) if meta.get('variable') else ''}")
            for w in meta.get("warnings", []):
                self._log(f"  警告: {w}")
        except Exception as exc:                          # noqa: BLE001
            self._log(f"网格文件 {os.path.basename(path)} 读不了:"
                      f"{type(exc).__name__}: {str(exc)[:300]}")

    def _on_point_cols_changed(self, _idx=0):
        for cb in (self.latcol_combo, self.loncol_combo):
            cb.setToolTip(f"当前:{cb.currentText()}")

    def _mode_changed(self, _checked=False):
        order = [k for k, _ in _INPUT_MODES]
        idx = next((i for i, k in enumerate(order)
                    if self.mode_radios[k].isChecked()), 0)
        self.mode_stack.setCurrentIndex(idx)

    def _on_coeffs_path_changed(self):
        # 路径一变,之前读进来的系数就作废了(单历元解算要看这个框)。
        # ⚠️ 但**切到批量模式时会主动清空这个框**(见 _clear_single_run_paths),
        # 那一刻 self._coeffs 装的是**批量序列**,必须留住 —— 否则『综合整条序列』
        # 会以为"还没读系数"而直接返回。所以清除期间设 _keep_coeffs 挡住这次作废。
        if getattr(self, "_keep_coeffs", False):
            return
        self.time_spin.setEnabled(False)
        self._coeffs = None

    def current_mode(self) -> str:
        return next((k for k, rb in self.mode_radios.items() if rb.isChecked()),
                    "global")

    # =====================================================================
    # 参数收集
    # =====================================================================
    def collect_spec(self) -> SynthRequest:
        mode = self.current_mode()
        spec = SynthRequest(
            coeffs_path=self.coeffs_edit.text().strip(),
            coeffs_layout=self.layout_combo.currentData(),
            coeffs_nmax=(self.coeff_nmax_spin.value()
                         if self.coeff_nmax_spin.value() > 0 else None),
            time=(self.time_spin.value() if self.time_spin.isEnabled() else None),
            truncate_nmax=(self.truncate_spin.value()
                           if self.truncate_spin.value() > 0 else None),
            gaussian_km=float(self.gauss_spin.value()),
            gaussian_method=self.gauss_method.currentData(),
            target_unit=self.unit_combo.currentData(),
            chunk=int(self.chunk_spin.value()),
            out_path=self.out_edit.text().strip(),
            out_var=self.outvar_edit.text().strip() or "value",
            out_units=self.outunits_edit.text().strip(),
            out_coeffs_path=self.outcoeffs_edit.text().strip(),
            out_coeffs_layout=self.outcoeffs_layout.currentData(),
            figure_kind=self.figkind_combo.currentData(),
            figure_path=self.figfile_edit.text().strip(),
            figure_dpi=int(self.dpi_spin.value()),
            cmap=self.cmap_combo.currentText(),
            contour=self.chk_contour.isChecked(),
            coast=self.chk_coast.isChecked(),
            symmetric=self.chk_symmetric.isChecked(),
        )
        if mode == "global":
            spec.mode = "grid"
            spec.grid_source = "global"
            spec.lat_step = spec.lon_step = float(self.global_step.value())
        elif mode == "range":
            spec.mode = "grid"
            spec.grid_source = "range"
            spec.lat_min, spec.lat_max = self.lat_min.value(), self.lat_max.value()
            spec.lon_min, spec.lon_max = self.lon_min.value(), self.lon_max.value()
            spec.lat_step, spec.lon_step = self.lat_step.value(), self.lon_step.value()
        elif mode == "gridfile":
            spec.mode = "grid"
            spec.grid_source = "file"
            spec.grid_file = self.gridfile_edit.text().strip()
            spec.grid_var = self.gridvar_edit.text().strip() or None
        elif mode == "points":
            spec.mode = "points"
            spec.points_file = self.pointsfile_edit.text().strip()
            spec.lat_col = self.latcol_combo.currentData()
            spec.lon_col = self.loncol_combo.currentData()
        else:
            spec.mode = "points"
            spec.points_file = ""
            spec.n_sphere_points = int(self.sphere_n.value())
        return spec

    # =====================================================================
    # 动作
    # =====================================================================
    def browse_coeffs(self):
        pats = " ".join(f"*{e}" for e in COEFF_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择球谐系数文件", self._start_dir(),
            f"球谐系数 ({pats});;所有文件 (*)")
        if path:
            self.coeffs_edit.setText(path.replace("/", os.sep))
            self.load_coeffs_info()

    def _start_dir(self) -> str:
        for cand in (self.coeffs_edit.text(), self.gridfile_edit.text(),
                     self.pointsfile_edit.text()):
            if cand and os.path.isdir(os.path.dirname(cand)):
                return os.path.dirname(cand)
        return os.getcwd()

    def _browse_into(self, edit: QLineEdit, title: str, filt: str):
        """**输入**文件:用『打开』对话框。"""
        path, _ = QFileDialog.getOpenFileName(self, title, self._start_dir(), filt)
        if path:
            edit.setText(path.replace("/", os.sep))

    def _browse_save_into(self, edit: QLineEdit, title: str, filt: str,
                          default_name: str = ""):
        """**输出**文件:用『保存』对话框。

        ⚠️ 输出路径以前错用了『打开』对话框 —— "选一个还不存在的输出文件"
        在语义上就别扭,而且对话框会把已存在的文件当成"选中的输入"。
        所有**要写出去**的路径都必须走这里。
        """
        start = edit.text().strip() or self._start_dir()
        if start and not os.path.isabs(start):
            start = os.path.abspath(start)
        if not default_name and start and os.path.isdir(os.path.dirname(start)):
            default_name = os.path.basename(start)
        target = (os.path.join(os.path.dirname(start), default_name)
                  if default_name else start)
        path, _ = QFileDialog.getSaveFileName(self, title, target, filt)
        if path:
            edit.setText(path.replace("/", os.sep))

    def _browse_out(self):
        filt = ("结果文件 (*.nc *.grd *.csv *.txt *.dat *.npy *.tsv);;"
                "netCDF (*.nc);;Surfer 网格 (*.grd);;CSV (*.csv);;"
                "文本 (*.txt);;numpy (*.npy)")
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", self._start_dir(), filt)
        if path:
            self.out_edit.setText(path.replace("/", os.sep))
            if not self.figfile_edit.text():
                root, _ext = os.path.splitext(path)
                self.figfile_edit.setText(root + ".png")

    def show_plan(self):
        spec = self.collect_spec()
        self._log("—— 参数预览 ——")
        self._log(plan_text(spec))
        self.tabs.setCurrentWidget(self.log_view)

    def load_coeffs_info(self):
        path = self.coeffs_edit.text().strip()
        if not path:
            QMessageBox.information(self, "尚未选择", "请先选择球谐系数文件。")
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "文件不存在", path)
            return
        self.status_label.setText("读取系数信息…")
        nmax = self.coeff_nmax_spin.value() or None
        self._info_worker = CoeffInfoWorker(path, self.layout_combo.currentData(),
                                            nmax)
        self._info_thread = make_thread(self._info_worker)
        self._info_worker.finished.connect(self._on_coeffs_loaded)
        self._info_worker.failed.connect(self._on_worker_failed)
        # ⚠️ 清引用必须在**线程真的结束之后**(thread.finished);
        # 在 worker 的 finished 槽里清会把还在跑的 QThread 析构掉,
        # Qt 会直接报 "QThread: Destroyed while thread is still running"。
        self._info_thread.finished.connect(self._on_info_thread_finished)
        self._info_thread.start()

    def _on_info_thread_finished(self):
        self._info_thread = None
        self._info_worker = None

    def _on_coeffs_loaded(self, coeffs):
        self._coeffs = coeffs
        info = coeff_info(coeffs)
        self.time_spin.setEnabled(coeffs.ntime > 1)
        self.time_spin.setRange(0, max(coeffs.ntime - 1, 0))
        # ---- v2.0:历元范围控件 + 时间轴摘要 --------------------------
        multi = coeffs.ntime > 1
        self.epoch_combo.setEnabled(multi)
        for s in (self.epoch_from, self.epoch_to):
            s.setRange(1, max(coeffs.ntime, 1))
            s.setEnabled(multi and self.epoch_combo.currentData() == "range")
        self.epoch_from.setValue(1)
        self.epoch_to.setValue(max(coeffs.ntime, 1))
        if coeffs.times is not None:
            sm = coeffs.times.summary()
            first = sm.splitlines()
            self.time_info_label.setText(
                "；".join(x.strip() for x in first[:3])
                + ("；…(详见日志)" if len(first) > 3 else ""))
        else:
            self.time_info_label.setText(
                "(无时间轴:只有时次序号。想按真实日期批量综合,"
                "请用上面的『读序列』按钮读 gfc 目录)")
        if self.coeff_nmax_spin.value() == 0:
            self.coeff_nmax_spin.setSpecialValueText(
                f"自动（文件: {coeffs.nmax} 阶）")
        self._log("—— 系数信息 ——")
        self._log(f"文件        : {info['source_file']}")
        self._log(f"识别布局    : {info['layout']}（自动识别;适配 SHKit 全部输出格式）")
        self._log(f"最高阶      : nmax = {coeffs.nmax}   独立系数 {coeffs.ncoef}")
        self._log(f"时次数      : ntime = {coeffs.ntime}")
        self._log(f"物理量声明  : "
                  f"{FIELD_UNIT_LABELS.get(info['field_unit'], info['field_unit'])}")
        if coeffs.times is not None:
            self._log("—— 时间轴 ——")
            for ln in coeffs.times.summary().splitlines():
                self._log("  " + ln)
        if info.get("gaussian_radius_km") or info.get("gaussian_km"):
            self._log(f"文件头高斯  : "
                      f"{info.get('gaussian_radius_km') or info.get('gaussian_km')} km")
        for w in info["warnings"]:
            self._log(f"警告        : {w}")
        # 自动填一个合理的默认截断/输出名
        if not self.truncate_spin.value() and coeffs.nmax <= 2190:
            self.truncate_spin.setValue(0)
        if not self.out_edit.text():
            root, _ext = os.path.splitext(os.path.basename(coeffs.meta.get(
                "source_file", "field")))
            self.out_edit.setText(os.path.join("out", f"{root}_synth.nc"))
        if not self.figfile_edit.text():
            self.figfile_edit.setText(os.path.join("out", f"{root}_synth.png"))
        self.status_label.setText(f"已读取:{os.path.basename(str(info['source_file']))}")
        # 逐阶谱先画出来(多时次用时间平均,避免把时间揉进一条曲线)
        self.spectrum_canvas.show_curves(
            {"系数逐阶 RMS": coeffs.degree_rms(time="mean")},
            title=f"系数逐阶振幅（nmax={coeffs.nmax}）")

    # =====================================================================
    # v2.0:时间轴控件 / 批量序列 / 矢量 / 诊断表
    # =====================================================================
    def _on_epoch_mode(self, _idx=0):
        mode = self.epoch_combo.currentData()
        on = self.epoch_combo.isEnabled() and mode == "range"
        for s in (self.epoch_from, self.epoch_to):
            s.setEnabled(on)

    def _on_component(self, _idx=0):
        """选了水平形变就把『输出物理量』也切过去(因子不同,不能混用)。"""
        comp = self.component_combo.currentData()
        if comp != "scalar":
            idx = self.unit_combo.findData("horizontal_displacement")
            if idx >= 0:
                self.unit_combo.setCurrentIndex(idx)
        self.vector_canvas.setVisible(comp in ("horizontal", "north", "east"))

    def browse_series_read(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择装着 GSM-*.gfc 的目录(或取消后手动填 .txt 清单)")
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "或选择一个 .txt 清单 / 多时间系数文件", self._start_dir(),
                "清单或多时间文件 (*.txt *.nc *.dat);;所有文件 (*)")
        if not path:
            return
        self._log(f"—— 读序列 ——\n来源: {path}")
        self.status_label.setText("读序列中…")
        self._series_read_worker = SeriesReadWorker(path)
        self._series_read_thread = make_thread(self._series_read_worker)
        self._series_read_worker.finished.connect(self._on_series_read)
        self._series_read_worker.failed.connect(self._on_worker_failed)
        self._series_read_thread.finished.connect(
            self._on_series_read_thread_finished)
        self._series_read_thread.start()

    def _on_series_read_thread_finished(self):
        self._series_read_thread = None
        self._series_read_worker = None

    def _clear_single_run_paths(self, why: str) -> None:
        """载入**批量序列**后,把"单历元那一套"的路径清空。

        为什么必须清:单文件模式与批量模式的**输入/输出语义完全不同** ——
        单文件那套是「一个系数文件 → 一个结果文件 + 一张图」,批量那套是
        「一条序列 → 场序列 nc(.nc 带时间坐标)+ 逐历元诊断 csv」。
        留着旧路径会让人以为批量结果会写到那个文件里,实际不会;
        更糟的是下次点『开始解算』会把**整条序列的第一个历元**写成那个文件。

        只清"路径",不动几何/物理量/绘图选项 —— 那些两边通用。

        ⚠️ 清 ``coeffs_edit`` 会触发 ``_on_coeffs_path_changed``(它把 ``_coeffs``
        作废)。这里必须先把 ``_coeffs`` 保住:切到批量模式后 ``_coeffs`` 装的正是
        批量序列,作废了『综合整条序列』就会以为没读系数。
        """
        cleared = []
        self._keep_coeffs = True
        try:
            for edit, label in ((self.coeffs_edit, "单文件"),
                                (self.out_edit, "结果文件"),
                                (self.figfile_edit, "图片文件")):
                if edit.text().strip():
                    cleared.append(label)
                    edit.clear()
        finally:
            self._keep_coeffs = False
        if cleared:
            self._log(f"已切到批量模式({why}):清空单历元那一套的路径"
                      f"({ '、'.join(cleared) });"
                      "批量结果请用下面的『场序列输出』『诊断表输出』。")

    def _on_series_read(self, coeffs):
        self._coeffs = coeffs
        self._log(f"读出序列:nmax={coeffs.nmax}  ntime={coeffs.ntime}")
        if coeffs.times is not None:
            for ln in coeffs.times.summary().splitlines():
                self._log("  " + ln)
        for w in (coeffs.meta.get("warnings") or []):
            self._log(f"警告: {w}")
        self._on_coeffs_loaded(coeffs)
        self._update_mean_note()
        if coeffs.ntime > 1:
            self._clear_single_run_paths(f"ntime={coeffs.ntime}")
            idx = self.epoch_combo.findData("all")
            self.epoch_combo.setCurrentIndex(idx)
            self._on_epoch_mode()
        self.status_label.setText(f"序列已读入:{coeffs.ntime} 个历元")

    # ---------------------------------------------------------- 去均值口径
    def mean_mode(self):
        """当前去均值口径:``None`` / ``'grace'`` / ``'all'`` / ``'custom'``。"""
        return self.mean_combo.currentData()

    def mean_choice(self) -> dict:
        """把界面上的去均值选择整成 ``remove_mean_window`` 的参数。

        "只填一边"的补全(``2005`` → ``2005-01-01`` / ``2009`` → ``2009-12-31``)
        由 :func:`shsynth.series.mean_window` 统一负责 —— 命令行与界面走**同一套**
        规则,不会出现"界面对、命令行错"这种分叉。
        """
        mode = self.mean_mode()
        if mode != "custom":
            return {"mode": mode} if mode else {"mode": None}
        return {"mode": "custom",
                "from_date": self.mean_from.text().strip() or None,
                "to_date": self.mean_to.text().strip() or None}

    def _on_mean_mode(self, *_a):
        custom = self.mean_mode() == "custom"
        for w in (self.mean_from, self.mean_to):
            w.setEnabled(custom)
        if custom and not self.mean_from.text().strip():
            self.mean_from.setText(GRACE_MEAN_FROM)
        if custom and not self.mean_to.text().strip():
            self.mean_to.setText(GRACE_MEAN_TO)
        self._update_mean_note()

    def _update_mean_note(self):
        """把"这一次到底会拿哪一段做基准"直接算给用户看,不让他猜。"""
        mode = self.mean_mode()
        if mode is None:
            self.mean_note.setText(
                "⚠️ 不去均值:GSM 含静态场,换成 EWH 会出现万米量级常量。"
                "做异常场请选一种去均值口径。")
            return
        coeffs = getattr(self, "_coeffs", None)
        if coeffs is None or coeffs.times is None:
            self.mean_note.setText("(读入带日期的序列后会算出实际使用的历元区间)")
            return
        try:
            from ..series import mean_window
            rep = mean_window(coeffs.times, **self.mean_choice())
            self.mean_note.setText(
                f"→ 将用 {rep['n_epochs']}/{rep['n_total']} 个历元"
                f"({rep['from_date']} .. {rep['to_date']},"
                f"占 {rep['coverage'] * 100:.0f}%)做基准")
            if rep["warnings"]:
                self.mean_note.setText(self.mean_note.text()
                                       + "  ! " + rep["warnings"][0])
        except Exception as exc:                             # noqa: BLE001
            self.mean_note.setText(f"⚠️ {exc}")

    def _series_target(self):
        """按当前『求值位置』给出批量序列的目标几何。"""
        from .. import targets as T
        spec = self.collect_spec()
        if spec.mode == "points":
            if spec.points_file:
                la, lo, _ = T.points_from_file(spec.points_file,
                                               lat_col=spec.lat_col,
                                               lon_col=spec.lon_col)
                return None, None, (la, lo)
            la, lo, _ = T.spherical_points(spec.n_sphere_points)
            return None, None, (la, lo)
        if spec.grid_source == "file":
            latv, lonv, _ref, _ = T.grid_from_file(spec.grid_file,
                                                   var=spec.grid_var)
            return latv, lonv, None
        if spec.grid_source == "global":
            latv, lonv, _ = T.global_grid(spec.lat_step)
            return latv, lonv, None
        latv, lonv, _ = T.grid_from_spec(spec.lat_min, spec.lat_max,
                                         spec.lon_min, spec.lon_max,
                                         spec.lat_step, spec.lon_step)
        return latv, lonv, None

    def start_series_run(self):
        if getattr(self, "_series_thread", None) is not None:
            return
        self._anim_stop()          # 重新算之前先停下播放,免得对着旧序列放
        coeffs = getattr(self, "_coeffs", None)
        # ⚠️ 这些前置条件**不用模态对话框**:模态框在无人值守/离屏环境里会把
        # 界面(以及自动化测试)永久卡住。改成状态栏 + 日志,信息一样清楚。
        if coeffs is None:
            self.status_label.setText("还没读系数:请先『读取系数信息』或『读序列』")
            self._log("! 批量综合需要先读入系数:点『读取系数信息』或『读序列』。")
            self.tabs.setCurrentWidget(self.log_view)
            return
        if coeffs.ntime < 2:
            self.status_label.setText("这不是序列(ntime=1)")
            self._log("! 批量综合需要多时次:请用『读序列』选一个装着 "
                      "GSM-*.gfc 的目录(会带上真实日期)。")
            self.tabs.setCurrentWidget(self.log_view)
            return
        spec = self.collect_spec()
        target = self._series_target()
        comp = self.component_combo.currentData()
        tgt_unit = self.unit_combo.currentData()
        sub = None
        mode = self.epoch_combo.currentData()
        if mode == "current":
            sub = coeffs.time_slice(self.time_spin.value())
        elif mode == "range":
            a = self.epoch_from.value() - 1
            b = self.epoch_to.value()
            sub = coeffs.select_times(idx=np.arange(a, b))
        if sub is not None:
            if sub.ntime < 2:
                self.status_label.setText("按当前设置只选了 1 个历元")
                self._log("! 批量综合至少要 2 个历元:把『历元范围』改成"
                          "『整条序列』,或把起止拉开。")
                self.tabs.setCurrentWidget(self.log_view)
                return
            coeffs = sub
        from ..series import remove_mean_window
        choice = self.mean_choice()
        if choice["mode"] is not None:
            # 口径错误(窗口里没历元、自定义没填全、没有时间轴)要在这里**报出来**
            # 并且不启动计算 —— 绝不悄悄换成全时段(那会换掉静态场基准)。
            try:
                coeffs, mean_rep = remove_mean_window(coeffs, **choice)
            except ValueError as exc:
                self.status_label.setText("去均值口径有问题")
                self._log(f"! 去均值失败:{exc}")
                self.tabs.setCurrentWidget(self.log_view)
                return
            self._log("  去均值      : " + mean_rep["summary"])
            for w in mean_rep["warnings"]:
                self._log("  ! " + w)
        else:
            self._log("  去均值      : 不做(输出是含静态场的原始场)")
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.series_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("批量综合中…")
        self._log(f"—— 批量序列综合 ——\n"
                  f"  历元数 {coeffs.ntime} | 分量 {comp} | 目标 {'散点' if target[2] is not None else '网格'}")
        self._series_worker = SeriesSynthWorker(
            coeffs, target, nmax=spec.truncate_nmax or None,
            gaussian_km=spec.gaussian_km, gaussian_method=spec.gaussian_method,
            target_unit=tgt_unit if comp == "scalar" else None,
            component=comp, chunk=spec.chunk)
        self._series_thread = make_thread(self._series_worker)
        self._series_worker.progress.connect(self._on_progress)
        self._series_worker.finished.connect(self._on_series_finished)
        self._series_worker.failed.connect(self._on_worker_failed)
        self._series_worker.cancelled.connect(self._on_cancelled)
        self._series_thread.finished.connect(self._on_thread_finished)
        self._series_thread.start()

    def _on_series_finished(self, res):
        self._series_result = res
        self._log("")
        self._log(res.summary)
        # 落盘
        out = self.series_out_edit.text().strip()
        if out:
            try:
                v = res.values
                if res.kind == "grid":
                    extra = None
                    if res.components:
                        extra = {("north_displacement" if k == "north"
                                  else "east_displacement"): np.asarray(a)
                                 for k, a in res.components.items()}
                    p = fieldio.write_grid(
                        out, res.lat, res.lon, v, var="value",
                        units=self.out_units() or "m", times=res.times,
                        extra_vars=extra,
                        meta={"method": res.stats.get("method"),
                              "component": res.stats.get("component")})
                else:
                    p = fieldio.write_points(out, res.lat, res.lon, v)
                self._log(f"已写出场序列: {p}")
            except Exception as exc:                     # noqa: BLE001
                self._log(f"写出失败: {type(exc).__name__}: {exc}")
        diag = self.series_diag_edit.text().strip()
        if diag:
            try:
                self._log(f"已写出诊断表: {res.to_csv(diag)}")
            except Exception as exc:                     # noqa: BLE001
                self._log(f"诊断表写出失败: {type(exc).__name__}: {exc}")
        self._show_series_tab(res)
        self._show_diag_table(res)
        self._render_series_frame(res)
        self._show_anim_tab(res)
        self.status_label.setText(
            f"批量完成:{len(res.times)} 个历元 × {res.stats.get('eval_points', 0):,} 点")

    def out_units(self) -> str:
        return self.outunits_edit.text().strip()

    def focus_mode(self) -> str:
        """当前的地图聚焦模式(与 MapCanvas 的用法一致)。"""
        return "auto" if self.chk_focus.isChecked() else "global"

    def _show_series_tab(self, res):
        """时间序列页:网格给面积加权平均,散点给前几个点。"""
        try:
            v = np.asarray(res.values, dtype=float)
            curves = {}
            if res.kind == "grid" and v.ndim == 3:
                w = np.cos(np.deg2rad(np.asarray(res.lat, dtype=float)))[:, None]
                flat = v.reshape(-1, v.shape[-1])
                ww = np.broadcast_to(w, v.shape[:2]).reshape(-1)
                ok = np.isfinite(flat).all(axis=0)
                num = np.nansum(flat * ww[:, None], axis=0)
                den = np.nansum(np.where(np.isfinite(flat), ww[:, None], 0.0),
                                axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
                curves["区域加权平均"] = mean
                curves["逐历元最小"] = np.nanmin(flat, axis=0)
                curves["逐历元最大"] = np.nanmax(flat, axis=0)
                del ok
            elif v.ndim == 2:
                n = min(v.shape[0], 6)
                for i in range(n):
                    curves[f"点 {i + 1}"] = v[i]
            else:
                curves["序列"] = v
            unit = self.outunits_edit.text().strip() or "值"
            self.series_canvas.show_series(
                res.times, curves,
                title=f"时间序列({len(res.times)} 个历元)",
                ylabel=unit)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"时间序列页绘制失败: {type(exc).__name__}: {exc}")

    def _render_series_frame(self, res):
        """地图/矢量页画**第一个历元**(便于立刻看到形状)。"""
        try:
            if res.kind != "grid":
                return
            comp = res.stats.get("component", "scalar")
            lat = np.asarray(res.lat, dtype=float)
            lon = np.asarray(res.lon, dtype=float)
            if comp in ("horizontal", "north", "east") and res.components:
                self.vector_canvas.show_vectors(
                    lat, lon, np.asarray(res.components["north"])[:, :, 0],
                    np.asarray(res.components["east"])[:, :, 0],
                    magnitude=(np.asarray(res.values)[:, :, 0]
                               if comp == "horizontal" else None),
                    title=f"水平形变(第 1 个历元 {str(res.times.values[0])[:10]})",
                    stride=8, cb_label=self.outunits_edit.text().strip() or "m")
            else:
                from ..plotting import make_map_figure
                make_map_figure(lat, lon, np.asarray(res.values)[:, :, 0],
                                title=f"第 1 个历元 {str(res.times.values[0])[:10]}",
                                focus=self.focus_mode(),
                                fig=self.map_canvas.figure)
                self.map_canvas.refresh()
        except Exception as exc:                         # noqa: BLE001
            self._log(f"帧绘制失败: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------- 动画(v2.0)
    def _show_anim_tab(self, res):
        """把刚算完的场序列接到动画页(网格才有帧可放)。"""
        self._anim_stop()
        self._anim_res = None
        self._anim_values = None
        self._anim_idx = []
        self._anim_pos = 0
        self.anim_slider.blockSignals(True)
        self.anim_slider.setRange(0, 0)
        self.anim_slider.setValue(0)
        self.anim_slider.blockSignals(False)
        ok = False
        try:
            if res is not None and res.kind == "grid" and res.values.ndim == 3:
                v = np.asarray(res.values, dtype=float)
                from ..plotting import _symmetric_range
                # 配色范围在**整段序列**上定一次(逐帧定标会闪,见 §4.6)
                a, b = _symmetric_range(v, symmetric=True)
                self._anim_vmin, self._anim_vmax = float(a), float(b)
                self._anim_res = res
                self._anim_values = v
                self._anim_idx = list(range(v.shape[2]))
                self._anim_pos = 0
                self.anim_slider.blockSignals(True)
                self.anim_slider.setRange(0, len(self._anim_idx) - 1)
                self.anim_slider.setValue(0)
                self.anim_slider.blockSignals(False)
                self._anim_draw()
                ok = True
                self._log(f"动画页就绪:{len(self._anim_idx)} 帧,"
                          f"配色固定 {self._anim_vmin:.4g} .. {self._anim_vmax:.4g}"
                          "(全序列稳健范围,不逐帧定标)")
                if res.stats.get("component", "scalar") in ("horizontal",
                                                            "north", "east"):
                    self._log("  这一页播的是『大小』(旋转不变量);"
                              "要看北/东分量与矢量箭头请用『水平形变』页")
        except Exception as exc:                             # noqa: BLE001
            self._log(f"动画页准备失败: {type(exc).__name__}: {exc}")
        for w in (self.anim_play_btn, self.anim_prev_btn, self.anim_next_btn,
                  self.anim_slider, self.anim_gif_btn):
            w.setEnabled(ok)
        if not ok:
            self.anim_label.setText("—")
            self._log("(动画页需要网格序列;散点序列没有规则帧可画)")

    def _anim_draw(self):
        if self._anim_res is None or not self._anim_idx:
            return
        k = self._anim_idx[self._anim_pos]
        res = self._anim_res
        unit = self.outunits_edit.text().strip() or "值"
        lab = str(res.times.values[k])[:10] if len(res.times) else f"#{k}"
        try:
            # 水平形变模式下 res.values 是 magnitude(旋转不变量),逐帧播放它
            # 是有意义的;分量/矢量请用『水平形变』页。这个提示在
            # _show_anim_tab 里说一次就够了 —— 放在这里会按帧率刷屏。
            arr = np.asarray(res.values)
            from ..plotting import make_map_figure
            make_map_figure(np.asarray(res.lat, dtype=float),
                            np.asarray(res.lon, dtype=float),
                            arr[:, :, k],
                            title=f"SHSynth {lab}",
                            cb_label=unit,
                            vmin=self._anim_vmin, vmax=self._anim_vmax,
                            focus=self.focus_mode(),
                            fig=self.anim_canvas.figure)
            self.anim_canvas.refresh()
        except Exception as exc:                             # noqa: BLE001
            self._log(f"动画帧绘制失败: {type(exc).__name__}: {exc}")
        self.anim_label.setText(f"{self._anim_pos + 1}/{len(self._anim_idx)}  {lab}")
        self.anim_slider.blockSignals(True)
        self.anim_slider.setValue(self._anim_pos)
        self.anim_slider.blockSignals(False)

    def _anim_step(self, d: int):
        if not self._anim_idx:
            return
        n = len(self._anim_idx)
        p = self._anim_pos + d
        if p >= n:
            p = 0 if self.chk_anim_loop.isChecked() else n - 1
        elif p < 0:
            p = n - 1 if self.chk_anim_loop.isChecked() else 0
        self._anim_pos = p
        self._anim_draw()

    def _anim_next(self):
        self._anim_step(+1)

    def _on_anim_slider(self, value: int):
        if not self._anim_idx:
            return
        self._anim_pos = max(0, min(int(value), len(self._anim_idx) - 1))
        self._anim_draw()

    def _on_anim_fps(self, _v):
        if self._anim_timer.isActive():
            self._anim_timer.start(self._anim_interval())

    def _anim_interval(self) -> int:
        return max(20, int(round(1000.0 / max(self.anim_fps.value(), 0.2))))

    def _on_anim_play(self, on: bool):
        self.anim_play_btn.setText("⏸ 暂停" if on else "▶ 播放")
        if on and self._anim_idx:
            self._anim_timer.start(self._anim_interval())
        else:
            self._anim_timer.stop()

    def _anim_stop(self):
        self._anim_timer.stop()
        if self.anim_play_btn.isChecked():
            self.anim_play_btn.blockSignals(True)
            self.anim_play_btn.setChecked(False)
            self.anim_play_btn.blockSignals(False)
        self.anim_play_btn.setText("▶ 播放")

    def export_animation(self):
        """把当前场序列导出成 GIF(子线程渲染,不冻结界面)。"""
        import os as _os
        if self._anim_res is None:
            self.status_label.setText("先在『批量序列』里综合一条序列,再导出动画")
            return
        if self._anim_export_thread is not None:
            self.status_label.setText("动画正在导出中,请稍候(或点『停止』取消)")
            return
        from PySide6.QtWidgets import QFileDialog
        default = ""
        if self.series_out_edit.text().strip():
            default = _os.path.splitext(self.series_out_edit.text().strip())[0] + ".gif"
        path, _f = QFileDialog.getSaveFileName(
            self, "导出动画", default or "shsynth_animation.gif",
            "GIF 动图 (*.gif);;MP4 视频 (*.mp4,需 imageio)")
        if not path:
            return
        self._anim_stop()
        from .workers import AnimationExportWorker, make_thread
        unit = self.outunits_edit.text().strip()
        worker = AnimationExportWorker(
            self._anim_res, path, fps=self.anim_fps.value(),
            values=self._anim_values, focus=self.focus_mode(),
            title="SHSynth", cb_label=unit or "value")
        thread = make_thread(worker)
        self._anim_export_worker, self._anim_export_thread = worker, thread
        worker.progress.connect(self._on_anim_export_progress)
        worker.finished.connect(self._on_anim_export_done)
        worker.failed.connect(self._on_anim_export_failed)
        worker.cancelled.connect(self._on_anim_export_cancelled)
        thread.finished.connect(self._on_anim_export_thread_finished)
        thread.start()
        self.status_label.setText(f"正在导出动画({len(self._anim_idx)} 帧)…")
        self._log(f"开始导出动画: {path}")

    def _on_anim_export_progress(self, done: int, total: int):
        self.status_label.setText(f"正在渲染动画 {done}/{total} …")

    def _on_anim_export_done(self, path: str, info: dict):
        mb = info.get("bytes", 0) / 1e6
        self._log(f"动画已写出: {path} ({info.get('frames')} 帧, {mb:.1f} MB)")
        got, want = info.get("fps_effective"), info.get("fps_requested")
        if got is not None and want is not None and abs(got - want) > 1e-6:
            self._log(f"  注意:GIF 延时只能取 10 ms 整数倍,实际帧率 "
                      f"{got:.3f} fps(你给的是 {want:g})")
        self.status_label.setText(f"动画已写出({info.get('frames')} 帧)")

    def _on_anim_export_failed(self, msg: str, tb: str):
        self._log(f"动画导出失败: {msg}")
        self._log(tb)
        self.status_label.setText("动画导出失败(原因见日志)")

    def _on_anim_export_cancelled(self):
        self._log("动画导出已取消")
        self.status_label.setText("动画导出已取消")

    def _on_anim_export_thread_finished(self):
        # 必须先断开再清引用 —— 在 worker 的 finished 槽里直接销毁还在跑的
        # QThread 会触发 "QThread: Destroyed while thread is still running"。
        self._anim_export_thread = None
        self._anim_export_worker = None

    def _show_diag_table(self, res):
        st = res.stats
        cols = [c for c in ("epoch", "time", "decimal_year", "rms", "mean",
                            "min", "max", "finite_frac") if c in st]
        self.diag_table.clear()
        self.diag_table.setColumnCount(len(cols))
        self.diag_table.setRowCount(len(res.times))
        self.diag_table.setHorizontalHeaderLabels(cols)
        for i in range(len(res.times)):
            for j, c in enumerate(cols):
                val = st[c][i] if isinstance(st[c], (list, tuple)) else st[c]
                txt = f"{val:.6g}" if isinstance(val, float) else str(val)
                self.diag_table.setItem(i, j, QTableWidgetItem(txt))
        out = res.outliers()
        for i in out:
            for j in range(len(cols)):
                it = self.diag_table.item(i, j)
                if it is not None:
                    it.setBackground(QColor(255, 235, 205))
                    it.setToolTip("MAD 稳健离群(**只报告,不剔除**)")
        self._log(f"逐历元诊断表已填:{len(res.times)} 行"
                  + (f";可疑历元 {len(out)} 个(高亮,不剔除)" if out else ""))

    def _on_diag_row(self, row, _col=0):
        res = self._series_result
        if res is None or res.kind != "grid":
            return
        try:
            from ..plotting import make_map_figure
            make_map_figure(np.asarray(res.lat, float), np.asarray(res.lon, float),
                            np.asarray(res.values)[:, :, row],
                            title=f"第 {row + 1} 个历元 {str(res.times.values[row])[:10]}",
                            focus=self.focus_mode(),
                            fig=self.map_canvas.figure)
            self.map_canvas.refresh()
            self.tabs.setCurrentWidget(self.map_canvas)
        except Exception as exc:                         # noqa: BLE001
            self._log(f"跳转失败: {type(exc).__name__}: {exc}")
        # 动画页也跟着跳:同一件事(定位到某历元)在两页应当一致
        if self._anim_idx and row < len(self._anim_idx):
            self._anim_stop()
            self._anim_pos = int(row)
            self._anim_draw()

    def start_run(self):
        spec = self.collect_spec()
        if not spec.coeffs_path:
            QMessageBox.warning(self, "缺少系数文件", "请先选择球谐系数文件。")
            return
        if spec.mode == "grid" and spec.grid_source == "file" and not spec.grid_file:
            QMessageBox.warning(self, "缺少网格文件", "请选择要借用格点的网格文件。")
            return
        if spec.mode == "points" and spec.figure_kind == "none" and \
                not spec.points_file and not spec.out_path:
            QMessageBox.information(self, "没有输出", "既不出图也不写文件,无事可做。")
            return

        self._log("")
        self._log("=" * 62)
        self._log(plan_text(spec))
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText("解算中…")

        self._worker = SynthWorker(spec)
        self._thread = make_thread(self._worker)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    def stop_run(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("正在停止…")
        if self._series_worker is not None:
            self._series_worker.cancel()
            self.status_label.setText("正在停止批量综合…")

    def _on_progress(self, msg: str, frac: float):
        self.progress.setValue(int(frac * 100))
        if msg != self._last_progress_msg:
            self._last_progress_msg = msg
            self.status_label.setText(msg)

    def _on_finished(self, result):
        self._last_result = result
        self._log(result.summary)
        self.progress.setValue(100)
        self.status_label.setText(
            f"完成:{result.stats['n_points']:,} 点   "
            f"RMS={result.stats['rms']:.6g}")
        self._render(result)
        self._after_run(result)

    def _after_run(self, result):
        """解算结束后的落点(纯界面行为,不影响算出来的数和存的图)。

        默认**自动打开地图页**:区域结果在全球视角下太小,而四联报告图里那张
        地图只有半幅,所以"跑完就能看到放大的结果范围"是默认路径;取消勾选
        『运行后自动打开地图』则停在『图类型』选的那张图上。
        """
        if getattr(result, "figure", None) is not None:
            if self.chk_autoshow.isChecked():
                self.tabs.setCurrentWidget(self.map_canvas)
                if self.chk_focus.isChecked():
                    self.status_label.setText(
                        self.status_label.text() + "　视图:地图(已聚焦结果范围)")
            else:
                # 与上面的 docstring 一致:不自动开地图时,停在**刚才选的图类型**上
                spec = getattr(result, "spec", None)
                kind = getattr(spec, "figure_kind", "report")
                target = {"report": self.report_canvas,
                          "map": self.map_canvas,
                          "points": self.map_canvas,
                          "spectrum": self.spectrum_canvas,
                          "hist": self.hist_canvas}.get(kind)
                if target is not None:
                    self.tabs.setCurrentWidget(target)
        if self.chk_raise.isChecked():
            self.raise_()
            self.activateWindow()

    def toggle_focus(self):
        """在『聚焦结果范围』与『全球视图』之间切换,并立刻重画当前结果。"""
        self.chk_focus.setChecked(not self.chk_focus.isChecked())
        if self._last_result is None:
            self.status_label.setText(
                "已设为『" + ("聚焦结果范围" if self.chk_focus.isChecked()
                              else "全球视图") + "』;解算后按此绘图")
            return
        self._render(self._last_result)
        self.status_label.setText(
            "视图:" + ("聚焦结果范围" if self.chk_focus.isChecked() else "全球视图"))

    def _render(self, result):
        """在主线程里把结果画到各画布自带的 Figure 上,并按需存盘。"""
        spec = result.spec
        kind = spec.figure_kind
        flat = result.values
        cb_label = spec.out_units or {
            "geoid": "m", "ewh": "m (EWH)", "surface_density": "kg/m²",
            "radial_displacement": "m",
        }.get(result.coeffs.field_unit, "")
        title = (f"{os.path.basename(spec.coeffs_path)}  nmax="
                 f"{result.coeffs.nmax}"
                 + (f"  高斯 {spec.gaussian_km:g} km" if spec.gaussian_km else ""))
        curves = {"系数逐阶 RMS": result.coeffs.degree_rms(time="mean")}
        styles = {"系数逐阶 RMS": {"color": "0.45", "linestyle": "--"}}
        lat_vec = result.lat if result.kind == "grid" else None
        lon_vec = result.lon if result.kind == "grid" else None
        plotted = None
        # 聚焦:界面上勾了『自动聚焦到结果范围』就自动缩放到数据范围,
        # 否则画全球视图。覆盖接近全球时 data_extent 返回 None(等于不聚焦)。
        focus = "auto" if self.chk_focus.isChecked() else "global"

        try:
            if kind == "none":
                plotted = None
            elif kind == "report":
                self.report_canvas.show_report(
                    result.lat, result.lon, result.values, coeffs=result.coeffs,
                    curves=curves, title=title, cmap=spec.cmap, cb_label=cb_label,
                    contour=spec.contour,
                    annotation=[f"n={result.stats['n_points']:,}",
                                f"RMS={result.stats['rms']:.4g}",
                                f"min={result.stats['min']:.4g}",
                                f"max={result.stats['max']:.4g}"],
                    lat_vec=lat_vec, lon_vec=lon_vec, focus=focus)
                plotted = self.report_canvas.figure
                self.tabs.setCurrentWidget(self.report_canvas)
            elif kind == "spectrum":
                self.spectrum_canvas.show_curves(curves, styles=styles)
                plotted = self.spectrum_canvas.figure
                self.tabs.setCurrentWidget(self.spectrum_canvas)
            elif kind == "hist":
                self.hist_canvas.show_values(flat, title=title, xlabel=cb_label)
                plotted = self.hist_canvas.figure
                self.tabs.setCurrentWidget(self.hist_canvas)
            else:
                self.map_canvas.show_field(
                    result.lat, result.lon, result.values, title=title,
                    cmap=spec.cmap, cb_label=cb_label, contour=spec.contour,
                    coast=spec.coast, symmetric=spec.symmetric, focus=focus)
                plotted = self.map_canvas.figure
                self.tabs.setCurrentWidget(self.map_canvas)

            # 地图与分布无论选了哪种图都更新一下,方便切页签看
            if kind != "none" and kind != "map":
                self.map_canvas.show_field(
                    result.lat, result.lon, result.values, title=title,
                    cmap=spec.cmap, cb_label=cb_label, contour=spec.contour,
                    coast=spec.coast, symmetric=spec.symmetric, focus=focus)
                self.hist_canvas.show_values(flat, title=title, xlabel=cb_label)

            if plotted is not None and spec.figure_path:
                p = plotting.save_figure(plotted, spec.figure_path,
                                         dpi=spec.figure_dpi)
                self._log(f"已保存图      : {p}")
            # 把"界面上正在显示的那张 Figure"记回结果对象,后续『保存当前图』
            # 或二次导出都直接用同一张,不会出现"显示与存盘不是一张图"。
            result.figure = plotted
        except Exception as exc:                         # noqa: BLE001
            self._log(f"绘图失败: {type(exc).__name__}: {exc}")
            self._log(traceback.format_exc())

    def save_current_figure(self):
        w = self.tabs.currentWidget()
        fig = getattr(w, "figure", None)
        if fig is None or not fig.get_axes():
            QMessageBox.information(self, "没有可保存的图", "当前页签里还没有图。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前图", self._start_dir(),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;JPEG (*.jpg)")
        if not path:
            return
        try:
            p = plotting.save_figure(fig, path, dpi=int(self.dpi_spin.value()))
            self._log(f"已保存当前图  : {p}")
            self.status_label.setText(f"已保存 {os.path.basename(p)}")
        except Exception as exc:                         # noqa: BLE001
            QMessageBox.critical(self, "保存失败", f"{type(exc).__name__}: {exc}")

    def _on_cancelled(self):
        self._log("已停止。")
        self.status_label.setText("已停止")

    def _on_worker_failed(self, msg: str, tb: str):
        self._log(f"失败:{msg}")
        if tb:
            self._log(tb)
        self.progress.setValue(0)
        self.status_label.setText("失败")
        QMessageBox.critical(self, "解算失败", msg)

    def _on_thread_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.series_btn.setEnabled(True)
        self._thread = None
        self._worker = None
        self._series_thread = None
        self._series_worker = None

    # =====================================================================
    # 其它
    # =====================================================================
    def _log(self, text: str):
        self.log_view.appendPlainText(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _show_formats(self):
        from ..coeffio import FORMAT_SPECS
        lines = []
        for sp in FORMAT_SPECS:
            lines.append(f"[{sp['layout']}] {sp['name']}")
            lines.append(f"   扩展名: {', '.join(sp['ext'])}")
            lines.append(f"   {sp['detail']}")
        lines.append("")
        lines.append("另可读:.gz 压缩文本/gfc、MATLAB 稠密方阵文本")
        QMessageBox.information(self, "支持的系数格式", "\n".join(lines))

    def show_guide(self):
        """在**窗口里**打开说明书(安装包里带 HTML;源码树里退回 Markdown)。"""
        from .docs_window import GuideDialog
        html = author.guide_html_path()
        dlg = GuideDialog(self, html_path=html)
        if not html:
            self._log("未找到随包的 HTML 说明书;窗口里已给出提示。"
                      "源码运行时先跑 tools/make_help_html.py")
        else:
            self._log(f"已打开说明书窗口: {html}")
        dlg.exec()

    def _about(self):
        AboutDialog(self).exec()

    # ---------------------------------------------------------------- 拖放
    def dragEnterEvent(self, event):                     # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):                          # noqa: N802
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in COEFF_EXTENSIONS:
                self.coeffs_edit.setText(path)
                self._log(f"拖入系数文件: {path}")
                self.load_coeffs_info()
            elif ext in POINT_EXTENSIONS:
                self.pointsfile_edit.setText(path)
                self.mode_radios["points"].setChecked(True)
                self._log(f"拖入散点文件: {path}")
            elif ext in GRID_EXTENSIONS:
                self.gridfile_edit.setText(path)
                self.mode_radios["gridfile"].setChecked(True)
                self._log(f"拖入网格文件: {path}")
            else:
                self._log(f"拖入的文件扩展名无法识别: {path}")
        event.acceptProposedAction()

    def closeEvent(self, event):                         # noqa: N802
        if self._worker is not None:
            self._worker.cancel()
        # 关窗前把动画定时器和导出线程收干净,否则会留一个还在跑的 QThread
        self._anim_stop()
        if self._anim_export_worker is not None:
            self._anim_export_worker.cancel()
        if self._anim_export_thread is not None:
            self._anim_export_thread.quit()
            self._anim_export_thread.wait(5000)
        super().closeEvent(event)


#: 优先使用的中文字体(与 shsynth.plotting 的候选表一致)
_CJK_FONTS = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
              "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB",
              "WenQuanYi Zen Hei", "Arial Unicode MS", "MS Gothic")


def _apply_cjk_font(app) -> Optional[str]:
    """把界面字体设成本机确实装了的中文字体,避免中文显示成方框。

    实测:默认 ``windows`` 平台下有 213 个字体族,含 Microsoft YaHei / SimHei /
    SimSun,本来就没问题;但离屏(``offscreen``)插件不带字体库,Qt 控件文字会变成
    方框(matplotlib 不受影响,它自己管字体)。显式指定一次,两种情况都稳。
    """
    try:
        from PySide6.QtGui import QFont, QFontDatabase
        families = set(QFontDatabase.families())
    except Exception:                                    # pragma: no cover
        return None
    for name in _CJK_FONTS:
        if name in families:
            f = QFont(name)
            f.setPointSize(9)
            app.setFont(f)
            return name
    return None


#: 关于对话框里的 6 步快速上手(与「使用说明」文档同源)
_QUICK_START = [
    ("1  选系数", "「系数文件」→ 浏览(或把文件拖进窗口)。"
                  "布局保持 <b>auto</b> 即可自动识别 SHKit 的五种输出格式;"
                  "点「读取系数信息」看识别结果。"),
    ("2  选位置", "全球网格 / 范围网格 / <b>借用网格文件</b> / 散点文件 / 球面散点。"
                  "区域结果默认会自动聚焦,不用手动放大。"),
    ("3  定阶数", "「截断阶数」留 0 表示用系数自身的阶数。"
                  "需要更高阶就改系数文件或用 <code>--nmax</code>。"),
    ("4  平滑换算", "「高斯平滑」在综合时施加(不改动系数文件);"
                    "「输出物理量」是不换算 / geoid / EWH / 面密度 / 径向形变 —— "
                    "换算因子是<b>逐阶</b>的(Aₙ 从 1.17e7 到 1.68e8)。"),
    ("5  解算", "点「开始解算」。结果写成 .nc / .grd / .csv / .npy,"
                "同时给出地图、四联报告图、逐阶谱与数值分布。"),
    ("6  出图", "解算完会<b>自动打开地图页并聚焦到结果范围</b>(可在「出图」组里关掉);"
                "「自动聚焦到结果范围」可切换全球/区域视图;"
                "「保存当前图…」导出 PNG/PDF/SVG;图与界面显示的是同一张。"),
]


class AboutDialog(QDialog):
    """关于 / 作者信息 + 公众号二维码 + 快速上手。

    个人信息(作者、单位、邮箱、电话、公众号)与二维码都在这一屏里:左侧是联系
    方式表,右侧直接显示二维码(不用再点开二级对话框),底部有直通说明书的按钮。
    与同课题组的 SHKit / GRACE Downloader 系列版式一致。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"关于 {author.APP_NAME} / About")
        self.resize(900, 820)
        self.setMinimumSize(640, 480)
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(18, 16, 18, 16)

        header = QLabel(
            f"<h2>{author.APP_NAME} {__version__} — {author.APP_NAME_CN}</h2>"
            f"<p>{author.APP_TAGLINE}<br>"
            "<small>适配 SHKit 的全部球谐系数输出格式;"
            "综合结果与 SHKit <b>逐位一致</b>。</small></p>")
        header.setWordWrap(True)
        root.addWidget(header)

        root.addWidget(self._author_group())
        root.addWidget(self._quick_start_group())

        credit = QLabel(
            "<small>图形界面:Qt for Python(<b>PySide6-Essentials</b>,GNU LGPL v3)"
            "—— 动态链接、未修改;刻意未安装 Qt Charts / Qt Data Visualization 等 "
            "GPL-only 模块,绘图改用 matplotlib。LGPLv3 全文见 "
            "<code>licenses/LGPL-3.0.txt</code>,第三方声明见 "
            "<code>licenses/NOTICE.txt</code>。SHSynth 自身代码采用 MIT 许可。<br>"
            "载荷勒夫数表来自 PREM-LLNs.dat(Wang et al. 2012);"
            "海岸线来自 Natural Earth 110m(公有领域)。仅画海岸线、不画国界。<br>"
            "本软件仅供科研与教学使用。</small>")
        credit.setWordWrap(True)
        credit.setStyleSheet("color:#666;")
        root.addWidget(credit)
        root.addStretch(1)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        root.addWidget(box)

    # ------------------------------------------------------------- 子部件
    def _author_group(self) -> QGroupBox:
        grp = QGroupBox("作者信息  /  Author")
        outer = QHBoxLayout(grp)
        outer.setSpacing(16)

        left = QWidget()
        grid = QGridLayout(left)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        rows = [
            ("作者  Author",
             f"{author.AUTHOR_NAME_CN}  （{author.AUTHOR_NAME_EN}）"),
            ("单位  Affiliation",
             f"{author.AUTHOR_AFFILIATION_CN}<br>{author.AUTHOR_AFFILIATION_EN}"),
            ("邮箱  E-mail",
             f'<a href="mailto:{author.AUTHOR_EMAIL}">{author.AUTHOR_EMAIL}</a>'),
            ("电话  Phone", author.AUTHOR_PHONE),
            ("公众号  WeChat",
             f"{author.WECHAT_ACCOUNT}（{author.WECHAT_ACCOUNT_EN}）"),
            ("版本  Version", f"v{__version__}"),
        ]
        for r, (key, value) in enumerate(rows):
            k = QLabel(f"<b>{key}</b>")
            k.setAlignment(Qt.AlignRight | Qt.AlignTop)
            v = QLabel(value)
            v.setWordWrap(True)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse
                                      | Qt.LinksAccessibleByMouse)
            v.setOpenExternalLinks(True)
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
        grid.setColumnStretch(1, 1)

        btns = QHBoxLayout()
        b_mail = QPushButton("发邮件  (E-mail)")
        b_mail.setMinimumHeight(30)
        b_mail.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl(f"mailto:{author.AUTHOR_EMAIL}")))
        b_wx = QPushButton("放大二维码")
        b_wx.setMinimumHeight(30)
        b_wx.clicked.connect(self._show_wechat)
        b_guide = QPushButton("使用说明（F1）")
        b_guide.setMinimumHeight(30)
        b_guide.setToolTip("打开随包的 HTML 使用说明")
        b_guide.clicked.connect(self._open_guide)
        for b in (b_mail, b_wx, b_guide):
            btns.addWidget(b)
        btns.addStretch(1)
        grid.addLayout(btns, len(rows), 0, 1, 2)
        outer.addWidget(left, stretch=1)
        outer.addWidget(self._qr_panel(), stretch=0)
        return grp

    def _qr_panel(self) -> QWidget:
        """二维码小面板:有图就显示,没图就说明该把文件放哪儿。"""
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        cap = QLabel(f"<b>{author.WECHAT_ACCOUNT}</b><br>"
                     f"<small style='color:#666;'>{author.WECHAT_ACCOUNT_EN} · "
                     "微信扫一扫</small>")
        cap.setAlignment(Qt.AlignCenter)
        lay.addWidget(cap)

        path = author.qr_image_path()
        pix = QPixmap(path) if path else QPixmap()
        if not pix.isNull():
            img = QLabel()
            img.setPixmap(pix.scaled(180, 180, Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))
            img.setAlignment(Qt.AlignCenter)
            img.setToolTip("点击放大二维码")
            img.setCursor(Qt.PointingHandCursor)
            img.mousePressEvent = lambda _ev: self._show_wechat()  # noqa: ARG005
            lay.addWidget(img)
        else:
            miss = QLabel(
                f"未找到二维码图片。<br>把 <code>{author.WECHAT_QR_FILENAME}</code> "
                "放到程序目录或 <code>docs/</code> 下即可显示。")
            miss.setWordWrap(True)
            miss.setFixedWidth(190)
            miss.setAlignment(Qt.AlignCenter)
            miss.setStyleSheet("color:#777;font-size:8.5pt;")
            lay.addWidget(miss)
        return panel

    def _quick_start_group(self) -> QGroupBox:
        grp = QGroupBox("使用说明（快速上手）  /  Quick start")
        lay = QVBoxLayout(grp)
        for title, body in _QUICK_START:
            lbl = QLabel(f"<b>{title}</b>　{body}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size:9.5pt;")
            lay.addWidget(lbl)
        hint = QLabel(
            "完整说明:<b>帮助 → 使用说明</b>(快捷键 <b>F1</b>);"
            "命令行帮助 <code>python -m shsynth synth --help</code>。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#357;font-size:9pt;")
        lay.addWidget(hint)
        return grp

    # ------------------------------------------------------------- 动作
    def _open_guide(self):
        """在窗口里显示说明书(与主界面「帮助 → 使用说明」同一个窗口)。"""
        from .docs_window import GuideDialog
        GuideDialog(self, html_path=author.guide_html_path()).exec()

    def _show_wechat(self):
        path = author.qr_image_path()
        if not path:
            QMessageBox.information(
                self, author.WECHAT_QR_CAPTION,
                f"未找到二维码图片 {author.WECHAT_QR_FILENAME}。\n"
                f"公众号:{author.WECHAT_ACCOUNT}({author.WECHAT_ACCOUNT_EN})")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(author.WECHAT_QR_CAPTION)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)
        pix = QPixmap(path)
        lab = QLabel()
        lab.setPixmap(pix.scaled(360, 360, Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation))
        lab.setAlignment(Qt.AlignCenter)
        lay.addWidget(lab)
        cap = QLabel(author.WECHAT_QR_CAPTION)
        cap.setAlignment(Qt.AlignCenter)
        lay.addWidget(cap)
        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(dlg.reject)
        lay.addWidget(box)
        dlg.exec()


def run_gui(argv: Optional[list] = None, preload: Optional[str] = None) -> int:
    """启动界面;``preload`` 是启动时预载的系数文件。"""
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("SHSynth")
    _apply_cjk_font(app)
    win = MainWindow()
    if preload:
        win.coeffs_edit.setText(preload)
        win.load_coeffs_info()
    win.show()
    return app.exec()


if __name__ == "__main__":                               # pragma: no cover
    sys.exit(run_gui())
