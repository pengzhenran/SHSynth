# -*- coding: utf-8 -*-
"""
SHSynth — 球谐系数解算(综合)
=============================

输入:**球谐系数** + **网格或散点位置**
输出:**网格或散点值** + **图**

它只做"系数 → 场"这一个方向(综合/synthesis),但把这一件事做完整:

* **系数读取**:适配 SHKit 的**全部**球谐系数输出格式
  (``triangle`` / ``gmfcsv`` / ``gfc``(ICGEM) / ``npy`` / ``npz``,含 ``.gz``、
  中文表头、文件头元数据),``layout='auto'`` 自动识别;
* **位置输入**:规则经纬网格(给范围+步长)、已有网格文件(nc/grd/npy/csv)、
  散点文件(csv/txt/npy/xlsx)、Fibonacci 全球准均匀点;
* **综合引擎**:4π 归一化、无 CS 相位,与 SHKit / SHTOOLS(``norm=1, csphase=1``)
  逐位一致;只保留两列勒让德函数 + 自动分块,百万点也不爆内存;支持多时次;
* **物理量换算**:geoid / EWH / 面密度 σ / 径向形变 u_r,逐阶因子(不是常数!),
  标签不明时**拒绝**换算而不是给出差 1e7 倍的结果;
* **高斯平滑**:综合时逐阶乘 W_n,不改动存下来的系数;
* **输出**:nc / grd(Surfer) / csv / txt / npy,以及系数再导出(SHKIT 五种布局);
* **绘图**:地图(带离线海岸线)、散点、逐阶谱、差值图、直方图,PNG/PDF/SVG。

模块
----
``shsynth.coeffs``     :class:`SHCoeffs` 容器与三角布局
``shsynth.coeffio``    **所有 SHKit 系数格式**的读写
``shsynth.engine``     勒让德递推与综合(任意点 / 任意网格 / 分块)
``shsynth.filters``    高斯平滑
``shsynth.units``      物理量标签与逐阶换算
``shsynth.lovenumbers``载荷勒夫数表 h′/l′/k′
``shsynth.fieldio``    散点与网格文件的读写
``shsynth.targets``    目标几何(网格 / 散点)解析
``shsynth.plotting``   matplotlib 绘图
``shsynth.workflow``   一站式流程(读系数 → 综合 → 落盘 → 出图)
``shsynth.cli``        命令行
``shsynth.gui``        PySide6 桌面界面
"""

from __future__ import annotations

__version__ = "2.0.0"

from .coeffs import SHCoeffs
from .coeffio import (COEFF_EXTENSIONS, COEFF_LAYOUTS, detect_coeff_layout,
                      read_coeffs, read_coeffs_series, read_series_dat,
                      read_series_nc, scan_gfc_dir, write_coeffs,
                      write_series_dat, write_series_nc)
from .engine import (DEFAULT_CHUNK, DEFAULT_TIME_CHUNK, evaluate,
                     evaluate_horizontal, fft_path_applicable,
                     fibonacci_points, horizontal_grid_fft,
                     legendre_columns, legendre_columns_vec, legendre_pbar,
                     love_horizontal_factors, q_at_pole_m1, regular_grid,
                     synthesize, synthesize_horizontal, synthesis_grid,
                     synthesis_grid_fft)
from .filters import (EARTH_RADIUS_M, RHO_AVE, RHO_WATER,
                      apply_degree_filter, apply_gaussian,
                      gaussian_coefficients)
from .series import (GRACE_MEAN_FROM, GRACE_MEAN_TO, MEAN_MODES, FitResult,
                     SeriesResult, basin_average, basin_compare,
                     check_degree0_trap, check_static_dominance, drop_degree0,
                     estimate_memory, fit_trend_seasonal, mean_window,
                     remove_mean, remove_mean_window, remove_reference,
                     series_at_points, synth_series)
from .timeaxis import TimeAxis, dec_year, parse_filename_epoch
from .units import (CANONICAL, FIELD_UNIT_LABELS, FIELD_UNITS, convert,
                    degree_factors, normalise_unit)

__all__ = [
    "__version__",
    # 容器
    "SHCoeffs",
    # 时间轴(v2.0)
    "TimeAxis", "dec_year", "parse_filename_epoch",
    # 系数 I/O(全部 SHKit 格式 + v2.0 序列布局)
    "read_coeffs", "write_coeffs", "detect_coeff_layout",
    "COEFF_EXTENSIONS", "COEFF_LAYOUTS",
    "read_coeffs_series", "scan_gfc_dir",
    "read_series_nc", "write_series_nc", "read_series_dat", "write_series_dat",
    # 综合
    "synthesize", "synthesis_grid", "evaluate", "regular_grid",
    "fibonacci_points", "legendre_columns", "legendre_pbar",
    "DEFAULT_CHUNK",
    # 综合 —— v2.0:FFT 快路径
    "synthesis_grid_fft", "fft_path_applicable", "DEFAULT_TIME_CHUNK",
    # 综合 —— v2.0:水平形变
    "legendre_columns_vec", "q_at_pole_m1", "love_horizontal_factors",
    "synthesize_horizontal", "evaluate_horizontal", "horizontal_grid_fft",
    # 批量与产品(v2.0)
    "synth_series", "SeriesResult", "series_at_points", "basin_average",
    "basin_compare",
    "fit_trend_seasonal", "FitResult", "estimate_memory",
    "remove_reference", "remove_mean", "drop_degree0",
    # 去均值口径(v2.0:GRACE 惯例 / 全时段 / 自定义)
    "remove_mean_window", "mean_window", "MEAN_MODES",
    "GRACE_MEAN_FROM", "GRACE_MEAN_TO",
    "check_degree0_trap", "check_static_dominance",
    # 滤波与物理量
    "gaussian_coefficients", "apply_gaussian", "apply_degree_filter",
    "convert", "degree_factors", "normalise_unit",
    "FIELD_UNITS", "FIELD_UNIT_LABELS", "CANONICAL",
    "EARTH_RADIUS_M", "RHO_AVE", "RHO_WATER",
]
