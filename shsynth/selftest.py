# -*- coding: utf-8 -*-
"""
shsynth.selftest
================

**冻结版自检**:进程"活着"不代表没崩 —— 如果崩在导入期,窗口根本不会出现,
而 ``Start-Process`` 可能已经拿到句柄、``HasExited`` 还是 ``false``。所以打包好的
exe 必须自己走一遍关键路径并以退出码表态。

这个模块被两处共用:

* ``SHSynth.exe --self-test``(安装包/绿色版里的冻结程序);
* ``python -m shsynth.selftest``(源码树里,CI/回归也能跑)。

覆盖的关键路径(全部走**真实代码**,不做假):

1. 数据文件:勒夫数表、离线海岸线、公众号二维码、许可文本是否真在包里;
2. 系数读写:五种布局往返 + 与写入值逐位比对;
3. 综合:解析解校验(``C₁₀ = 1/√3 → sinφ``)+ 全球加权平均 = ``C₀₀``;
4. 物理量换算与高斯平滑(与"先换算/先平滑再综合"逐位一致);
5. 结果落盘与回读:``.nc``(含 HDF5 网络盘回退)、``.grd``、``.npy``、三列 csv;
6. 绘图:地图/报告图/谱/直方图都能渲染出坐标轴,且能存成 PNG;
7. 界面:建主窗口 → 载入系数 → 走后台线程解算 → 四个画布都有内容 → 存图
   (用 offscreen 平台,不需要真的看得到窗口)。

任何一项失败都会打印 ``[FAIL]`` 并让退出码非零,打包脚本据此拒绝出包。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

import numpy as np

__all__ = ["run_self_test", "SelfTestResult"]

_PASS = "[PASS]"
_FAIL = "[FAIL]"


class SelfTestResult:
    """收集自检结果(供打包脚本解析)。

    ``verbose=True`` 时**每一条检查都实时打印** —— 打包脚本按 ``[PASS]``/``[FAIL]``
    前缀统计,日志里也就能直接看到跑了哪些项(只打印汇总的话,出问题时无从查起)。
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.lines: list = []
        self.passed = 0
        self.failed = 0

    def _emit(self, line: str) -> None:
        self.lines.append(line)
        if self.verbose:
            print(line, flush=True)

    def ok(self, label: str, cond: bool, detail: str = "") -> bool:
        line = f"{_PASS if cond else _FAIL} {label}"
        if detail:
            line += f"  {detail}"
        self._emit(line)
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        return bool(cond)

    def section(self, title: str) -> None:
        self._emit(f"---- {title} ----")

    def failed_lines(self) -> list:
        return [ln for ln in self.lines if ln.startswith(_FAIL)]


def _utf8_stdout() -> None:
    """冻结版(windowed)的 stdout 是 GBK 管道,中文会乱码 —— 强制 UTF-8。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # pragma: no cover
        pass


def run_self_test(verbose: bool = True, keep_files: bool = False) -> int:
    """跑完整自检;返回 0(全过)或 1(有失败)。"""
    _utf8_stdout()
    res = SelfTestResult(verbose=verbose)
    out = (lambda s: print(s, flush=True)) if verbose else (lambda s: None)
    tmp = tempfile.mkdtemp(prefix="shsynth-selftest-")
    try:
        _check_environment(res, out)
        _check_data_files(res, out)
        coeffs = _check_coeffs_io(res, out, tmp)
        _check_physics(res, out, coeffs)
        _check_timeseries(res, out, tmp, coeffs)
        _check_horizontal(res, out, coeffs)
        _check_fft_path(res, out, coeffs)
        _check_animation(res, out, tmp, coeffs)
        _check_outputs(res, out, tmp, coeffs)
        _check_plots(res, out, tmp, coeffs)
        _check_gui(res, out, tmp, coeffs)
    except Exception as exc:                             # noqa: BLE001
        res.ok(f"自检过程中未抛异常({type(exc).__name__}: {exc})", False)
        if verbose:
            traceback.print_exc()
    finally:
        if not keep_files:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    if verbose:
        print("=" * 60, flush=True)
        print(f"自检:{res.passed} 项通过,{res.failed} 项失败", flush=True)
        for ln in res.failed_lines():
            print("  " + ln, flush=True)
    return 1 if res.failed else 0


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------
def _check_environment(res: SelfTestResult, out) -> None:
    import platform
    from . import __version__, author
    res.section("环境")
    res.ok(f"版本 {__version__}({author.APP_NAME} {author.APP_NAME_CN})", True)
    res.ok(f"Python {platform.python_version()} / {platform.machine()}",
           sys.version_info >= (3, 10))
    res.ok("运行方式是冻结版" if getattr(sys, "frozen", False) else
           "运行方式是源码(非冻结)", True,
           os.path.abspath(sys.executable))
    import numpy
    res.ok(f"numpy {numpy.__version__}", hasattr(numpy, "ndarray"))
    for mod in ("scipy", "matplotlib", "PySide6"):
        try:
            m = __import__(mod)
            res.ok(f"{mod} {getattr(m, '__version__', '?')}", True)
        except Exception as exc:                         # noqa: BLE001
            res.ok(f"导入 {mod}", False, f"{type(exc).__name__}: {exc}")
    for mod in ("xarray", "netCDF4", "pandas"):
        try:
            m = __import__(mod)
            res.ok(f"可选依赖 {mod} {getattr(m, '__version__', '?')}", True)
        except Exception:                                # noqa: BLE001
            res.ok(f"可选依赖 {mod}(缺失,相关格式不可用)", False)
    out("  环境检查完成")


def _check_data_files(res: SelfTestResult, out) -> None:
    from . import author
    from .lovenumbers import load_lln, load_love_numbers
    res.section("随包数据")
    kl = load_love_numbers()
    res.ok(f"载荷勒夫数 k′ 表({kl.size} 阶)", kl.size > 50)
    lln = load_lln()
    res.ok("载荷勒夫数 h′/l′/k′ 齐全",
           all(k in lln and np.size(lln[k]) > 50 for k in ("h", "l", "k")))
    res.ok(f"k′₂ ≈ -0.30516(实测 {kl[2]:.6f})", abs(kl[2] + 0.30516) < 1e-4)
    try:
        from . import plotting
        c = plotting.load_coastlines("coastline")
        res.ok(f"离线海岸线({0 if c is None else c[0].size} 顶点)", c is not None)
        res.ok("国界数据已移除(只画海岸线)",
               plotting.load_coastlines("borders") is None)
    except Exception as exc:                             # noqa: BLE001
        res.ok("离线海岸线", False, f"{type(exc).__name__}: {exc}")
    qr = author.qr_image_path()
    res.ok(f"公众号二维码({os.path.basename(qr) if qr else '缺失'})", qr is not None)
    from . import coeffio
    res.ok(f"系数格式清单({len(coeffio.FORMAT_SPECS)} 种布局)",
           len(coeffio.FORMAT_SPECS) >= 5)
    out("  数据文件检查完成")


def _make_coeffs(nmax: int = 8):
    from .coeffs import SHCoeffs
    rng = np.random.default_rng(20260913)
    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    for n in range(nmax + 1):
        for m in range(n + 1):
            C[n, m] = rng.normal(0, 1e-4)
            S[n, m] = rng.normal(0, 1e-4)
    C[0, 0] = -1.25
    return SHCoeffs(C, S, {"field_unit": "geopotential", "modelname": "selftest"})


def _check_coeffs_io(res: SelfTestResult, out, tmp: str):
    from . import read_coeffs, write_coeffs
    from .coeffio import detect_coeff_layout
    res.section("系数读写(五种布局)")
    src = _make_coeffs()
    cases = [(".sh", "triangle"), (".csv", "gmfcsv"), (".gfc", "gfc"),
             (".npy", "npy"), (".npz", "npz"), (".txt.gz", "triangle")]
    for ext, layout in cases:
        p = os.path.join(tmp, f"selftest_{layout}{ext.replace('.', '_')}{ext}")
        try:
            write_coeffs(src, p, layout=layout)
            back = read_coeffs(p)
            same = (np.array_equal(back.C, src.C) and np.array_equal(back.S, src.S))
            det = detect_coeff_layout(p)
            res.ok(f"{layout:<8}{ext:<8} 往返逐位相同(识别为 {det})", same)
        except Exception as exc:                         # noqa: BLE001
            res.ok(f"{layout}{ext} 往返", False, f"{type(exc).__name__}: {exc}")
    out("  五种布局往返完成")
    return src


def _check_physics(res: SelfTestResult, out, coeffs) -> None:
    from . import evaluate, synthesis_grid
    from .coeffs import SHCoeffs
    from .units import convert
    res.section("数值内核")

    # 1) 解析解:C10 = 1/√3 → 场 = sinφ
    C = np.zeros((2, 2))
    S = np.zeros((2, 2))
    C[1, 0] = 1.0 / np.sqrt(3.0)
    lat = np.array([-75.0, -30.0, 0.0, 33.3, 60.0, 89.0])
    lon = np.linspace(0.0, 300.0, lat.size)
    v = evaluate(lat, lon, SHCoeffs(C, S, {}))
    err = float(np.max(np.abs(v - np.sin(np.deg2rad(lat)))))
    res.ok(f"解析解 C₁₀=1/√3 → sinφ(偏差 {err:.2e})", err < 1e-13)

    # 2) 全球加权平均 = C00(球带中心采样,中点法则)
    latv = -90.0 + (np.arange(90) + 0.5) * 2.0
    lonv = np.arange(0.0, 360.0, 2.0)
    g = synthesis_grid(latv, lonv, coeffs)
    w = np.cos(np.deg2rad(latv))
    mean = float(np.sum(g * w[:, None]) / (np.sum(w) * g.shape[1]))
    d = abs(mean - coeffs.C[0, 0])
    res.ok(f"全球加权平均 = C₀₀({mean:.8f} vs {coeffs.C[0, 0]:.8f},差 {d:.1e})",
           d < 1e-5)

    # 3) 高斯平滑与物理量换算:两条路径必须逐位一致
    a = evaluate(lat, lon, coeffs, gaussian_km=800.0)
    from .filters import apply_gaussian
    b = evaluate(lat, lon, apply_gaussian(coeffs, 800.0))
    res.ok("综合时高斯 = 先平滑再综合", np.array_equal(a, b))
    c1 = evaluate(lat, lon, coeffs, target_unit="ewh")
    c2 = evaluate(lat, lon, convert(coeffs, "ewh"))
    res.ok("综合时换算 = 先换算再综合",
           bool(np.allclose(c1, c2, rtol=1e-12, atol=0)))
    res.ok(f"EWH 因子量级正确(A₀≈1.17e7,实测 {c1[0] / max(abs(v[0]), 1e-30):.3g} 倍场)",
           abs(c1[0]) > 100.0)
    out("  数值内核检查完成")


# ---------------------------------------------------------------------------
# v2.0:时间轴 / 序列 / FFT / 水平形变
# ---------------------------------------------------------------------------
def _check_timeseries(res: SelfTestResult, out, tmp: str, coeffs) -> None:
    """时间轴口径 + 序列落盘往返 + 批量 ≡ 逐历元。"""
    from . import (TimeAxis, read_series_nc, synth_series, write_series_nc)
    res.section("v2.0 时间轴与批量")
    try:
        # 十进制年口径:闰年分母必须是 366(与 legacy *_TimeInfo.dat 一致)
        d2002 = np.datetime64("2002-04-18")
        d2004 = np.datetime64("2004-04-18")
        from .timeaxis import dec_year
        ok365 = abs(dec_year(d2004) - (2004 + 108 / 365.0)) > 1e-5
        res.ok("十进制年在闰年用 366 天做分母(与 legacy 一致)",
               abs(dec_year(d2002) - (2002 + 107 / 365.0)) < 1e-12 and ok365)
        r = __import__("shsynth.timeaxis", fromlist=["x"]).parse_filename_epoch
        r1 = r("GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc")
        res.ok("文件名规则1(年积日区间)", str(r1["start"]) == "2002-04-05")
        r2 = r("ITSG-Grace2018_n60_2002-04.gfc")
        res.ok("文件名规则2(年月)", r2["rule"] == "ym")
        r3 = r("ITSG-Grace2018_Kalman_n40_2002-04-01.gfc")
        res.ok("文件名规则3(年月日)", r3["rule"] == "ymd")
        hdr = {"time_coverage_start": ": 2002-04-05T00:00:00.00",
               "time_coverage_end": ": 2002-05-01T00:00:00.00",
               "time_period_of_data": "20020405 - 20020430 (mid: 20020418)"}
        ax = TimeAxis.from_gfc_headers([hdr], paths=["GSM-2_2002095-2002120_X.gfc"])
        res.ok("gfc 头 (mid: YYYYMMDD) 按日历日期解析(不是年积日)",
               str(ax.values[0])[:10] == "2002-04-18",
               str(ax.values[0])[:10])
        # 序列落盘往返
        nt = 6
        ax2 = TimeAxis.from_datetimes(
            [np.datetime64(f"2002-{m:02d}-15") for m in range(1, nt + 1)])
        C = np.repeat(coeffs.C[:, :, None], nt, axis=2)
        S = np.repeat(coeffs.S[:, :, None], nt, axis=2)
        c = type(coeffs)(C, S, dict(coeffs.meta), ax2)
        p = write_series_nc(c, os.path.join(tmp, "series.nc"))
        b = read_series_nc(p)
        res.ok("series_nc 往返:系数逐位、日期逐值",
               bool(np.array_equal(b.C, c.C)) and
               bool(np.array_equal(b.times.values, ax2.values)))
        # 批量 ≡ 逐历元
        from .engine import synthesis_grid
        latv = np.arange(-60.0, 61.0, 30.0)
        lonv = 360.0 * np.arange(24) / 24
        r = synth_series(c, lat_vec=latv, lon_vec=lonv, use_fft="no")
        worst = 0.0
        for k in range(nt):
            one = c.time_slice(k)
            worst = max(worst, float(np.abs(
                r.values[:, :, k] - synthesis_grid(latv, lonv, one,
                                                   chunk=10 ** 7)).max()))
        res.ok(f"批量一次算完 ≡ 逐历元循环(最大差 {worst:.1e})", worst < 1e-12)
        # 拟合一致性
        from .series import fit_trend_seasonal
        f = fit_trend_seasonal(c, periods=(1.0,), trend="linear")
        # 截距 + 趋势 + cos + sin = 4 个参数
        res.ok(f"系数域趋势拟合(满秩 {f.rank}/4)", f.rank == 4)
        # 零阶陷阱诊断
        from .series import check_degree0_trap
        C0 = np.zeros((3, 3)); C0[0, 0] = 1.0
        msg = check_degree0_trap(type(coeffs)(C0, np.zeros((3, 3))), "ewh")
        res.ok("C₀₀=1 → 报出 EWH 常量偏移 1.17e7", msg is not None
               and "1.173e+07" in msg)
    except Exception as exc:                             # noqa: BLE001
        res.ok(f"时间序列检查未抛异常({type(exc).__name__}: {exc})", False)
        if res.verbose:
            traceback.print_exc()
    out("  时间序列检查完成")


def _check_horizontal(res: SelfTestResult, out, coeffs) -> None:
    """水平形变:解析锚点 + 极点(不许 0/NaN)+ 因子量级。"""
    from . import SHCoeffs, love_horizontal_factors, synthesize_horizontal
    res.section("v2.0 水平形变 u_N/u_E")
    try:
        C = np.zeros((2, 2)); S = np.zeros((2, 2)); C[1, 0] = 1.0
        lat = np.array([-90.0, -60.0, 0.0, 30.0, 90.0])
        lon = np.zeros_like(lat)
        d = synthesize_horizontal(lat, lon, C, S, nmax=1, weights_scale=np.ones(2))
        err = float(np.abs(d["north"]
                           - np.sqrt(3.0) * np.cos(np.deg2rad(lat))).max())
        res.ok(f"解析锚点 纯 C₁₀:u_N = √3·cosφ(偏差 {err:.1e})", err < 1e-13)
        res.ok("纯 C₁₀ 的 u_E ≡ 0", float(np.abs(d["east"]).max()) < 1e-14)
        C2 = np.zeros((2, 2)); C2[1, 1] = 1.0
        lat2 = np.array([90.0, 45.0, 0.0])
        lon2 = np.array([90.0, 90.0, 180.0])
        d2 = synthesize_horizontal(lat2, lon2, C2, np.zeros((2, 2)), nmax=1,
                                  weights_scale=np.ones(2))
        pole = float(d2["east"][0])
        res.ok(f"极点上 u_E = −√3(实测 {pole:.9f},不是 0/NaN)",
               abs(pole + np.sqrt(3.0)) < 1e-12)
        F = love_horizontal_factors(60)
        ratio = abs(F[1]) / abs(6378136.46 * (-0.285668) / (1 + 0.026168))
        res.ok(f"水平/径向因子同量级(比 {ratio:.3f},不是 1e7 倍)",
               1e-3 < ratio < 1.0)
    except Exception as exc:                             # noqa: BLE001
        res.ok(f"水平形变检查未抛异常({type(exc).__name__}: {exc})", False)
        if res.verbose:
            traceback.print_exc()
    out("  水平形变检查完成")


def _check_fft_path(res: SelfTestResult, out, coeffs) -> None:
    """FFT 经度路径:与直接法一致 + 经度对齐 + 适用条件回退。"""
    from . import (fft_path_applicable, synthesis_grid, synthesis_grid_fft)
    res.section("v2.0 FFT 经度快路径")
    try:
        L = min(coeffs.nmax, 20)
        lat = -90.0 + (np.arange(19) + 0.5) * (180.0 / 19)
        lon = 360.0 * np.arange(72) / 72
        a = synthesis_grid_fft(lat, lon, coeffs, nmax=L)
        b = synthesis_grid(lat, lon, coeffs, nmax=L, chunk=10 ** 7)
        rel = float(np.abs(a - b).max() / np.abs(b).max())
        res.ok(f"FFT 路径 ≡ 直接法(相对 {rel:.1e})", rel < 1e-12)
        ok1, _ = fft_path_applicable(np.arange(70.0, 141.0), L)
        res.ok("区域经度 → 自动判定不可用(回退直接法)", not ok1)
        ok2, why2 = fft_path_applicable(360.0 * np.arange(20) / 20, 20)
        res.ok("nlon ≤ 2·nmax → 判定不可用(Nyquist)", not ok2
               and "Nyquist" in why2)
        # 经度对齐:必须用 m≥1 的系数(m=0 对经度旋转不敏感)
        C1 = np.zeros((L + 1, L + 1)); C1[1, 1] = 1.0
        c1 = type(coeffs)(C1, np.zeros((L + 1, L + 1)), {})
        g = synthesis_grid_fft(lat, lon, c1)
        ref = synthesis_grid(lat, lon, c1, chunk=10 ** 7)
        d0 = float(np.abs(g - ref).max())
        d1 = float(np.abs(np.roll(g, 1, axis=1) - ref).max())
        res.ok(f"经度零格平移下才一致(对齐 {d0:.1e},平移一格 {d1:.1e})",
               d0 < 1e-12 and d1 > 1e-3)
    except Exception as exc:                             # noqa: BLE001
        res.ok(f"FFT 检查未抛异常({type(exc).__name__}: {exc})", False)
        if res.verbose:
            traceback.print_exc()
    out("  FFT 快路径检查完成")


def _check_animation(res: SelfTestResult, out, tmp: str, coeffs) -> None:
    """动画(v2.0):帧数 = 历元数、配色固定、可复现、缺依赖不静默。"""
    import os
    from . import TimeAxis, SHCoeffs
    from .plotting import make_series_frame, save_animation
    res.section("v2.0 动画(GIF)")
    try:
        L = min(coeffs.nmax, 12)
        nt = 4
        ax = TimeAxis.from_datetimes(
            [np.datetime64(f"2005-{k + 1:02d}-15") for k in range(nt)])
        C = np.repeat(coeffs.C[:L + 1, :L + 1, None], nt, axis=2) \
            * (1.0 + 0.3 * np.arange(nt))[None, None, :]
        S = np.repeat(coeffs.S[:L + 1, :L + 1, None], nt, axis=2) \
            * (1.0 + 0.3 * np.arange(nt))[None, None, :]
        seq = SHCoeffs(C, S, dict(coeffs.meta), ax)
        from .engine import synthesis_grid
        lat = -90.0 + (np.arange(19) + 0.5) * (180.0 / 19)
        lon = 360.0 * np.arange(36) / 36
        v = synthesis_grid(lat, lon, seq)
        if v.ndim == 2:
            v = v[:, :, None]

        p1 = os.path.join(tmp, "anim1.gif")
        p2 = os.path.join(tmp, "anim2.gif")
        info = {}
        save_animation(
            p1, (make_series_frame(ax, lat, lon, v, k, title="自检",
                                   vmin=-1.0, vmax=1.0) for k in range(nt)),
            fps=5.0, info=info)
        try:
            from PIL import Image
            with Image.open(p1) as im:
                res.ok(f"GIF 帧数 = 历元数({im.n_frames}/{nt})", im.n_frames == nt)
                res.ok(f"GIF 默认无限循环(loop={im.info.get('loop')})",
                       im.info.get("loop") == 0)
        except ImportError:
            res.ok("Pillow 缺失(≤)但 save_animation 未抛异常", True)
        # 可复现:同样的输入两次编码应当逐字节一致
        save_animation(
            p2, (make_series_frame(ax, lat, lon, v, k, title="自检",
                                   vmin=-1.0, vmax=1.0) for k in range(nt)),
            fps=5.0)
        with open(p1, "rb") as f1, open(p2, "rb") as f2:
            same = f1.read() == f2.read()
        res.ok("同一输入两次导出的 GIF 逐字节一致(可复现)", same)
        res.ok(f"帧率如实回报({info.get('fps_effective')} fps)",
               abs(info.get("fps_effective", 0) - 5.0) < 1e-9)

        # 帧号越界必须报错,不许静默画第一帧
        try:
            make_series_frame(ax, lat, lon, v, nt)
            res.ok("帧号越界报错", False)
        except IndexError:
            res.ok("帧号越界报错(不静默回绕)", True)
        # 视频格式缺 imageio 时必须**明确报错**,不降级成静止图
        try:
            save_animation(os.path.join(tmp, "anim.mp4"),
                           [np.zeros((8, 8, 3), np.uint8)] * 2)
            res.ok("缺 imageio 时写 mp4 明确报错(不静默降级)", False)
        except RuntimeError as exc:
            res.ok("缺 imageio 时写 mp4 明确报错,并指路 .gif",
                   ".gif" in str(exc))
    except Exception as exc:                             # noqa: BLE001
        res.ok(f"动画检查未抛异常({type(exc).__name__}: {exc})", False)
        if res.verbose:
            traceback.print_exc()
    out("  动画检查完成")


def _check_outputs(res: SelfTestResult, out, tmp: str, coeffs) -> None:
    from . import fieldio, read_coeffs
    from .engine import synthesis_grid
    res.section("结果落盘与回读")
    latv = np.arange(-80.0, 81.0, 4.0)
    lonv = np.arange(0.0, 360.0, 4.0)
    g = synthesis_grid(latv, lonv, coeffs)
    for ext in (".nc", ".grd", ".npy", ".csv"):
        p = os.path.join(tmp, f"field{ext}")
        try:
            fieldio.write_grid(p, latv, lonv, g, var="value",
                               meta={"lat_order_flipped": False,
                                     "producer": "selftest"})
            la, lo, g2, meta = fieldio.read_grid(p)
            g2 = g2[:, :, 0] if np.ndim(g2) == 3 else g2
            tol = 1e-6 if ext in (".csv", ".grd") else 1e-12
            same = (la.size == latv.size and lo.size == lonv.size
                    and float(np.max(np.abs(g2 - g))) <= tol)
            res.ok(f"网格 {ext:<5} 写出+回读一致(tol {tol:g})", same)
            if ext == ".nc" and meta.get("warnings"):
                res.ok("netCDF 走了临时文件中转(网络盘回退生效)", True,
                       "; ".join(str(w)[:40] for w in meta["warnings"]))
        except Exception as exc:                         # noqa: BLE001
            res.ok(f"网格 {ext} 写出+回读", False,
                   f"{type(exc).__name__}: {exc}")
    # 散点
    p = os.path.join(tmp, "points.csv")
    pts_lat = latv[:6]
    pts_lon = lonv[:6]
    try:
        fieldio.write_points(p, pts_lat, pts_lon, np.arange(6.0))
        la, lo, vals, meta = fieldio.read_points(p)
        res.ok(f"散点 csv 写出+回读一致({la.size} 点)",
               la.size == 6 and np.allclose(vals, np.arange(6.0)))
    except Exception as exc:                             # noqa: BLE001
        res.ok("散点 csv 写出+回读", False, f"{type(exc).__name__}: {exc}")
    # 系数再导出
    from . import write_coeffs
    p = os.path.join(tmp, "again.gfc")
    write_coeffs(coeffs, p, layout="gfc")
    back = read_coeffs(p)
    res.ok("系数再导出 .gfc 并读回(nmax 保持)",
           back.nmax == coeffs.nmax
           and np.allclose(back.C, coeffs.C, atol=1e-12))
    out("  落盘检查完成")


def _check_plots(res: SelfTestResult, out, tmp: str, coeffs) -> None:
    from . import plotting
    from .engine import synthesis_grid
    res.section("绘图")
    latv = np.arange(20.0, 51.0, 1.0)
    lonv = np.arange(100.0, 141.0, 1.0)
    g = synthesis_grid(latv, lonv, coeffs)
    try:
        fig = plotting.make_map_figure(latv, lonv, g, title="自检")
        ax = fig.get_axes()[0]
        xlim = ax.get_xlim()
        res.ok("区域地图自动聚焦(不是全球视图)",
               (xlim[1] - xlim[0]) < 60, f"x 范围 {xlim[0]:.0f}..{xlim[1]:.0f}")
        res.ok(f"地图上有 {len(ax.lines)} 条海岸线", len(ax.lines) > 0)
        p = os.path.join(tmp, "map.png")
        plotting.save_figure(fig, p, dpi=110)
        res.ok(f"地图存盘({os.path.getsize(p) // 1024} KB)",
               os.path.getsize(p) > 10_000)
        rpt = plotting.make_report_figure(latv, lonv, g, coeffs=coeffs,
                                          lat_vec=latv, lon_vec=lonv)
        res.ok(f"四联报告图 {len(rpt.get_axes())} 个坐标轴",
               len(rpt.get_axes()) >= 4)
        spec = plotting.make_spectrum_figure({"系数 RMS": coeffs.degree_rms()})
        res.ok("逐阶谱可渲染", len(spec.get_axes()) >= 1)
        hist = plotting.make_hist_figure(g.ravel())
        res.ok("数值分布可渲染", len(hist.get_axes()) >= 1)
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception as exc:                             # noqa: BLE001
        res.ok("绘图", False, f"{type(exc).__name__}: {exc}")
    out("  绘图检查完成")


def _pump_until(app, cond, timeout_s: float) -> bool:
    """跑事件循环直到 ``cond()`` 为真或超时(界面自检用)。"""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return bool(cond())


def _check_gui(res: SelfTestResult, out, tmp: str, coeffs) -> None:
    """界面路径:建窗口 → 载入系数 → 后台解算 → 渲染 → 存图(offscreen)。"""
    res.section("图形界面")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
        from . import author, write_coeffs
        from .gui.app import MainWindow
    except Exception as exc:                             # noqa: BLE001
        res.ok("导入界面模块", False, f"{type(exc).__name__}: {exc}")
        return
    app = QApplication.instance() or QApplication(sys.argv)
    win = None
    try:
        win = MainWindow()
        win.show()
        res.ok(f"主窗口建立({win.width()}x{win.height()})",
               win.width() > 600 and win.height() > 400)
    except Exception as exc:                             # noqa: BLE001
        res.ok("主窗口建立", False, f"{type(exc).__name__}: {exc}")
        return

    try:
        # 系数文件落盘后走界面真实路径(后台线程读)
        cin = os.path.join(tmp, "gui_coeffs.sh")
        write_coeffs(coeffs, cin, layout="triangle")
        win.coeffs_edit.setText(cin)
        win.load_coeffs_info()
        # 读系数走的是**另一个**后台线程(_info_thread),别只等解算线程
        _pump_until(app, lambda: win._info_thread is None
                    and win._coeffs is not None, 60)
        res.ok("界面读取系数(后台线程)", win._coeffs is not None,
               f"nmax={getattr(win._coeffs, 'nmax', '?')}")

        # 真跑一次解算(区域网格 + 报告图)
        win.mode_radios["range"].setChecked(True)
        win.lat_min.setValue(20.0)
        win.lat_max.setValue(50.0)
        win.lon_min.setValue(100.0)
        win.lon_max.setValue(140.0)
        win.lat_step.setValue(1.0)
        win.lon_step.setValue(1.0)
        win.figkind_combo.setCurrentIndex(0)
        out_nc = os.path.join(tmp, "gui_out.nc")
        fig_png = os.path.join(tmp, "gui_report.png")
        win.out_edit.setText(out_nc)
        win.figfile_edit.setText(fig_png)
        win.start_run()
        _pump_until(app, lambda: win._thread is None, 180)
        r = win._last_result
        res.ok("界面解算完成并拿到结果", r is not None)
        if r is not None:
            res.ok(f"求值点数 {r.stats['n_points']:,}", r.stats["n_points"] > 0)
            res.ok("界面写出结果文件", bool(r.out_path) and os.path.exists(r.out_path))
            res.ok(f"界面报告图 {len(win.report_canvas.figure.get_axes())} 个坐标轴",
                   len(win.report_canvas.figure.get_axes()) >= 4)
            res.ok(f"地图画布已渲染({len(win.map_canvas.figure.get_axes())} 坐标轴)",
                   len(win.map_canvas.figure.get_axes()) >= 1)
            res.ok("报告图存盘", os.path.exists(fig_png)
                   and os.path.getsize(fig_png) > 10_000,
                   f"{os.path.getsize(fig_png) // 1024 if os.path.exists(fig_png) else 0} KB")
            res.ok("结果对象带回绘图对象", r.figure is not None)
        # 运行后自动打开地图(默认开):上面选的是四联报告图,结束时应当停在地图页
        cur = win.tabs.currentWidget()
        res.ok("『运行后自动打开地图』默认开启",
               hasattr(win, "chk_autoshow") and win.chk_autoshow.isChecked())
        res.ok("解算后自动停在地图页(聚焦结果)", cur is win.map_canvas,
               f"当前页签 {type(cur).__name__}")
        if r is not None:
            # 关掉它 + 再走一次收尾 → 停在『图类型』选的那张图上
            win.chk_autoshow.setChecked(False)
            win._after_run(r)
            app.processEvents()
            res.ok("关掉后停在所选图类型(四联报告图)",
                   win.tabs.currentWidget() is win.report_canvas)
            win.chk_autoshow.setChecked(True)
        # 聚焦切换
        win.focus_btn.click()
        app.processEvents()
        res.ok("聚焦/全球视图切换可用", win.chk_focus.isChecked() is False)
        # 关于对话框(含作者信息与二维码)
        from PySide6.QtWidgets import QGroupBox, QLabel
        from .gui import app as gui_app
        dlg = gui_app.AboutDialog(win)
        text = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel)) + \
            " ".join(g.title() for g in dlg.findChildren(QGroupBox))
        res.ok("关于对话框含作者信息",
               author.AUTHOR_NAME_CN in text and author.AUTHOR_EMAIL in text)
        res.ok("关于对话框含公众号二维码", author.qr_image_path() is not None)
        dlg.close()
    except Exception as exc:                             # noqa: BLE001
        res.ok("界面解算路径", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        try:
            if win is not None:
                win.close()
        except Exception:                                # pragma: no cover
            pass
    out("  界面检查完成")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep = "--keep" in argv
    return run_self_test(verbose=True, keep_files=keep)


if __name__ == "__main__":                               # pragma: no cover
    sys.exit(main())
