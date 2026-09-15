# -*- coding: utf-8 -*-
"""
shsynth.cli
===========

命令行入口::

    python -m shsynth synth --coeffs model.sh --global-grid 1 \
        --out out/field.nc --figure out/map.png

    python -m shsynth synth --coeffs model.gfc --points stations.csv \
        --out out/points.csv --figure-kind report --figure out/report.png

    python -m shsynth info    --coeffs model.sh --spectrum
    python -m shsynth convert --coeffs model.sh --to-unit ewh --layout gfc \
        --out model_ewh.gfc
    python -m shsynth formats
    python -m shsynth selftest            # 与 SHKit 逐位比对(装了才跑)

装了包之后可以直接 ``shsynth ...``。
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Optional

import numpy as np

from . import __version__
from .coeffio import (COEFF_EXTENSIONS, COEFF_LAYOUTS, FORMAT_SPECS, coeff_info,
                      detect_coeff_layout, read_coeffs, read_coeffs_series,
                      write_coeffs)
from .engine import DEFAULT_CHUNK, DEFAULT_TIME_CHUNK
from .fieldio import GRID_EXTENSIONS, POINT_EXTENSIONS
from .units import (FIELD_UNITS, FIELD_UNIT_LABELS, convert,
                    describe_conversion, field_unit, normalise_unit)
from .workflow import FIGURE_KINDS, SynthRequest, plan_text, run

PROG = "shsynth"


class CliError(Exception):
    """命令行用法/参数错误(打印中文说明并以退出码 2 结束)。"""


#: stdout 写不出去时收集到的文本(退出前尽力交出去)
_LOST_OUTPUT: list = []


def _out(msg: str = "") -> None:
    """打印一行。**写不出去也绝不抛异常。**

    ⚠️ 冻结版是 windowed exe:在 cmd 里跑 ``SHSynth.exe --cli …`` 时,
    ``sys.stdout`` 可能是一个非 None 但无效的句柄,``print()`` 会抛
    ``OSError: [Errno 22] Invalid argument``。这一刻通常是**计算开始之前**
    或**结果已算完、正要汇报**的时候 —— 让一次"打印"把整条命令带走,
    用户看到的就是"报错了,什么也没得到",即使结果文件其实能写出来。

    所以这里把打印降级为**尽力而为**:失败就把文本攒进 :data:`_LOST_OUTPUT`,
    由 :func:`main` 在退出前写到临时日志并给出路径。
    """
    try:
        print(msg, flush=True)
    except (OSError, ValueError, UnicodeError):
        # OSError: 句柄无效 / 管道已关;UnicodeError: 控制台编码放不下
        _LOST_OUTPUT.append(str(msg))


def _out_err(msg: str, exc: Optional[BaseException] = None) -> None:
    """往 stderr 写一行(同样尽力而为,失败再退回 stdout 的通道)。"""
    try:
        print(msg, file=sys.stderr, flush=True)
    except (OSError, ValueError, UnicodeError):
        _LOST_OUTPUT.append(str(msg))


def _hr(title: str = "", width: int = 68) -> None:
    _out("=" * width if not title else f"{title}".center(width, "="))


# ---------------------------------------------------------------------------
# synth
# ---------------------------------------------------------------------------
def cmd_synth(args) -> int:
    spec = SynthRequest(
        coeffs_path=args.coeffs,
        coeffs_layout=args.layout,
        coeffs_nmax=args.coeff_nmax,
        time=args.time,
        mode=args.mode,
        grid_source=args.grid_source,
        lat_min=args.lat_min, lat_max=args.lat_max,
        lon_min=args.lon_min, lon_max=args.lon_max,
        lat_step=args.lat_step, lon_step=args.lon_step,
        grid_file=args.grid_file or "", grid_var=args.grid_var,
        points_file=args.points or "",
        lat_col=args.lat_col, lon_col=args.lon_col,
        n_sphere_points=args.sphere_points,
        truncate_nmax=args.nmax,
        gaussian_km=args.gaussian_km,
        gaussian_method=args.gaussian_method,
        target_unit=args.target_unit,
        allow_unit_mismatch=args.allow_unit_mismatch,
        chunk=args.chunk,
        out_path=args.out or "",
        out_var=args.var,
        out_units=args.units or "",
        out_comment=args.comment or "",
        out_coeffs_path=args.out_coeffs or "",
        out_coeffs_layout=args.out_coeffs_layout,
        figure_path=args.figure or "",
        figure_kind=args.figure_kind,
        figure_dpi=args.dpi,
        cmap=args.cmap,
        contour=args.contour,
        symmetric=not args.asymmetric_range,
        coast=not args.no_coast,
        focus=args.focus,
    )

    if not args.quiet:
        _hr("SHSynth 球谐系数解算")
        _out(plan_text(spec))
        _out("")

    last = [0.0]

    def progress(msg: str, frac: float):
        if args.quiet:
            return
        if frac - last[0] >= 0.05 or frac >= 1.0:
            last[0] = frac
            sys.stdout.write(f"\r  [{frac * 100:5.1f}%] {msg[:64]:<64}")
            sys.stdout.flush()

    try:
        result = run(spec, progress=progress)
    finally:
        if not args.quiet:
            sys.stdout.write("\r" + " " * 78 + "\r")

    if not args.quiet:
        _out(result.summary)
    else:
        _out(f"{result.stats['n_points']} 点  "
             f"RMS={result.stats['rms']:.6g}  "
             f"-> {result.out_path or '(未写出)'}")
    return 0


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def cmd_info(args) -> int:
    coeffs = read_coeffs(args.coeffs, nmax=args.coeff_nmax, layout=args.layout)
    info = coeff_info(coeffs)
    _hr("系数信息")
    _out(f"文件          : {info['source_file']}")
    _out(f"识别布局      : {info['layout']}   (自动识别;适配 SHKit 全部输出格式)")
    _out(f"最高阶 nmax   : {info['nmax']}   独立系数 {info['ncoef']}")
    _out(f"时次数 ntime  : {info['ntime']}")
    _out(f"物理量声明    : {FIELD_UNIT_LABELS.get(info['field_unit'], info['field_unit'])}")
    for k in ("modelname", "norm", "gaussian_radius_km", "gaussian_km",
              "truncated_from", "converted_from", "max_degree_in_file"):
        if info.get(k) is not None:
            _out(f"{k:<14}: {info[k]}")
    if info["degree_rms_first"]:
        n0, v0 = info["degree_rms_first"]
        n1, v1 = info["degree_rms_last"]
        _out(f"逐阶 RMS      : n={n0} -> {v0:.6e}   ...   n={n1} -> {v1:.6e}")
    for w in info["warnings"]:
        _out(f"警告          : {w}")

    if args.spectrum:
        multi = coeffs.ntime > 1
        rms = coeffs.degree_rms(time="mean" if multi else None)
        pwr = coeffs.power(time="mean" if multi else None)
        _hr("逐阶谱(degree RMS / power)")
        if multi:
            _out(f"  (源文件含 {coeffs.ntime} 个时次;下表为**时间平均**谱;"
                 "逐历元谱见 series-* 子命令)")
        _out(f"{'n':>4} {'RMS':>18} {'power':>18} {'半波长 (km)':>14}")
        for n in range(coeffs.nmax + 1):
            half = 20015.0 / n if n > 0 else float("inf")
            _out(f"{n:>4} {rms[n]:>18.8e} {pwr[n]:>18.8e} "
                 f"{'--' if n == 0 else f'{half:>14.1f}'}")

    if args.unit_table:
        _out()
        _out(describe_conversion(min(coeffs.nmax, 8)))
    if args.comments and coeffs.meta.get("comment_header"):
        _hr("文件头注释")
        for ln in coeffs.meta["comment_header"]:
            _out("  " + str(ln))
    return 0


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------
def cmd_convert(args) -> int:
    coeffs = read_coeffs(args.coeffs, nmax=args.coeff_nmax, layout=args.layout)
    src = field_unit(coeffs)
    if args.nmax is not None and args.nmax != coeffs.nmax:
        coeffs = coeffs.truncate(args.nmax)
        _out(f"已截断到 nmax = {coeffs.nmax}"
             + (f"(原 {args.nmax} 阶)" if args.nmax > coeffs.nmax else ""))
    if args.gaussian_km:
        from .filters import apply_gaussian
        coeffs = apply_gaussian(coeffs, args.gaussian_km,
                                method=args.gaussian_method)
        _out(f"已施加高斯平滑: {args.gaussian_km:g} km "
             f"({args.gaussian_method})")
    if args.to_unit:
        target = normalise_unit(args.to_unit)
        coeffs = convert(coeffs, target)
        _out(f"物理量换算: {FIELD_UNIT_LABELS.get(src, src)} → "
             f"{FIELD_UNIT_LABELS.get(target, target)}")
    if args.set_unit:
        coeffs = coeffs.with_unit(normalise_unit(args.set_unit))
        _out(f"已打标签 field_unit = {coeffs.field_unit}")
    if not args.out:
        raise CliError("请给出 --out(要写成哪个文件;扩展名决定布局)")
    p = write_coeffs(coeffs, args.out, layout=args.layout_out,
                     comment=args.comment, fmt=args.fmt,
                     meta_extra={"producer": f"SHSynth {__version__}"})
    _out(f"已写出: {p}(布局 {detect_coeff_layout(p)})")
    return 0


# ---------------------------------------------------------------------------
# formats / selftest
# ---------------------------------------------------------------------------
def cmd_formats(args) -> int:
    _hr("SHSynth 支持的系数格式(与 SHKit 输出完全对齐)")
    for spec in FORMAT_SPECS:
        _out(f"[{spec['layout']:<8}] {spec['name']}")
        _out(f"    扩展名 : {', '.join(spec['ext'])}")
        _out(f"    来源   : {spec['source']}")
        _out(f"    说明   : {spec['detail']}")
        _out()
    _hr("支持的其它文件")
    _out(f"位置(散点) : {', '.join(POINT_EXTENSIONS)}")
    _out(f"网格       : {', '.join(GRID_EXTENSIONS)}")
    _out(f"系数       : {', '.join(COEFF_EXTENSIONS)}(文本与 gfc 还支持 .gz)")
    _hr("物理量(target-unit)")
    for k in FIELD_UNITS:
        _out(f"  {k:<20} {FIELD_UNIT_LABELS[k]}")
    return 0


def cmd_selftest(args) -> int:
    """内部自检:与 SHKit 逐位比对(装了 SHKit 才做跨软件比对)。"""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shkit_dir = args.shkit or os.path.join(os.path.dirname(here), "SHKit")
    ok = True

    _hr("SHSynth 自检")
    # 1) 解析式:线性场 P̄_10 = sqrt(3) sin φ
    from .engine import evaluate, legendre_pbar
    from .coeffs import SHCoeffs
    C = np.zeros((2, 2)); S = np.zeros((2, 2))
    C[1, 0] = 1.0 / np.sqrt(3.0)
    lat = np.array([-60.0, -10.0, 0.0, 33.3, 75.0])
    lon = np.zeros_like(lat)
    c = SHCoeffs(C, S, {})
    v = evaluate(lat, lon, c)
    ref = np.sin(np.deg2rad(lat))
    err = float(np.max(np.abs(v - ref)))
    _out(f"解析校验 P̄_10 线性场        : 最大偏差 {err:.3e}  -> "
         f"{'OK' if err < 1e-14 else 'FAIL'}")
    ok &= err < 1e-14

    # 2) 与 scipy 的 4π 归一化勒让德函数对照
    try:
        from math import factorial

        from scipy.special import lpmv            # 稳定的 ufunc 接口
        n = 8
        P = legendre_pbar(lat, n)[0]                 # 第一个点的所有 (n,m)
        from .coeffs import triangle_order
        m_vec, n_vec = triangle_order(n)
        worst = 0.0
        for k in range(len(m_vec)):
            m, nn = int(m_vec[k]), int(n_vec[k])
            x = np.sin(np.deg2rad(lat[0]))
            # scipy 的 lpmv 带 Condon–Shortley 相位,乘 (-1)^m 去掉
            val = float(lpmv(m, nn, x)) * (-1) ** m
            if m == 0:
                norm = np.sqrt(2 * nn + 1)
            else:
                norm = np.sqrt(2.0 * (2 * nn + 1) * factorial(nn - m)
                               / factorial(nn + m))
            worst = max(worst, abs(float(P[k]) - val * norm))
        _out(f"与 scipy 的 4π 归一化对照    : 最大偏差 {worst:.3e}  -> "
             f"{'OK' if worst < 1e-10 else 'FAIL'}")
        ok &= worst < 1e-10
    except ImportError:                              # pragma: no cover
        _out("scipy 不可用,跳过勒让德对照")

    # 3) 与 SHKit 比对
    if os.path.isdir(shkit_dir):
        # SHKit 的包在 ``<shkit_dir>/shkit``,所以 import 根是 **shkit_dir 本身**,
        # 不是它的父目录。以前这里插的是父目录,``import shkit`` 必然
        # ModuleNotFoundError —— 跨软件比对等于从来没跑过(静默变成"失败")。
        # 顺便兼容"用户把 import 根当成 --shkit"的给法。
        import_root = None
        for cand in (shkit_dir, os.path.dirname(os.path.abspath(shkit_dir))):
            if os.path.isdir(os.path.join(cand, "shkit")):
                import_root = cand
                break
        if import_root is None:
            _out(f"SHKit 目录里没有 shkit 包({shkit_dir}),跳过跨软件比对")
        else:
            sys.path.insert(0, import_root)
        try:
            import shkit.io as shio
            # 注意:SHKit 的 __init__ 把 synthesis 这个名字导出成了**函数**,
            # 所以 `import shkit.synthesis as x` 拿到的是函数而不是模块。
            from shkit.synthesis import synthesis as sh_synthesis
            cands = [os.path.join(shkit_dir, f) for f in
                     ("shkit_coeffs.sh", "shkit_coeffs.gfc",
                      "yantze_shkit_coeffs.gfc", "sample_data/truth_coeffs_20.sh")]
            cands = [p for p in cands if os.path.exists(p)]
            if not cands:
                _out("SHKit 目录里没有找到系数样例,跳过比对")
            rng = np.random.default_rng(7)
            la = rng.uniform(-90, 90, 300)
            lo = rng.uniform(0, 360, 300)
            for f in cands:
                mine, theirs = read_coeffs(f), shio.read_coeffs(f)
                same = np.array_equal(mine.C, theirs.C) and \
                    np.array_equal(mine.S, theirs.S)
                dv = float(np.max(np.abs(evaluate(la, lo, mine)
                                         - np.asarray(sh_synthesis(la, lo, theirs)))))
                good = same and dv == 0.0
                _out(f"与 SHKit 比对 {os.path.basename(f):<28}: "
                     f"系数逐位相同={same}  综合最大差={dv:.3g}  -> "
                     f"{'OK' if good else 'FAIL'}")
                ok &= good
            # 再比一次高斯平滑后的结果
            g1 = evaluate(la, lo, read_coeffs(cands[0]), gaussian_km=300.0)
            g2 = np.asarray(sh_synthesis(la, lo, shio.read_coeffs(cands[0]),
                                         gaussian_km=300.0))
            dg = float(np.max(np.abs(g1 - g2)))
            _out(f"与 SHKit 比对 高斯 300 km            : 最大差 {dg:.3g}  -> "
                 f"{'OK' if dg == 0.0 else 'FAIL'}")
            ok &= dg == 0.0
        except Exception as exc:                      # noqa: BLE001
            _out(f"SHKit 比对失败: {type(exc).__name__}: {exc}")
            ok = False
    else:
        _out(f"未找到 SHKit 目录({shkit_dir}),跳过跨软件比对")

    _hr("自检" + ("通过" if ok else "未通过"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# v2.0 series-*
# ---------------------------------------------------------------------------
def cmd_series_read(args) -> int:
    prod = tuple(p.strip().upper() for p in str(args.product).split(",") if p.strip())
    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        _hr("SHSynth 序列读取(gfc 目录 → 系数序列)")
        _out(f"输入          : {args.indir}")
        _out(f"产品前缀      : {', '.join(prod)}"
             + ("(混入其它产品会被拒绝)" if prod else "(不过滤)"))
    s = read_coeffs_series(
        args.indir, nmax=args.nmax, pattern=args.pattern, product=prod,
        mission=args.mission, center=args.center, rl=args.rl,
        recursive=not args.no_recursive, epoch_from=args.epoch_from,
        sort=args.sort, on_duplicate=args.on_duplicate,
        strict_nmax=not args.lenient_nmax)
    layout = "series_nc" if str(args.out).lower().endswith(".nc") else "series_dat"
    p = write_coeffs(s, args.out, layout=layout,
                     meta_extra={"producer": f"SHSynth {__version__}"})
    if quiet:
        _out(f"nmax={s.nmax} ntime={s.ntime} -> {p}")
        return 0
    _out(f"读出          : nmax={s.nmax}  ntime={s.ntime}  "
         f"物理量={FIELD_UNIT_LABELS.get(s.field_unit, s.field_unit)}")
    _out("")
    _out(s.times.summary() if s.times is not None else "(无时间轴)")
    _out("")
    _out(f"已写出序列    : {p}  (布局 {layout})")
    if layout == "series_dat":
        _out(f"  并写出时间表: {os.path.splitext(p)[0]}_TimeInfo.dat")
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write(s.times.summary() + "\n")
        _out(f"已写出摘要    : {args.summary}")
    ws = list(dict.fromkeys(str(w) for w in (s.meta.get("warnings") or []) if w))
    if ws:
        _out("")
        _out(f"警告 ({len(ws)} 条):")
        for w in ws:
            _out(f"  ! {w}")
    return 0


def _series_targets(args):
    """把 series-synth 的位置参数解析成 (lat_vec, lon_vec, points) 三者之一。"""
    from . import targets as T
    chosen = [bool(args.points), bool(args.grid_file),
              args.global_grid is not None]
    if sum(chosen) > 1:
        raise CliError("--points、--grid-file、--global-grid 只能给一个")
    if args.points:
        lat, lon, _meta = T.points_from_file(args.points, lat_col=args.lat_col,
                                             lon_col=args.lon_col)
        return None, None, (lat, lon)
    if args.grid_file:
        latv, lonv, _ref, _meta = T.grid_from_file(args.grid_file, var=args.grid_var)
        return latv, lonv, None
    if args.global_grid is not None:
        step = float(args.global_grid)
        latv, lonv, _ = T.grid_from_spec(-90.0, 90.0, 0.0, 360.0, step, step)
        return latv, lonv, None
    latv, lonv, _ = T.grid_from_spec(args.lat_min, args.lat_max, args.lon_min,
                                     args.lon_max, args.lat_step, args.lon_step)
    return latv, lonv, None


def _apply_prep(s, args):
    """显式预处理:``--drop-degree0`` 与去均值。默认**什么都不做**。

    去均值有三种口径(设计原则 7:不静默改数,所以都要用户显式选):

    * ``--remove-mean``            全时段(与 v2.0 早期行为**完全一致**)
    * ``--remove-mean-mode grace`` **GRACE 惯例**:2004-01-01 .. 2010-12-31
    * ``--remove-mean-mode custom --mean-from A --mean-to B``  自定义时段

    给了 ``--remove-mean-mode`` 就**隐含**要做去均值(不必再写 ``--remove-mean``)。
    """
    from .series import drop_degree0, remove_mean_window
    notes = []
    if getattr(args, "drop_degree0", None) is not None:
        s = drop_degree0(s, nmax_drop=int(args.drop_degree0))
        notes.append(f"已把 n<={args.drop_degree0} 的系数置零")
    mode = getattr(args, "remove_mean_mode", None)
    want = bool(getattr(args, "remove_mean", False)) or mode is not None
    if want:
        # --remove-mean(不带 mode)保持"全时段",与旧行为逐位一致
        if mode is None:
            mode = "all"
        try:
            s, rep = remove_mean_window(
                s, mode=mode,
                from_date=getattr(args, "mean_from", None),
                to_date=getattr(args, "mean_to", None))
        except ValueError as exc:
            # 窗口不合法(空窗口 / 缺日期 / 没有时间轴)是**用法问题**,不是崩溃:
            # 走 CliError → 退出码 2 + 中文说明,不要甩一个调用栈给用户。
            raise CliError(str(exc)) from exc
        notes.append(rep["summary"])
        for w in rep["warnings"]:
            notes.append(f"! {w}")
    return s, notes


def _add_prep_args(parser) -> None:
    """把「显式预处理」参数挂到一个序列子命令上(四个序列命令共用一套)。

    去均值的三种口径对**所有**序列产品都有意义(异常场的静态基准由它决定),
    所以别只给 series-synth 开。
    """
    g = parser.add_argument_group("预处理(默认都不做)")
    g.add_argument("--remove-mean", action="store_true",
                   help="先去掉**全时段**时间平均(得到异常场)。默认**不做**;"
                        "想按 GRACE 惯例(2004-2010)或自定义时段请用 "
                        "--remove-mean-mode")
    g.add_argument("--remove-mean-mode", default=None,
                   choices=("grace", "all", "custom"), metavar="MODE",
                   help="去均值口径(给了它就隐含要做去均值):"
                        "grace=GRACE 惯例(2004-01-01..2010-12-31)、"
                        "all=全时段(等同 --remove-mean)、"
                        "custom=自定义(必须同时给 --mean-from/--mean-to)")
    g.add_argument("--mean-from", default=None, metavar="DATE",
                   help="自定义去均值窗口起点 YYYY-MM-DD(也接受 YYYY-MM)")
    g.add_argument("--mean-to", default=None, metavar="DATE",
                   help="自定义去均值窗口终点 YYYY-MM-DD")
    g.add_argument("--drop-degree0", type=int, default=None, metavar="N",
                   help="把 n<=N 的系数置零。注意:GSM 里 C00=1、C20 等是**静态场**"
                        "(换成 EWH 约 4 万米),光去零阶不够;正式做法是去参考历元"
                        "或去均值")


def _report_prep(notes) -> None:
    """把预处理说明逐条打印(多行也只占一条,便于与其它行对齐)。"""
    for n in notes:
        _out(f"  预处理      : {n}")


def _read_series_input(path, layout: str = "auto", nmax=None, **kw):
    """序列子命令的输入:单个序列文件,**或者**一个装满 gfc 的目录。

    给目录时自动走 :func:`read_coeffs_series`(和 ``series-read`` 同一套识别
    逻辑),省得为了看一眼流域平均先手动导出一次。给文件时行为与 v1.0 一致。
    """
    if os.path.isdir(path):
        return read_coeffs_series(path, nmax=nmax, **kw)
    return read_coeffs(path, nmax=nmax, layout=layout)


def cmd_series_synth(args) -> int:
    from .series import (check_degree0_trap, estimate_memory, synth_series)
    s0 = _read_series_input(args.coeffs, args.layout, nmax=args.coeff_nmax)
    s, prep = _apply_prep(s0, args)
    latv, lonv, pts = _series_targets(args)
    if s.times is None:
        raise CliError(
            "这份系数没有时间轴,无法做序列综合。请先用 series-read 从 gfc 目录读入。")
    tgt = args.target_unit
    if args.component in ("north", "east", "horizontal"):
        if tgt is not None and normalise_unit(tgt) != "horizontal_displacement":
            raise CliError(
                f"--component {args.component} 是水平形变,但 --target-unit {tgt} "
                "不是 horizontal_displacement。\n  水平形变的逐阶因子是 "
                "R·l′ₙ/(1+k′ₙ),与 EWH/geoid 不同,混用会差好几个数量级。")
        tgt = "horizontal_displacement"
    _hr("SHSynth 批量综合(序列)")
    _out(f"系数          : {args.coeffs}")
    _out(f"  阶数/时次   : nmax={s.nmax}  ntime={s.ntime}")
    _out(f"  输出分量    : {args.component}")
    _out(f"  综合路径    : {args.use_fft}(auto = 能走 FFT 就走)")
    _report_prep(prep)
    _out("")
    last = [0.0]

    def prog(msg, frac):
        if args.quiet:
            return
        if frac - last[0] >= 0.05 or frac >= 1.0:
            last[0] = frac
            sys.stdout.write(f"\r  [{frac * 100:5.1f}%] {msg[:58]:<58}")
            sys.stdout.flush()

    res = synth_series(s, lat_vec=latv, lon_vec=lonv, points=pts,
                       nmax=args.nmax, gaussian_km=args.gaussian_km,
                       gaussian_method=args.gaussian_method,
                       target_unit=tgt, component=args.component,
                       chunk=args.chunk,
                       time_chunk=args.time_chunk, use_fft=args.use_fft,
                       progress=prog)
    if not args.quiet:
        sys.stdout.write("\r" + " " * 72 + "\r")
        _out(res.summary)
    scale = 1000.0 if args.scale == "mm" else 1.0
    units = args.units or ("mm" if args.scale == "mm" else "m")
    if args.out:
        v = res.values * scale
        if res.kind == "grid":
            from . import fieldio
            extra = None
            if res.components:
                extra = {}
                for k, a in res.components.items():
                    nm = "north_displacement" if k == "north" else "east_displacement"
                    extra[nm] = np.asarray(a) * scale
            p = fieldio.write_grid(args.out, res.lat, res.lon, v, var=args.var,
                                   units=units, times=res.times,
                                   time_encoding=args.time_encoding,
                                   extra_vars=extra,
                                   long_name="SHSynth series synthesis",
                                   meta={"method": res.stats.get("method"),
                                         "component": args.component,
                                         "epochs": len(res.times),
                                         "source_coeffs": os.path.basename(
                                             str(args.coeffs))})
            if extra:
                _out(f"已写出场序列  : {p}(含 {', '.join(extra)} 两个分量)")
            else:
                _out(f"已写出场序列  : {p}")
        else:
            from . import fieldio
            p = fieldio.write_points(args.out, pts[0], pts[1], v)
            _out(f"已写出场序列  : {p}")
    if args.out_summary:
        p = res.to_csv(args.out_summary)
        _out(f"已写出诊断表  : {p}")
    if not args.quiet:
        mem = estimate_memory(*( (res.lat.size, res.lon.size, len(res.times), s.nmax)
                                if res.kind == "grid" else (1,1,1,s.nmax) ))
        _out(f"内存参考      : 输出 {mem['output_bytes']/1e6:.1f} MB,"
             f"FFT 临时 {mem['fft_tmp_bytes']/1e6:.1f} MB")
        d0 = check_degree0_trap(s, tgt)
        if d0 and not prep:
            _out("")
            _out("⚠️ " + d0)
        elif d0:
            _out("")
            _out("(提示)源系数含 C₀₀=1,但已做预处理("
                 + "、".join(n.split(":")[0] for n in prep)
                 + "),本次输出是异常场)")

    if args.anim_out:
        _tgt_lab = FIELD_UNIT_LABELS.get(tgt or s.field_unit, tgt or "")
        comp_lab = {"scalar": "", "radial": "径向", "north": "北向",
                    "east": "东向", "horizontal": "水平"}[args.component]
        rc = _write_series_animation(
            args, res, res.values * scale,
            title=args.anim_title if args.anim_title is not None
            else " ".join(x for x in ("SHSynth", comp_lab, _tgt_lab) if x),
            cb_label=args.units or (f"{_tgt_lab}" if _tgt_lab else units))
        if rc:
            return rc
    return 0


def _write_series_animation(args, res, values, *, title: str = "",
                            cb_label: str = "") -> int:
    """把场序列写成 GIF(方案 §4.6 的 ``save_animation``)。

    ⚠️ **配色范围必须在整段序列上定一次**。逐帧各自定标的话,动画会随
    每历元的极值"闪",眼睛看到的"变化"其实是配色在变 —— 这是动画最常踩的坑,
    所以这里默认用**全序列**的稳健对称范围,并把实际范围打印出来。
    """
    from .plotting import make_series_frame, save_animation, _symmetric_range
    if res.kind != "grid":
        raise CliError("动画只支持网格序列;散点序列没有规则帧可画")
    stride = max(1, int(args.anim_stride))
    idx = list(range(0, values.shape[2], stride))
    vmin, vmax = args.anim_vmin, args.anim_vmax
    if vmin is None or vmax is None:
        a, b = _symmetric_range(values[:, :, idx], symmetric=True)
        vmin = a if vmin is None else vmin
        vmax = b if vmax is None else vmax
    _out("")
    _out(f"动画          : {len(idx)} 帧(stride={stride}),"
         f"{args.anim_fps:g} fps,配色固定 {vmin:.4g} .. {vmax:.4g}(全序列稳健范围)")

    def _frames(k):
        return make_series_frame(res.times, res.lat, res.lon, values, idx[k],
                                 title=title, cb_label=cb_label,
                                 vmin=vmin, vmax=vmax, focus=args.anim_focus)

    info = {}
    try:
        p = save_animation(args.anim_out, _frames, n_frames=len(idx),
                           fps=args.anim_fps, info=info,
                           progress=(None if args.quiet else
                                     (lambda d, n: sys.stdout.write(
                                         f"\r  渲染动画 {d}/{n} ..."))),
                           cancel=(lambda: False))
    except KeyboardInterrupt:
        _out("")
        _out("动画导出已取消")
        return 1
    if not args.quiet:
        sys.stdout.write("\r" + " " * 40 + "\r")
    _out(f"已写出动画    : {p}  ({info.get('bytes', 0) / 1e6:.1f} MB,"
         f"{info.get('frames')} 帧)")
    got, want = info.get("fps_effective"), info.get("fps_requested")
    if got is not None and want is not None and abs(got - want) > 1e-6:
        _out(f"  注意:GIF 延时只能取 10 ms 整数倍,实际帧率 {got:.3f} fps"
             f"(你给的是 {want:g})")
    return 0


def cmd_series_points(args) -> int:
    from . import targets as T
    s0 = _read_series_input(args.coeffs, args.layout)
    if s0.times is None:
        raise CliError("这份系数没有时间轴;请先用 series-read 读入带日期的序列")
    s, prep = _apply_prep(s0, args)
    _report_prep(prep)
    lat, lon, _ = T.points_from_file(args.points, lat_col=args.lat_col,
                                     lon_col=args.lon_col)
    scale = 1000.0 if args.scale == "mm" else 1.0
    if args.component in ("north", "east", "horizontal"):
        from .engine import evaluate_horizontal
        h = evaluate_horizontal(lat, lon, s, gaussian_km=args.gaussian_km)
        names = ["north", "east", "magnitude", "azimuth"]
        data = {k: np.asarray(h[k]) * (scale if k != "azimuth" else 1.0)
                for k in names}
    else:
        from .engine import evaluate
        v = np.asarray(evaluate(lat, lon, s, gaussian_km=args.gaussian_km,
                                target_unit=args.target_unit)) * scale
        names = [args.target_unit or "value"]
        data = {names[0]: v}
    npt = int(np.size(lat))
    nt = len(s.times)
    for k, a in data.items():
        if a.ndim == 1:
            data[k] = np.repeat(a[:, None], nt, axis=1)
    unit_note = "mm" if args.scale == "mm" else "m"
    p = os.fspath(args.out)
    if os.path.dirname(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    head = ["lon", "lat", "epoch", "time", "decimal_year"] + names
    with open(p, "w", encoding="utf-8-sig") as f:
        f.write(f"# SHSynth 站点时间序列  units: {unit_note}"
                f"  component: {args.component}\n")
        f.write(",".join(head) + "\n")
        latf = np.asarray(lat).ravel()
        lonf = np.asarray(lon).ravel()
        for i in range(npt):
            for k in range(nt):
                row = [f"{lonf[i]:.6f}", f"{latf[i]:.6f}", str(k),
                       str(s.times.values[k])[:10],
                       f"{s.times.decimal_years[k]:.6f}"]
                row += [f"{data[nm][i, k]:.10g}" for nm in names]
                f.write(",".join(row) + "\n")
    _out(f"站点 {npt} 个 × {nt} 个历元 → {p}(长表:一行 = 一个点一个历元)")
    from .series import check_static_dominance
    st = check_static_dominance(data[names[0]], s.times, "ewh" if
                                args.component == "scalar" else
                                "horizontal_displacement")
    if st:
        _out("")
        _out("⚠️ " + st)
    return 0


def cmd_series_fit(args) -> int:
    from . import fieldio
    from . import targets as T
    from .engine import synthesis_grid
    from .series import fit_trend_seasonal
    s, prep = _apply_prep(_read_series_input(args.coeffs, args.layout), args)
    periods = tuple(float(x) for x in str(args.periods).split(",") if x.strip())
    f = fit_trend_seasonal(s, periods=periods, trend=args.trend, sigma=args.sigma)
    _hr("系数域时间拟合")
    _report_prep(prep)
    _out(f.summary())
    latv, lonv, _ = T.grid_from_spec(-90.0, 90.0, 0.0, 360.0,
                                     args.lat_step, args.lat_step)
    if args.out_trend:
        if args.trend == "none":
            _out("(trend=none,不写趋势场)")
        else:
            g = synthesis_grid(latv, lonv, f.slope_coeffs())
            fieldio.write_grid(args.out_trend, latv, lonv, g, var="trend",
                               long_name="least-squares trend (coefficient/yr)")
            _out(f"已写出趋势场  : {args.out_trend}")
    if args.out_seasonal:
        for k, per in enumerate(periods):
            tag = f"{per:g}".replace(".", "p")
            base = args.out_seasonal
            stem = base[:-3] if base.lower().endswith(".nc") else base
            for j, lab in enumerate(("cos", "sin")):
                g = synthesis_grid(latv, lonv, f.field(f.period_offset + 2 * k + j))
                pp = f"{stem}_{lab}{tag}.nc"
                fieldio.write_grid(pp, latv, lonv, g, var=lab,
                                   long_name=f"{lab} term of {per:g}-year cycle")
                _out(f"已写出周期 {per:g} 年 {lab} 场: {pp}")
    if args.out_table:
        import csv
        with open(args.out_table, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["param", "index", "min", "max"])
            for i, name in enumerate(f.param_names):
                a = f.coef[i][:, :, 0]
                w.writerow([name, i, f"{np.nanmin(a):.10g}", f"{np.nanmax(a):.10g}"])
        _out(f"已写出参数表  : {args.out_table}")
    return 0


def _tlabel(t, i: int) -> str:
    """时间轴上第 ``i`` 个历元的短标签(给逐行表格用)。"""
    if getattr(t, "labels", None) and i < len(t.labels):
        return str(t.labels[i])[:12]
    if getattr(t, "kind", "") == "index":
        return f"#{i}"
    return str(t.values[i])[:10]


def cmd_series_basin(args) -> int:
    from .series import basin_average
    s, prep = _apply_prep(_read_series_input(args.coeffs, args.layout), args)
    mask = args.mask
    lat_vec = lon_vec = None
    if args.mask_lat_step or args.mask_lon_step:
        # 掩膜是"只有数值没有坐标轴"的裸 .npy 时,坐标轴只能由用户给出;
        # 少给的那个按"刚好铺满全球"推(dlat=180/nlat, dlon=360/nlon)。
        if not str(mask).lower().endswith(".npy"):
            raise ValueError("--mask-lat-step/--mask-lon-step 只用于裸 .npy 掩膜;"
                             "带坐标轴的文件(.nc/.grd/.csv)不需要")
        mask = np.load(str(mask))
        if mask.ndim == 3 and mask.shape[0] >= 3:
            raise ValueError(
                f"这张 .npy 是 {mask.shape} 的 [lon; lat; value] 堆叠布局,"
                "自带坐标轴 —— 请去掉 --mask-lat-step/--mask-lon-step")
        if mask.ndim != 2:
            raise ValueError(f"裸 .npy 掩膜必须是二维,实际 {mask.shape}")
        nlat, nlon = mask.shape
        dlat = args.mask_lat_step or (180.0 / nlat)
        dlon = args.mask_lon_step or (360.0 / nlon)
        lat_vec = -90.0 + (np.arange(nlat) + 0.5) * dlat
        lon_vec = np.arange(nlon) * dlon

    _hr("区域平均时间序列")
    _report_prep(prep)
    res = basin_average(s, mask, nmax=args.nmax, method=args.method,
                        mask_kernel=args.mask_kernel,
                        lat_vec=lat_vec, lon_vec=lon_vec)
    if args.method == "both":
        t, sp, co, diff = res
        scale = float(np.nanmax(np.abs(sp))) or 1.0
        _out(f"{'日期':<12} {'spatial':>16} {'coeff':>16} {'coeff-spatial':>16}")
        for i in range(len(t)):
            _out(f"{_tlabel(t, i):<12} {sp[i]:16.8g} {co[i]:16.8g} "
                 f"{diff[i]:16.3e}")
        _out("")
        _out(f"最大差 {np.nanmax(np.abs(diff)):.4g}  "
             f"RMS 差 {np.sqrt(np.nanmean(diff ** 2)):.4g}  "
             f"相对场量级 {np.nanmax(np.abs(diff)) / scale * 100:.4g}%")
        if args.mask_kernel is None:
            _out("提示:没给 --mask-kernel 时两个口径用的是同一套格点,此时它们")
            _out("      **代数上恒等**,差异应当只是浮点舍入(~1e-15 相对)。")
            _out("      要看真实差异,请用 --mask-kernel 给一张更细的掩膜建核。")
    else:
        t, vals = res
        _out(f"{'日期':<12} {'值':>16}")
        for i in range(len(t)):
            _out(f"{_tlabel(t, i):<12} {vals[i]:16.8g}")
    if args.out:
        import csv
        # 注意:不能用 float(t.values[i]) —— values 是 datetime64[ns],
        # 转 float 会得到纳秒整数(1.019e18),不是十进制年。
        dy = t.decimal_years
        with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            if args.method == "both":
                w.writerow(["date", "decimal_year", "spatial", "coeff", "diff"])
                for i in range(len(t)):
                    w.writerow([_tlabel(t, i), f"{dy[i]:.10f}",
                                f"{sp[i]:.10g}", f"{co[i]:.10g}", f"{diff[i]:.10g}"])
            else:
                w.writerow(["date", "decimal_year", "value"])
                for i in range(len(t)):
                    w.writerow([_tlabel(t, i), f"{dy[i]:.10f}",
                                f"{vals[i]:.10g}"])
        _out(f"已写出: {args.out}")
    return 0


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="SHSynth — 球谐系数解算:输入球谐系数 + 网格/散点位置,"
                    "输出网格/散点值并绘图。系数支持 SHKit 的全部输出格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m shsynth synth --coeffs model.sh --global-grid 1 \\\n"
            "      --out out/field.nc --figure out/map.png\n"
            "  python -m shsynth synth --coeffs model.gfc --gaussian-km 300 \\\n"
            "      --target-unit ewh --global-grid 0.5 --out out/ewh.nc\n"
            "  python -m shsynth synth --coeffs model.sh --points stations.csv \\\n"
            "      --out out/points.csv --figure-kind report --figure out/r.png\n"
            "  python -m shsynth info --coeffs model.sh --spectrum\n"
            "  python -m shsynth convert --coeffs model.sh --layout-out gfc \\\n"
            "      --to-unit ewh --out model_ewh.gfc\n"
            "  python -m shsynth formats\n"))
    p.add_argument("--version", action="version",
                   version=f"SHSynth {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="子命令")

    # ---------------------------------------------------------- synth
    s = sub.add_parser(
        "synth", help="综合:系数 + 位置 -> 值 + 图(主功能)",
        description="读入球谐系数,在规则网格、已有网格格点或散点上求值,"
                    "写出结果并绘图。")
    s.add_argument("--coeffs", required=True, metavar="FILE",
                   help="球谐系数文件(.sh .txt .csv .dat .tsv .gfc .npy .npz)")
    s.add_argument("--layout", default="auto",
                   choices=("auto", "triangle", "gmfcsv", "matrix", "gfc",
                            "npy", "npz"),
                   help="系数文件布局(默认 auto,自动识别 SHKit 全部输出格式)")
    s.add_argument("--coeff-nmax", type=int, default=None, metavar="N",
                   help="读取时按该阶截断(默认用文件里的最高阶)")
    s.add_argument("--time", type=int, default=None, metavar="K",
                   help="多时次时用第 K 个(默认第 0 个)")

    g = s.add_argument_group("目标位置")
    g.add_argument("--global-grid", type=float, metavar="DEG", default=None,
                   help="全球网格(0..360, 不含端点),给步长,例如 1 或 0.5")
    g.add_argument("--grid-file", default=None, metavar="FILE",
                   help="借用已有网格文件的格点作为求值位置")
    g.add_argument("--grid-var", default=None, metavar="NAME",
                   help="网格文件的变量名(netCDF)")
    g.add_argument("--lat-min", type=float, default=-90.0, metavar="DEG",
                   help="范围网格:起始纬度(默认 -90)")
    g.add_argument("--lat-max", type=float, default=90.0, metavar="DEG",
                   help="范围网格:终止纬度(默认 90)")
    g.add_argument("--lon-min", type=float, default=0.0, metavar="DEG",
                   help="范围网格:起始经度(默认 0)")
    g.add_argument("--lon-max", type=float, default=360.0, metavar="DEG",
                   help="范围网格:终止经度(默认 360;给 360 时按不含端点处理)")
    g.add_argument("--lat-step", type=float, default=1.0, metavar="DEG",
                   help="纬度步长(默认 1°)")
    g.add_argument("--lon-step", type=float, default=None, metavar="DEG",
                   help="经度步长(默认与 --lat-step 相同)")
    g.add_argument("--points", default=None, metavar="FILE",
                   help="散点位置文件(csv/txt/dat/tsv/npy/xlsx);"
                        "用它的经纬度作为求值点")
    g.add_argument("--lat-col", default=None, metavar="COL")
    g.add_argument("--lon-col", default=None, metavar="COL")
    g.add_argument("--sphere-points", type=int, default=20000, metavar="N",
                   help="不给 --points 时,在全球 Fibonacci 球面 N 点上求值")

    o = s.add_argument_group("综合选项")
    o.add_argument("--nmax", type=int, default=None, metavar="N",
                   help="截断到该阶(默认用系数自身的阶数)")
    o.add_argument("--gaussian-km", type=float, default=0.0, metavar="KM",
                   help="高斯平滑半径(km),在综合时施加,不改动系数文件")
    o.add_argument("--gaussian-method", default="glq", choices=("glq", "frc"))
    o.add_argument("--target-unit", default=None, metavar="UNIT",
                   help="输出物理量:不换算(默认)| geoid | ewh | surface_density "
                        "| radial_displacement,也接受中文(等效水高/水准面/面密度/"
                        "径向形变)")
    o.add_argument("--allow-unit-mismatch", action="store_true",
                   help="物理量声明矛盾时仍然强行换算(仅用于实验)")
    o.add_argument("--chunk", type=int, default=200000, metavar="N",
                   help="每块点数(内存控制,默认 200000)")

    w = s.add_argument_group("输出")
    w.add_argument("--out", default=None, metavar="FILE",
                   help="结果文件,扩展名决定格式:网格 .nc/.grd/.csv/.txt/.npy;"
                        "散点 .csv/.txt/.dat/.tsv/.npy")
    w.add_argument("--var", default="value", metavar="NAME",
                   help="netCDF 变量名(默认 value)")
    w.add_argument("--units", default=None, metavar="TEXT",
                   help="写进 netCDF/图上的单位说明(仅标注,不做换算)")
    w.add_argument("--comment", default=None, metavar="TEXT")
    w.add_argument("--out-coeffs", default=None, metavar="FILE",
                   help="顺带把(截断/平滑后的)系数写成另一个文件")
    w.add_argument("--out-coeffs-layout", default="auto",
                   choices=("auto", "triangle", "gmfcsv", "gfc", "npy", "npz"))

    f = s.add_argument_group("出图")
    f.add_argument("--figure", default=None, metavar="FILE",
                   help="图文件(.png/.pdf/.svg)")
    f.add_argument("--figure-kind", default="report",
                   choices=FIGURE_KINDS,
                   help="report=四联报告图(默认);map=地图;spectrum=逐阶谱;"
                        "hist=直方图;none=不出图")
    f.add_argument("--dpi", type=int, default=160)
    f.add_argument("--cmap", default="RdBu_r", metavar="NAME")
    f.add_argument("--contour", action="store_true", help="网格图上叠加等值线")
    f.add_argument("--no-coast", action="store_true", help="不画海岸线")
    f.add_argument("--focus", default="auto", choices=("auto", "global"),
                   help="地图聚焦:auto(默认)在区域结果上自动缩放到结果范围,"
                        "覆盖接近全球时画全球;global 强制全球视图")
    f.add_argument("--asymmetric-range", action="store_true",
                   help="配色不关于 0 对称(默认对称,异常场更清楚)")
    f.add_argument("--quiet", action="store_true", help="只打印一行结果")
    s.set_defaults(func=cmd_synth)

    # ----------------------------------------------------------- info
    i = sub.add_parser("info", help="查看系数文件信息(布局/阶数/物理量/逐阶谱)")
    i.add_argument("--coeffs", required=True, metavar="FILE")
    i.add_argument("--layout", default="auto",
                   choices=("auto", "triangle", "gmfcsv", "matrix", "gfc",
                            "npy", "npz"))
    i.add_argument("--coeff-nmax", type=int, default=None, metavar="N")
    i.add_argument("--spectrum", action="store_true", help="打印逐阶 RMS / power 表")
    i.add_argument("--unit-table", action="store_true",
                   help="打印各物理量的逐阶换算因子")
    i.add_argument("--comments", action="store_true", help="打印文件头注释")
    i.set_defaults(func=cmd_info)

    # -------------------------------------------------------- convert
    c = sub.add_parser("convert", help="转换系数:改布局 / 换物理量 / 截断 / 平滑")
    c.add_argument("--coeffs", required=True, metavar="FILE")
    c.add_argument("--layout", default="auto",
                   choices=("auto", "triangle", "gmfcsv", "matrix", "gfc",
                            "npy", "npz"), help="输入布局")
    c.add_argument("--layout-out", default="auto",
                   choices=("auto", "triangle", "gmfcsv", "gfc", "npy", "npz"),
                   help="输出布局(默认按扩展名)")
    c.add_argument("--coeff-nmax", type=int, default=None, metavar="N",
                   help="读取时截断")
    c.add_argument("--nmax", type=int, default=None, metavar="N",
                   help="写出前再截断到该阶")
    c.add_argument("--to-unit", default=None, metavar="UNIT",
                   help="换成目标物理量(geoid/ewh/surface_density/"
                        "radial_displacement/geopotential)")
    c.add_argument("--set-unit", default=None, metavar="UNIT",
                   help="只打标签,不换算(声明源系数到底是什么)")
    c.add_argument("--gaussian-km", type=float, default=0.0, metavar="KM")
    c.add_argument("--gaussian-method", default="glq", choices=("glq", "frc"))
    c.add_argument("--out", required=True, metavar="FILE")
    c.add_argument("--comment", default=None, metavar="TEXT")
    c.add_argument("--fmt", default="%.16g", metavar="FMT")
    c.set_defaults(func=cmd_convert)

    sub.add_parser("formats", help="列出支持的全部文件格式与物理量"
                   ).set_defaults(func=cmd_formats)

    # ------------------------------------------------- v2.0 series-* 系列
    sr = sub.add_parser(
        "series-read",
        help="[v2.0] 把一个 gfc 目录 / 清单 -> 带日期的系数序列(.nc)",
        description="读一个目录(或清单、或已有多时间文件)里的全部历元,"
                    "拼成 (nmax+1, nmax+1, ntime) 的系数序列,并挂上时间轴。")
    sr.add_argument("--indir", required=True, metavar="DIR",
                    help="gfc 目录 / glob / .txt 清单 / 单个多时间文件")
    sr.add_argument("--pattern", default=None, metavar="GLOB")
    sr.add_argument("--product", default="GSM", metavar="P1,P2",
                    help="只收这些前缀(默认 GSM);混入 GAC/GAD 会被拒绝")
    sr.add_argument("--mission", default=None, metavar="GRAC|GRFO")
    sr.add_argument("--center", default=None, metavar="UTCSR|GFZOP|JPL")
    sr.add_argument("--rl", default=None, metavar="0600|0603")
    sr.add_argument("--no-recursive", action="store_true",
                    help="不递归子目录(ITSG 日解按年分目录,默认要递归)")
    sr.add_argument("--nmax", type=int, default=None, metavar="N")
    sr.add_argument("--epoch-from", default="auto",
                    choices=("auto", "header", "filename", "index"))
    sr.add_argument("--sort", default="time", choices=("time", "name", "none"))
    sr.add_argument("--on-duplicate", default="error",
                    choices=("error", "first", "last", "report"))
    sr.add_argument("--lenient-nmax", action="store_true",
                    help="各文件最高阶不一致时不报错(按最大阶零填充)")
    sr.add_argument("--out", required=True, metavar="FILE",
                    help="输出的系数序列(.nc 推荐;.dat 为 legacy 布局)")
    sr.add_argument("--summary", default=None, metavar="FILE",
                    help="把时间轴摘要写到该文件")
    sr.add_argument("--quiet", action="store_true", help="只打印一行结果")
    sr.set_defaults(func=cmd_series_read)

    ss = sub.add_parser(
        "series-synth",
        help="[v2.0] 批量综合:序列 -> 场序列 nc + 逐历元诊断 csv + 图/动画")
    ss.add_argument("--coeffs", required=True, metavar="FILE")
    ss.add_argument("--layout", default="auto", choices=COEFF_LAYOUTS)
    ss.add_argument("--coeff-nmax", type=int, default=None, metavar="N")
    ss.add_argument("--nmax", type=int, default=None, metavar="N",
                    help="综合时截断到该阶")
    g2 = ss.add_argument_group("目标位置")
    g2.add_argument("--global-grid", type=float, default=None, metavar="DEG")
    g2.add_argument("--grid-file", default=None, metavar="FILE")
    g2.add_argument("--grid-var", default=None, metavar="NAME")
    g2.add_argument("--lat-min", type=float, default=-90.0)
    g2.add_argument("--lat-max", type=float, default=90.0)
    g2.add_argument("--lon-min", type=float, default=0.0)
    g2.add_argument("--lon-max", type=float, default=360.0)
    g2.add_argument("--lat-step", type=float, default=1.0)
    g2.add_argument("--lon-step", type=float, default=None)
    g2.add_argument("--points", default=None, metavar="FILE")
    g2.add_argument("--lat-col", default=None)
    g2.add_argument("--lon-col", default=None)
    o2 = ss.add_argument_group("综合选项")
    o2.add_argument("--gaussian-km", type=float, default=0.0, metavar="KM")
    o2.add_argument("--gaussian-method", default="glq", choices=("glq", "frc"))
    o2.add_argument("--target-unit", default=None, metavar="UNIT")
    o2.add_argument("--component", default="scalar",
                    choices=("scalar", "radial", "north", "east", "horizontal"))
    o2.add_argument("--scale", default="1", choices=("1", "mm"),
                    help="数值标度:1(默认,SI)或 mm(×1e3 并改写 units)")
    o2.add_argument("--use-fft", default="auto", choices=("auto", "yes", "no"))
    _add_prep_args(ss)
    o2.add_argument("--time-chunk", type=int, default=DEFAULT_TIME_CHUNK)
    o2.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    w2 = ss.add_argument_group("输出")
    w2.add_argument("--out", default=None, metavar="FILE",
                    help="场序列(.nc 推荐)")
    w2.add_argument("--out-summary", default=None, metavar="FILE",
                    help="逐历元诊断表(.csv)")
    w2.add_argument("--var", default="value", metavar="NAME")
    w2.add_argument("--units", default=None, metavar="TEXT")
    w2.add_argument("--time-encoding", default="datetime64",
                    choices=("datetime64", "cf"))
    a2 = ss.add_argument_group("动画(v2.0)")
    a2.add_argument("--anim-out", default=None, metavar="FILE",
                    help="写成动图:.gif(只需 Pillow,推荐)或 .mp4"
                         "(需要 imageio + imageio-ffmpeg)")
    a2.add_argument("--anim-fps", type=float, default=5.0, metavar="FPS",
                    help="帧率。GIF 延时只能取 10 ms 整数倍,实际值会打印出来")
    a2.add_argument("--anim-stride", type=int, default=1, metavar="N",
                    help="每隔 N 个历元取一帧(默认 1 = 每个历元一帧)")
    a2.add_argument("--anim-vmin", type=float, default=None)
    a2.add_argument("--anim-vmax", type=float, default=None,
                    help="配色范围;**默认取全序列的稳健对称范围**,"
                         "不逐帧定标(否则动画会随历元极值闪)")
    a2.add_argument("--anim-focus", default="global",
                    choices=("auto", "global", "none"),
                    help="动画默认全球视图(逐帧聚焦会让视野一直晃)")
    a2.add_argument("--anim-title", default=None, metavar="TEXT",
                    help="每帧标题前缀(默认按分量+物理量自动拼)")
    ss.add_argument("--quiet", action="store_true")
    ss.set_defaults(func=cmd_series_synth)

    sp = sub.add_parser("series-points",
                        help="[v2.0] 站点/散点的时间序列(可含水平形变两个分量)")
    sp.add_argument("--coeffs", required=True, metavar="FILE")
    sp.add_argument("--layout", default="auto", choices=COEFF_LAYOUTS)
    sp.add_argument("--points", required=True, metavar="FILE")
    sp.add_argument("--lat-col", default=None)
    sp.add_argument("--lon-col", default=None)
    sp.add_argument("--gaussian-km", type=float, default=0.0)
    sp.add_argument("--target-unit", default=None, metavar="UNIT")
    sp.add_argument("--component", default="scalar",
                    choices=("scalar", "radial", "north", "east", "horizontal"))
    sp.add_argument("--scale", default="1", choices=("1", "mm"))
    sp.add_argument("--out", required=True, metavar="FILE",
                    help="csv:lon,lat[,分量列...] + 每个历元一列")
    _add_prep_args(sp)
    sp.set_defaults(func=cmd_series_points)

    sf = sub.add_parser("series-fit",
                        help="[v2.0] 系数域趋势/周年拟合 -> 趋势场 + 周年振幅相位场")
    sf.add_argument("--coeffs", required=True, metavar="FILE")
    sf.add_argument("--layout", default="auto", choices=COEFF_LAYOUTS)
    sf.add_argument("--periods", default="1,0.5", metavar="P1,P2")
    sf.add_argument("--trend", default="linear",
                    choices=("none", "linear", "quadratic"))
    sf.add_argument("--sigma", action="store_true", help="同时估 1σ")
    sf.add_argument("--lat-step", type=float, default=1.0)
    sf.add_argument("--out-trend", default=None, metavar="FILE")
    sf.add_argument("--out-seasonal", default=None, metavar="FILE")
    sf.add_argument("--out-table", default=None, metavar="FILE")
    _add_prep_args(sf)
    sf.set_defaults(func=cmd_series_fit)

    sb = sub.add_parser(
        "series-basin",
        help="[v2.0] 流域/区域平均时间序列(两种口径都算并对比)")
    sb.add_argument("--coeffs", required=True, metavar="FILE")
    sb.add_argument("--layout", default="auto", choices=COEFF_LAYOUTS)
    sb.add_argument("--mask", required=True, metavar="FILE",
                    help="掩膜:二维数组的 .nc/.grd/.csv,或 [lon;lat;value] "
                         "堆叠的 .npy;裸二维 .npy 需配 --mask-lat-step")
    sb.add_argument("--mask-lat-step", type=float, default=None,
                    help="裸二维 .npy 掩膜的纬度步长(度);不给则按 180/nlat 推")
    sb.add_argument("--mask-lon-step", type=float, default=None,
                    help="裸二维 .npy 掩膜的经度步长(度);不给则按 360/nlon 推")
    sb.add_argument("--mask-kernel", default=None, metavar="FILE",
                    help="**只用来建球谐核**的更细掩膜(给了才有真实差异可比)")
    sb.add_argument("--nmax", type=int, default=None)
    sb.add_argument("--method", default="both",
                    choices=("coeff", "spatial", "both"))
    sb.add_argument("--out", default=None, metavar="FILE")
    _add_prep_args(sb)
    sb.set_defaults(func=cmd_series_basin)

    t = sub.add_parser("selftest", help="自检:解析解 + scipy + 与 SHKit 逐位比对")
    t.add_argument("--shkit", default=None, metavar="DIR",
                   help="SHKit 目录(默认 ../SHKit)")
    t.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0

    # 目标来源互斥:--points / --grid-file / --global-grid / 范围+步长
    if args.cmd == "synth":
        chosen = [bool(args.points), bool(args.grid_file),
                  args.global_grid is not None]
        if sum(chosen) > 1:
            parser.error("--points、--grid-file、--global-grid 三者只能给一个")
        if args.points:
            args.mode, args.grid_source = "points", "range"
        elif args.grid_file:
            args.mode, args.grid_source = "grid", "file"
        elif args.global_grid is not None:
            args.mode, args.grid_source = "grid", "global"
            args.lat_step = float(args.global_grid)
            args.lon_step = float(args.global_grid)
        else:
            args.mode, args.grid_source = "grid", "range"

    try:
        rc = int(args.func(args))
    except CliError as exc:
        _report_lost_output()
        _out_err(f"\n[SHSynth] 参数错误: {exc}")
        return 2
    except InterruptedError as exc:
        _report_lost_output()
        _out_err(f"\n[SHSynth] 已中止: {exc}")
        return 130
    except KeyboardInterrupt:                        # pragma: no cover
        _report_lost_output()
        _out_err("\n[SHSynth] 用户中断")
        return 130
    except Exception as exc:                         # noqa: BLE001
        _report_lost_output()
        _out_err(f"\n[SHSynth] 失败: {type(exc).__name__}: {exc}")
        _out_err("[SHSynth] 完整调用栈:")
        try:
            traceback.print_exc(file=sys.stderr)
        except (OSError, ValueError):                # pragma: no cover
            _LOST_OUTPUT.append(traceback.format_exc())
        return 1
    # 成功路径也要汇报"输出丢了":结果文件可能已经写好,但用户一句都没看到
    _report_lost_output()
    return rc


def _report_lost_output() -> Optional[str]:
    """把写不出去的控制台输出落到临时日志,返回文件路径(没丢就返回 None)。

    只在**确实写不出去**时才发生(冻结版 windowed exe 没有接上父控制台)。
    这时用户什么都看不到,所以至少要把内容留在磁盘上并告诉他路径 ——
    "结果算出来了但一句话都没打出来"是最难排查的情况。
    """
    if not _LOST_OUTPUT:
        return None
    import tempfile
    try:
        path = os.path.join(tempfile.gettempdir(), "SHSynth_cli_output.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"SHSynth {__version__} —— 这些输出无法写到控制台\n")
            fh.write("(冻结版是 GUI 程序;在 cmd 里跑请用 SHSynth.exe --cli …)\n")
            fh.write("=" * 60 + "\n")
            fh.write("\n".join(_LOST_OUTPUT) + "\n")
    except OSError:
        return None
    _LOST_OUTPUT.clear()
    _LOST_OUTPUT.append(f"[SHSynth] 控制台写不出去,输出已存到: {path}")
    _out_err(f"\n[SHSynth] 控制台写不出去,输出已存到: {path}")
    return path


if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main())
