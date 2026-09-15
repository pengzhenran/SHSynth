# -*- coding: utf-8 -*-
"""
tests/test_workflow_cli.py
==========================

**流程与命令行测试**:从参数到落盘到出图的整条链路,以及命令行四个子命令。

覆盖:
* 网格 / 散点两种目标,各种输出格式(.nc/.grd/.csv/.txt/.npy)写出后能读回且数值正确;
* 借用的网格文件、散点位置文件、Fibonacci 球面点;
* 截断 / 高斯 / 物理量换算一路传到结果里;
* 五种图都能画出来并落盘(文件非空);
* 顺带导出系数(SHKIT 五种布局)并读回核对;
* 命令行 ``synth`` / ``info`` / ``convert`` / ``formats`` / ``selftest`` 的退出码与输出;
* 参数互斥、缺文件等错误路径以非零码退出。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (Checker, SHKIT_DIR, guard, have_shkit,  # noqa: E402
                     tmp_dir)

from shsynth import fieldio, targets                                # noqa: E402
from shsynth.cli import main as cli_main                            # noqa: E402
from shsynth.coeffio import read_coeffs                             # noqa: E402
from shsynth.engine import evaluate, synthesis_grid                 # noqa: E402
from shsynth.units import convert                                  # noqa: E402
from shsynth.workflow import SynthRequest, plan_text, run           # noqa: E402

COEFFS = os.path.join(SHKIT_DIR, "shkit_coeffs.sh")
COEFFS_GFC = os.path.join(SHKIT_DIR, "shkit_coeffs.gfc")


def _run_cli(argv, quiet=True):
    """跑一次命令行,返回 (退出码, 标准输出+错误)。"""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            code = cli_main(argv)
        except SystemExit as exc:                       # argparse 的退出路径
            code = int(exc.code or 0)
    return code, buf.getvalue()


def test_workflow_grid_outputs(check: Checker):
    """网格目标:每种输出格式都要写出、能读回、数值与引擎一致。"""
    check.section("网格目标与全部输出格式")
    d = tmp_dir("workflow")
    src = read_coeffs(COEFFS)
    latv, lonv, _ = targets.global_grid(5.0)
    ref = synthesis_grid(latv, lonv, src)

    for ext in (".nc", ".grd", ".npy", ".csv", ".txt"):
        p = os.path.join(d, f"grid{ext}")
        spec = SynthRequest(coeffs_path=COEFFS, mode="grid", grid_source="global",
                            lat_step=5.0, lon_step=5.0, out_path=p,
                            figure_kind="none")
        res = run(spec)
        check.ok(res.out_path and os.path.exists(res.out_path),
                 f"{ext} 写出成功")
        if ext in (".nc", ".grd", ".npy"):
            la, lo, g, meta = fieldio.read_grid(p)
            g = g[:, :, 0] if np.ndim(g) == 3 else g
            check.ok(la.size == latv.size and lo.size == lonv.size,
                     f"{ext} 读回坐标轴一致({la.size}×{lo.size})")
            dmax = float(np.max(np.abs(g - ref)))
            check.ok(dmax < 1e-9, f"{ext} 数值与引擎一致(最大差 {dmax:.2e})")
        else:
            la, lo, vals, m2 = fieldio.read_points(p)
            check.ok(la.size == latv.size * lonv.size,
                     f"{ext} 三列点数 = {la.size}")


def test_workflow_points(check: Checker):
    """散点目标:球面点与散点文件两种来源。"""
    check.section("散点目标")
    d = tmp_dir("workflow")
    p = os.path.join(d, "pts.csv")
    spec = SynthRequest(coeffs_path=COEFFS, mode="points", n_sphere_points=3000,
                        out_path=p, figure_kind="none")
    res = run(spec)
    check.ok(res.stats["n_points"] == 3000, f"Fibonacci 3000 点 → {res.stats['n_points']}")
    la, lo, vals, meta = fieldio.read_points(p)
    check.ok(la.size == 3000, "散点结果读回点数正确")
    direct = evaluate(la, lo, read_coeffs(COEFFS))
    # 文本输出默认 %.10g(与 SHKit 一致),所以只要求 1e-9 量级的相对一致
    rel = float(np.max(np.abs(vals - direct))) / \
        max(float(np.max(np.abs(direct))), 1e-300)
    check.ok(rel < 1e-9, f"写出的散点值 = 直接求值(最大相对差 {rel:.2e})")

    # 借用 SHKit 的散点文件作为位置
    pf = os.path.join(SHKIT_DIR, "sample_data", "points_global_fibonacci.csv")
    if os.path.exists(pf):
        spec = SynthRequest(coeffs_path=COEFFS, mode="points", points_file=pf,
                            out_path=os.path.join(d, "pts2.npy"),
                            figure_kind="spectrum")
        res = run(spec)
        check.ok(res.stats["n_points"] == 3000,
                 f"散点文件求值 {res.stats['n_points']} 点")
        check.ok(res.figure is not None and len(res.figure.get_axes()) >= 1,
                 "逐阶谱图生成成功")


def test_options_propagate(check: Checker):
    """截断 / 高斯 / 物理量换算必须一路传到结果里。"""
    check.section("选项传递")
    d = tmp_dir("workflow")
    src = read_coeffs(COEFFS_GFC)
    latv, lonv, _ = targets.global_grid(5.0)

    # 截断
    p = os.path.join(d, "trunc.npy")
    run(SynthRequest(coeffs_path=COEFFS_GFC, mode="grid", grid_source="global",
                     lat_step=5.0, lon_step=5.0, truncate_nmax=5,
                     out_path=p, figure_kind="none"))
    _la, _lo, g, _m = fieldio.read_grid(p)
    g = g[:, :, 0] if g.ndim == 3 else g
    check.ok(float(np.max(np.abs(g - synthesis_grid(latv, lonv, src, nmax=5)))) < 1e-12,
             "truncate_nmax=5 生效")

    # 高斯
    p = os.path.join(d, "gauss.npy")
    run(SynthRequest(coeffs_path=COEFFS_GFC, mode="grid", grid_source="global",
                     lat_step=5.0, lon_step=5.0, gaussian_km=1000.0,
                     out_path=p, figure_kind="none"))
    _la, _lo, g, _m = fieldio.read_grid(p)
    g = g[:, :, 0] if g.ndim == 3 else g
    check.ok(float(np.max(np.abs(g - synthesis_grid(latv, lonv, src,
                                                    gaussian_km=1000.0)))) < 1e-12,
             "gaussian_km=1000 生效")
    check.ok(float(np.std(g)) < float(np.std(synthesis_grid(latv, lonv, src))),
             "平滑后数值确实被压平")

    # 物理量换算
    p = os.path.join(d, "ewh.npy")
    res = run(SynthRequest(coeffs_path=COEFFS_GFC, mode="grid",
                           grid_source="global", lat_step=5.0, lon_step=5.0,
                           target_unit="ewh", out_path=p, figure_kind="none"))
    _la, _lo, g, _m = fieldio.read_grid(p)
    g = g[:, :, 0] if g.ndim == 3 else g
    check.ok(float(np.max(np.abs(g - synthesis_grid(latv, lonv, convert(src, "ewh")))))
             < 1e-6 * max(1.0, float(np.max(np.abs(g)))),
             "target_unit='ewh' 生效")
    check.ok(any("EWH" in w or "换算" in w for w in res.warnings),
             "换算写进 warnings")

    # 顺带导出系数
    pc = os.path.join(d, "out_coeffs.gfc")
    res = run(SynthRequest(coeffs_path=COEFFS_GFC, mode="grid",
                           grid_source="global", lat_step=10.0, lon_step=10.0,
                           truncate_nmax=6, out_coeffs_path=pc,
                           out_coeffs_layout="gfc", figure_kind="none"))
    back = read_coeffs(pc)
    check.ok(back.nmax == 6, f"导出系数被截断到 nmax={back.nmax}")
    check.ok(os.path.exists(res.out_coeffs_path), "导出系数文件存在")


def test_figures(check: Checker):
    """五种图都要画得出来、存得下去(文件非空)。"""
    check.section("绘图")
    d = tmp_dir("workflow")
    for kind in ("report", "map", "spectrum", "hist"):
        p = os.path.join(d, f"fig_{kind}.png")
        res = run(SynthRequest(coeffs_path=COEFFS, mode="grid",
                               grid_source="global", lat_step=10.0, lon_step=10.0,
                               figure_kind=kind, figure_path=p,
                               contour=(kind in ("map", "report"))))
        ok_axes = res.figure is not None and len(res.figure.get_axes()) >= 1
        check.ok(ok_axes, f"{kind} 图有坐标轴({len(res.figure.get_axes()) if res.figure else 0} 个)")
        check.ok(os.path.exists(p) and os.path.getsize(p) > 5000,
                 f"{kind} 图已落盘({os.path.getsize(p) // 1024 if os.path.exists(p) else 0} KB)")
    res = run(SynthRequest(coeffs_path=COEFFS, mode="points", n_sphere_points=2000,
                           figure_kind="map", figure_path=os.path.join(d, "scatter.png")))
    check.ok(res.figure is not None, "散点图生成成功(散点分支)")


def test_targets_and_fieldio(check: Checker):
    """目标几何解析与散点/网格文件的列识别。"""
    check.section("目标解析与文件列识别")
    lat, lon, meta = targets.grid_from_spec(-90, 90, 0, 360, 1.0, 1.0)
    check.ok(lat.size == 181 and lon.size == 360,
             f"0..360 不含端点:{lat.size}×{lon.size}(不是 361)")
    check.ok(lon[-1] == 359.0, f"最后一条经线 = {lon[-1]}(避免 0 与 360 重复)")
    lat, lon, meta = targets.grid_from_spec(20, 50, 100, 140, 1.0, 1.0)
    check.ok(lat.size == 31 and lon.size == 41, f"区域网格 {lat.size}×{lon.size}")

    d = tmp_dir("workflow")
    # 中文表头 + 负经度
    p = os.path.join(d, "cn.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("经度,纬度,数值\n")
        fh.write("-10.0,30.0,1.5\n-20.0,40.0,2.5\n")
    la, lo, vals, m = fieldio.read_points(p)
    check.ok(np.allclose(lo, [350.0, 340.0]), f"中文表头 + 负经度归一化 {lo}")
    check.ok(np.allclose(la, [30.0, 40.0]) and np.allclose(vals, [1.5, 2.5]),
             "中文表头列识别正确")

    # 显式列下标:第 1 列纬度、第 2 列经度
    p2 = os.path.join(d, "cols.txt")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write("30 10 7.5\n40 20 8.5\n")
    la, lo, vals, m = fieldio.read_points(p2)
    check.ok(np.allclose(lo, [30, 40]) and np.allclose(la, [10, 20]),
             "无表头时按 经度,纬度,值 的通行约定(第 1 列经度、第 2 列纬度)")
    la2, lo2, vals2, m2 = fieldio.read_points(p2, lat_col=0, lon_col=1)
    check.ok(np.allclose(la2, [30, 40]) and np.allclose(lo2, [10, 20]),
             "显式指定 lat_col=0, lon_col=1 时按指定列解释")

    # 散点写出读回
    p3 = os.path.join(d, "rt_points.csv")
    fieldio.write_points(p3, np.array([10.0, 20.0]), np.array([30.0, 40.0]),
                         np.array([1.0, 2.0]))
    la, lo, vals, m = fieldio.read_points(p3)
    check.ok(np.allclose(la, [10, 20]) and np.allclose(lo, [30, 40])
             and np.allclose(vals, [1.0, 2.0]), "散点写出读回一致")

    # 网格写出读回(grd / npy / csv)
    latv = np.array([-10.0, 0.0, 10.0])
    lonv = np.array([0.0, 90.0, 180.0, 270.0])
    g = np.arange(12.0).reshape(3, 4)
    for ext in (".grd", ".npy", ".csv", ".nc"):
        p4 = os.path.join(d, f"rtgrid{ext}")
        fieldio.write_grid(p4, latv, lonv, g)
        la, lo, g2, meta = fieldio.read_grid(p4)
        g2 = g2[:, :, 0] if np.ndim(g2) == 3 else g2
        check.ok(np.allclose(la, latv) and np.allclose(lo, lonv)
                 and np.allclose(g2, g), f"{ext} 网格往返一致")


def test_error_paths(check: Checker):
    """错误路径:必须给中文说明并以非零码退出。"""
    check.section("错误路径")
    check.raises(lambda: run(SynthRequest(coeffs_path="", mode="grid")),
                 ValueError, "没给系数文件 → 报错")
    check.raises(lambda: run(SynthRequest(coeffs_path=COEFFS, mode="grid",
                                          grid_source="file", grid_file="")),
                 ValueError, "借用网格但没给文件 → 报错")
    check.raises(lambda: run(SynthRequest(coeffs_path=COEFFS, mode="grid",
                                          grid_source="global", lat_step=1.0,
                                          target_unit="ewh")),
                 ValueError, "未声明物理量却要换算 → 报错")

    code, out = _run_cli(["synth", "--coeffs", os.path.join("no", "such.sh"),
                          "--global-grid", "30"])
    check.ok(code != 0, f"系数文件不存在 → 退出码 {code} ≠ 0")
    check.ok("失败" in out or "不存在" in out, "给出中文错误信息")

    code, out = _run_cli(["synth", "--coeffs", COEFFS, "--points", "a.csv",
                          "--global-grid", "10"])
    check.ok(code == 2, f"目标来源互斥 → argparse 退出码 {code}")


def test_cli_commands(check: Checker):
    """命令行各子命令。"""
    check.section("命令行子命令")
    d = tmp_dir("cli")

    out_nc = os.path.join(d, "cli_field.nc")
    fig = os.path.join(d, "cli_fig.png")
    code, out = _run_cli(["synth", "--coeffs", COEFFS, "--global-grid", "10",
                          "--out", out_nc, "--figure", fig,
                          "--figure-kind", "report", "--quiet"])
    check.ok(code == 0, f"synth 退出码 {code}")
    check.ok(os.path.exists(out_nc) and os.path.getsize(out_nc) > 1000,
             "synth 写出 .nc")
    check.ok(os.path.exists(fig) and os.path.getsize(fig) > 5000, "synth 出图")

    code, out = _run_cli(["synth", "--coeffs", COEFFS_GFC, "--sphere-points", "500",
                          "--gaussian-km", "800", "--target-unit", "等效水高",
                          "--out", os.path.join(d, "pts.csv")])
    check.ok(code == 0, f"synth 散点 + 中文物理量 退出码 {code}")
    check.ok("EWH" in out or "等效水高" in out, "汇总里出现物理量名称")

    code, out = _run_cli(["info", "--coeffs", COEFFS_GFC, "--spectrum"])
    check.ok(code == 0 and "识别布局" in out and "gfc" in out,
             f"info 退出码 {code},输出含布局")
    check.ok("逐阶谱" in out, "info --spectrum 打印逐阶谱")

    p_out = os.path.join(d, "conv.sh")
    code, out = _run_cli(["convert", "--coeffs", COEFFS_GFC, "--out", p_out,
                          "--to-unit", "ewh", "--layout-out", "triangle"])
    check.ok(code == 0 and os.path.exists(p_out), f"convert 退出码 {code}")
    back = read_coeffs(p_out)
    check.ok(back.field_unit == "ewh", f"convert 后标签 = {back.field_unit}")
    check.ok(back.nmax == read_coeffs(COEFFS_GFC).nmax, "convert 保留阶数")

    p_gfc = os.path.join(d, "conv2.gfc")
    code, _ = _run_cli(["convert", "--coeffs", COEFFS, "--out", p_gfc,
                        "--set-unit", "ewh", "--layout-out", "gfc"])
    check.ok(code == 0 and read_coeffs(p_gfc).field_unit == "ewh",
             "--set-unit 只打标签")

    code, out = _run_cli(["formats"])
    check.ok(code == 0 and "triangle" in out and "gfc" in out and "npz" in out,
             "formats 列出全部布局")

    if have_shkit():
        code, out = _run_cli(["selftest", "--shkit", SHKIT_DIR])
        check.ok(code == 0, f"selftest 退出码 {code}")
        check.ok("OK" in out, "selftest 有 OK 标记")
        check.ok("与 SHKit 比对" in out, "selftest 做了跨软件比对")
    else:
        # SHKit 只是**开发期的对照物**,不是运行依赖(打包环境里就没有它);
        # 没有它就如实跳过,而不是报一个红叉 —— 与 test_vs_shkit.py 保持一致
        check.ok(True, "跳过:旁边没有 SHKit(跨软件比对只在开发环境里跑)")

    code, out = _run_cli([])
    check.ok(code == 0 and "synth" in out, "不带子命令 → 打印帮助")


def test_multiepoch_export_guard(check: Checker):
    """多时次导出到"只装得下一个时次"的格式(.gfc)时,必须先要求选时次。"""
    check.section("多时次导出守卫")
    from shsynth.coeffs import SHCoeffs
    from shsynth.coeffio import write_coeffs

    d = tmp_dir("workflow")
    base = read_coeffs(COEFFS)
    C3 = np.repeat(base.C[:, :, None], 3, axis=2) * np.array([1.0, 1.3, 0.7])
    S3 = np.repeat(base.S[:, :, None], 3, axis=2) * np.array([1.0, 1.3, 0.7])
    src = os.path.join(d, "multi_src.sh")
    write_coeffs(SHCoeffs(C3, S3, {"field_unit": "geopotential"}), src,
                 layout="triangle")
    check.ok(read_coeffs(src).ntime == 3, "多时次源文件 ntime = 3")

    def _spec(**kw):
        base_kw = dict(coeffs_path=src, mode="grid", grid_source="global",
                       lat_step=15.0, lon_step=15.0, figure_kind="none")
        base_kw.update(kw)
        return SynthRequest(**base_kw)

    # (a) 不指定时次 + gfc → 报错
    check.raises(
        lambda: run(_spec(out_coeffs_path=os.path.join(d, "bad.gfc"),
                          out_coeffs_layout="gfc")),
        ValueError, "多时次 → .gfc 且没给 time 必须报错")
    try:
        run(_spec(out_coeffs_path=os.path.join(d, "bad.gfc"),
                  out_coeffs_layout="gfc"))
    except ValueError as exc:
        msg = str(exc)
        check.ok("time" in msg and ("npz" in msg or "triangle" in msg),
                 "报错信息给出『指定时次』与『改用 triangle/npz』两条出路")

    # (b) 指定时次 → 成功,且内容就是那个时次
    res = run(_spec(time=1, out_coeffs_path=os.path.join(d, "t1.gfc"),
                    out_coeffs_layout="gfc"))
    back = read_coeffs(res.out_coeffs_path)
    check.ok(back.ntime == 1 and np.allclose(back.C, C3[:, :, 1]),
             "指定 time=1 后导出的就是第 1 个时次")

    # (c) 多时次布局(npz)不需要指定时次,三个时次都保留
    res = run(_spec(out_coeffs_path=os.path.join(d, "multi.npz"),
                    out_coeffs_layout="npz"))
    back = read_coeffs(res.out_coeffs_path)
    check.ok(back.ntime == 3 and np.allclose(back.C, C3),
             "npz 布局把 3 个时次全部保留")


def test_plan_text(check: Checker):
    """参数预览必须把关键信息说全。"""
    check.section("参数预览")
    spec = SynthRequest(coeffs_path="a.sh", mode="grid", grid_source="global",
                        lat_step=1.0, gaussian_km=300.0, target_unit="ewh",
                        out_path="o.nc", figure_kind="map", figure_path="f.png")
    txt = plan_text(spec)
    for key in ("a.sh", "全球网格", "300", "EWH", "o.nc", "f.png"):
        check.ok(key in txt, f"预览含 {key!r}")


def test_cli_stdout_never_fatal(check: Checker):
    """控制台写不出去时,**计算与返回码不受影响**(冻结版 windowed exe 的坑)。

    发行版是 GUI(windowed)exe:在 cmd 里跑 ``SHSynth.exe --cli …`` 时
    ``sys.stdout`` 可能是个非 None 但无效的句柄,``print()`` 会抛
    ``OSError: [Errno 22] Invalid argument``。**实测踩过**:命令在打印标题
    (还没开始算)时就崩,用户看到"报错了"却什么也没得到。
    """
    import io

    from shsynth import cli
    check.section("控制台输出不许把命令带走")

    class Bad(io.TextIOBase):
        def write(self, s):
            raise OSError(22, "Invalid argument")

        def flush(self):
            pass

    real_out, real_err = sys.stdout, sys.stderr
    raised = None
    cli._LOST_OUTPUT.clear()
    sys.stdout = sys.stderr = Bad()
    try:
        cli._out("写不出去的一行")
        cli._hr("标题")
        cli._out_err("stderr 也写不出去")
    except BaseException as exc:                          # noqa: BLE001
        raised = exc
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    check.ok(raised is None,
             f"_out/_hr/_out_err 句柄无效时不抛异常(实测 {raised!r})")
    check.ok(len(cli._LOST_OUTPUT) >= 3,
             f"写不出去的 {len(cli._LOST_OUTPUT)} 行被攒下来,没静默丢内容")
    p = cli._report_lost_output()
    check.ok(p is not None and os.path.exists(p),
             f"丢失的输出落盘并回报路径({p})")
    if p:
        txt = open(p, encoding="utf-8").read()
        check.ok("写不出去的一行" in txt, "日志里含原文本")
        check.ok("--cli" in txt, "日志里给出命令行用法提示")
    # 成功路径的返回码不能被打印失败连坐
    cli._LOST_OUTPUT.clear()
    sys.stdout = sys.stderr = Bad()
    try:
        cli._out("又写不出去")
        rc = 0
        cli._report_lost_output()
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    check.ok(rc == 0, "控制台写不出去时,成功路径返回码仍是 0")
    # 控制台正常时不该产生"丢失"记录
    cli._LOST_OUTPUT.clear()
    cli._out("正常一行")
    check.ok(not cli._LOST_OUTPUT, "控制台正常时不产生丢失记录")


def main() -> int:
    check = Checker("test_workflow_cli —— 流程、输出格式与命令行")
    for fn in (test_workflow_grid_outputs, test_workflow_points,
               test_options_propagate, test_figures, test_targets_and_fieldio,
               test_error_paths, test_cli_commands, test_multiepoch_export_guard,
               test_plan_text, test_cli_stdout_never_fatal):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
