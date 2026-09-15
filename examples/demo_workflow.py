# -*- coding: utf-8 -*-
"""
examples/demo_workflow.py
=========================

端到端演示:把 SHSynth 的主要用法各跑一遍,并把结果(数据 + 图)写到
``examples/out/`` 下,方便直接翻看。

    python examples/demo_workflow.py            # 跑全部场景
    python examples/demo_workflow.py --fast     # 全球网格改粗一点,更快

六个场景:

1. **格式适配巡礼** —— 读遍 SHKit 目录里所有系数文件,打印识别出的布局;
2. **全球网格 + 报告图** —— 最常用的一条路;
3. **区域网格 + 散点** —— 只在一小块区域、或只有几个台站时;
4. **高斯平滑 + 等效水高** —— 物理量换算(逐阶因子,不是常数!);
5. **借用别人的网格文件** —— 直接用它已有的格点求值,并出差值图;
6. **系数再导出** —— 截断/平滑后的系数按 SHKit 五种布局各写一份。

若 SHKit 就在上一级目录,还会顺带做一次**逐位比对**。
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SHKIT = os.path.join(os.path.dirname(ROOT), "SHKit")
OUT = os.path.join(HERE, "out")

from shsynth import __version__, read_coeffs, write_coeffs          # noqa: E402
from shsynth.coeffio import FORMAT_SPECS                            # noqa: E402
from shsynth.engine import evaluate, synthesis_grid                 # noqa: E402
from shsynth.plotting import save_figure                            # noqa: E402
from shsynth.workflow import SynthRequest, plan_text, run           # noqa: E402

FAST = "--fast" in sys.argv
STEP = 2.0 if FAST else 1.0


def hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def scene1_formats() -> str:
    """① 读遍 SHKit 的系数文件,看布局识别。"""
    hr("① 格式适配巡礼 —— 读遍 SHKit 目录里的系数文件")
    cands = [
        os.path.join(SHKIT, "shkit_coeffs.sh"),
        os.path.join(SHKIT, "shkit_coeffs.gfc"),
        os.path.join(SHKIT, "yantze_shkit_coeffs.gfc"),
        os.path.join(SHKIT, "shkit_coeffs.npy"),
        os.path.join(SHKIT, "shkit_coeffs.npz"),
        os.path.join(SHKIT, "sample_data", "truth_coeffs_20.sh"),
    ]
    found = [p for p in cands if os.path.exists(p)]
    if not found:
        print(f"没找到 SHKit 样例目录({SHKIT});改用本目录的示例数据。")
        found = [p for p in (os.path.join(OUT, "demo_coeffs.sh"),)
                 if os.path.exists(p)]
    for p in found:
        c = read_coeffs(p)
        print(f"  {os.path.basename(p):<28} 布局={c.meta['layout']:<9} "
              f"nmax={c.nmax:<3} ntime={c.ntime}  "
              f"物理量={c.field_unit}")
    print()
    print("  本软件支持的系数布局:")
    for spec in FORMAT_SPECS:
        print(f"    [{spec['layout']:<8}] {spec['name']:<16} "
              f"{', '.join(spec['ext'])}")
    if not found:
        return ""
    src = found[0]
    # 若目录里没有 .sh 之外的布局,现场生成一份,后面的场景也有得用
    demo = os.path.join(OUT, "demo_coeffs.sh")
    write_coeffs(read_coeffs(src), demo, layout="triangle",
                 comment="SHSynth 演示用系数")
    return src


def scene2_global_grid(src: str) -> None:
    """② 全球网格 + 四联报告图。"""
    hr(f"② 全球网格({STEP:g}°)+ 报告图")
    spec = SynthRequest(
        coeffs_path=src, mode="grid", grid_source="global",
        lat_step=STEP, lon_step=STEP,
        out_path=os.path.join(OUT, "global_field.nc"),
        figure_kind="report", figure_path=os.path.join(OUT, "global_report.png"),
        contour=False)
    print(plan_text(spec))
    res = run(spec)
    print()
    print(res.summary)
    print()
    print(f"  图的坐标轴数 = {len(res.figure.get_axes())}"
          "(地图 + 色标 + 谱 + 分布)")


def scene3_region_and_points(src: str) -> None:
    """③ 区域网格与散点。"""
    hr("③ 区域网格(东亚)+ 散点(台站)")
    spec = SynthRequest(
        coeffs_path=src, mode="grid", grid_source="range",
        lat_min=5.0, lat_max=55.0, lon_min=60.0, lon_max=140.0,
        lat_step=0.5, lon_step=0.5,
        out_path=os.path.join(OUT, "eastasia.csv"),
        figure_kind="map", figure_path=os.path.join(OUT, "eastasia.png"),
        contour=True)
    res = run(spec)
    print(f"  区域网格 {res.lat.size}×{res.lon.size} = "
          f"{res.stats['n_points']:,} 点   值范围 "
          f"[{res.stats['min']:.4g}, {res.stats['max']:.4g}]")

    pts_file = os.path.join(SHKIT, "sample_data", "points_global_fibonacci.csv")
    spec = SynthRequest(coeffs_path=src, mode="points", points_file=pts_file,
                        out_path=os.path.join(OUT, "points_synth.csv"),
                        figure_kind="map",
                        figure_path=os.path.join(OUT, "points_scatter.png"),
                        cmap="viridis", symmetric=False)
    if not os.path.exists(pts_file):
        spec.points_file = ""
        spec.n_sphere_points = 3000
    res = run(spec)
    print(f"  散点 {res.stats['n_points']:,} 个   值 RMS = {res.stats['rms']:.6g}"
          f"   -> {res.out_path}")


def scene4_smooth_ewh(src_gfc: str) -> None:
    """④ 高斯平滑 + 等效水高。"""
    hr("④ 高斯平滑 + 换算成等效水高(逐阶因子)")
    from shsynth.units import degree_factors
    c = read_coeffs(src_gfc)
    A = degree_factors("ewh", min(c.nmax, 6))
    print("  Aₙ 逐阶因子(由无量纲位系数到 EWH):")
    for n in range(min(c.nmax, 6) + 1):
        print(f"    n={n:<2} Aₙ = {A[n]:.6e}")
    print(f"    → 0→6 阶就跨了 {A[6] / A[0]:.2f} 倍,所以它绝不是"
          "『乘一个常数』")
    spec = SynthRequest(
        coeffs_path=src_gfc, mode="grid", grid_source="global",
        lat_step=STEP, lon_step=STEP, gaussian_km=300.0,
        target_unit="ewh",
        out_path=os.path.join(OUT, "ewh_300km.nc"),
        out_units="m (EWH)",
        figure_kind="map", figure_path=os.path.join(OUT, "ewh_map.png"),
        cmap="BrBG")
    res = run(spec)
    print(f"  平滑 300 km 后 EWH 范围 "
          f"[{res.stats['min']:.4g}, {res.stats['max']:.4g}] m")
    for w in res.warnings:
        print(f"  警告: {w}")


def scene5_borrow_grid(src: str) -> None:
    """⑤ 借用网格文件 + 差值图(平滑前后之差)。"""
    hr("⑤ 借用已有网格文件的格点 + 差值图")
    gfile = os.path.join(SHKIT, "sample_data", "grid_global_2deg.nc")
    if not os.path.exists(gfile):
        print("  没找到 SHKit 的示例网格,跳过。")
        return
    spec = SynthRequest(
        coeffs_path=src, mode="grid", grid_source="file", grid_file=gfile,
        out_path=os.path.join(OUT, "borrowed.npy"), figure_kind="none")
    res = run(spec)
    print(f"  借用格点 {res.lat.size}×{res.lon.size} = "
          f"{res.stats['n_points']:,} 点   -> {res.out_path}")

    from shsynth import fieldio, plotting
    la, lo, ref, meta = fieldio.read_grid(gfile)
    print(f"  (该文件的变量是 {meta.get('variable')!r}、"
          f"{np.ndim(ref)} 维;这里只借它的格点,不动它的数值)")

    raw = np.asarray(res.values)
    raw = raw[:, :, 0] if raw.ndim == 3 else raw
    sm = synthesis_grid(res.lat, res.lon, res.coeffs, gaussian_km=1000.0)
    sm = np.asarray(sm)
    sm = sm[:, :, 0] if sm.ndim == 3 else sm
    fig = plotting.make_diff_figure(
        res.lat, res.lon, raw, sm,
        title="平滑 1000 km − 未平滑(同一套系数、同一批格点)")
    p = save_figure(fig, os.path.join(OUT, "borrowed_diff.png"), dpi=140)
    print(f"  平滑引入的差值 RMS = {float(np.sqrt(np.mean((sm - raw) ** 2))):.4g}"
          f"(原场 RMS {float(np.sqrt(np.mean(raw ** 2))):.4g})")
    print(f"  差值图 -> {p}")


def scene6_export_coeffs(src: str) -> None:
    """⑥ 处理后的系数按 SHKit 五种布局各导一份。"""
    hr("⑥ 系数再导出(截断 + 平滑,五种布局)")
    for layout, ext in (("triangle", ".sh"), ("gmfcsv", ".csv"),
                        ("gfc", ".gfc"), ("npy", ".npy"), ("npz", ".npz")):
        p = os.path.join(OUT, f"demo_export{ext}" if layout != "gmfcsv"
                         else os.path.join(OUT, "gmf", "demo_export.csv"))
        spec = SynthRequest(coeffs_path=src, mode="grid", grid_source="global",
                            lat_step=30.0, lon_step=30.0, truncate_nmax=8,
                            gaussian_km=500.0, figure_kind="none",
                            out_coeffs_path=p, out_coeffs_layout=layout)
        res = run(spec)
        back = read_coeffs(res.out_coeffs_path)
        print(f"  {layout:<9} -> {os.path.relpath(res.out_coeffs_path, ROOT)}"
              f"  读回 nmax={back.nmax} ntime={back.ntime} "
              f"布局={back.meta['layout']}")


def scene7_compare_shkit(src: str) -> None:
    """⑦ (可选)与 SHKit 逐位比对。"""
    hr("⑦ 与 SHKit 逐位比对")
    if not os.path.isdir(SHKIT):
        print("  没找到 SHKit 目录,跳过。")
        return
    sys.path.insert(0, os.path.dirname(SHKIT))
    try:
        import shkit.io as shio
        from shkit.synthesis import synthesis as sh_synthesis
    except Exception as exc:                             # noqa: BLE001
        print(f"  导入 SHKit 失败({type(exc).__name__}: {exc}),跳过。")
        return
    rng = np.random.default_rng(0)
    lat = rng.uniform(-90, 90, 1000)
    lon = rng.uniform(0, 360, 1000)
    for f in (src, os.path.join(SHKIT, "shkit_coeffs.gfc")):
        if not os.path.exists(f):
            continue
        mine, theirs = read_coeffs(f), shio.read_coeffs(f)
        same = np.array_equal(mine.C, theirs.C) and np.array_equal(mine.S, theirs.S)
        dv = float(np.max(np.abs(evaluate(lat, lon, mine)
                                 - np.asarray(sh_synthesis(lat, lon, theirs)))))
        print(f"  {os.path.basename(f):<28} 系数逐位相同={same}  "
              f"1000 点综合最大差={dv}")
    # 顺便看看本软件写的文件 SHKit 能不能读
    p = os.path.join(OUT, "demo_export.sh")
    if os.path.exists(p):
        back = shio.read_coeffs(p)
        print(f"  SHKit 读我们写的 demo_export.sh: nmax={back.nmax}  OK")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print(f"SHSynth {__version__} 端到端演示")
    print(f"输出目录: {OUT}")
    src = scene1_formats()
    gfc = os.path.join(SHKIT, "shkit_coeffs.gfc")
    if not os.path.exists(gfc):
        gfc = src
    scene2_global_grid(src)
    scene3_region_and_points(src)
    scene4_smooth_ewh(gfc)
    scene5_borrow_grid(src)
    scene6_export_coeffs(src)
    scene7_compare_shkit(src)
    hr("演示结束")
    print(f"所有结果都在: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
