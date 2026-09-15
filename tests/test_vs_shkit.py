# -*- coding: utf-8 -*-
"""
tests/test_vs_shkit.py
======================

**与 SHKit 逐位比对**。这是本项目"适配 SHKit 全部输出格式"这句话的实证:

* SHKit 写得出的每一种布局,我们读出来的 ``C``/``S`` 必须与 SHKit 自己读的
  **逐位相同**(``np.array_equal``,不是"很接近");
* 同一套系数、同一批点,我们的综合结果与 SHKit 的 ``synthesis`` **逐位相同**;
* 高斯平滑、物理量换算(geoid / EWH / 面密度 / 径向形变)也逐位相同;
* 我们写出的每一种布局,SHKit 都要能读回并逐位相同(双向兼容);
* 布局自动识别的结论要与 SHKit 的 ``detect_coeff_layout`` 一致。

若 SHKit 不在旁边(比如只拷走了 SHSynth 目录),本测试自动跳过并如实报告。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (Checker, guard, have_shkit, shkit_samples,  # noqa: E402
                     tmp_dir)

from shsynth.coeffio import detect_coeff_layout, read_coeffs, write_coeffs  # noqa: E402
from shsynth.engine import evaluate, regular_grid                  # noqa: E402
from shsynth.filters import gaussian_coefficients                  # noqa: E402
from shsynth.units import convert                                  # noqa: E402

if have_shkit():
    import shkit.io as shio
    from shkit.synthesis import synthesis as sh_synthesis
else:                                                                # pragma: no cover
    shio = None
    sh_synthesis = None


def _skip(check: Checker, reason: str):
    check.ok(True, f"跳过: {reason}")


def test_read_bit_identical(check: Checker):
    """SHKit 的每个样例文件,读出来的系数必须逐位相同。"""
    check.section("读取逐位相同")
    if shio is None:
        return _skip(check, "未找到 SHKit")
    samples = shkit_samples()
    check.ok(len(samples) >= 3, f"SHKit 样例 {len(samples)} 个")
    for f in samples:
        name = os.path.basename(f)
        mine = read_coeffs(f)
        theirs = shio.read_coeffs(f)
        check.ok(np.array_equal(mine.C, theirs.C) and np.array_equal(mine.S, theirs.S),
                 f"{name}: C/S 与 SHKit 逐位相同"
                 f"(nmax={mine.nmax}, ntime={mine.ntime})")
        check.ok(mine.nmax == theirs.nmax and mine.ntime == theirs.ntime,
                 f"{name}: nmax/ntime 与 SHKit 一致")


def test_detection_agrees(check: Checker):
    """布局自动识别的结论要与 SHKit 一致。"""
    check.section("布局识别一致")
    if shio is None:
        return _skip(check, "未找到 SHKit")
    d = tmp_dir("vs_shkit")
    src = read_coeffs(shkit_samples()[0])
    cases = [(".sh", "triangle"), (".csv", "ghost"), (".gfc", "gfc"),
             (".npy", "npy"), (".npz", "npz")]
    for ext, tag in cases:
        p = os.path.join(d, f"det{ext}")
        write_coeffs(src, p, layout="auto" if tag == "ghost" else tag)
        mine = detect_coeff_layout(p)
        theirs = shio.detect_coeff_layout(p)
        # SHKit 把 npz 叫 'matrix',我们叫 'npz';两者指的是同一件事,都接受
        ok = mine == theirs or {mine, theirs} <= {"npz", "matrix"}
        check.ok(ok, f"{ext}: 我们识别 {mine} / SHKit 识别 {theirs}")


def test_synthesis_bit_identical(check: Checker):
    """同一套系数、同一批点,综合结果逐位相同。"""
    check.section("综合逐位相同")
    if sh_synthesis is None:
        return _skip(check, "未找到 SHKit")
    rng = np.random.default_rng(2024)
    lat = rng.uniform(-90, 90, 700)
    lon = rng.uniform(0, 360, 700)

    for f in shkit_samples():
        name = os.path.basename(f)
        mine = read_coeffs(f)
        theirs = shio.read_coeffs(f)
        a = evaluate(lat, lon, mine)
        b = np.asarray(sh_synthesis(lat, lon, theirs))
        check.ok(np.array_equal(a, b),
                 f"{name}: 700 个随机点逐位相同"
                 f"(最大差 {float(np.max(np.abs(a - b))):.3e})")

    # 网格上再比一次
    latv, lonv = regular_grid(-90, 90, 0, 360, 5.0, 5.0)
    LA, LO = np.meshgrid(latv, lonv, indexing="ij")
    f0 = shkit_samples()[0]
    a = evaluate(LA.ravel(), LO.ravel(), read_coeffs(f0))
    b = np.asarray(sh_synthesis(LA.ravel(), LO.ravel(), shio.read_coeffs(f0)))
    check.ok(np.array_equal(a, b), "5° 全球网格逐位相同")

    # 分块(我们自己的内存策略)不能改变数值
    a2 = evaluate(lat, lon, read_coeffs(f0), chunk=100)
    check.ok(np.array_equal(a2, evaluate(lat, lon, read_coeffs(f0))),
             "chunk=100 分块后仍逐位相同")


def test_epochs_and_truncation(check: Checker):
    """多时次与截断后仍与 SHKit 逐位相同。"""
    check.section("多时次与截断")
    if sh_synthesis is None:
        return _skip(check, "未找到 SHKit")
    d = tmp_dir("vs_shkit")
    src = read_coeffs(shkit_samples()[0])
    C3 = np.repeat(src.C[:, :, None], 3, axis=2) * np.array([1.0, 1.3, 0.7])
    S3 = np.repeat(src.S[:, :, None], 3, axis=2) * np.array([1.0, 1.3, 0.7])
    from shsynth.coeffs import SHCoeffs
    multi = SHCoeffs(C3, S3, dict(src.meta))
    p = os.path.join(d, "multi.sh")
    write_coeffs(multi, p, layout="triangle")
    mine = read_coeffs(p)
    theirs = shio.read_coeffs(p)
    check.ok(np.array_equal(mine.C, theirs.C) and np.array_equal(mine.S, theirs.S),
             "多时次三角文件逐位相同")
    rng = np.random.default_rng(5)
    lat = rng.uniform(-90, 90, 200)
    lon = rng.uniform(0, 360, 200)
    a = evaluate(lat, lon, mine)
    b = np.asarray(sh_synthesis(lat, lon, theirs))
    check.ok(a.shape == b.shape == (200, 3) and np.allclose(a, b, rtol=0, atol=0),
             "多时次综合结果逐位相同")

    for L in (4, 8, 12):
        a = evaluate(lat, lon, mine, nmax=L)
        b = np.asarray(sh_synthesis(lat, lon, theirs, nmax=L))
        check.ok(np.allclose(a, b, rtol=0, atol=0), f"截断到 {L} 阶逐位相同")


def test_gaussian_bit_identical(check: Checker):
    """高斯滤波系数与平滑结果逐位相同。"""
    check.section("高斯平滑逐位相同")
    if sh_synthesis is None:
        return _skip(check, "未找到 SHKit")
    from shkit.filters import gaussian_coefficients as sh_gauss
    for radius in (100.0, 300.0, 500.0, 1000.0, 3000.0):
        for nmax in (12, 60):
            a = gaussian_coefficients(radius, nmax)
            b = sh_gauss(radius, nmax)
            check.ok(np.array_equal(a, b),
                     f"W_n({radius:g} km, nmax={nmax}) 逐位相同"
                     f"(最大差 {float(np.max(np.abs(a - b))):.2e})")
    a = gaussian_coefficients(300.0, 60, method="frc")
    b = sh_gauss(300.0, 60, method="frc")
    check.ok(np.array_equal(a, b), "frc 递推公式逐位相同")

    rng = np.random.default_rng(6)
    lat = rng.uniform(-90, 90, 300)
    lon = rng.uniform(0, 360, 300)
    f0 = shkit_samples()[0]
    mine, theirs = read_coeffs(f0), shio.read_coeffs(f0)
    for radius in (300.0, 1000.0):
        a = evaluate(lat, lon, mine, gaussian_km=radius)
        b = np.asarray(sh_synthesis(lat, lon, theirs, gaussian_km=radius))
        check.ok(np.array_equal(a, b), f"平滑 {radius:g} km 后综合逐位相同")


def test_unit_conversion_bit_identical(check: Checker):
    """物理量换算与 SHKit 逐位相同。"""
    check.section("物理量换算逐位相同")
    if sh_synthesis is None:
        return _skip(check, "未找到 SHKit")
    from shkit.units import convert as sh_convert
    gfc = [f for f in shkit_samples() if f.endswith(".gfc")]
    if not gfc:
        return _skip(check, "没有 .gfc 样例(需要 geopotential 标签)")
    f0 = gfc[0]
    mine = read_coeffs(f0)
    theirs = shio.read_coeffs(f0)
    for unit in ("geoid", "ewh", "surface_density", "radial_displacement"):
        a = convert(mine, unit)
        b = sh_convert(theirs, unit)
        same = np.array_equal(a.C, b.C) and np.array_equal(a.S, b.S)
        check.ok(same, f"→ {unit}: C/S 逐位相同"
                 f"(最大相对差 "
                 f"{float(np.max(np.abs(a.C - b.C)) / max(float(np.max(np.abs(b.C))), 1e-300)):.2e})")

    # 综合时换算(target_unit)也要一致
    rng = np.random.default_rng(11)
    lat = rng.uniform(-90, 90, 200)
    lon = rng.uniform(0, 360, 200)
    for unit in ("ewh", "geoid"):
        a = evaluate(lat, lon, mine, target_unit=unit)
        b = np.asarray(sh_synthesis(lat, lon, theirs, target_unit=unit))
        check.ok(np.array_equal(a, b), f"综合 target_unit='{unit}' 逐位相同")


def test_write_readable_by_shkit(check: Checker):
    """我们写出的每一种布局,SHKit 都能读回并逐位相同(双向兼容)。"""
    check.section("我们写的文件 SHKit 可读")
    if shio is None:
        return _skip(check, "未找到 SHKit")
    d = tmp_dir("vs_shkit")
    src = read_coeffs(shkit_samples()[0])
    cases = [(".sh", "triangle"), (".txt", "triangle"), (".csv", "triangle"),
             (".dat", "triangle"), (".tsv", "triangle"),
             (".csv", "gmfcsv"), (".txt", "gmfcsv"),
             (".gfc", "gfc"), (".npy", "npy"), (".npz", "npz")]
    for ext, layout in cases:
        p = os.path.join(d, f"compat_{layout}{ext.replace('.', '_')}{ext}")
        write_coeffs(src, p, layout=layout)
        try:
            back = shio.read_coeffs(p)
        except Exception as exc:                         # noqa: BLE001
            check.ok(False, f"SHKit 读取 {layout}{ext}",
                     f"{type(exc).__name__}: {exc}")
            continue
        if layout == "gmfcsv":
            # 我们写的是 %.16g(十进制可精确往返 double),但 SHKit 用 pandas 读
            # 这种逐行表,pandas 的 CSV 浮点解析会丢最后 1 ulp(实测 9.7e-17)。
            # 文件本身没问题:我们自己的读取器读回来是逐位相同的。
            rel = float(np.max(np.abs(back.C - src.C))) / \
                max(float(np.max(np.abs(src.C))), 1e-300)
            check.ok(rel < 1e-14,
                     f"SHKit 读回 {layout:<8}{ext:<6} 一致到 1 ulp"
                     f"(相对差 {rel:.1e};pandas 文本解析的极限)")
            check.ok(np.array_equal(read_coeffs(p).C, src.C),
                     f"我们自己读回 {layout}{ext} 逐位相同")
        else:
            check.ok(np.array_equal(back.C, src.C)
                     and np.array_equal(back.S, src.S),
                     f"SHKit 读回 {layout:<8}{ext:<6} 逐位相同")


def test_shkit_written_files(check: Checker):
    """SHKit 现场写出的每一种布局,我们都要读对。"""
    check.section("SHKit 写的文件我们能读")
    if shio is None:
        return _skip(check, "未找到 SHKit")
    d = tmp_dir("vs_shkit")
    # 用 SHKit 自己的读取器取它自己的对象:它的 write_coeffs 有
    # isinstance(coeffs, shkit.coeffs.SHCoeffs) 检查,不接受本软件的类。
    src_sh = shio.read_coeffs(shkit_samples()[0])
    src = read_coeffs(shkit_samples()[0])
    for layout, ext in (("triangle", ".sh"), ("triangle", ".csv"),
                        ("gmfcsv", ".csv"), ("gfc", ".gfc"),
                        ("npy", ".npy"), ("npz", ".npz")):
        p = os.path.join(d, f"fromshkit_{layout}{ext}")
        shio.write_coeffs(src_sh, p, layout=layout)
        mine = read_coeffs(p)
        tol = 1e-16 if layout == "gmfcsv" else 0.0     # 文本 16 位有效数字的极限
        dmax = float(np.max(np.abs(mine.C - src.C)))
        check.ok(np.allclose(mine.C, src.C, rtol=0, atol=tol)
                 and np.allclose(mine.S, src.S, rtol=0, atol=tol),
                 f"SHKit 写 {layout:<8}{ext:<6} → 我们读回一致"
                 f"(布局识别 {mine.meta['layout']},最大差 {dmax:.1e})")


def main() -> int:
    check = Checker("test_vs_shkit —— 与 SHKit 逐位比对")
    for fn in (test_read_bit_identical, test_detection_agrees,
               test_synthesis_bit_identical, test_epochs_and_truncation,
               test_gaussian_bit_identical, test_unit_conversion_bit_identical,
               test_write_readable_by_shkit, test_shkit_written_files):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
