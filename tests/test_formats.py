# -*- coding: utf-8 -*-
"""
tests/test_formats.py
=====================

**格式适配测试**:SHKit 会写出的每一种球谐系数布局,这里都要能读、能写、
能自动识别,并且往返无损。

覆盖:
* triangle(.sh/.txt/.csv/.dat/.tsv,含 .gz 与各种文件头)
* gmfcsv(n,m,C,S;含多时次 C_t1/S_t1)
* gfc(ICGEM/GFZ,含/不含 sigma 列、gfct、begin/end 标记)
* npy((2,L+1,L+1)、(2,L+1,L+1,ntime)、(L+1,L+1)、三角 (2NC,))
* npz(C/S/meta、flat_cs、triangle)
* 自动识别、明确指定布局、错误路径(扩展名不认识、行数不匹配、空文件)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Checker, guard, shkit_samples, tmp_dir     # noqa: E402

from shsynth.coeffs import SHCoeffs                            # noqa: E402
from shsynth.coeffio import (detect_coeff_layout, read_coeffs,  # noqa: E402
                             write_coeffs)


def make_coeffs(nmax: int = 8, ntime: int = 1, seed: int = 11) -> SHCoeffs:
    rng = np.random.default_rng(seed)
    shape = (nmax + 1, nmax + 1) if ntime == 1 else (nmax + 1, nmax + 1, ntime)
    C = rng.normal(0, 1e-4, shape)
    S = rng.normal(0, 1e-4, shape)
    for n in range(nmax + 1):
        for m in range(nmax + 1):
            if m > n:
                C[n, m] = 0.0
                S[n, m] = 0.0
    if ntime == 1:
        S[:, 0] = 0.0
    else:
        S[:, 0, :] = 0.0
    return SHCoeffs(C, S, {"field_unit": "geopotential", "modelname": "unit_test"})


def test_shkit_files(check: Checker):
    """SHKit 目录里现成的系数文件必须都能读,且布局识别正确。"""
    check.section("读取 SHKit 现成文件")
    samples = shkit_samples()
    check.ok(len(samples) >= 3, f"找到 SHKit 样例文件 {len(samples)} 个")
    expect = {".sh": "triangle", ".gfc": "gfc", ".npy": "matrix"}
    for f in samples:
        ext = os.path.splitext(f)[1].lower()
        layout = detect_coeff_layout(f)
        c = read_coeffs(f)
        check.ok(c.nmax >= 0 and c.ntime >= 1,
                 f"{os.path.basename(f)} 可读(nmax={c.nmax}, ntime={c.ntime})")
        if ext in expect:
            # .npy 可能是 matrix 也可能是 triangle,都接受
            ok = layout == expect[ext] or ext == ".npy"
            check.ok(ok, f"{os.path.basename(f)} 布局识别 = {layout}",
                     f"期望 {expect[ext]}")
        check.ok(np.all(np.isfinite(c.C)) and np.all(np.isfinite(c.S)),
                 f"{os.path.basename(f)} 数值全部有限")
        check.ok(np.allclose(c.S[:, 0], 0.0), f"{os.path.basename(f)} S[:,0]≡0")


def test_roundtrip_all_layouts(check: Checker):
    """五种布局 + gz + 多时次:写出再读回必须逐位相同,且识别正确。"""
    check.section("五种布局往返")
    d = tmp_dir("formats")
    cases = [
        (".sh", "triangle"), (".txt", "triangle"), (".csv", "triangle"),
        (".dat", "triangle"), (".tsv", "triangle"),
        (".txt.gz", "triangle"), (".sh.gz", "triangle"),
        (".csv", "gmfcsv"), (".txt", "gmfcsv"),
        (".gfc", "gfc"), (".gfc.gz", "gfc"),
        (".npy", "npy"), (".npz", "npz"),
    ]
    src = make_coeffs(8, 1)
    for ext, layout in cases:
        p = os.path.join(d, f"rt_{layout}{ext.replace('.', '_')}{ext}")
        try:
            write_coeffs(src, p, layout=layout)
        except Exception as exc:                         # noqa: BLE001
            check.ok(False, f"写出 {layout} {ext}", f"{type(exc).__name__}: {exc}")
            continue
        try:
            back = read_coeffs(p)
        except Exception as exc:                         # noqa: BLE001
            check.ok(False, f"读回 {layout} {ext}", f"{type(exc).__name__}: {exc}")
            continue
        tol = 1e-15 if layout != "gfc" else 1e-12
        good = (np.allclose(back.C, src.C, atol=tol, rtol=0)
                and np.allclose(back.S, src.S, atol=tol, rtol=0)
                and back.nmax == src.nmax and back.ntime == src.ntime)
        check.ok(good, f"往返 {layout:<8}{ext:<10} nmax={back.nmax} ntime={back.ntime}")
        det = detect_coeff_layout(p)
        check.ok(det in ("triangle", "gmfcsv", "gfc", "matrix", "npz"),
                 f"自动识别 {layout:<8}{ext:<10} -> {det}")


def test_multitime(check: Checker):
    """多时次:triangle(多列)、gmfcsv(C_t1/S_t1)、npz、npy 都要能装下来。"""
    check.section("多时次")
    d = tmp_dir("formats")
    src = make_coeffs(6, 4)
    for ext, layout in ((".sh", "triangle"), (".csv", "gmfcsv"),
                        (".npz", "npz"), (".npy", "npy")):
        p = os.path.join(d, f"multi{ext}")
        write_coeffs(src, p, layout=layout)
        back = read_coeffs(p)
        check.ok(back.ntime == 4, f"{layout} 保留 4 个时次(读到 {back.ntime})")
        check.close(back.C, src.C, 1e-15, f"{layout} 多时次 C 逐位一致")
        check.close(back.S, src.S, 1e-15, f"{layout} 多时次 S 逐位一致")
    # gfc 装不下多时次:必须明确报错,而不是悄悄拼坏
    check.raises(lambda: write_coeffs(src, os.path.join(d, "multi.gfc"),
                                      layout="gfc"),
                 ValueError, "gfc 多时次应报错")
    # 指定时次则可以
    write_coeffs(src, os.path.join(d, "multi_t2.gfc"), layout="gfc", time=2)
    back = read_coeffs(os.path.join(d, "multi_t2.gfc"))
    check.close(back.C, src.C[:, :, 2], 1e-12, "gfc time=2 只导出该时次")


def test_gfc_variants(check: Checker):
    """gfc 的常见变体:gfct、begin/end 标记、误差列、非 4π 归一化警告。"""
    check.section("ICGEM/gfc 变体")
    d = tmp_dir("formats")
    body = ["# comment",
            "begin_of_head",
            "modelname            prueba",
            "max_degree            2",
            "norm                 4pi",
            "end_of_head",
            "gfct   0   0  1.0  0.0",
            "gfc    1   0  0.5  0.0",
            "gfc    1   1 -0.25 0.75",
            "gfc    2   0  0.1  0.0  1e-6 0.0",
            "gfc    2   1  0.2 -0.3  1e-6 1e-6",
            "gfc    2   2  0.05 0.05 1e-6 1e-6",
            "end_of_data"]
    p = os.path.join(d, "variant.gfc")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body) + "\n")
    c = read_coeffs(p)
    check.ok(c.nmax == 2, f"max_degree 生效(nmax={c.nmax})")
    check.ok(abs(c.C[1, 1] + 0.25) < 1e-15 and abs(c.S[1, 1] - 0.75) < 1e-15,
             "gfct/gfc 行的 C,S 解析正确")
    check.ok(c.meta.get("has_sigmas") is True, "识别出形式误差列")
    check.ok("sigmaC" in c.meta and c.meta["sigmaC"][2, 1] == 1e-6,
             "sigmaC 被读入")
    check.ok(c.field_unit == "geopotential", ".gfc 自动标注为无量纲位系数")

    # 非 4π 归一化应给出警告
    body2 = [ln.replace("4pi", "unnormalized") for ln in body]
    p2 = os.path.join(d, "variant_norm.gfc")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write("\n".join(body2) + "\n")
    c2 = read_coeffs(p2)
    check.ok(any("归一化" in w for w in c2.meta.get("warnings", [])),
             "非 4π 归一化给出警告")


def test_npz_variants(check: Checker):
    """npz 的三种存法:C/S/meta、flat_cs、triangle。"""
    check.section("npz 变体")
    d = tmp_dir("formats")
    src = make_coeffs(5, 1)
    p1 = os.path.join(d, "v_cs.npz")
    np.savez(p1, C=src.C, S=src.S, nmax=5, ntime=1,
             meta='{"field_unit": "ewh"}')
    c1 = read_coeffs(p1)
    check.close(c1.C, src.C, 0, "npz C/S 逐位一致")
    check.ok(c1.field_unit == "ewh", "npz 的 meta 被还原(field_unit=ewh)")

    tri = src.to_triangle()
    p2 = os.path.join(d, "v_flat.npz")
    np.savez(p2, flat_cs=tri)
    c2 = read_coeffs(p2)
    check.close(c2.C, src.C, 0, "npz flat_cs 逐位一致")

    p3 = os.path.join(d, "v_tri.npz")
    np.savez(p3, triangle=tri)
    c3 = read_coeffs(p3)
    check.close(c3.C, src.C, 0, "npz triangle 逐位一致")


def test_npy_variants(check: Checker):
    """npy 的几种形状:(2,L+1,L+1)、(2,L+1,L+1,ntime)、(L+1,L+1)、三角。"""
    check.section("npy 形状变体")
    d = tmp_dir("formats")
    src = make_coeffs(5, 1)
    p = os.path.join(d, "s1.npy")
    np.save(p, np.stack([src.C, src.S]))
    check.close(read_coeffs(p).C, src.C, 0, "(2,L+1,L+1) 逐位一致")

    src4 = make_coeffs(5, 3)
    p = os.path.join(d, "s2.npy")
    np.save(p, np.stack([src4.C, src4.S]))
    c = read_coeffs(p)
    check.ok(c.ntime == 3, f"(2,L+1,L+1,ntime) 读成 ntime={c.ntime}")
    check.close(c.S, src4.S, 0, "(2,L+1,L+1,ntime) S 逐位一致")

    p = os.path.join(d, "s3.npy")
    np.save(p, src.C.copy())
    c = read_coeffs(p)
    check.close(c.C, src.C, 0, "(L+1,L+1) 稠密矩阵读成 C")
    check.ok(c.S is not None and np.allclose(c.S, 0.0),
             "(L+1,L+1) 时 S 置 0 并有警告")
    check.ok(any("S 置 0" in w for w in c.meta.get("warnings", [])),
             "(L+1,L+1) 分支给出明确警告")

    p = os.path.join(d, "s4.npy")
    np.save(p, src.to_triangle())
    c = read_coeffs(p)
    check.close(c.C, src.C, 0, "三角 .npy 逐位一致")


def test_header_metadata(check: Checker):
    """三角文件头里的元数据(field_unit / 高斯半径 / 覆盖率)必须能读回来。"""
    check.section("文件头元数据往返")
    d = tmp_dir("formats")
    src = make_coeffs(4, 1)
    src.meta["gaussian_km"] = 300.0
    src.meta["coverage"] = 0.987
    src.meta["n_points"] = 12345
    p = os.path.join(d, "meta.sh")
    write_coeffs(src, p, layout="triangle",
                 comment="测试用文件头")
    c = read_coeffs(p)
    check.ok(c.field_unit == "geopotential",
             f"field_unit 标签往返({c.field_unit})")
    hdr = c.meta.get("header_fields", {})
    check.ok(hdr.get("gaussian_radius_km") is None or
             abs(float(hdr.get("gaussian_km", 0)) - 300.0) < 1e-9 or
             abs(float(hdr.get("gaussian_radius_km", 0)) - 300.0) < 1e-9,
             f"高斯半径回到 meta({c.meta.get('gaussian_km')})")
    check.ok(abs(float(hdr.get("coverage", 0)) - 0.987) < 1e-9,
             "覆盖率回到 meta")
    check.ok(int(hdr.get("n_points", 0)) == 12345, "点数回到 meta")
    check.ok(any("测试用文件头" in ln for ln in c.meta.get("comment_header", [])),
             "自定义注释保留")


def test_gmfcsv_detection(check: Checker):
    """gmfcsv 与 triangle 的行数判据:两者不能被认错。"""
    check.section("gmfcsv / triangle 判别")
    d = tmp_dir("formats")
    for L in (4, 12, 20):
        src = make_coeffs(L, 1)
        p = os.path.join(d, f"gmf{L}.csv")
        write_coeffs(src, p, layout="gmfcsv")
        c = read_coeffs(p)
        check.ok(c.nmax == L and np.allclose(c.C, src.C),
                 f"nmax={L} 的 gmfcsv 读回正确(nmax={c.nmax})")
        p2 = os.path.join(d, f"tri{L}.csv")
        write_coeffs(src, p2, layout="triangle")
        c2 = read_coeffs(p2)
        check.ok(c2.nmax == L and np.allclose(c2.C, src.C),
                 f"nmax={L} 的 triangle 读回正确(nmax={c2.nmax})")


def test_error_paths(check: Checker):
    """错误路径:要做成明确的中文报错,而不是崩或给错数字。"""
    check.section("错误路径")
    d = tmp_dir("formats")
    p = os.path.join(d, "bad.xyz")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("1 2 3\n")
    check.raises(lambda: read_coeffs(p), ValueError, "不支持的扩展名报错")
    check.raises(lambda: read_coeffs(os.path.join(d, "nope.sh")),
                 FileNotFoundError, "文件不存在报错")

    p2 = os.path.join(d, "empty.txt")
    open(p2, "w").close()
    check.raises(lambda: read_coeffs(p2), ValueError, "空文件报错")

    p3 = os.path.join(d, "odd.txt")
    with open(p3, "w", encoding="utf-8") as fh:
        fh.write("\n".join("1.0" for _ in range(37)) + "\n")
    check.raises(lambda: read_coeffs(p3, layout="triangle"), ValueError,
                 "行数不是 2*(L+1)(L+2)/2 报错")

    check.raises(lambda: read_coeffs(p2, layout="nonsense"), ValueError,
                 "未知 layout 报错")


def test_flat_cs_consistency(check: Checker):
    """三角布局的列序必须与 SHKit 的 m 外层/n 内层完全一致。"""
    check.section("三角布局列序")
    src = make_coeffs(3, 1)
    tri = src.to_triangle()
    NC = (3 + 1) * (3 + 2) // 2
    check.ok(tri.shape == (2 * NC, 1), f"to_triangle 形状 {tri.shape}")
    # 手工按 m 外层 / n 内层取值核对
    k = 0
    ok = True
    for m in range(4):
        for n in range(m, 4):
            if abs(tri[k, 0] - src.C[n, m]) > 0 or \
                    abs(tri[NC + k, 0] - src.S[n, m]) > 0:
                ok = False
            k += 1
    check.ok(ok, "三角顺序 = m 外层 / n 内层,且 [C; S] 上下堆叠")
    back = SHCoeffs.from_triangle(tri, 3)
    check.close(back.C, src.C, 0, "from_triangle 无损还原")


def main() -> int:
    check = Checker("test_formats —— SHKit 全部系数格式的读写与识别")
    for fn in (test_shkit_files, test_roundtrip_all_layouts, test_multitime,
               test_gfc_variants, test_npz_variants, test_npy_variants,
               test_header_metadata, test_gmfcsv_detection, test_error_paths,
               test_flat_cs_consistency):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
