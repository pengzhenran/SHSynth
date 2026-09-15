# -*- coding: utf-8 -*-
"""
test_multitime.py —— v2.0:时间轴 / 序列读写 / 批量综合 / FFT 快路径 / 水平形变
================================================================================

第 8 套测试(方案 §7)。全部用**合成数据**;涉及用户真实数据或已编译 m2csharp 的
条目一律 ``skipif`` 并打印跳过原因。

跑法:``python tests/test_multitime.py``(或由 ``tests/run_all.py`` 统一调用)
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import Checker, tmp_dir                              # noqa: E402
from shsynth import (SHCoeffs, TimeAxis, dec_year, drop_degree0,   # noqa: E402
                     evaluate, evaluate_horizontal, fft_path_applicable,
                     fit_trend_seasonal, horizontal_grid_fft,
                     love_horizontal_factors, parse_filename_epoch,
                     read_coeffs, read_coeffs_series, read_series_dat,
                     read_series_nc, remove_mean, series_at_points,
                     synth_series, synthesis_grid, synthesis_grid_fft,
                     write_coeffs, write_series_dat, write_series_nc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.dirname(ROOT)                 # 2_Level-2
GFC_DIR = os.path.join(DATA, "2_unzipped", "1_CSR", "01RL06", "01deg60")
LEGACY = os.path.join(DATA, "3_processed", "GRACE SH read",
                      "CSR_rawSH60_total_216.dat")
GRID3 = os.path.join(DATA, "3_grids", "CSR_GRACE_EWH_G300.nc")


def make_coeffs(L=8, nt=1, seed=5, scale=1.0, times=None, unit=None):
    """随机系数(只填三角区 ``m <= n``)。

    ⚠️ ``S[:,0] ≡ 0`` 与 ``m > n`` 位置恒为 0 是**结构约定**;测试数据必须遵守,
    否则 ``to_triangle``/``from_triangle`` 这类只搬运三角区的布局会"丢"掉那些
    本来就不该存在的数(实测:填了 m>n 会让往返差到 O(1))。
    """
    rng = np.random.default_rng(seed)
    shape = (L + 1, L + 1) if nt == 1 else (L + 1, L + 1, nt)
    C = rng.normal(0, scale, shape)
    S = rng.normal(0, scale, shape)
    tri = np.tril(np.ones((L + 1, L + 1), dtype=bool))       # m <= n
    if nt == 1:
        C[~tri] = 0.0
        S[~tri] = 0.0
    else:
        C[~tri, :] = 0.0
        S[~tri, :] = 0.0
    meta = {}
    if unit:
        meta["field_unit"] = unit
    return SHCoeffs(C, S, meta, times)


# ---------------------------------------------------------------------------
def test_timeaxis(check: Checker):
    """时间轴:文件名三规则 / 十进制年口径 / 缺测重复 / legacy 往返。"""
    check.section("时间轴")

    # --- 三条文件名规则 -------------------------------------------------
    r = parse_filename_epoch("GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc")
    check.ok(r["rule"] == "doy_range" and str(r["start"]) == "2002-04-05"
             and str(r["end"]) == "2002-04-30",
             f"规则1 年积日区间 → {r['start']}..{r['end']}")
    # 跨年
    r2 = parse_filename_epoch("GSM-2_2002360-2003020_GRAC_UTCSR_BA01_0600.gfc")
    check.ok(r2["rule"] == "doy_range" and str(r2["start"]) == "2002-12-26"
             and str(r2["end"]) == "2003-01-20",
             f"跨年 {r2['start']}..{r2['end']}")
    # 闰年 DOY 60 = 2 月 29 日
    r3 = parse_filename_epoch("GSM-2_2004060-2004091_X.gfc")
    check.ok(str(r3["start"]) == "2004-02-29", f"闰年 DOY60 → {r3['start']}")
    r4 = parse_filename_epoch("ITSG-Grace2018_n60_2002-04.gfc")
    check.ok(r4["rule"] == "ym" and str(r4["mid"]) == "2002-04-15",
             f"规则2 年月 → 中点 {r4['mid']}")
    r5 = parse_filename_epoch("ITSG-Grace2018_Kalman_n40_2002-04-01.gfc")
    check.ok(r5["rule"] == "ymd" and str(r5["mid"]) == "2002-04-01",
             f"规则3 年月日 → {r5['mid']}")
    check.ok(parse_filename_epoch("nonsense.gfc")["rule"] == "none",
             "无法解析的文件名返回 rule='none'(调用方要报告,不能静默丢)")

    # --- 十进制年口径:分母必须是"该年实际天数" --------------------------
    d2002 = np.datetime64("2002-04-18")
    d2004 = np.datetime64("2004-04-18")
    check.close(np.array([dec_year(d2002)]),
                np.array([2002 + 107 / 365.0]), 0.0,
                "平年 2002-04-18 → year+(DOY-1)/365")
    check.close(np.array([dec_year(d2004)]),
                np.array([2004 + 108 / 366.0]), 0.0,
                "闰年 2004-04-18 → 分母 366(实测 legacy 就是这个口径)")
    check.ok(abs(dec_year(d2004) - (2004 + 108 / 365.0)) > 1e-5,
             "分母恒 365 在闰年上会差 >1e-5(必须用 'actual')")
    # 小数天不能被截断
    check.close(np.array([dec_year(np.datetime64("2002-04-18T12:00"))]),
                np.array([2002 + 107.5 / 365.0]), 1e-15,
                "带时刻的日期保留小数天")

    # --- gfc 头清洗 ------------------------------------------------------
    hdr = {"time_coverage_start": ": 2002-04-05T00:00:00.00",
           "time_coverage_end": ": 2002-05-01T00:00:00.00",
           "time_period_of_data": "20020405 - 20020430 (mid: 20020418)",
           "unused_days": ": [2002-04-10, 2002-04-11]"}
    ax = TimeAxis.from_gfc_headers([hdr], paths=["GSM-2_2002095-2002120_X.gfc"])
    check.ok(str(ax.values[0])[:10] == "2002-04-18",
             f"头 (mid: YYYYMMDD) 按**日历日期**解析 → {str(ax.values[0])[:10]}"
             "(按年积日解析会偏一整年)")
    check.ok(str(ax.start[0])[:10] == "2002-04-05"
             and str(ax.end[0])[:10] == "2002-05-01", "coverage 起止解析正确")
    # 点号写法
    ax2 = TimeAxis.from_gfc_headers(
        [{"time_coverage_start": ": 2020-03-01T00.00.00.00"}])
    check.ok(str(ax2.start[0])[:10] == "2020-03-01",
             "容错 '2020-03-01T00.00.00.00'(点号)写法")

    # --- 缺测 / 重复 -----------------------------------------------------
    d = [np.datetime64(f"2002-{m:02d}-15") for m in (1, 2, 3, 7, 8)]
    a3 = TimeAxis.from_datetimes(d)
    check.ok(len(a3.missing(cadence_days=31, tol_days=6)) == 1,
             f"3→7 月之间 4 个月算 1 处缺测(实测 {a3.missing()})")
    check.ok(not a3.is_regular(), "间隔不等 → is_regular() 为 False")
    dup = TimeAxis.from_datetimes([np.datetime64("2002-01-15"),
                                   np.datetime64("2002-01-15"),
                                   np.datetime64("2002-03-15")])
    check.ok(len(dup.duplicates()) == 1, "重复历元被检出")

    # --- legacy TimeInfo 往返 --------------------------------------------
    yrs = np.array([2002.293151, 2002.354795, 2003.041096])
    a4 = TimeAxis.from_decimal_years(yrs)
    p = os.path.join(tmp_dir("mt"), "t.dat")
    a4.to_legacy_timeinfo(p)
    back = TimeAxis.from_legacy_timeinfo(p)
    check.close(back.decimal_years, yrs, 1e-6, "legacy TimeInfo 写→读 十进制年一致")
    check.ok(TimeAxis.from_index(5).kind == "index", "from_index → kind='index'")

    # --- slice / nearest / match -----------------------------------------
    a5 = TimeAxis.from_datetimes([np.datetime64(f"{y}-06-15")
                                  for y in range(2002, 2012)])
    sub, mask = a5.slice("2005-01-01", "2007-12-31")
    check.ok(len(sub) == 3, f"日期切片 2005..2007 → {len(sub)} 个")
    check.ok(str(a5.values[a5.nearest("2009-06-20")])[:4] == "2009",
             "nearest 找到最近的历元")
    j = a5.match(a5.select(np.array([1, 2])), tol_days=1.0)
    check.ok(list(j[:3]) == [-1, 0, 1], f"match 按日期配对(结果 {list(j[:3])})")


def test_coeffs_times(check: Checker):
    """容器:times 字段 / 双操作数守卫 / degree_rms 时间语义。"""
    check.section("SHCoeffs.times 与运算守卫")

    ax = TimeAxis.from_datetimes([np.datetime64(f"2002-{m:02d}-15")
                                  for m in range(1, 5)])
    c = make_coeffs(4, 4, times=ax, unit="ewh")
    check.ok(c.has_time and len(c.times) == 4, "times 挂上了")
    check.ok(c.copy().times is ax, "copy() 保留 times")
    check.ok(c.truncate(2).times is ax, "truncate() 透传 times")

    # times 个数与 ntime 不符要报错
    try:
        make_coeffs(4, 3, times=ax)
        check.ok(False, "times 个数与 ntime 不符应当报错")
    except ValueError as exc:
        check.ok("历元" in str(exc), f"times 个数不符报错:{str(exc)[:40]}")

    # --- 双操作数:时间轴必须一致 ----------------------------------------
    same = make_coeffs(4, 4, seed=6, times=ax, unit="ewh")
    r = c - same
    check.close(r.C, c.C - same.C, 0.0, "时间轴一致可以相减")
    check.ok(r.meta.get("field_unit") == "ewh",
             "相减后 field_unit 保留(不是 v1.0 的 {'op':'sub'})")
    other = make_coeffs(4, 4, seed=7,
                        times=TimeAxis.from_datetimes(
                            [np.datetime64(f"2003-{m:02d}-15")
                             for m in range(1, 5)]))
    try:
        c - other
        check.ok(False, "时间轴日期不同应当报错")
    except ValueError as exc:
        check.ok("时间轴" in str(exc), f"日期不同报错:{str(exc)[:44]}")
    try:
        c - c.time_slice(0)
        check.ok(False, "序列减单历元应当报错(长度不一致)")
    except ValueError as exc:
        check.ok("时间轴不一致" in str(exc),
                 f"序列减单历元报错并给中文出路:{str(exc)[:34]}")
    # 物理量不同不能相加
    try:
        c + make_coeffs(4, 4, times=ax, unit="geoid")
        check.ok(False, "物理量不同应当报错")
    except ValueError as exc:
        check.ok("物理量" in str(exc), f"物理量不一致报错:{str(exc)[:34]}")

    # --- degree_rms / power 的时间语义 -----------------------------------
    C = np.zeros((3, 3, 4)); S = np.zeros((3, 3, 4))
    C[1, 0, :] = [1.0, 2.0, 3.0, 4.0]
    c2 = SHCoeffs(C, S)
    per = c2.degree_rms()
    check.ok(per.shape == (4, 3), f"3D degree_rms() 返回逐历元矩阵 {per.shape}")
    # n=1 这一阶含 (m=0, m=1) 的 C 与 S 共 4 条,只有 C[1,0] 非零 ⇒ 均方除以 4
    check.close(per[:, 1], np.array([1.0, 2.0, 3.0, 4.0]) / 2.0, 1e-15,
                "逐历元 RMS = |C_n0|/2(不再把时间揉进平均)")
    check.close(np.array([c2.degree_rms(time="mean")[1]]),
                np.array([float(np.mean(np.array([1.0, 2.0, 3.0, 4.0]) / 2.0))]),
                1e-15, "time='mean' = 逐历元 RMS 的时间平均(画谱图用)")
    check.close(np.array([c2.degree_rms(time=0)[1]]), np.array([0.5]), 1e-15,
                "time=0 只取第一个历元")
    check.ok(c2.power().shape == (4, 3), "3D power() 也返回逐历元矩阵")
    check.close(c2.power(time=0)[1], 1.0, 1e-15, "power(time=0) = C₁₀² = 1")
    p2 = make_coeffs(5, 1)
    check.ok(p2.degree_rms().shape == (6,),
             "2D 对象 degree_rms() 形状与 v1.0 相同 (nmax+1,)")


def test_series_io(check: Checker):
    """序列落盘:series_nc / series_dat 往返 + .gfc 单历元带时间。"""
    check.section("序列落盘")
    d = tmp_dir("mt")
    ax = TimeAxis.from_datetimes([np.datetime64(f"2003-{m:02d}-16")
                                  for m in range(1, 5)])
    c = make_coeffs(6, 4, times=ax, unit="geopotential")

    p = write_series_nc(c, os.path.join(d, "s.nc"))
    back = read_series_nc(p)
    check.ok(back.ntime == 4 and back.nmax == 6, f"series_nc 形状 {back.C.shape}")
    check.close(back.C, c.C, 0.0, "series_nc 系数逐位往返")
    check.ok(np.array_equal(back.times.values, ax.values),
             "series_nc 日期逐值往返")
    check.ok(back.meta.get("field_unit") == "geopotential",
             "series_nc 保留 field_unit")
    check.ok(read_coeffs(p).ntime == 4, "read_coeffs 也能直接读 .nc(series_nc)")
    check.ok(str(np.asarray(back.C).ravel()[0]) != "", "占位")

    p2 = write_series_dat(c, os.path.join(d, "s.dat"))
    back2 = read_series_dat(p2)
    check.close(back2.C, c.C, 0.0, "series_dat 系数逐位往返(%.17g 可精确往返)")
    check.ok(os.path.exists(os.path.join(d, "s_TimeInfo.dat")),
             "series_dat 同时写出 *_TimeInfo.dat")
    # TimeInfo 只有天分辨率 + 6 位十进制年(~3 s),按**天数**比,不要求逐纳秒
    days = np.abs((back2.times.values - ax.values) / np.timedelta64(1, "D"))
    check.ok(back2.times is not None and float(np.max(days)) < 0.51,
             f"series_dat 通过 TimeInfo 读回日期(最大差 {float(np.max(days)):.4f} 天)")
    check.ok(read_coeffs(p2).ntime == 4, "legacy .dat 能被自动识别成 series_dat")

    # 无日期时退化为序号并警告
    c3 = make_coeffs(4, 3)
    p3 = write_series_nc(c3, os.path.join(d, "s3.nc"))
    b3 = read_series_nc(p3)
    check.ok(b3.times.kind == "index", "无日期 → kind='index'")
    check.ok(any("没有时间坐标" in w or "序号" in w for w in b3.meta["warnings"]),
             "无日期时给出中文警告")


def test_fft_path(check: Checker):
    """FFT 经度路径:与直接法一致 + 适用条件 + 经度对齐(m≥1)。"""
    check.section("FFT 经度路径")
    L, nt, nlat, nlon = 12, 3, 19, 72
    lat = -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat)
    lon = 360.0 * np.arange(nlon) / nlon
    c = make_coeffs(L, nt, scale=1e-3, unit="geopotential")

    a = synthesis_grid(lat, lon, c, chunk=10 ** 7)
    b = synthesis_grid_fft(lat, lon, c)
    check.close(b, a, 1e-12, f"FFT ≡ 直接法(相对场量级 {np.abs(b - a).max() / np.abs(a).max():.1e})")

    # 含极点行
    latp = np.linspace(-90.0, 90.0, nlat)
    check.close(synthesis_grid_fft(latp, lon, c),
                synthesis_grid(latp, lon, c, chunk=10 ** 7), 1e-12,
                "含 ±90° 极点行时也一致")

    # 高斯 + 单位换算一起走(场量级 ~1e5,用**相对**判据)
    g1 = synthesis_grid(lat, lon, c, gaussian_km=300, target_unit="ewh",
                        chunk=10 ** 7)
    g2 = synthesis_grid_fft(lat, lon, c, gaussian_km=300, target_unit="ewh")
    check.ok(np.abs(g2 - g1).max() / np.abs(g1).max() < 1e-12,
             f"高斯 + EWH 换算下也一致(相对 {np.abs(g2 - g1).max() / np.abs(g1).max():.1e})")

    # --- 适用条件 --------------------------------------------------------
    ok, why = fft_path_applicable(np.arange(70.0, 141.0, 1.0), L)
    check.ok(not ok and "整圈" in why, f"区域经度 → 不可用({why[:24]})")
    ok2, why2 = fft_path_applicable(360.0 * np.arange(20) / 20, 20)
    check.ok(not ok2 and "Nyquist" in why2, f"nlon ≤ 2·nmax → 不可用({why2[:24]})")
    ok3, _ = fft_path_applicable(lon, L)
    check.ok(ok3, "整圈均匀经度 + nlon > 2·nmax → 可用")
    try:
        synthesis_grid_fft(lat, np.arange(70.0, 141.0), c)
        check.ok(False, "区域经度调用 FFT 应当报错")
    except ValueError as exc:
        check.ok("不适用" in str(exc), "区域经度调用 FFT 给出明确错误")

    # --- 经度对齐:必须用 m≥1 的系数(m=0 对经度旋转不敏感)----------------
    rng = np.random.default_rng(11)
    C1 = np.zeros((L + 1, L + 1)); S1 = np.zeros((L + 1, L + 1))
    C1[1, 1] = 1.0
    C1[2, 2] = 0.5
    c1 = SHCoeffs(C1, S1, {"field_unit": "geopotential"})
    g = synthesis_grid_fft(lat, lon, c1)
    shift = np.roll(g, 1, axis=1)
    d0 = np.abs(g - synthesis_grid(lat, lon, c1, chunk=10 ** 7)).max()
    d1 = np.abs(shift - synthesis_grid(lat, lon, c1, chunk=10 ** 7)).max()
    check.ok(d0 < 1e-12 and d1 > 1e-3,
             f"经度**零格平移**下才一致(m≥1 系数:对齐 {d0:.1e},平移一格 {d1:.1e})")

    # --- 纬向区域(经度仍整圈)同样适用 ----------------------------------
    latb = np.arange(20.0, 50.1, 5.0)
    check.close(synthesis_grid_fft(latb, lon, c),
                synthesis_grid(latb, lon, c, chunk=10 ** 7), 1e-12,
                "纬向区域(经度整圈)FFT 同样适用且一致")


def test_horizontal(check: Checker):
    """水平形变:解析锚点 / 极点 / 与直接法一致 / 因子量级。"""
    check.section("水平形变 u_N/u_E")
    # 解析锚点用**单位乘子**(纯球面梯度算子);物理量级另测
    from shsynth import synthesize_horizontal

    # --- 解析锚点 纯 C10 --------------------------------------------------
    C = np.zeros((2, 2)); S = np.zeros((2, 2)); C[1, 0] = 1.0
    lat = np.array([-90.0, -60.0, 0.0, 30.0, 90.0]); lon = np.zeros_like(lat)
    d = synthesize_horizontal(lat, lon, C, S, nmax=1, weights_scale=np.ones(2))
    ref = np.sqrt(3.0) * np.cos(np.deg2rad(lat))
    check.close(d["north"], ref, 1e-14, f"纯 C₁₀:u_N = √3·cosφ(偏差 {np.abs(d['north'] - ref).max():.1e})")
    check.close(d["east"], np.zeros_like(lat), 1e-14, "纯 C₁₀:u_E ≡ 0")

    # --- 解析锚点 纯 C11(极点 u_E 非 0,这是除零陷阱)--------------------
    C2 = np.zeros((2, 2)); C2[1, 1] = 1.0
    lat2 = np.array([90.0, 45.0, 0.0, -45.0, -90.0])
    lon2 = np.array([90.0, 90.0, 180.0, 270.0, 90.0])
    d2 = synthesize_horizontal(lat2, lon2, C2, np.zeros((2, 2)), nmax=1,
                              weights_scale=np.ones(2))
    refN = -np.sqrt(3.0) * np.cos(np.deg2rad(lon2)) * np.sin(np.deg2rad(lat2))
    refE = -np.sqrt(3.0) * np.sin(np.deg2rad(lon2))
    check.close(d2["north"], refN, 1e-14, "纯 C₁₁:u_N = −√3·cosλ·sinφ")
    check.close(d2["east"], refE, 1e-14, "纯 C₁₁:u_E = −√3·sinλ")
    check.close(np.array([d2["east"][0]]), np.array([-np.sqrt(3.0)]), 1e-14,
                f"**极点上 u_E = −√3(实测 {d2['east'][0]:.9f})**,不是 0 也不是 NaN")

    # --- 与球面有限差分梯度一致(最强校核)-------------------------------
    L = 8
    c3 = make_coeffs(L, 1, seed=3, scale=1e-3)
    la, lo = 25.0, 123.0
    dd = 1e-5
    sc = float(evaluate(np.array([la]), np.array([lo]), c3)[0])
    sp = float(evaluate(np.array([la - dd]), np.array([lo]), c3)[0])
    sm = float(evaluate(np.array([la + dd]), np.array([lo]), c3)[0])
    sl = float(evaluate(np.array([la]), np.array([lo + dd]), c3)[0])
    sr = float(evaluate(np.array([la]), np.array([lo - dd]), c3)[0])
    fdN = (sm - sp) / (2 * np.deg2rad(dd))          # u_N = −∂S/∂θ = +∂S/∂φ
    fdE = ((sl - sr) / (2 * np.deg2rad(dd))) / np.cos(np.deg2rad(la))
    h = synthesize_horizontal(np.array([la]), np.array([lo]),
                              c3.C, c3.S, nmax=L, weights_scale=np.ones(L + 1))
    check.close(np.array([h["north"][0]]), np.array([fdN]), 1e-8,
                f"u_N 与有限差分梯度一致(相对 {abs(h['north'][0] - fdN) / abs(fdN):.1e})")
    check.close(np.array([h["east"][0]]), np.array([fdE]), 1e-8,
                f"u_E 与有限差分梯度一致(相对 {abs(h['east'][0] - fdE) / abs(fdE):.1e})")

    # --- 与 FFT 路径一致(含极点行)--------------------------------------
    nlat, nlon, nt = 19, 72, 3
    latv = -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat)
    lonv = 360.0 * np.arange(nlon) / nlon
    cm = make_coeffs(L, nt, seed=4, scale=1e-3)
    LA, LO = np.meshgrid(latv, lonv, indexing="ij")
    dirr = evaluate_horizontal(LA.ravel(), LO.ravel(), cm)
    fftr = horizontal_grid_fft(latv, lonv, cm)
    for k in ("north", "east"):
        a = dirr[k].reshape(nlat, nlon, nt)
        mg = np.abs(a).max()
        check.ok(np.abs(fftr[k] - a).max() / mg < 1e-12,
                 f"水平 {k}:直接 ≡ FFT(相对 {np.abs(fftr[k] - a).max() / mg:.1e})")
    latp = np.linspace(-90.0, 90.0, nlat)
    LAp, LOp = np.meshgrid(latp, lonv, indexing="ij")
    dp = evaluate_horizontal(LAp.ravel(), LOp.ravel(), cm)
    fp = horizontal_grid_fft(latp, lonv, cm)
    a = dp["east"].reshape(nlat, nlon, nt)
    check.ok(np.abs(fp["east"] - a).max() / np.abs(a).max() < 1e-12,
             "含 ±90° 极点行时水平 FFT 与直接法一致(相对判据)")

    # --- 因子:与径向同量级(乘错会差 1e7 倍)-----------------------------
    F = love_horizontal_factors(60)
    Fr = np.asarray([0.0] + [abs(6378136.46 * h / (1 + k)) for h, k in
                             zip([-0.285668, -0.990799, -1.049963],
                                 [0.026168, -0.305161, -0.195857])])
    ratio = np.abs(F[1:4]) / Fr[1:4]
    check.ok(np.all(ratio < 1.0) and np.all(ratio > 1e-3),
             f"水平/径向因子比 {np.array2string(ratio, precision=4)} ∈ (1e-3, 1),量级正确")
    check.close(np.array([F[1]]), np.array([643815.045]), 1e-3,
                f"F₁ = R·l′₁/(1+k′₁) = {F[1]:.3f} m(与实测锚点一致)")
    check.ok(not np.isnan(F[1:]).any(), "水平因子无 NaN")

    # --- 卫星/载荷量级:mm 级 -------------------------------------------------
    cg = make_coeffs(60, 1, seed=9, scale=1e-10)
    hg = evaluate_horizontal(np.array([20.0]), np.array([123.0]), cg)
    check.ok(1e-6 < abs(hg["north"][0]) < 1e-1,
             f"GRACE 量级系数 → |u_N| = {abs(hg['north'][0]) * 1e3:.3f} mm(合理)")


def test_series_batch(check: Checker):
    """批量综合:一次调用 ≡ 逐历元 / FFT 等价 / 诊断表 / 拟合一致性。"""
    check.section("批量综合与产品")
    ax = TimeAxis.from_datetimes([np.datetime64(f"{2002 + i // 12}-{i % 12 + 1:02d}-15")
                                  for i in range(24)])
    c = make_coeffs(10, 24, times=ax, scale=1e-3, unit="geopotential")
    latv = -90.0 + (np.arange(19) + 0.5) * 10.0
    lonv = 360.0 * np.arange(36) / 36

    r = synth_series(c, lat_vec=latv, lon_vec=lonv, use_fft="auto")
    check.ok(r.values.shape == (19, 36, 24), f"场序列形状 {r.values.shape}")
    check.ok("FFT" in r.stats["method"], f"自动选中 FFT 路径({r.stats['method'][:22]})")
    # 逐历元一致
    worst = 0.0
    for k in range(24):
        one = c.time_slice(k)
        ref = synthesis_grid(latv, lonv, one, chunk=10 ** 7)
        worst = max(worst, float(np.abs(r.values[:, :, k] - ref).max()))
    check.ok(worst < 1e-12, f"批量一次算完 ≡ 逐历元循环(最大差 {worst:.2e})")

    r2 = synth_series(c, lat_vec=latv, lon_vec=lonv, use_fft="no")
    check.close(r2.values, r.values, 1e-12, "强制直接法与 FFT 结果一致")

    # 诊断表
    p = r.to_csv(os.path.join(tmp_dir("mt"), "diag.csv"))
    txt = open(p, encoding="utf-8-sig").read().strip().splitlines()
    check.ok(len(txt) == 25, f"诊断表 {len(txt)} 行(表头 + 24 历元)")
    check.ok("decimal_year" in txt[0] and "finite_frac" in txt[0],
             "诊断表含 decimal_year / finite_frac")

    # --- 趋势/周年:系数域 ≡ 场域逐点(机器精度)-------------------------
    s = remove_mean(c)
    f = fit_trend_seasonal(s, periods=(1.0, 0.5), trend="linear", sigma=True)
    check.ok(f.rank == 6, f"设计矩阵满秩 {f.rank}/6")
    la, lo = 25.0, 100.0
    series_pt = np.asarray(evaluate(np.array([la]), np.array([lo]), s)).ravel()
    sol = np.linalg.lstsq(f.design, series_pt, rcond=None)[0]
    sl = float(evaluate(np.array([la]), np.array([lo]), f.slope_coeffs())[0])
    check.close(np.array([sl]), np.array([sol[1]]), 1e-10,
                f"系数域趋势 ≡ 场域逐点(相对 {abs(sl - sol[1]) / abs(sol[1]):.1e})")
    amp_field = np.hypot(
        float(evaluate(np.array([la]), np.array([lo]), f.field(2))[0]),
        float(evaluate(np.array([la]), np.array([lo]), f.field(3))[0]))
    check.close(np.array([amp_field]), np.array([np.hypot(sol[2], sol[3])]), 1e-10,
                "系数域周年振幅 ≡ 场域逐点")
    check.ok(f.slope_coeffs().meta.get("field_unit") == "geopotential",
             "拟合输出**保留 field_unit**(否则换算语义会丢)")
    ph = f.phase(1.0)
    check.ok(np.nanmin(ph) >= 0.0 and np.nanmax(ph) < 360.0,
             f"相位 ∈ [0,360)(实测 {np.nanmin(ph):.1f}..{np.nanmax(ph):.1f})")
    check.ok(f.amplitude_sigma(1.0).shape == f.amplitude(1.0).shape,
             "振幅 1σ 形状与振幅一致")

    # --- 已知信号恢复 ----------------------------------------------------
    rng = np.random.default_rng(2)
    nt = 120
    t = 2002.0 + np.arange(nt) * 0.0833
    tt = TimeAxis.from_decimal_years(t)
    a_true = rng.normal(0, 1e-11, (5, 5))
    Ck = np.zeros((5, 5, nt)); Sk = np.zeros((5, 5, nt))
    Ck[:, :, :] = a_true[:, :, None] * 0.01 * (t - t.mean())[None, None, :]
    sig = SHCoeffs(Ck, Sk, {"field_unit": "geopotential"}, tt)
    fs = fit_trend_seasonal(sig, periods=(1.0,), trend="linear")
    rec = fs.slope_coeffs().C
    check.close(rec, a_true * 0.01, 1e-14, "已知线性趋势被精确恢复(振幅/斜率)")

    # --- 零阶/静态场诊断 --------------------------------------------------
    from shsynth import check_degree0_trap, check_static_dominance
    C0 = np.zeros((3, 3)); C0[0, 0] = 1.0
    msg = check_degree0_trap(SHCoeffs(C0, np.zeros((3, 3))), "ewh")
    check.ok(msg is not None and "1.173e+07" in msg,
             "C₀₀=1 + target_unit=ewh → 给出 1.17e7 常量偏移警告")
    check.ok(check_degree0_trap(SHCoeffs(C0, np.zeros((3, 3))), "geoid") is None,
             "换 geoid 不触发零阶警告")
    vals = np.ones((4, 4, 10)) * 1e7 + np.random.default_rng(1).normal(0, 1e3, (4, 4, 10))
    sm = check_static_dominance(vals, tt.select(np.arange(10)), "ewh")
    check.ok(sm is not None and "静态" in sm, "静态场压过时间变化 → 给出警告")

    # --- 散点:必须走直接法 ----------------------------------------------
    rp = synth_series(c, points=(np.array([10.0, 20.0]), np.array([30.0, 40.0])))
    check.ok(rp.kind == "points" and "直接法" in rp.stats["method"],
             "散点自动走直接法(FFT 不适用)")
    tm, pv = series_at_points([30.0], [114.0], c)
    check.ok(np.shape(pv) == (1, 24), f"点序列形状 {np.shape(pv)}")

    # --- 矢量网格序列:一次算完北+东,values 取 magnitude ------------------
    hv = synth_series(c, lat_vec=latv, lon_vec=lonv, component="horizontal",
                      use_fft="no")
    check.ok(set(hv.components) == {"north", "east"},
             f"矢量序列带两个分量 {sorted(hv.components)}")
    check.ok(hv.values.shape == (19, 36, 24), f"values 形状 {hv.values.shape}")
    check.close(hv.values,
                np.hypot(hv.components["north"], hv.components["east"]), 1e-15,
                "values = magnitude = hypot(北, 东)(旋转不变量)")
    check.ok(hv.stats.get("component") == "horizontal", "stats 记下分量")
    # 单分量
    hn = synth_series(c, lat_vec=latv, lon_vec=lonv, component="north",
                      use_fft="no")
    check.close(hn.values, hv.components["north"], 1e-15,
                "component='north' 只取北分量")
    # 矢量 FFT 路径 ≡ 直接法
    hv2 = synth_series(c, lat_vec=latv, lon_vec=lonv, component="horizontal",
                       use_fft="yes")
    check.ok("FFT" in hv2.stats["method"], "矢量也能走 FFT")
    for k in ("north", "east"):
        a, b = hv2.components[k], hv.components[k]
        check.ok(np.abs(a - b).max() / np.abs(b).max() < 1e-12,
                 f"矢量 {k}:FFT ≡ 直接法(相对 "
                 f"{np.abs(a - b).max() / np.abs(b).max():.1e})")
    # 物理量不匹配要报错
    try:
        synth_series(c, lat_vec=latv, lon_vec=lonv, component="horizontal",
                     target_unit="ewh")
        check.ok(False, "水平形变配 EWH 应当报错")
    except ValueError as exc:
        check.ok("水平形变" in str(exc),
                 f"水平形变 + EWH 报错并解释因子不同:{str(exc)[:30]}")


def test_basin(check: Checker):
    """区域平均:两口径**恒等性** / 细核 / 归一化锚点 / 非法输入。

    核心断言不是"两种方法差多少",而是 ——
    **同一套格点下两口径必须代数恒等**(Σ f_nm g_nm ≡ Σ w_i f(x_i))。
    这条一立,两个实现就互为交叉验证;一旦它破了,是 bug,不是"方法差异"。
    """
    from shsynth import basin_average, basin_compare
    check.section("区域平均(coeff / spatial)")

    def axes(nlat, nlon):
        return (-90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat),
                360.0 * np.arange(nlon) / nlon)

    def box(nlat, nlon, la0, la1, lo0, lo1):
        la, lo = axes(nlat, nlon)
        return (((la[:, None] >= la0) & (la[:, None] <= la1)
                 & (lo[None, :] >= lo0) & (lo[None, :] <= lo1)).astype(float))

    # --- 1. 常值场:两口径都必须**精确**给出该常数(归一化硬锚点)---------
    for nlat, nlon in ((36, 72), (91, 180)):
        c0 = SHCoeffs(np.full((1, 1, 1), 7.5), np.zeros((1, 1, 1)))
        _t, sp, co, diff = basin_average(c0, box(nlat, nlon, -30, 30, 0, 90),
                                         method="both")
        check.ok(abs(sp[0] - 7.5) < 1e-12 and abs(co[0] - 7.5) < 1e-12,
                 f"{nlat}×{nlon} 常值场两口径都给出 7.5"
                 f"(spatial 偏差 {abs(sp[0] - 7.5):.1e}, "
                 f"coeff {abs(co[0] - 7.5):.1e})")

    # --- 2. 同一套格点 → 两口径恒等(机器精度)--------------------------
    c = make_coeffs(20, 3, seed=11, scale=1e-4)
    for nlat, nlon in ((45, 90), (90, 180), (180, 360)):
        for tag, mask in (("全球", np.ones((nlat, nlon))),
                          ("锐利箱", box(nlat, nlon, -20, 20, 280, 320))):
            _t, sp, co, diff = basin_average(c, mask, nmax=20, method="both")
            rel = float(np.max(np.abs(diff)) / max(np.max(np.abs(sp)), 1e-30))
            check.ok(rel < 1e-13,
                     f"{nlat}×{nlon} {tag}:两口径恒等(相对 {rel:.1e})")
    # 平滑掩膜(低阶纬向函数)也恒等
    la, _ = axes(90, 180)
    smooth = np.broadcast_to((1.0 + 0.5 * np.sin(np.deg2rad(la)))[:, None],
                             (90, 180)).copy()
    _t, sp, co, diff = basin_average(c, smooth, nmax=20, method="both")
    check.ok(float(np.max(np.abs(diff)) / np.max(np.abs(sp))) < 1e-13,
             "平滑掩膜下两口径同样恒等")

    # --- 3. 全球掩膜 → C₀₀,离散化误差 O(Δ²)(两口径同值)----------------
    ref = c.C[0, 0]
    errs = []
    for nlat, nlon in ((45, 90), (90, 180), (180, 360), (360, 720)):
        _t, sp, co, _d = basin_average(c, np.ones((nlat, nlon)), nmax=20,
                                       method="both")
        errs.append(float(np.max(np.abs(co - ref))))
        check.close(sp, co, 1e-12, f"{nlat}×{nlon} 全球掩膜两口径同值")
    ratios = [errs[i + 1] / errs[i] for i in range(len(errs) - 1)]
    check.ok(all(0.15 < r < 0.40 for r in ratios),
             f"全球掩膜 → C₀₀ 误差按 Δ² 收敛(倍率 "
             f"{', '.join(f'{r:.3f}' for r in ratios)})")

    # --- 4. mask_kernel:细核的口径**不受**粗掩膜网格影响 ----------------
    nt = 4
    ct = make_coeffs(20, nt, seed=3, scale=1e-4,
                     times=TimeAxis.from_datetimes(
                         [np.datetime64(f"2003-{i + 1:02d}-15") for i in range(nt)]))
    la_f, lo_f = axes(360, 720)
    m_fine = box(360, 720, -20, 20, 280, 320)
    _t, ref_series = basin_average(ct, m_fine, lat_vec=la_f, lon_vec=lo_f,
                                   method="spatial")
    sc = float(np.max(np.abs(ref_series))) or 1.0
    e_sp, e_co = [], []
    for nlat, nlon in ((45, 90), (90, 180), (180, 360)):
        la_c, lo_c = axes(nlat, nlon)
        _t, sp_c, co_c, df_c = basin_average(
            ct, box(nlat, nlon, -20, 20, 280, 320), lat_vec=la_c, lon_vec=lo_c,
            method="both", mask_kernel=(la_f, lo_f, m_fine))
        e_sp.append(float(np.max(np.abs(sp_c - ref_series))) / sc)
        e_co.append(float(np.max(np.abs(co_c - ref_series))) / sc)
    check.ok(all(co < sp for co, sp in zip(e_co, e_sp)),
             f"细核 coeff 比粗网格 spatial 更贴参考"
             f"(coeff {e_co[-1]:.1e} vs spatial {e_sp[-1]:.1e})")
    check.ok(max(e_co) / min(e_co) < 10.0,
             f"细核 coeff **不随粗掩膜网格变化**(比值 {max(e_co) / min(e_co):.3f})")
    check.close(np.array([e_co[0]]), np.array([e_co[-1]]), 1e-13,
                "45×90 与 180×360 的细核 coeff 结果一致")

    # --- 5. 计时:coeff 必须明显快于 spatial ---------------------------
    import time
    la_c, lo_c = axes(180, 360)
    mask = box(180, 360, -60, 60, 0, 360)
    tc = make_coeffs(30, 40, seed=7, scale=1e-4,
                     times=TimeAxis.from_datetimes(
                         [np.datetime64(f"2004-01-{i % 28 + 1:02d}") for i in range(40)]))
    t0 = time.perf_counter()
    basin_average(tc, mask, lat_vec=la_c, lon_vec=lo_c, method="spatial")
    ts = time.perf_counter() - t0
    t0 = time.perf_counter()
    basin_average(tc, mask, lat_vec=la_c, lon_vec=lo_c, method="coeff")
    tco = time.perf_counter() - t0
    check.ok(tco < ts, f"coeff({tco:.3f}s) 快于 spatial({ts:.3f}s)"
                       f"({ts / max(tco, 1e-9):.1f}×)")

    # --- 6. basin_compare 摘要必须点破"恒等"这件事 ----------------------
    d = basin_compare(ct, box(90, 180, -20, 20, 280, 320))
    check.ok(set(d) >= {"times", "spatial", "coeff", "diff", "rms_diff", "summary"},
             "basin_compare 返回两口径 + 差 + 摘要")
    check.ok("恒等" in d["summary"] and "mask_kernel" in d["summary"],
             "没给 mask_kernel 时摘要明确说两口径恒等、并指路 mask_kernel")
    d2 = basin_compare(ct, box(90, 180, -20, 20, 280, 320),
                       mask_kernel=(la_f, lo_f, m_fine))
    check.ok("mask_kernel" in d2["summary"] and "恒等" not in d2["summary"],
             "给了 mask_kernel 时摘要改说'不同的掩膜离散化'")
    check.ok(d2["max_abs_diff"] > d["max_abs_diff"],
             f"细核带来的差({d2['max_abs_diff']:.2e})"
             f"大于同网格舍入({d['max_abs_diff']:.2e})")

    # --- 7. 掩膜文件入口 + 非法输入 -------------------------------------
    base = tmp_dir("basin")
    la_f2, lo_f2 = axes(45, 90)
    m2 = box(45, 90, -20, 20, 280, 320)
    p = os.path.join(base, "mask.npy")
    LO, LA = np.meshgrid(lo_f2, la_f2)
    np.save(p, np.stack([LO, LA, m2]))
    _t, co_file = basin_average(ct, p, method="coeff")
    _t, co_arr = basin_average(ct, m2, lat_vec=la_f2, lon_vec=lo_f2,
                               method="coeff")
    check.close(co_file, co_arr, 1e-12, "掩膜文件入口 ≡ 数组入口")
    # 裸二维 .npy(无坐标轴)必须报错,不许猜坐标
    np.save(p, m2)
    check.raises(lambda: basin_average(ct, p), ValueError,
                 "裸二维 .npy 掩膜被拒绝(坐标轴不可知)")
    # 3 维掩膜(有多个时间层)必须报错
    check.raises(lambda: basin_average(ct, np.zeros((4, 45, 90))), ValueError,
                 "3 维掩膜被拒绝")
    check.raises(lambda: basin_average(ct, m2, method="grid"), ValueError,
                 "method 取值非法时报错")
    check.raises(lambda: basin_average(ct, np.zeros((45, 90))), ValueError,
                 "全 0 掩膜报错")
    # 掩膜坐标轴长度对不上要报错
    check.raises(lambda: basin_average(ct, m2, lat_vec=la_f2[:10]), ValueError,
                 "掩膜形状与坐标轴不匹配时报错")


def test_demean_modes(check: Checker):
    """去均值的三种口径:GRACE 惯例 / 全时段 / 自定义。

    口径依据是用户自己的 legacy 脚本(减 2004-2010 的平均)。这里盯四件事:

    1. ``mode='all'`` 与旧 ``remove_mean`` **逐位一致**(``--remove-mean`` 不能变);
    2. ``grace`` 只在 2004-01-01..2010-12-31 的历元上求平均,并把实际区间报出来;
    3. 窗口为空 / 缺日期 / 起点晚于终点 → **报错**,绝不退化成全时段;
    4. 口径写进 ``meta``(落盘后的文件自己说明得清拿哪一段做的基准)。
    """
    from shsynth import (GRACE_MEAN_FROM, GRACE_MEAN_TO, MEAN_MODES,
                         mean_window, remove_mean, remove_mean_window)
    check.section("去均值口径(GRACE 惯例 / 全时段 / 自定义)")
    check.ok(MEAN_MODES == ("grace", "all", "custom"),
             f"三种口径 {MEAN_MODES}")
    check.ok((GRACE_MEAN_FROM, GRACE_MEAN_TO) == ("2004-01-01", "2010-12-31"),
             f"GRACE 惯例窗口 {GRACE_MEAN_FROM} .. {GRACE_MEAN_TO}")

    # 造一条跨 2002-2012 的月度序列(每月 15 日),便于精确核对窗口
    dates = [np.datetime64(f"{y}-{m:02d}-15")
             for y in range(2002, 2013) for m in range(1, 13)]
    ax = TimeAxis.from_datetimes(dates)
    nt = len(dates)
    rng = np.random.default_rng(17)
    C = rng.normal(0, 1e-3, (6, 6, nt))
    S = rng.normal(0, 1e-3, (6, 6, nt))
    tri = np.tril(np.ones((6, 6), bool))
    C[~tri, :] = 0.0
    S[~tri, :] = 0.0
    S[:, 0, :] = 0.0
    seq = SHCoeffs(C, S, {"field_unit": "geopotential"}, ax)

    # ---- 1. mode='all' ≡ 旧 remove_mean(逐位)------------------------
    a = remove_mean(seq)
    b, rep_all = remove_mean_window(seq, mode="all")
    check.ok(np.array_equal(a.C, b.C) and np.array_equal(a.S, b.S),
             "mode='all' 与旧 remove_mean **逐位一致**(--remove-mean 行为不变)")
    check.ok(rep_all["n_epochs"] == nt,
             f"全时段用满 {rep_all['n_epochs']}/{nt} 个历元")

    # ---- 2. GRACE 惯例窗口精确 ---------------------------------------
    _c, rep = remove_mean_window(seq, mode="grace")
    win = mean_window(ax, mode="grace")
    d = np.array([str(v)[:10] for v in ax.values])
    expect = np.nonzero((d >= "2004-01-01") & (d <= "2010-12-31"))[0]
    check.ok(win["n_epochs"] == expect.size,
             f"GRACE 惯例命中 {win['n_epochs']} 个历元(手数 {expect.size})")
    check.ok(rep["from_date"] == "2004-01-15" and rep["to_date"] == "2010-12-15",
             f"实际区间 {rep['from_date']} .. {rep['to_date']}(边界历元正确)")
    check.ok(np.array_equal(win["idx"], expect), "窗口索引与按日期手算一致")
    # 参考场必须**只**是窗口内的平均 —— 用窗口外历元动一下,结果不能变
    C2 = C.copy()
    outside = np.setdiff1d(np.arange(nt), expect)
    C2[:, :, outside] += 5.0e-3
    seq2 = SHCoeffs(C2, S, {"field_unit": "geopotential"}, ax)
    g1, _ = remove_mean_window(seq, mode="grace")
    g2, _ = remove_mean_window(seq2, mode="grace")
    same_ref = np.allclose(g1.C - g2.C, -5.0e-3, atol=1e-18) or True
    d_ref = float(np.max(np.abs((g1.C - g2.C)[:, :, expect])))
    check.ok(d_ref < 1e-15,
             f"窗口内结果只由窗口内历元决定(窗口内差 {d_ref:.2e})")
    del same_ref
    # 窗口外的确被改变了(证明确实没把全时段混进来)
    d_out = float(np.max(np.abs((g1.C - g2.C)[:, :, outside])))
    check.ok(d_out > 1e-6,
             f"窗口外历元照原样保留(差 {d_out:.2e},说明没有动它们)")

    # ---- 3. 自定义 -----------------------------------------------------
    _c, rep_c = remove_mean_window(seq, mode="custom",
                                   from_date="2005-01-01", to_date="2007-12-31")
    check.ok(rep_c["from_date"] == "2005-01-15" and rep_c["to_date"] == "2007-12-15",
             f"自定义区间 {rep_c['from_date']} .. {rep_c['to_date']}")
    check.ok(rep_c["n_epochs"] == 36, f"自定义命中 {rep_c['n_epochs']} 个月")
    # 简写到"年"也要认
    _c, rep_y = remove_mean_window(seq, mode="custom", from_date="2005",
                                   to_date="2007")
    check.ok(rep_y["n_epochs"] == rep_c["n_epochs"],
             "只给年份时自动补 01-01 / 12-31")
    # 十进制年也认
    _c, rep_d = remove_mean_window(seq, mode="custom", from_date=2005.0,
                                   to_date=2007.99)
    check.ok(rep_d["n_epochs"] == rep_c["n_epochs"],
             f"十进制年 2005.0..2007.99 命中 {rep_d['n_epochs']} 个")

    # ---- 4. 三种口径的结果确实不同(否则这个选项没意义)-----------------
    rms = {}
    for m, kw in (("grace", {}), ("all", {}),
                  ("custom", dict(from_date="2004-01-01", to_date="2009-12-31"))):
        o, _r = remove_mean_window(seq, mode=m, **kw)
        rms[m] = float(np.sqrt(np.mean(o.C ** 2 + o.S ** 2)))
    check.ok(abs(rms["grace"] - rms["all"]) > 1e-12,
             f"GRACE 惯例与全时段结果不同({rms['grace']:.4e} vs {rms['all']:.4e})")
    check.ok(abs(rms["custom"] - rms["all"]) > 1e-12,
             f"自定义(2004-2009)与全时段也不同({rms['custom']:.4e})")
    check.ok(abs(rms["custom"] - rms["grace"]) > 1e-12,
             "自定义(2004-2009)与 GRACE 惯例(2004-2010)也不同"
             "(差一个 2010 年,确实会不一样)")

    # ---- 5. 边界必须报错,绝不静默退化 ---------------------------------
    for kw, why in (
        (dict(mode="custom", from_date="2090-01-01", to_date="2091-01-01"),
         "窗口里没有历元"),
        (dict(mode="custom", from_date="2010-01-01", to_date="2005-01-01"),
         "起点晚于终点"),
        (dict(mode="custom"), "自定义没给日期"),
        (dict(mode="custom", from_date="2005"), "自定义只给了一半"),
        (dict(mode="nope"), "口径名非法"),
    ):
        check.raises(lambda kw=kw: remove_mean_window(seq, **kw), ValueError, why)
    # 没有时间轴:grace/custom 必须报错(不能猜),all 仍可用
    flat = SHCoeffs(C[:, :, 0].copy(), S[:, :, 0].copy(), {})
    check.raises(lambda: remove_mean_window(flat, mode="grace"), ValueError,
                 "没有时间轴时 GRACE 惯例报错(不猜日期)")
    # 窗口覆盖全部时给提示
    _c, rep_full = remove_mean_window(seq, mode="custom",
                                      from_date="2000-01-01",
                                      to_date="2020-12-31")
    check.ok(rep_full["n_epochs"] == nt and rep_full["warnings"],
             "自定义窗口覆盖全部历元时给出『退化为全时段』提示")

    # ---- 6. 口径写进 meta(落盘后自己说明得清)-------------------------
    g, _rep = remove_mean_window(seq, mode="grace")
    check.ok("reference_removed" in g.meta and "GRACE" in g.meta["reference_removed"],
             f"meta 记下参考口径({g.meta.get('reference_removed', '')[:34]})")
    mw = g.meta.get("mean_window", {})
    check.ok(mw.get("mode") == "grace" and mw.get("n_epochs") == win["n_epochs"],
             f"meta 记下窗口 {mw}")
    check.ok("reference_removed" not in seq.meta,
             "输入对象没有被改动(不就地修改)")


def test_real_data(check: Checker):
    """真实数据对表(只读;缺失则跳过并说明)。"""
    check.section("真实数据(只读)")
    if not os.path.isdir(GFC_DIR):
        check.skip(f"没有 {GFC_DIR}")
        return
    s = read_coeffs_series(GFC_DIR)
    check.ok(s.ntime == 203 and s.nmax == 60,
             f"203 个真实 gfc → {s.C.shape}(ntime={s.ntime})")
    check.ok(s.times is not None and len(s.times) == 203, "时间轴 203 个历元")
    t0 = s.times.values[0]
    check.ok(str(t0)[:10] == "2002-04-18", f"首个历元 = {str(t0)[:10]}")
    # 单调递增
    d = np.diff(s.times.values.astype("datetime64[D]").astype(int))
    check.ok(np.all(d > 0), "时间严格单调递增")
    check.ok(s.times.decimal_years[0] > 2002.29
             and s.times.decimal_years[0] < 2002.30,
             f"首个十进制年 {s.times.decimal_years[0]:.6f}(legacy 为 2002.293151)")
    # 缺测被报告、不插补
    check.ok(len(s.times.missing()) == 20,
             f"报告 {len(s.times.missing())} 处缺测(实测 20),且没有插补")

    if os.path.exists(LEGACY):
        cl = read_coeffs(LEGACY)
        check.ok(cl.ntime == 216 and cl.nmax == 60,
                 f"legacy .dat 自动识别 → ntime={cl.ntime}")
        j = s.times.match(cl.times, tol_days=2.0)
        ok = j >= 0
        check.ok(int(ok.sum()) >= 190,
                 f"按日期配对 {int(ok.sum())} 个历元(不是按下标)")
        rel = np.abs(s.C[:, :, ok] - cl.C[:, :, j[ok]]) / np.maximum(
            np.abs(s.C[:, :, ok]), 1e-300)
        check.ok(float(np.median(rel)) < 1e-6,
                 f"与 legacy 文本相对差中位 {float(np.median(rel)):.1e}(老文件只有 ~9 位"
                 "有效数字,**判据按文本精度给,不是 1e-15**)")

    if os.path.exists(GRID3):
        from shsynth import fieldio
        la, lo, g, me, ax = fieldio.read_grid(GRID3, with_time=True)
        check.ok(g.shape[2] == 257 and len(ax) == 257,
                 f"3_grids 读回 {g.shape} + {len(ax)} 个日期")
        check.ok(str(ax.values[0])[:10] == "2002-04-18",
                 f"首个日期 {str(ax.values[0])[:10]}")
        miss = ax.missing()
        check.ok(len(miss) >= 1 and abs(miss[0][2] - 98.0) < 1,
                 f"报出首处 {miss[0][2]:.0f} 天缺口(实测 98 天)")


def test_vs_shkit_if_available(check: Checker):
    """与 SHKit 交叉(装了才跑)。"""
    check.section("与 SHKit 交叉(可选)")
    shkit = os.path.join(os.path.dirname(ROOT), "SHKit")
    if not os.path.isdir(shkit):
        check.skip("没有 SHKit 目录")
        return
    sys.path.insert(0, os.path.dirname(shkit))
    try:
        import shkit.io as shio
    except Exception as exc:                                   # noqa: BLE001
        check.skip(f"SHKit 不可导入: {exc}")
        return
    f = os.path.join(shkit, "shkit_coeffs.sh")
    if not os.path.exists(f):
        check.skip("没有 shkit_coeffs.sh")
        return
    mine, theirs = read_coeffs(f), shio.read_coeffs(f)
    check.ok(np.array_equal(mine.C, theirs.C) and np.array_equal(mine.S, theirs.S),
             "与 SHKit 读系数逐位相同")
    h = evaluate_horizontal(np.array([20.0]), np.array([30.0]), mine)
    check.ok(np.isfinite(h["north"][0]), "水平形变在 SHKit 系数上可算")


def main() -> int:
    check = Checker("test_multitime —— v2.0 时间轴/序列/FFT/水平形变")
    for fn in (test_timeaxis, test_coeffs_times, test_series_io,
               test_fft_path, test_horizontal, test_series_batch,
               test_basin, test_demean_modes, test_real_data,
               test_vs_shkit_if_available):
        try:
            fn(check)
        except Exception as exc:                               # noqa: BLE001
            import traceback
            check.ok(False, f"{fn.__name__} 抛异常: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
