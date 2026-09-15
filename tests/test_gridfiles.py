# -*- coding: utf-8 -*-
"""
tests/test_gridfiles.py
=======================

**网格文件形式与绘图聚焦测试**。

针对用户实际踩到的坑:

* Surfer ASCII 网格会**按每行 10 个数字折行**(不是一行一个数据行)——
  按行读会直接失败,这是"我试了一些 grd 都不支持"的真正原因;
* 扩展名会撒谎:GMT 的 ``.grd`` 其实是 netCDF;Esri ``.asc`` 头部是
  ``ncols/nrows``;
* 平面/投影坐标(米)的网格不能当经纬度用 —— 必须明确报错而不是给一张
  位置全错的地图;
* 地图**只画海岸线、不画国界**;
* 区域结果要能自动聚焦到结果范围;
* 三列网格的数值列不能选错(表头 ``lat,value,lon`` 这种);
* 经纬度 0..360 含端点时不能留下重复列。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Checker, guard, tmp_dir                        # noqa: E402

from shsynth import fieldio, plotting                              # noqa: E402


def _write_surfer(path, nx, ny, xlo, xhi, ylo, yhi, wrap=10, magic="DSAA",
                  crlf=True):
    """写一个 Surfer ASCII 网格;``wrap`` 模拟 Surfer 的每行折行数。"""
    nl = "\r\n" if crlf else "\n"
    z = np.arange(nx * ny, dtype=float).reshape(ny, nx) / 10.0
    lines = [magic, f"{nx} {ny}", f"{xlo} {xhi}", f"{ylo} {yhi}",
             f"{z.min()} {z.max()}"]
    flat = z.ravel()
    for i in range(0, flat.size, wrap):
        lines.append(" ".join(f"{v:.6f}" for v in flat[i:i + wrap]))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(nl.join(lines) + nl)
    return z


def test_surfer_wrapped(check: Checker):
    """Surfer ASCII:折行(每行 10 个数字)与不折行都要读对。"""
    check.section("Surfer ASCII 折行")
    d = tmp_dir("gridfiles")
    for wrap, tag in ((10, "每行10个数字(Surfer 的写法)"),
                      (4, "每行4个"),
                      (100000, "一行一个数据行")):
        p = os.path.join(d, f"s_{wrap}.grd")
        z = _write_surfer(p, 56, 47, 100.0, 155.0, 20.0, 66.0, wrap=wrap)
        la, lo, g, meta = fieldio.read_grid(p)
        g = g[:, :, 0] if g.ndim == 3 else g
        check.ok(g.shape == (47, 56), f"{tag}: 形状 {g.shape}")
        check.close(g, z, 1e-9, f"{tag}: 数值逐位一致")
        check.ok(la[0] == 20.0 and abs(lo[-1] - 155.0) < 1e-9,
                 f"{tag}: 坐标轴正确({la[0]:g}..{la[-1]:g})")

    # CRLF + UTF-8 BOM 也要能读(实测用户的 .grd 就是这两种)
    p = os.path.join(d, "bom.grd")
    _write_surfer(p, 12, 9, 0.0, 110.0, -40.0, 40.0, wrap=3)
    z = np.arange(12 * 9, dtype=float).reshape(9, 12) / 10.0
    raw = open(p, "rb").read()
    with open(p, "wb") as fh:
        fh.write(b"\xef\xbb\xbf" + raw)
    la, lo, g, meta = fieldio.read_grid(p)
    g = g[:, :, 0] if g.ndim == 3 else g
    check.close(g, z, 1e-9, "UTF-8 BOM + CRLF 也能读")

    # 数据不足要报错
    p = os.path.join(d, "short.grd")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("DSAA\n56 47\n0 1\n0 1\n0 1\n1 2 3\n")
    check.raises(lambda: fieldio.read_grid(p), ValueError, "数据不足时报错")


def test_content_sniffing(check: Checker):
    """扩展名撒谎时按内容判型:.grd 其实是 netCDF、.txt 其实是 Surfer。"""
    check.section("按内容判型")
    d = tmp_dir("gridfiles")
    try:
        import xarray as xr
    except ImportError:                                   # pragma: no cover
        check.ok(True, "没有 xarray,跳过 netCDF 判型")
        return
    # 造一个 netCDF,存成 .grd 名字(GMT 的 .grd 就是这样)
    z = np.arange(12.0).reshape(3, 4)
    da = xr.DataArray(z, dims=("lat", "lon"),
                      coords={"lat": [-10.0, 0.0, 10.0],
                              "lon": [0.0, 90.0, 180.0, 270.0]}, name="value")
    p = os.path.join(d, "gmt_like.grd")
    try:
        da.to_dataset().to_netcdf(p, engine="netcdf4")
    except Exception as exc:                              # pragma: no cover
        check.ok(True, f"本机写不了 netCDF({type(exc).__name__}),跳过")
        return
    la, lo, g, meta = fieldio.read_grid(p)
    check.ok(meta.get("format") == "netcdf",
             f"内容是 netCDF 的 .grd 被正确识别(format={meta.get('format')})")
    check.ok(any("netCDF" in w for w in meta.get("warnings", [])),
             "给了『内容是 netCDF 但扩展名是 .grd』的提示")

    # Esri ASCII
    p2 = os.path.join(d, "esri.asc")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write("ncols 4\nnrows 3\nxllcorner 0\n yllcorner -10\n"
                 "cellsize 90\nNODATA_value -9999\n")
        fh.write("1 2 3 4\n5 997 7 8\n9 10 11 12\n".replace("997", "6"))
    la2, lo2, g2, meta2 = fieldio.read_grid(p2)
    g2 = g2[:, :, 0] if g2.ndim == 3 else g2
    check.ok(g2.shape == (3, 4), f"Esri ASCII 形状 {g2.shape}")
    check.close(g2, np.arange(1.0, 13.0).reshape(3, 4), 1e-12,
                "Esri ASCII 数值正确")

    # 二进制 Surfer:必须给出可操作的说明,而不是含糊失败
    p3 = os.path.join(d, "binary.grd")
    with open(p3, "wb") as fh:
        fh.write(b"DSRB" + bytes(range(256)) * 4)
    try:
        fieldio.read_grid(p3)
        check.ok(False, "Surfer 二进制应报错")
    except ValueError as exc:
        msg = str(exc)
        check.ok("二进制" in msg and ("ASCII" in msg or "netCDF" in msg),
                 "Surfer 二进制给出了转换办法")


def test_projected_coords_refused(check: Checker):
    """平面/投影坐标(米)的网格必须明确拒绝,而不是给错位置。"""
    check.section("平面坐标拒绝")
    d = tmp_dir("gridfiles")
    p = os.path.join(d, "hainan_like.grd")
    _write_surfer(p, 56, 47, 369200.0, 396700.0, 2044500.0, 2067500.0, wrap=10)
    try:
        fieldio.read_grid(p)
        check.ok(False, "平面坐标应报错")
    except ValueError as exc:
        msg = str(exc)
        check.ok("投影" in msg or "平面" in msg, "报错说明了是平面坐标")
        check.ok("经纬度" in msg, "报错说明了需要经纬度")
        check.ok("--global-grid" in msg or "散点" in msg,
                 "报错给出了可行做法")
    # 真正经纬度的 Surfer 网格要能读(month.NNN.grd 那种)
    p2 = os.path.join(d, "month_like.grd")
    z_ref = _write_surfer(p2, 65, 30, -75.0, -10.0, 55.0, 85.0, wrap=8)
    la, lo, g, meta = fieldio.read_grid(p2)
    g = g[:, :, 0] if g.ndim == 3 else g
    check.ok(la[0] == 55.0 and la[-1] == 85.0,
             f"经纬度 Surfer 网格正常({la[0]:g}..{la[-1]:g})")
    check.close(g, z_ref, 1e-9, "经纬度 Surfer 网格数值也正确")


def test_three_column_value_pick(check: Checker):
    """三列网格:数值列必须取"既不是纬度也不是经度"的那一列。"""
    check.section("三列网格列识别")
    d = tmp_dir("gridfiles")
    lat = np.array([30.0, 31.0])
    lon = np.array([100.0, 101.0, 102.0])
    val = {(0, 0): 1.5, (0, 1): 2.5, (0, 2): 3.5,
           (1, 0): 4.5, (1, 1): 5.5, (1, 2): 6.5}
    for header, order in (("lat,value,lon", ("lat", "value", "lon")),
                          ("lon,lat,value", ("lon", "lat", "value")),
                          ("value,lat,lon", ("value", "lat", "lon")),
                          ("lat,lon,value", ("lat", "lon", "value"))):
        p = os.path.join(d, "t_" + header.replace(",", "_") + ".csv")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(header + "\n")
            for i, a in enumerate(lat):
                for j, b in enumerate(lon):
                    row = {"lat": a, "lon": b, "value": val[(i, j)]}
                    fh.write(",".join(f"{row[k]:g}" for k in order) + "\n")
        la, lo, g, meta = fieldio.read_grid(p)
        g = g[:, :, 0] if g.ndim == 3 else g
        expect = np.array([[val[(i, j)] for j in range(3)]
                           for i in range(2)])
        check.close(g, expect, 1e-12, f"表头 {header!r} 数值列选对")


def test_lon_dedup_and_lat_check(check: Checker):
    """经度 0..360 含端点要去重;纬度越界要报警。"""
    check.section("经度去重与纬度校验")
    d = tmp_dir("gridfiles")
    # lon = 0,1,...,360(361 个,含两端)→ 去重成 360 个
    p = os.path.join(d, "lon360.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("lat,lon,value\n")
        for a in (-10.0, 0.0):
            for b in range(0, 361):
                fh.write(f"{a:g},{b:g},1.0\n")
    la, lo, g, meta = fieldio.read_grid(p)
    check.ok(lo.size == 360, f"经度去重后 nlon = {lo.size}(应为 360)")
    check.ok(np.all(np.diff(lo) > 0), "去重后经度严格递增")
    check.ok(any("重复" in w for w in meta.get("warnings", [])),
             "重复经度给了警告")

    # 纬度 100..110 → 不是经纬度,应报错
    p2 = os.path.join(d, "badlat.csv")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write("lat,lon,value\n")
        for a in (100.0, 110.0):
            for b in (0.0, 1.0):
                fh.write(f"{a:g},{b:g},1.0\n")
    check.raises(lambda: fieldio.read_grid(p2), ValueError,
                 "纬度 100..110 被拒绝")


def test_nc_bool_attribute(check: Checker):
    """写 .nc 时 meta 里的 bool 属性不能把写入搞崩。"""
    check.section("netCDF 布尔属性")
    d = tmp_dir("gridfiles")
    try:
        import importlib
        importlib.import_module("xarray")
    except ImportError:                                   # pragma: no cover
        check.ok(True, "没有 xarray,跳过")
        return
    latv = np.array([30.0, 20.0, 10.0])          # 递减 → 会触发 lat_order_flipped
    lonv = np.array([0.0, 90.0, 180.0, 270.0])
    g = np.arange(12.0).reshape(3, 4)
    p = os.path.join(d, "bool_meta.nc")
    try:
        fieldio.write_grid(p, latv, lonv, g,
                           meta={"lat_order_flipped": True,
                                 "flag_bool": False, "n_points": np.int64(12)})
    except Exception as exc:                              # noqa: BLE001
        check.ok(False, "写 .nc(含 bool 属性)不应报错",
                 f"{type(exc).__name__}: {exc}")
        return
    check.ok(os.path.exists(p) and os.path.getsize(p) > 500,
             f"含 bool 属性的 .nc 写出成功({os.path.getsize(p)} 字节)")
    la, lo, g2, meta = fieldio.read_grid(p)
    g2 = g2[:, :, 0] if g2.ndim == 3 else g2
    check.ok(la[0] < la[-1], "纬度被翻成递增")
    check.close(g2, g[::-1, :], 1e-12, "递减纬度已连数据一起翻转")


def test_no_borders_and_focus(check: Checker):
    """地图只画海岸线(不画国界),并支持聚焦。"""
    check.section("海岸线 / 聚焦")
    # 1) 数据文件里不应再有国界数据
    borders = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "shsynth", "data", "borders_110m.npz")
    check.ok(not os.path.exists(borders),
             "已移除 borders_110m.npz(不再画国界)")
    check.ok(plotting.load_coastlines("borders") is None,
             "load_coastlines('borders') 返回 None")
    check.ok(plotting.load_coastlines("coastline") is not None,
             "海岸线数据仍在")

    latv = np.arange(20.0, 51.0, 1.0)
    lonv = np.arange(100.0, 141.0, 1.0)
    g = np.ones((latv.size, lonv.size))
    # 2) 区域数据 → 自动聚焦
    fig = plotting.make_map_figure(latv, lonv, g, title="区域")
    ax = fig.get_axes()[0]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    check.ok((xlim[1] - xlim[0]) < 60, f"自动聚焦后 x 范围 {xlim[0]:.1f}..{xlim[1]:.1f}")
    check.ok((ylim[1] - ylim[0]) < 45, f"自动聚焦后 y 范围 {ylim[0]:.1f}..{ylim[1]:.1f}")
    n_lines = len(ax.lines)
    check.ok(n_lines > 0, f"画了 {n_lines} 条海岸线")
    # 3) focus='global' → 全球
    fig2 = plotting.make_map_figure(latv, lonv, g, focus="global")
    ax2 = fig2.get_axes()[0]
    check.ok(abs(ax2.get_xlim()[1] - ax2.get_xlim()[0] - 360) < 1,
             "focus='global' 时是全球视图")
    # 4) 全球数据 → 不聚焦
    glat = np.arange(-90.0, 91.0, 10.0)
    glon = np.arange(0.0, 360.0, 10.0)
    gg = np.ones((glat.size, glon.size))
    fig3 = plotting.make_map_figure(glat, glon, gg)
    ax3 = fig3.get_axes()[0]
    check.ok(abs(ax3.get_xlim()[1] - ax3.get_xlim()[0] - 360) < 1,
             "全球数据不会被聚焦")
    # 5) 换日线附近的窗口
    ext = plotting.data_extent(np.array([-10.0, 10.0]),
                               np.array([170.0, -170.0]))
    check.ok(ext is not None and 150 < ext[0] < 180 and 180 < ext[1] < 210,
             f"跨换日线的窗口 = {ext}")
    check.ok(plotting.data_extent(np.array([-90.0, 90.0]),
                                  np.array([0.0, 359.0]),
                                  np.ones((2, 2))) is None,
             "覆盖全球时返回 None(不聚焦)")
    import matplotlib.pyplot as plt
    plt.close("all")


def test_points_columns_and_non_numeric(check: Checker):
    """散点表:非数值列(台站名)不应该把读表搞崩;列信息要能读出。"""
    check.section("散点列与非数值列")
    d = tmp_dir("gridfiles")
    p = os.path.join(d, "stations.csv")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("station,longitude,latitude,value\n")
        fh.write("A,100.5,30.5,1.0\nB,110.0,35.0,2.0\nC,120.0,40.0,3.0\n")
    info = fieldio.peek_columns(p)
    check.ok(info["has_header"] and info["columns"] ==
             ["station", "longitude", "latitude", "value"],
             f"读到列名 {info['columns']}")
    check.ok(info["suggested_lat"] == 2 and info["suggested_lon"] == 1,
             f"自动认出 lat={info['suggested_lat']} lon={info['suggested_lon']}")
    check.ok("station" in info["non_numeric_columns"],
             f"识别出台站名列不是数值: {info['non_numeric_columns']}")
    la, lo, vals, meta = fieldio.read_points(p)
    check.close(la, [30.5, 35.0, 40.0], 1e-12, "纬度列正确")
    check.close(lo, [100.5, 110.0, 120.0], 1e-12, "经度列正确")
    check.close(vals, [1.0, 2.0, 3.0], 1e-12, "数值列正确")
    check.ok(any("不是数值" in w for w in meta.get("warnings", [])),
             "非数值列给了警告")

    # 无表头 → 下拉框显示序号
    p2 = os.path.join(d, "noheader.csv")
    with open(p2, "w", encoding="utf-8") as fh:
        fh.write("100.5,30.5,1.0\n110.0,35.0,2.0\n")
    info2 = fieldio.peek_columns(p2)
    check.ok(not info2["has_header"] and info2["columns"][:2] ==
             ["第1列", "第2列"],
             f"无表头时列名是序号 {info2['columns']}")
    check.ok(info2["suggested_lon"] == 0 and info2["suggested_lat"] == 1,
             "无表头时按通行约定猜(1=经度,2=纬度)")

    # 只有两列的 .npy 在 require_values=True 时应报错
    p3 = os.path.join(d, "two_col.npy")
    np.save(p3, np.array([[100.0, 30.0], [110.0, 35.0]]))
    check.raises(lambda: fieldio.read_points(p3), ValueError,
                 "两列 .npy 且要数值列时报错")
    la4, lo4, meta4 = fieldio.read_positions(p3)
    check.close(la4, [30.0, 35.0], 1e-12, "read_positions 能读两列 .npy")


def main() -> int:
    check = Checker("test_gridfiles —— 网格文件形式、聚焦与列识别")
    for fn in (test_surfer_wrapped, test_content_sniffing,
               test_projected_coords_refused, test_three_column_value_pick,
               test_lon_dedup_and_lat_check, test_nc_bool_attribute,
               test_no_borders_and_focus, test_points_columns_and_non_numeric):
        guard(fn)(check)
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
