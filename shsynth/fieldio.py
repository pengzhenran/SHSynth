# -*- coding: utf-8 -*-
"""
shsynth.fieldio
===============

散点与网格文件的读写(输入"位置",输出"值")。

``points``  散点 ``(lat, lon, value)``   → ``csv / txt / dat / tsv / npy / xlsx``
``grid``    规则网格 ``(lat_vec, lon_vec, value)`` → ``nc / grd / npy / csv / txt``

单位与约定(全模块统一)
----------------------
* 纬度  度,地心纬度,``-90 <= lat <= 90``;
* 经度  度,读入后按 ``mod 360`` 归一化到 ``[0, 360)``(原本是 ``-180..180`` 时记一条警告);
* 值    调用方的场(EWH、水准面高、重力异常……),本模块**不做任何换算**;
* 网格形状  ``(nlat, nlon)`` 或 ``(nlat, nlon, ntime)``,``lat_vec`` 严格递增,
  盘中是递减纬度的会连数据一起翻过来并记 ``meta['lat_order_flipped'] = True``。

写出时列序为 ``lon, lat, value[, value2 ...]``(与 :func:`read_points` 的默认
假设一致,也与 SHKit 的输出一致);csv 带 BOM,Excel 打开中文表头不乱码。
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "POINT_EXTENSIONS",
    "GRID_EXTENSIONS",
    "read_points",
    "write_points",
    "read_positions",
    "peek_columns",
    "read_grid",
    "write_grid",
]

POINT_EXTENSIONS = (".csv", ".txt", ".dat", ".tsv", ".npy", ".xlsx")
GRID_EXTENSIONS = (".nc", ".nc4", ".cdf", ".grd", ".npy", ".csv", ".txt", ".dat")

COL_LON, COL_LAT, COL_VAL = "lon", "lat", "value"

_LON_NAMES = ("lon", "long", "longitude", "lon_deg", "longitude_deg", "x",
              "xlon", "glon", "经度", "东经")
_LAT_NAMES = ("lat", "latitude", "lat_deg", "latitude_deg", "y", "ylat",
              "glat", "纬度", "北纬")
_VAL_NAMES = ("value", "val", "z", "data", "field", "ewh", "height", "h",
              "sigma", "obs", "slm", "trend", "anomaly", "值", "数值")



def _pkg_version() -> str:
    """本包版本号。

    延迟到调用时再取,避免与 :mod:`shsynth` 的 ``__init__`` 循环导入
    (``__init__`` 会 import 本模块)。
    """
    try:
        from . import __version__
        return __version__
    except Exception:                                    # pragma: no cover
        return "1.0.0"


def _require_file(path) -> Path:
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{p} 是一个目录,不是数据文件")
    return p


def _suffix(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".gz":
        s = Path(p.stem).suffix.lower()
    return s


def _text_read(path) -> str:
    import gzip
    p = Path(os.fspath(path))
    raw = gzip.open(p, "rb").read() if str(p).lower().endswith(".gz") \
        else open(p, "rb").read()
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _text_write(path, text: str, bom: bool = True) -> Path:
    import gzip
    p = Path(os.fspath(path))
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8-sig" if bom else "utf-8")
    if str(p).lower().endswith(".gz"):
        with gzip.open(p, "wb") as fh:
            fh.write(data)
    else:
        with open(p, "wb") as fh:
            fh.write(data)
    return p


def _sniff_delimiter(text: str) -> Optional[str]:
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        for d in (",", ";", "\t", "|"):
            if d in s:
                return d
        return None
    return None


def _split(line: str, delim: Optional[str]) -> list:
    if delim is None:
        return [t for t in re.split(r"\s+", line.strip()) if t]
    return [t.strip() for t in line.split(delim) if t.strip() != ""]


def _read_table(path, delim: Optional[str] = None):
    """读文本表 → ``(comments, colnames, data)``。

    ``colnames`` 为表头(没有则 ``None``);``data`` 为 ``(nrow, ncol)`` float 数组。

    容错规则(实测必须这么处理):

    * 整行都是数字 → 数据行;
    * **部分单元格不是数字**但至少有数字(例如第一列是台站名
      ``A,100.5,30.5,1.0``)→ 仍然算数据行,非数字单元格记 ``NaN``,
      并把这些列的下标记进 ``non_numeric``(供调用方提示);
    * 整行都不是数字 → 只在还没有数据行时当表头,否则按"无法解析的行"跳过。
    """
    text = _text_read(path)
    if delim is None:
        delim = _sniff_delimiter(text)
    comments: list = []
    header: Optional[list] = None
    rows: list = []
    non_numeric: set = set()
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            comments.append(s.lstrip("#/ ").strip())
            continue
        toks = _split(s, delim)
        if not toks:
            continue
        vals: list = []
        bad_idx: list = []
        for i, t in enumerate(toks):
            try:
                vals.append(float(t))
            except ValueError:
                vals.append(np.nan)
                bad_idx.append(i)
        if len(bad_idx) == len(toks):        # 整行非数字
            if not rows and header is None:
                header = toks
            continue
        rows.append(vals)
        non_numeric.update(bad_idx)
    if not rows:
        raise ValueError(f"{Path(os.fspath(path))}: 没有可解析的数值行")
    ncol = max(len(r) for r in rows)
    clean = [r + [np.nan] * (ncol - len(r)) if len(r) < ncol else r[:ncol]
             for r in rows]
    data = np.asarray(clean, dtype=float)
    bad_cols: list = []
    if non_numeric:
        cols = [str(h) for h in header] if header else \
            [f"col{i + 1}" for i in range(ncol)]
        bad_cols = [cols[i] for i in sorted(non_numeric) if i < len(cols)]
        # 这些列整列置 NaN —— 它们本来就不是数值(台站名/标签),
        # 留着只会让别人误用
        data[:, sorted(non_numeric)] = np.nan
    return comments, header, data, bad_cols


def _pivot_grid(lat_c, lon_c, val_c, path, lat_name: str = "lat",
                lon_name: str = "lon"):
    """把三列 ``(lat, lon, value)`` pivot 成规则网格。

    重复格子(同一 ``(lat, lon)`` 出现多次)分两种处理:

    * **值一致** → 视为重复行(最常见的原因是经度同时给了 0 与 360,
      归一化后是同一根经线),去重并记一条说明;
    * **值不一致** → 报错:同一格点给了两个不同的数,谁对不知道,
      不能悄悄取后一个。
    """
    lat_u, lat_inv = np.unique(lat_c, return_inverse=True)
    lon_u, lon_inv = np.unique(lon_c, return_inverse=True)
    flat = lat_inv.astype(np.int64) * lon_u.size + lon_inv.astype(np.int64)
    n_cells = int(lat_u.size * lon_u.size)
    note = ""
    if np.unique(flat).size != flat.size:
        order = np.argsort(flat, kind="stable")
        f_sorted, v_sorted = flat[order], np.asarray(val_c, dtype=float)[order]
        bounds = np.flatnonzero(np.diff(f_sorted)) + 1
        groups = np.split(v_sorted, bounds)
        bad = 0
        for g in groups:
            g = g[np.isfinite(g)]
            if g.size > 1 and float(np.ptp(g)) > 1e-12 * max(
                    1.0, float(np.max(np.abs(g)))):
                bad += 1
        if bad:
            raise ValueError(
                f"{path}: 有 {bad} 个格点被赋了**互相矛盾**的值 —— "
                "同一个 (纬度, 经度) 在文件里出现了多次且数值不同,"
                "无法判断哪个才是对的。请先清理重复行")
        n_unique = int(np.unique(flat).size)
        note = (f"有 {flat.size - n_unique} 个重复格点(值一致,已去重);"
                "常见原因:经度同时写了 0 与 360,归一化后是同一根经线")
        if n_unique != n_cells:
            raise ValueError(
                f"{path}: 三列数据不构成完整规则网格"
                f"({lat_u.size} 个纬度 × {lon_u.size} 个经度 = {n_cells} 个格点,"
                f"实际只有 {n_unique} 个不同格点,{n_cells - n_unique} 个空缺)。"
                "散点数据请改用 read_points();网格有缺口时请补齐")
    elif flat.size != n_cells:
        raise ValueError(
            f"{path}: 三列数据不构成完整规则网格({lat_u.size} 个纬度 × "
            f"{lon_u.size} 个经度 = {n_cells},而文件有 {flat.size} 行)。"
            "散点数据请改用 read_points();网格有缺口时请补齐")
    grid = np.full(n_cells, np.nan)
    grid[flat] = np.asarray(val_c, dtype=float)
    return lat_u, lon_u, grid.reshape(lat_u.size, lon_u.size), note


def _resolve_col(spec, names: list, ncols: int, what: str, default: int) -> int:
    """把列选择(名字或 0 基下标)解析成下标。"""
    if spec is None:
        return default
    if isinstance(spec, str):
        s = spec.strip()
        if re.fullmatch(r"-?\d+", s):
            i = int(s)
        else:
            low = [str(n).strip().lower() for n in names]
            if s.lower() in low:
                return low.index(s.lower())
            raise ValueError(f"{what}列名 {spec!r} 不存在;可用列: {names}")
    else:
        i = int(spec)
    if i < 0:
        i += ncols
    if not 0 <= i < ncols:
        raise ValueError(f"{what}列下标 {spec!r} 超出范围(共 {ncols} 列)")
    return i


def _guess_named(names: list, candidates) -> Optional[int]:
    low = [str(n).strip().lower() for n in names]
    for cand in candidates:
        if cand in low:
            return low.index(cand)
    return None


def _normalise_lon(lon: np.ndarray, warnings: list) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    if np.any(lon < 0) or np.any(lon >= 360.0):
        warnings.append("经度已按 mod 360 归一化到 [0, 360)")
        lon = np.mod(lon, 360.0)
    return lon


# ---------------------------------------------------------------------------
# netCDF 的"网络盘"兼容层
# ---------------------------------------------------------------------------
# HDF5(netCDF4 的底层)在某些网络/虚拟文件系统上**既不能读也不能写**:文件明明
# 存在也会报 FileNotFoundError / PermissionError(实测:某 NAS 映射盘上
# ``xr.open_dataset`` 打不开 400 KB 的 .nc,而同一路径用普通文件 I/O 完全正常)。
# 这里的做法是:先直接读/写;失败就把文件搬到本机临时目录再读/写,最后拷回去。
# 对调用方透明,只在 meta['warnings'] 里记一条。

def _temp_copy(path):
    """把文件复制到本机临时目录;失败返回 ``None``。"""
    src = Path(os.fspath(path))
    tmp = os.path.join(tempfile.gettempdir(),
                       f"shsynth_{uuid.uuid4().hex}{src.suffix}")
    try:
        shutil.copyfile(src, tmp)
        return tmp
    except Exception:                                    # noqa: BLE001
        return None


@contextlib.contextmanager
def _netcdf_open(path, warnings=None):
    """打开 netCDF;HDF5 在目标文件系统上不可用时自动走临时副本。"""
    import xarray as xr

    tmp = None
    try:
        ds = xr.open_dataset(_require_file(path))
    except Exception as exc:                             # noqa: BLE001
        tmp = _temp_copy(path)
        if tmp is None:
            raise
        if warnings is not None:
            warnings.append(
                f"该路径上 HDF5 无法直接打开 netCDF({type(exc).__name__}),"
                "已复制到本机临时目录后读取;结果不受影响")
        ds = xr.open_dataset(tmp)
    try:
        yield ds
    finally:
        try:
            ds.close()
        except Exception:                                # pragma: no cover
            pass
        if tmp:
            with contextlib.suppress(OSError):
                os.remove(tmp)


def _netcdf_save(ds, path, warnings=None) -> None:
    """写 netCDF;直接写失败时改写到本机临时文件再拷回目标路径。"""
    p = Path(os.fspath(path))
    parent = p.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_netcdf(p, engine="netcdf4")
        return
    except Exception:                                    # noqa: BLE001
        pass                                             # 落到临时文件中转
    tmp = os.path.join(tempfile.gettempdir(),
                       f"shsynth_{uuid.uuid4().hex}.nc")
    try:
        ds.to_netcdf(tmp, engine="netcdf4")
    except Exception:                                    # pragma: no cover
        ds.to_netcdf(tmp)
    try:
        shutil.copyfile(tmp, p)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp)
    if warnings is not None:
        warnings.append(
            "HDF5 无法直接写入该路径,已先写到本机临时文件再拷回;文件内容一致")


def _clean_axis(v: np.ndarray, warnings: list, name: str) -> np.ndarray:
    """整理一根坐标轴:float32 造成的 ``-89.00000190734863`` 之类会被吸附回等间距。"""
    v = np.asarray(v, dtype=float).ravel()
    if v.size < 3:
        return v
    d = np.diff(v)
    if np.allclose(d, d[0], rtol=0, atol=1e-6 * max(1.0, abs(float(d[0])))):
        step = float(np.round(np.mean(d), 6))
        if step != 0.0:
            snapped = np.round(v / step) * step if abs(step - round(step)) < 1e-9 \
                else v
            if np.allclose(snapped, v, rtol=0, atol=1e-9 * max(1.0, abs(step))):
                v = snapped
    return v


# ---------------------------------------------------------------------------
# 散点:读
# ---------------------------------------------------------------------------
def read_points(path, lat_col=None, lon_col=None, val_col=None,
                delim=None, require_values: bool = True) -> tuple:
    """读散点 ``(lat, lon, value)``。

    Parameters
    ----------
    path : str | Path
        ``.csv .txt .dat .tsv .npy .xlsx``(文本允许 ``.gz``)。
    lat_col, lon_col, val_col : int | str, optional
        列下标(0 基,负数从末尾数)或列名。数字字符串按**下标**解释
        (``"1"`` = 第 2 列)。省略时若表头含 ``lat``/``latitude``/``纬度``、
        ``lon``/``longitude``/``经度``、``value``/``z``/``数值`` 就用表头,
        否则退回通行约定:第 1 列**经度**、第 2 列**纬度**、第 3 列起为值。
    require_values : bool
        为 ``False`` 时允许文件只有坐标(纯位置文件)。

    Returns
    -------
    lat, lon, values, meta : ndarray, ndarray, ndarray, dict
        ``lat``/``lon`` 为度(经度归一化到 ``[0,360)``);``values`` 形状
        ``(N,)`` 或 ``(N, ntime)``;``meta`` 记录解析出的列、分隔符与
        ``warnings`` —— **先看 warnings 再信数字**。
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in POINT_EXTENSIONS:
        raise ValueError(
            f"read_points: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(POINT_EXTENSIONS)}")
    warnings: list = []
    meta: dict = {"source_file": str(p), "kind": "points", "format": ext.lstrip(".")}

    if ext == ".npy":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                f"{p}: .npy 必须是 (N,2) 或 (N,>=3) 的二维数组,实际 {arr.shape}")
        lon = arr[:, 0].copy()
        lat = arr[:, 1].copy()
        vals = arr[:, 2:].copy()
        warnings.append(".npy 按第1列=经度、第2列=纬度、其余列=数值解释")
        meta.update({"columns": [f"col{i + 1}" for i in range(arr.shape[1])],
                     "has_header": False, "layout": "lon,lat,value[,...]"})
    elif ext == ".xlsx":
        try:
            import pandas as pd
        except ImportError as exc:                       # pragma: no cover
            raise ImportError("读 .xlsx 需要 pandas + openpyxl") from exc
        df = pd.read_excel(p)
        cols = [str(c) for c in df.columns]
        data = df.to_numpy(dtype=float)
        lat, lon, vals, meta2, w = _resolve_point_columns(
            data, cols, lat_col, lon_col, val_col, require_values)
        warnings.extend(w)
        meta.update(meta2)
        meta["has_header"] = True
    else:
        comments, header, data, bad_cols = _read_table(p, delim=delim)
        cols = [str(h) for h in header] if header else \
            [f"col{i + 1}" for i in range(data.shape[1])]
        if bad_cols:
            warnings.append("这些列不是数值,已忽略:"
                            + "、".join(repr(c) for c in bad_cols)
                            + "(台站名/标签这类列本来就不该参与计算)")
        lat, lon, vals, meta2, w = _resolve_point_columns(
            data, cols, lat_col, lon_col, val_col, require_values)
        warnings.extend(w)
        meta.update(meta2)
        meta["has_header"] = header is not None
        meta["non_numeric_columns"] = bad_cols
        if comments:
            meta["comment_header"] = comments

    lon = _normalise_lon(lon, warnings)
    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    vals = np.asarray(vals, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    if lat.size != lon.size:
        raise ValueError(f"{p}: lat 与 lon 长度不同({lat.size} vs {lon.size})")
    if vals.shape[1] == 0 and require_values:
        raise ValueError(
            f"{p}: 文件里只有经纬度两列,没有可用的数值列。"
            "如果只是要用它作为**求值位置**,请用 read_positions()"
            "(或界面上的『散点文件』/命令行 --points,那条路径不需要数值列)")
    if vals.shape[0] != lat.size and require_values:
        raise ValueError(f"{p}: 值与坐标长度不同"
                         f"({vals.shape[0]} vs {lat.size})")
    if vals.shape[1] == 1:
        vals = vals[:, 0]
    if np.any(~np.isfinite(lat)) or np.any(np.abs(lat) > 90.0):
        bad = int(np.sum(~np.isfinite(lat)) + np.sum(np.abs(lat) > 90.0))
        warnings.append(f"有 {bad} 个纬度超出 [-90, 90] 或非有限值;请核对列分配")
    meta.update({"n_points": int(lat.size),
                 "n_time": 1 if vals.ndim == 1 else int(vals.shape[1]),
                 "lat_range": [float(np.nanmin(lat)), float(np.nanmax(lat))],
                 "lon_range": [float(np.nanmin(lon)), float(np.nanmax(lon))],
                 "warnings": warnings})
    return lat, lon, vals, meta


def _resolve_point_columns(data: np.ndarray, cols: list, lat_col, lon_col,
                           val_col, require_values: bool):
    """列解析:表头优先,其次下标,最后退回 lon,lat,value 约定。"""
    warnings: list = []
    ncols = data.shape[1]
    if ncols < 2:
        raise ValueError(f"散点文件至少需要两列(经纬度),实际 {ncols} 列")

    i_lon = _guess_named(cols, _LON_NAMES) if lon_col is None else None
    i_lat = _guess_named(cols, _LAT_NAMES) if lat_col is None else None
    if lon_col is None:
        if i_lon is not None:
            note_lon = f"经度 = 列 {cols[i_lon]!r}"
        else:
            i_lon = 0
            note_lon = "经度 = 第 1 列(未找到经度列名,按通行约定)"
    else:
        i_lon = _resolve_col(lon_col, cols, ncols, "经度", 0)
        note_lon = f"经度 = 列 {cols[i_lon]!r}"
    if lat_col is None:
        if i_lat is not None:
            note_lat = f"纬度 = 列 {cols[i_lat]!r}"
        else:
            i_lat = 1
            note_lat = "纬度 = 第 2 列(未找到纬度列名,按通行约定)"
    else:
        i_lat = _resolve_col(lat_col, cols, ncols, "纬度", 1)
        note_lat = f"纬度 = 列 {cols[i_lat]!r}"
    if i_lat == i_lon:
        raise ValueError(
            f"经度列与纬度列指向同一列({cols[i_lon]!r});请显式指定 lat_col/lon_col")
    warnings.extend([note_lon, note_lat])

    if val_col is not None:
        i_val = [_resolve_col(val_col, cols, ncols, "数值", 2)]
    else:
        first = _guess_named(cols, _VAL_NAMES)
        if first is not None and first not in (i_lon, i_lat):
            chain = [first]
            low = [str(c).strip().lower() for c in cols]
            k = 1
            while f"{str(cols[first]).strip().lower()}{k}" in low:
                chain.append(low.index(f"{str(cols[first]).strip().lower()}{k}"))
                k += 1
            i_val = chain
            warnings.append("数值 = 列 " + ",".join(repr(str(cols[j])) for j in chain))
        else:
            rest = [i for i in range(ncols) if i not in (i_lon, i_lat)]
            if not rest:
                if not require_values:
                    i_val = []
                else:
                    raise ValueError("文件里只有经纬度两列,没有可用的数值列")
            else:
                i_val = rest
                warnings.append(
                    "数值 = 其余列 " + ",".join(repr(str(cols[j])) for j in i_val)
                    + "(未找到数值列名,按通行约定)")

    lat = data[:, i_lat]
    lon = data[:, i_lon]
    vals = data[:, i_val] if len(i_val) else np.zeros((data.shape[0], 0))
    meta = {"columns": cols, "lon_column": cols[i_lon], "lat_column": cols[i_lat],
            "value_columns": [cols[j] for j in i_val],
            "column_index": {"lon": i_lon, "lat": i_lat, "value": i_val}}
    return lat, lon, vals, meta, warnings


def read_positions(path, lat_col=None, lon_col=None, delim=None) -> tuple:
    """只读**位置**(经纬度),不需要数值列。返回 ``(lat, lon, meta)``。"""
    lat, lon, _vals, meta = read_points(path, lat_col=lat_col, lon_col=lon_col,
                                        delim=delim, require_values=False)
    return lat, lon, meta


def peek_columns(path, delim=None) -> dict:
    """只看一眼表格文件的列,用于界面做"纬度列 / 经度列"下拉框。

    Returns
    -------
    dict
        ``has_header``      是否有表头行
        ``columns``         列名列表(无表头时是 ``['第1列', '第2列', …]``)
        ``n_columns``       列数
        ``suggested_lat``   自动猜的纬度列下标(可能为 ``None``)
        ``suggested_lon``   自动猜的经度列下标
        ``n_rows``          数据行数(最多数 5000 行,够了就停)
        ``preview``         前 3 行数据(仅用于界面显示)
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext == ".npy":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
        ncol = int(arr.shape[1]) if arr.ndim > 1 else 1
        cols = [f"第{i + 1}列" for i in range(ncol)]
        return {"has_header": False, "columns": cols, "n_columns": ncol,
                "suggested_lat": 1 if ncol > 1 else None,
                "suggested_lon": 0 if ncol else None,
                "n_rows": int(arr.shape[0]) if arr.ndim else 0,
                "preview": np.atleast_2d(arr[:3]).tolist(),
                "notes": ".npy 按第1列经度、第2列纬度解释"}
    if ext == ".xlsx":
        try:
            import pandas as pd
            df = pd.read_excel(p, nrows=3)
            cols = [str(c) for c in df.columns]
            return {"has_header": True, "columns": cols, "n_columns": len(cols),
                    "suggested_lat": _guess_named(cols, _LAT_NAMES),
                    "suggested_lon": _guess_named(cols, _LON_NAMES),
                    "n_rows": None, "preview": df.to_numpy().tolist()}
        except Exception as exc:                         # noqa: BLE001
            return {"has_header": True, "columns": [], "n_columns": 0,
                    "suggested_lat": None, "suggested_lon": None,
                    "n_rows": None, "preview": [], "error": str(exc)}

    comments, header, data, bad_cols = _read_table(p, delim=delim)
    if header:
        cols = [str(h) for h in header]
        has_header = True
    else:
        ncol = int(data.shape[1])
        cols = [f"第{i + 1}列" for i in range(ncol)]
        has_header = False
    i_lat = _guess_named(cols, _LAT_NAMES)
    i_lon = _guess_named(cols, _LON_NAMES)
    if i_lat is None and i_lon is None and not has_header:
        i_lon, i_lat = 0 if len(cols) > 0 else None, 1 if len(cols) > 1 else None
    return {"has_header": has_header, "columns": cols,
            "non_numeric_columns": bad_cols,
            "n_columns": len(cols),
            "suggested_lat": i_lat, "suggested_lon": i_lon,
            "n_rows": int(data.shape[0]),
            "preview": np.asarray(data[:3]).tolist(),
            "comment_header": comments[:3]}


# ---------------------------------------------------------------------------
# 散点:写
# ---------------------------------------------------------------------------
def write_points(path, lat, lon, values, header: bool = True,
                 fmt: str = "%.10g", comment: Optional[str] = None,
                 delim=None, extra_cols: Optional[Mapping[str, Any]] = None) -> Path:
    """把散点写成 ``lon, lat, value[, value2 ...]``。

    Parameters
    ----------
    path : str | Path
        ``.csv``(逗号)、``.txt``/``.dat``(空白)、``.tsv``(制表符)、``.npy``
        (一个二维数组);文本允许 ``.gz``。
    lat, lon : array_like
        度,长度同为 ``N``。
    values : array_like
        ``(N,)`` 或 ``(N, ntime)``。
    header : bool
        写 ``lon,lat,value...`` 表头(``.npy`` 忽略)。
    fmt : str
        数字格式,默认 ``%.10g``。
    comment : str, optional
        额外 ``#`` 注释行。
    extra_cols : mapping, optional
        额外的 ``名字 -> 列``,追加在右侧。

    Returns
    -------
    pathlib.Path
        实际写出的路径。
    """
    p = Path(os.fspath(path))
    ext = _suffix(p)
    if ext not in (".csv", ".txt", ".dat", ".tsv", ".npy"):
        raise ValueError(
            f"write_points: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: .csv .txt .dat .tsv .npy(.gz 可用于文本)")

    lat = np.atleast_1d(np.asarray(lat, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon, dtype=float)).ravel()
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] != lat.size or lat.size != lon.size:
        raise ValueError(
            f"长度不一致: lat {lat.size}, lon {lon.size}, values {vals.shape[0]}")

    if ext == ".npy":
        if str(p).lower().endswith(".gz"):
            raise ValueError("write_points: .npy 不支持 .gz 后缀")
        if str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.column_stack([lon, lat, vals]))
        return p

    names = [COL_LON, COL_LAT] + [COL_VAL if i == 0 else f"{COL_VAL}{i}"
                                  for i in range(vals.shape[1])]
    mat = [lon, lat] + [vals[:, k] for k in range(vals.shape[1])]
    if extra_cols:
        for k, v in extra_cols.items():
            arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
            if arr.size != lat.size:
                raise ValueError(f"extra_cols[{k!r}] 长度 {arr.size} != {lat.size}")
            names.append(str(k))
            mat.append(arr)
    data = np.column_stack(mat)

    if delim is None:
        delim = {".csv": ",", ".tsv": "\t", ".txt": " ", ".dat": " "}.get(ext, ",")
    lines = []
    if comment:
        lines.append("# " + comment.replace("\n", "\n# "))
    if header:
        lines.append(delim.join(names))
    with np.errstate(all="ignore"):
        body = "\n".join(delim.join(fmt % v if np.isfinite(v) else "nan"
                                    for v in row) for row in data)
    lines.append(body)
    _text_write(p, "\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# 网格:读
# ---------------------------------------------------------------------------
def read_grid(path, var: Optional[str] = None,
              with_time: bool = False) -> tuple:
    """读规则经纬网格。

    Parameters
    ----------
    path : str | Path
        认可的形式(**按内容识别,扩展名不准也能认**):

        ==========================  ============================================
        netCDF                      ``.nc``/``.nc4``/``.cdf``(CF);也接受**内容
                                    是 netCDF 的 ``.grd``**(GMT 的 ``.grd``
                                    其实就是 netCDF,常见)
        Surfer ASCII 网格           ``.grd``(首行 ``DSAA``/``DSBB``);
                                    **按每行 10 个数字折行**也能读
        Esri ASCII 网格             ``.asc``(头部 ``ncols/nrows/cellsize``)
        numpy 堆叠                  ``.npy``,形状 ``(>=3, nlat, nlon)`` =
                                    ``[lon; lat; value]``
        三列文本                    ``.csv``/``.txt``/``.dat`` =
                                    ``lat,lon,value``,pivot 成网格
        ==========================  ============================================

        **不支持**:Surfer 二进制网格(``DSRB``/``DSI``)、GeoTIFF、投影/平面坐标
        网格。Surfer 二进制请用『网格 → 转换』另存为 ASCII 或 netCDF。
    var : str, optional
        netCDF 变量名(默认选第一个依赖 lat/lon 的变量)。
    with_time : bool
        ``True`` 时额外返回第 5 个值 :class:`~shsynth.timeaxis.TimeAxis`
        (v2.0 新增)。**默认 ``False``,返回值个数与 v1.0 完全一致。**
        文件里没有时间坐标时返回的轴是 ``kind='index'``(并带一条警告)。

    Returns
    -------
    lat_vec, lon_vec, grid, meta[, times]
        ``lat_vec`` 严格递增、``lon_vec`` 递增;``grid`` 形状 ``(nlat, nlon)``
        或 ``(nlat, nlon, ntime)``。坐标不是经纬度(例如平面米制坐标)时**报错**
        并说明原因,不会给出一个看着正常但位置全错的网格。
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in GRID_EXTENSIONS:
        raise ValueError(
            f"read_grid: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(GRID_EXTENSIONS)}")
    warnings: list = []
    meta: dict = {"source_file": str(p), "kind": "grid"}

    # 先按**内容**判型:扩展名经常撒谎(GMT 的 .grd 其实是 netCDF、
    # 有的 .txt 其实是 Surfer ASCII ……)
    sniffed = _sniff_grid_content(p)
    if sniffed == "netcdf":
        kind = "netcdf"
    elif sniffed == "surfer_ascii":
        kind = "surfer"
    elif sniffed == "surfer_binary":
        kind = "surfer_bin"
    elif sniffed == "esri_ascii":
        kind = "esri"
    else:
        kind = {"nc": "netcdf", "nc4": "netcdf", "cdf": "netcdf", "grd": "surfer",
                "npy": "npy", "asc": "esri"}.get(ext.lstrip("."), "text")

    if kind == "surfer_bin":                             # 先给出精确说明
        _read_grid_surfer(p)                             # 必抛错,消息更具体
    if kind == "netcdf":
        if ext not in (".nc", ".nc4", ".cdf"):
            warnings.append(
                f"文件内容是 netCDF,但扩展名是 {p.suffix!r}(GMT 的 .grd 常见如此);"
                "已按 netCDF 读取")
        lat, lon, grid, meta2, w = _read_grid_netcdf(p, var)
        meta.update(meta2)
        warnings.extend(w)
    elif kind == "surfer":
        lat, lon, grid, meta2, w = _read_grid_surfer(p)
        meta.update(meta2)
        warnings.extend(w)
    elif kind == "esri":
        lat, lon, grid, meta2, w = _read_grid_esri_ascii(p)
        meta.update(meta2)
        warnings.extend(w)
    elif kind == "npy":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
        meta["array_shape"] = list(arr.shape)
        if arr.ndim == 3 and arr.shape[0] >= 3:
            # 布局 (>=3, nlat, nlon):第 0 层经度、第 1 层纬度、其余层数值
            # (坐标层是 meshgrid 出来的二维阵列,要还原成一维轴并核对确实是网格)
            lon2, lat2 = arr[0], arr[1]
            grid = arr[2:]
            if grid.shape[0] == 1:
                grid = grid[0]
            lon_vec = lon2[0, :]
            lat_vec = lat2[:, 0]
            if not (np.allclose(lon2, lon_vec[None, :]) and
                    np.allclose(lat2, lat_vec[:, None])):
                raise ValueError(
                    f"{p}: .npy 的前两层不是规则经纬网格(看不到 meshgrid 结构)。"
                    "散点数据请用 read_points()")
            lat, lon = lat_vec, lon_vec
            warnings.append(".npy 按 (>=3, nlat, nlon) 的 [lon; lat; value] 解释")
        else:
            raise ValueError(
                f"{p}: .npy 网格需要 (>=3, nlat, nlon) 的 [lon; lat; value] 堆叠,"
                f"实际 {arr.shape}。只有数值没有坐标轴的网格请用 .nc / .csv,"
                "散点数据请用 read_points()")
        meta["format"] = "npy"
    else:
        _comments, header, data, bad_cols = _read_table(p)
        if bad_cols:
            warnings.append("这些列不是数值,已忽略:"
                            + "、".join(repr(c) for c in bad_cols))
        if data.shape[1] < 3:
            raise ValueError(f"{p}: 三列网格文件至少需要 lat,lon,value 三列")
        cols = [str(h) for h in header] if header else \
            [f"col{i + 1}" for i in range(data.shape[1])]
        i_lat = _guess_named(cols, _LAT_NAMES)
        i_lon = _guess_named(cols, _LON_NAMES)
        i_lat = 0 if i_lat is None else i_lat
        i_lon = 1 if i_lon is None else i_lon
        if i_lat == i_lon:
            raise ValueError(
                f"{p}: 纬度和经度解析到了同一列({cols[i_lat]!r});"
                "请把表头写清楚,或改用带坐标轴的 .nc/.npy")
        # 数值列 = **既不是纬度也不是经度**的第一列
        # (以前写死第 3 列,表头是 lat,value,lon 这种就会把纬度当数值读)
        rest = [i for i in range(data.shape[1]) if i not in (i_lat, i_lon)]
        if not rest:
            raise ValueError(f"{p}: 三列网格只有经纬度两列,没有数值列")
        i_val = rest[0]
        lat_c = data[:, i_lat]
        lon_c = _normalise_lon(data[:, i_lon], warnings)
        val_c = data[:, i_val]
        lat, lon, grid, dup_note = _pivot_grid(lat_c, lon_c, val_c, p,
                                               cols[i_lat], cols[i_lon])
        if dup_note:
            warnings.append(dup_note)
        meta["format"] = "text_grid"
        meta["column_index"] = {"lat": i_lat, "lon": i_lon, "value": i_val}
        warnings.append(
            f"三列网格:纬度=列 {cols[i_lat]!r}、经度=列 {cols[i_lon]!r}、"
            f"数值=列 {cols[i_val]!r}(已 pivot 成网格)")

    lat = _clean_axis(np.atleast_1d(np.asarray(lat, float)).ravel(), warnings, "lat")
    lon = _clean_axis(np.atleast_1d(np.asarray(lon, float)).ravel(), warnings, "lon")
    grid = np.asarray(grid, dtype=float)
    if grid.ndim not in (2, 3):
        raise ValueError(f"{p}: 网格必须是 2 维或 3 维,实际 {grid.ndim} 维")
    if grid.shape[0] != lat.size or grid.shape[1] != lon.size:
        raise ValueError(
            f"{p}: 网格 shape {grid.shape} 与坐标轴(nlat={lat.size}, "
            f"nlon={lon.size})不一致")
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        grid = grid[::-1, ...].copy()
        meta["lat_order_flipped"] = True
        warnings.append("盘中纬度是递减的,已连数据一起翻转成递增")
    if lat.size > 1 and np.any(np.diff(lat) <= 0):
        order = np.argsort(lat, kind="stable")
        lat = lat[order]
        grid = grid[order, ...]
        warnings.append("纬度轴有重复或乱序,已按稳定排序整理成严格递增")
    if np.any(np.diff(lon) < 0):
        order = np.argsort(lon, kind="stable")
        lon = lon[order]
        grid = grid[:, order, ...]
        warnings.append("经度轴已排序成递增")

    # 经度 0..360 **含端点**时会出现两列 0(360 ≡ 0):去重,否则 lon_vec 不是
    # 严格递增,后面画图/pivot 都会受影响。
    if lon.size > 1:
        dup = np.flatnonzero(np.diff(lon) == 0)
        if dup.size:
            keep = np.ones(lon.size, dtype=bool)
            keep[dup + 1] = False
            lon = lon[keep]
            grid = grid[:, keep, ...]
            warnings.append(
                f"经度轴有 {int(dup.size)} 个重复值(通常是 0 与 360 同时出现"
                f"≡ 同一根经线),已去掉重复列,nlon 变为 {lon.size}")

    # 坐标必须真的是经纬度:平面/投影坐标(米)在这里就停下,而不是给出一张
    # 看着正常、位置全错的地图。
    _guard_geographic_coords(p, lat, lon, meta, warnings)

    meta.update({
        "nlat": int(lat.size), "nlon": int(lon.size),
        "ntime": 1 if grid.ndim == 2 else int(grid.shape[2]),
        "lat_range": [float(lat.min()), float(lat.max())],
        "lon_range": [float(lon.min()), float(lon.max())],
        "lat_step_deg": float(np.median(np.diff(lat))) if lat.size > 1 else None,
        "lon_step_deg": float(np.median(np.diff(lon))) if lon.size > 1 else None,
        "warnings": warnings,
    })
    if lat.size > 1 and not np.allclose(np.diff(lat), np.diff(lat)[0],
                                        rtol=1e-6, atol=1e-9):
        warnings.append("纬度轴不是等间距的(球谐综合本身无所谓,但结果图会显示真实格点)")
    if int(meta["nlat"]) * int(meta["nlon"]) > 4_000_000:
        warnings.append(
            f"这是个很大的网格({meta['nlat']}×{meta['nlon']} = "
            f"{meta['nlat'] * meta['nlon']:,} 点);如果只是要借用它的格点,"
            "求值会比较久(可以用范围网格/粗一点的步长代替)")

    if not with_time:
        return lat, lon, grid, meta

    # ---- v2.0:把时间坐标交出去(不再只留一个 time_name)----------------
    from .timeaxis import TimeAxis
    tc = meta.pop("_time_coord", None)
    ntime = meta["ntime"]
    if tc is not None and np.asarray(tc).size == ntime:
        if np.issubdtype(np.asarray(tc).dtype, np.datetime64):
            ax = TimeAxis.from_datetimes(np.asarray(tc),
                                         source="netcdf_datetime64")
        else:
            try:
                ax = TimeAxis.from_netcdf_coord(tc)
            except Exception:                              # noqa: BLE001
                ax = TimeAxis.from_index(ntime)
                warnings.append("时间坐标无法解析,已退化为按序号当时间")
    else:
        ax = TimeAxis.from_index(ntime, label="网格序号")
        if ntime > 1:
            warnings.append(
                f"这个网格有 {ntime} 层,但文件里没有时间坐标 —— "
                "已退化为按序号当时间;日期信息不可恢复")
    return lat, lon, grid, meta, ax


def _read_grid_netcdf(path, var: Optional[str]):
    """用 xarray 读 netCDF;返回 ``(lat, lon, grid, meta, warnings)``。"""
    try:
        # 只确认依赖在,真正的打开在 _netcdf_open(它还要处理网络盘回退)
        import importlib
        importlib.import_module("xarray")
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "读 .nc 需要 xarray(+netCDF4);也可先把网格导成 .npy / .csv") from exc

    warnings: list = []
    with _netcdf_open(path, warnings) as ds:
        lat_name = next((n for n in ("lat", "latitude", "LAT", "Lat", "lat_deg",
                                     "y", "nav_lat")
                         if n in ds.coords or n in ds.variables), None)
        lon_name = next((n for n in ("lon", "longitude", "LON", "Lon", "lon_deg",
                                     "x", "nav_lon")
                         if n in ds.coords or n in ds.variables), None)
        if lat_name is None or lon_name is None:
            raise ValueError(f"{path}: 找不到 lat/lon 坐标变量")
        if var is None:
            cands = []
            for name, d in ds.data_vars.items():
                if set(d.dims) & {lat_name, lon_name}:
                    cands.append(name)
            if not cands:
                raise ValueError(f"{path}: 没有依赖 lat/lon 的数据变量")
            var = cands[0]
        if var not in ds:
            raise ValueError(
                f"{path}: 变量 {var!r} 不存在;可用: {list(ds.data_vars)}")
        da = ds[var]
        if lat_name not in da.dims or lon_name not in da.dims:
            raise ValueError(
                f"{path}: 变量 {var!r} 的维度 {list(da.dims)} 不含 lat/lon")
        time_name = next((n for n in ("time", "t", "nt", "epoch", "month")
                          if n in da.dims), None)
        order = [lat_name, lon_name] + ([time_name] if time_name else [])
        for d in da.dims:
            if d not in order:
                order.append(d)
        grid = np.asarray(da.transpose(*order).values, dtype=float)
        lat = np.asarray(ds[lat_name].values, dtype=float).ravel()
        lon = np.asarray(ds[lon_name].values, dtype=float).ravel()
        units = str(da.attrs.get("units", ""))
        meta = {
            "format": "netcdf", "variable": str(var),
            "data_variables": [str(v) for v in ds.data_vars],
            "dims": list(da.dims),
            "value_units": units,
            "value_long_name": str(da.attrs.get("long_name", "")),
            "attrs": {k: str(v) for k, v in ds.attrs.items()},
            "lat_name": lat_name, "lon_name": lon_name, "time_name": time_name,
        }
        # v2.0:把真正的时间坐标带出来(私有键,read_grid 消费后删掉)。
        # 优先级:同名坐标变量 → 任何带 time 维的坐标 → 全局属性里的字符串
        if time_name is not None:
            tc = None
            for cand in (time_name, "time", "Time", "t", "TIME"):
                if cand in ds.coords or cand in ds.variables:
                    tc = ds[cand]
                    break
            if tc is not None and tc.dims == (time_name,):
                meta["_time_coord"] = np.asarray(tc.values)
    if lon.size and (lon.min() < 0 or lon.max() >= 360.0):
        warnings.append("经度按 mod 360 归一化到 [0, 360)")
        order = np.argsort(np.mod(lon, 360.0))
        lon = np.mod(lon, 360.0)[order]
        grid = grid[:, order, ...]
    return lat, lon, grid, meta, warnings


def _read_grid_surfer(path):
    """读 Surfer ASCII 网格(``DSAA`` / ``DSBB``)。

    ⚠️ **Surfer 写 ASCII 网格时会把一行按每行 10 个数字折行**(不是一行一个
    数据行)。实测用户的数据:

    ```
    DSAA
    56 47                     ← 56 列 × 47 行
    369200 396700             ← x 范围
    2044500 2067500           ← y 范围
    -54.99 -5.33              ← z 范围
    -39.26 -40.05 … (每行 10 个,共 6 行才凑满 56 列的一行)
    ```

    所以这里**把所有数值读成一维流再 reshape**(``ny × nx``),折行与不折行都能读。
    """
    text = _text_read(path)
    lines = [ln for ln in text.splitlines()]
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"{path}: 空文件")
    magic = lines[idx].strip().upper()
    idx += 1
    warnings: list = []
    if magic.startswith("DSRB") or magic.startswith("DSI"):
        raise ValueError(
            f"{path}: 这是 **Surfer 二进制** 网格({magic}),本软件只读 ASCII 网格。"
            "转换方法(任选其一):\n"
            "  · Surfer 里『网格 → 转换 / Grid → Convert』另存为 "
            "'ASCII Grid (*.grd)' 或 'netCDF (*.nc)';\n"
            "  · 或用本软件能直接读的格式重新导出:.nc / .npy / .csv(三列 "
            "lat,lon,value)")
    if magic not in ("DSAA", "DSBB"):
        raise ValueError(
            f"{path}: 首行是 {magic!r},不是 Surfer ASCII 网格的 'DSAA'/'DSBB'。"
            "若这是 GMT 的 .grd(其实是 netCDF)请改名为 .nc 或直接指定 --grid-var;"
            "若不确定,用 shsynth info 无法判断时请提供文件头两行")

    def next_numeric():
        nonlocal idx
        while idx < len(lines):
            s = lines[idx].strip()
            idx += 1
            if not s or s.startswith("#"):
                continue
            try:
                return [float(t) for t in s.split()]
            except ValueError:
                continue
        raise ValueError(f"{path}: .grd 头部不完整")

    head = next_numeric()
    if len(head) < 2:
        raise ValueError(f"{path}: .grd 缺少行列数")
    nx, ny = int(head[0]), int(head[1])
    xlo, xhi = next_numeric()[:2]
    ylo, yhi = next_numeric()[:2]
    zlo, zhi = next_numeric()[:2]

    # 关键:把所有数值读成一维流,再按 (ny, nx) 重排 —— Surfer 会按每行 10 个
    # 数字折行,按"一行一个数据行"读会立刻失败或读错。
    values: list = []
    want = nx * ny
    while idx < len(lines) and len(values) < want:
        s = lines[idx].strip()
        idx += 1
        if not s or s.startswith("#"):
            continue
        for tok in s.split():
            try:
                values.append(float(tok))
            except ValueError:
                continue
    if len(values) < want:
        raise ValueError(
            f"{path}: .grd 数据不足:读到 {len(values)} 个数值,需要 {ny}×{nx}"
            f" = {want} 个(是否文件被截断?)")
    if len(values) > want:
        warnings.append(f".grd 末尾多出 {len(values) - want} 个数值,已忽略")
    grid = np.asarray(values[:want], dtype=float).reshape(ny, nx)
    blank = 1.701410009187828e38
    grid = np.where(np.abs(grid) >= blank * 0.5, np.nan, grid)
    lon = np.linspace(xlo, xhi, nx)
    lat = np.linspace(ylo, yhi, ny)                # Surfer 行序是 y 递增
    meta = {"format": "surfer_grd", "magic": magic,
            "z_range": [float(zlo), float(zhi)],
            "notes": "Surfer 行序 y 递增(南→北),读入后纬度轴为升序;"
                     "ASCII 网格常按每行 10 个数字折行,已按数值流重排"}
    return lat, lon, grid, meta, warnings


def _read_grid_esri_ascii(path):
    """读 Esri ASCII 网格(``.asc``;头部 ``ncols/nrows/xllcorner/cellsize``)。"""
    text = _text_read(path)
    warnings: list = []
    header: dict = {}
    values: list = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[0].lower() in (
                "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
                "yllcenter", "cellsize", "nodata_value"):
            key = parts[0].lower()
            try:
                header[key] = float(parts[1])
            except ValueError:
                pass
            continue
        for tok in parts:
            try:
                values.append(float(tok))
            except ValueError:
                continue
    need = ("ncols", "nrows", "cellsize")
    for k in need:
        if k not in header:
            raise ValueError(f"{path}: Esri ASCII 网格缺少头部字段 {k}")
    nx, ny = int(header["ncols"]), int(header["nrows"])
    cs = float(header["cellsize"])
    x0 = header.get("xllcorner", header.get("xllcenter", 0.0))
    y0 = header.get("yllcorner", header.get("yllcenter", 0.0))
    if len(values) < nx * ny:
        raise ValueError(f"{path}: Esri ASCII 数据不足({len(values)} < {nx * ny})")
    grid = np.asarray(values[:nx * ny], dtype=float).reshape(ny, nx)
    nodata = header.get("nodata_value")
    if nodata is not None:
        grid = np.where(grid == nodata, np.nan, grid)
    lon = x0 + (np.arange(nx) + 0.5) * cs
    lat = y0 + (np.arange(ny) + 0.5) * cs
    meta = {"format": "esri_ascii", "cellsize": cs,
            "notes": "Esri ASCII:按像元中心给出坐标,行序自北向南(已翻成递增)"}
    return lat, lon, grid, meta, warnings


def _guard_geographic_coords(path, lat, lon, meta: dict, warnings: list) -> None:
    """确认坐标真的是**经纬度**,而不是投影/平面坐标(米)。

    实测用户手里的 Surfer 网格大量是平面坐标(例如海南项目的
    ``369200..396700`` 米),直接当经纬度用会得到一张错误的地图,而且不报警。
    宁可停下来解释清楚。
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_bad = bool(lat.size and (np.nanmin(lat) < -90.0 or np.nanmax(lat) > 90.0))
    lon_bad = bool(lon.size and (np.nanmin(lon) < -360.0 or
                                 np.nanmax(lon) > 360.0))
    if not (lat_bad or lon_bad):
        return
    rng = (f"纬度轴 {float(np.nanmin(lat)):.6g}..{float(np.nanmax(lat)):.6g}, "
           f"经度轴 {float(np.nanmin(lon)):.6g}..{float(np.nanmax(lon)):.6g}")
    raise ValueError(
        f"{path}: 坐标不是经纬度({rng})——看起来是**投影/平面坐标(米)**。\n"
        "球谐综合只在球面上做,需要经纬度格点,所以这个网格不能直接当求值位置。"
        "可选做法:\n"
        "  · 只要它的**数值**、位置另给:改用 --global-grid / 范围网格 / 散点文件;\n"
        "  · 想做球面分析:先把该网格投影/插值成等经纬网格(格点经纬度),"
        "再存成 .nc/.npy/.csv 三列 lat,lon,value;"
        f"\n  (文件格式本身没有问题:{meta.get('format', '?')})")


def _nc_attr(v):
    """把 meta 的值转成 netCDF 属性**能接受**的类型。

    netCDF 属性只认 str / 数值数组;Python 的 ``bool`` 和 numpy 的 ``np.True_``
    会直接抛 ``TypeError: illegal data type for attribute``
    (``lat_order_flipped=True`` 就这么把一个正常的网格写成崩溃)。
    """
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return json.dumps(_jsonable(v), ensure_ascii=False)


def _sniff_grid_content(path) -> str:
    """按**文件内容**判断网格类型(扩展名经常撒谎)。

    返回 ``'netcdf'`` / ``'surfer_ascii'`` / ``'surfer_binary'`` /
    ``'esri_ascii'`` / ``''``(认不出来)。
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:                                      # pragma: no cover
        return ""
    if head[:3] == b"CDF":
        return "netcdf"                      # NetCDF-3 经典格式
    if head[:4] == b"\x89HDF":
        return "netcdf"                      # HDF5 / netCDF-4
    if head[:4] in (b"DSAA", b"DSBB"):
        return "surfer_ascii"
    if head[:4] in (b"DSRB", b"DSBB") or head[:3] == b"DSI":
        return "surfer_binary"
    # 可能带 UTF-8 BOM
    body = head[3:] if head[:3] == b"\xef\xbb\xbf" else head
    if body[:3] in (b"CDF",):
        return "netcdf"
    if body[:4] in (b"DSAA", b"DSBB"):
        return "surfer_ascii"
    low = body[:64].lower()
    if b"ncols" in low and b"nrows" in low:
        return "esri_ascii"
    return ""


# ---------------------------------------------------------------------------
# 网格:时间坐标(v2.0)
# ---------------------------------------------------------------------------
def _coerce_times(times):
    """``TimeAxis`` / ``datetime64`` 数组 / ``None`` → ``TimeAxis`` 或 ``None``。"""
    if times is None:
        return None
    from .timeaxis import TimeAxis
    if isinstance(times, TimeAxis):
        return times
    arr = np.atleast_1d(np.asarray(times))
    if np.issubdtype(arr.dtype, np.datetime64):
        return TimeAxis.from_datetimes(arr, source="user")
    return TimeAxis.from_decimal_years(np.asarray(arr, dtype=float))


def _attach_time_coord(ds, tv, *, encoding: str = "datetime64") -> None:
    """把 ``TimeAxis`` 挂到 xarray Dataset 上,并补齐 CF 属性。"""
    import xarray as xr
    n = len(tv)
    if "time" not in ds.dims:
        if n == 1:
            ds["time"] = xr.DataArray(tv.to_datetime64(), dims=("time",))
            return
        raise ValueError(
            f"times 有 {n} 个历元,但网格没有 time 维(只有 "
            f"{list(ds.sizes)});请确认 grid 是 3 维")
    if int(ds.sizes["time"]) != n:
        raise ValueError(
            f"times 有 {n} 个历元,但网格 time 维长度是 {ds.sizes['time']}")
    if tv.kind == "index" or encoding == "index":
        ds["time"] = xr.DataArray(np.arange(n, dtype="int32"), dims=("time",))
        ds["time"].attrs = {"long_name": "epoch index",
                            "units": "1",
                            "comment": "无真实日期,按历元序号"}
        ds.attrs["shsynth_time_note"] = "无真实日期:time 只是历元序号"
        return
    if encoding == "cf":
        origin = np.datetime64("2000-01-01", "D")
        days = ((tv.to_datetime64().astype("datetime64[D]") - origin)
                / np.timedelta64(1, "D"))
        ds["time"] = xr.DataArray(days.astype("int32"), dims=("time",))
        ds["time"].attrs = {"units": "days since 2000-01-01",
                            "calendar": "proleptic_gregorian",
                            "long_name": "time",
                            "standard_name": "time"}
    else:
        ds["time"] = xr.DataArray(tv.to_datetime64(), dims=("time",))
        ds["time"].attrs = {"long_name": "time", "standard_name": "time"}
    ys = tv.decimal_years
    ds["time"].attrs["shsynth_decimal_year"] = (
        f"{ys[0]:.6f} .. {ys[-1]:.6f}" if n else "")
    if tv.start is not None and tv.end is not None:
        try:
            ds.attrs["time_coverage_start"] = str(tv.start[0])[:19]
            ds.attrs["time_coverage_end"] = str(tv.end[-1])[:19]
        except Exception:                                  # pragma: no cover
            pass
    ds.attrs["shsynth_time_source"] = str(tv.meta.get("time_source", "unknown"))
    ds.attrs["shsynth_ntime"] = int(n)


# ---------------------------------------------------------------------------
# 网格:写
# ---------------------------------------------------------------------------
def write_grid(path, lat_vec, lon_vec, grid, var: str = "value",
               meta: Optional[Mapping[str, Any]] = None,
               long_name: Optional[str] = None,
               units: Optional[str] = None,
               comment: Optional[str] = None,
               fmt: str = "%.10g",
               times=None,
               time_encoding: str = "datetime64",
               extra_vars: Optional[Mapping[str, Any]] = None) -> Path:
    """把规则网格写成 ``nc`` / ``grd`` / ``csv`` / ``txt`` / ``npy``。

    Parameters
    ----------
    lat_vec, lon_vec : array_like
        度。``lat_vec`` 一个盘上递减的轴会连数据一起翻转(与 SHKit 行为一致)。
    grid : array_like
        ``(nlat, nlon)`` 或 ``(nlat, nlon, ntime)``。
    var : str
        netCDF 变量名。
    meta : mapping, optional
        自由来源信息,会写进 netCDF 全局属性和文本注释。
    long_name, units : str, optional
        数据变量的 CF 属性。
    times : TimeAxis | array_like, optional
        **v2.0 新增**:时间轴(或 ``datetime64`` 数组)。给了它就写出真正的
        ``time`` 坐标(CF),否则只是 ``dims=(lat,lon,time)`` 而**没有时间值** ——
        那正是 v1.0 的缺陷 A(写出即丢日期)。
    time_encoding : {'datetime64', 'cf'}
        ``'datetime64'``:直接写 ``datetime64[ns]`` 坐标(与用户 ``3_grids`` 同构);
        ``'cf'``:写 ``int32 + units="days since 2000-01-01"``,兼容面更宽。
    extra_vars : mapping, optional
        **v2.0 新增**:同网格的额外变量(水平形变用),
        例如 ``{"north_displacement": arr, "east_displacement": arr}``;
        与 ``grid`` 共享 ``lat/lon/time`` 三个坐标。只在 netCDF 上支持。
    """
    import json

    p = Path(os.fspath(path))
    ext = _suffix(p)
    if ext not in GRID_EXTENSIONS:
        raise ValueError(
            f"write_grid: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(GRID_EXTENSIONS)}")

    lat = np.atleast_1d(np.asarray(lat_vec, dtype=float)).ravel()
    lon = np.atleast_1d(np.asarray(lon_vec, dtype=float)).ravel()
    grid = np.asarray(grid, dtype=float)
    if grid.ndim not in (2, 3):
        raise ValueError(f"grid 必须是 2 维或 3 维,实际 {grid.ndim} 维")
    if grid.shape[0] != lat.size or grid.shape[1] != lon.size:
        raise ValueError(
            f"grid shape {grid.shape} 与坐标轴(nlat={lat.size}, "
            f"nlon={lon.size})不一致")
    m = dict(meta or {})
    if lat.size > 1 and lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        grid = grid[::-1, ...].copy()
        m["lat_order_flipped"] = True
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    if ext in (".nc", ".nc4", ".cdf"):
        try:
            import xarray as xr
        except ImportError as exc:                      # pragma: no cover
            raise ImportError("写 .nc 需要 xarray(+netCDF4)") from exc
        dims = ("lat", "lon") if grid.ndim == 2 else ("lat", "lon", "time")
        da = xr.DataArray(grid, dims=dims,
                          coords={"lat": lat, "lon": lon}, name=var)
        da["lat"].attrs = {"units": "degrees_north",
                           "standard_name": "latitude",
                           "long_name": "geocentric latitude"}
        da["lon"].attrs = {"units": "degrees_east",
                           "standard_name": "longitude",
                           "long_name": "longitude"}
        if units:
            da.attrs["units"] = units
        if long_name:
            da.attrs["long_name"] = long_name
        for k, v in m.items():
            try:
                da.attrs[f"shsynth_{k}"] = _nc_attr(v)
            except (TypeError, ValueError):
                da.attrs[f"shsynth_{k}"] = str(v)
        da.attrs["shsynth_version"] = _pkg_version()
        ds = da.to_dataset()
        # ---- v2.0:时间坐标(缺陷 A 的修复)-------------------------------
        tv = _coerce_times(times)
        if grid.ndim == 3 and tv is None:
            m.setdefault("time_missing", True)
            ds.attrs["shsynth_time_note"] = (
                "未提供 times:本文件只有 time 维长度,没有时间坐标值")
        if tv is not None:
            _attach_time_coord(ds, tv, encoding=time_encoding)
        # ---- v2.0:同网格的额外变量(水平形变北/东等)----------------------
        if extra_vars:
            if grid.ndim != 3 and any(
                    np.ndim(v) == 3 for v in extra_vars.values()):
                raise ValueError(
                    "extra_vars 里有 3 维数组,但主网格是 2 维;"
                    "请让两者形状一致")
            for name, arr in extra_vars.items():
                a = np.asarray(arr, dtype=float)
                if a.shape != grid.shape:
                    raise ValueError(
                        f"extra_vars[{name!r}] 形状 {a.shape} 与主网格 "
                        f"{grid.shape} 不一致")
                ds[str(name)] = (dims, a)
                if units:
                    ds[str(name)].attrs["units"] = units
                ds[str(name)].attrs["shsynth_version"] = _pkg_version()
        ds.attrs["Conventions"] = "CF-1.8"
        ds.attrs["title"] = f"SHSynth {var} on a regular lat/lon grid"
        ds.attrs["source"] = (f"SHSynth {_pkg_version()} "
                              "(shsynth.fieldio.write_grid)")
        ds.attrs["history"] = comment or "written by shsynth.fieldio.write_grid"
        ds.attrs["shsynth_meta"] = json.dumps(_jsonable(m), ensure_ascii=False)
        _netcdf_save(ds, p)
        return p

    if ext == ".grd":
        if grid.ndim == 3:
            raise ValueError(
                "write_grid: Surfer .grd 只有单层,装不下 3 维网格;"
                "请对每个时次各写一个文件,或改用 .nc")
        ny, nx = lat.size, lon.size
        lines = ["DSAA", f"{nx} {ny}",
                 f"{lon.min():.10g} {lon.max():.10g}",
                 f"{lat.min():.10g} {lat.max():.10g}"]
        finite = grid[np.isfinite(grid)]
        zlo = float(finite.min()) if finite.size else 1.0
        zhi = float(finite.max()) if finite.size else 1.0
        lines.append(f"{zlo:.10g} {zhi:.10g}")
        if comment:
            lines.extend(f"# {ln}" for ln in comment.splitlines())
        g = np.where(np.isfinite(grid), grid, 1.701410009187828e38)
        body = "\n".join(" ".join(fmt % v for v in row) for row in g)
        _text_write(p, "\n".join(lines) + "\n" + body + "\n")
        return p

    if ext == ".npy":
        # 统一写成 (3, nlat, nlon) 的 [lon; lat; value] 堆叠 —— read_grid 认这个。
        # (SHKit 对二维网格写的是 np.vstack([lon[None,:], lat[:,None], grid]),
        #  只在 nlon == nlat 时才不报错,这里不复刻这个缺陷。)
        if grid.ndim == 3 and grid.shape[2] != 1:
            raise ValueError(
                "write_grid: .npy 只有一层,装不下多时次网格;"
                "请对每个时次各写一个文件,或改用 .nc")
        g2 = grid[:, :, 0] if grid.ndim == 3 else grid
        arr = np.stack([
            np.repeat(lon[None, :], lat.size, axis=0),
            np.repeat(lat[:, None], lon.size, axis=1),
            g2,
        ])
        np.save(p, np.asarray(arr, dtype=float))
        return p

    # csv / txt / dat:三列 lat,lon,value
    if grid.ndim == 3:
        if grid.shape[2] != 1:
            raise ValueError(
                "write_grid: 三列文本只支持单个时次;多时次请用 .nc 或逐时次写出")
        grid = grid[:, :, 0]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    out = np.column_stack([LA.ravel(), LO.ravel(), grid.ravel()])
    lines = []
    if comment:
        lines.append("# " + comment.replace("\n", "\n# "))
    if m:
        lines.append("# shsynth_meta "
                     + json.dumps(_jsonable(m), ensure_ascii=False))
    lines.append("lat,lon,value")
    with np.errstate(all="ignore"):
        lines.append("\n".join(
            ",".join(fmt % v if np.isfinite(v) else "nan" for v in row)
            for row in out))
    _text_write(p, "\n".join(lines) + "\n")
    return p


def _jsonable(obj):
    """把 numpy 类型递归转成可 JSON 序列化的对象。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 256 else f"<array{obj.shape}>"
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
