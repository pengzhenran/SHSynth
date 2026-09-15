# -*- coding: utf-8 -*-
"""
shsynth.coeffio
===============

**适配 SHKit 全部球谐系数输出格式**的读取/写入模块。

SHKit 的 ``shkit.io.write_coeffs`` 会写出五种布局,这里全部支持,并且
``read_coeffs(layout='auto')`` 能自动认出它们(判据与 SHKit 的
``detect_coeff_layout`` 一致,另加了文件头标记等更强判据):

======================  =========================================  ==================
``layout``              写出的扩展名(SHKit)                        识别依据
======================  =========================================  ==================
``triangle``            ``.sh .txt .csv .dat .tsv``(+``.gz``)      ``#`` 头里含
                                                                  ``ncoef_triangle``,
                                                                  行数 = ``2*(L+1)(L+2)/2``
``gmfcsv``              ``.csv .txt``(``n,m,C,S`` 每行一个系数)  表头 ``n,m,C,S``,
                                                                  行数 = ``(L+1)(L+2)/2``
``gfc``                 ``.gfc``(ICGEM/GFZ 文本)                 首列 ``gfc``/``gfct``
``npy``                 ``.npy``(``(2,L+1,L+1[,ntime])``)        数组维度
``npz``                 ``.npz``(``C``/``S``/``meta``,或
                        ``flat_cs``/``triangle``)                 键名
======================  =========================================  ==================

另外还能读:

* SHKit 三角布局文件头里推上去的元数据(``field_unit``、``gaussian_radius_km``、
  ``coverage``、``n_points`` …),它们决定物理量换算与平滑是否会被重复施加;
* ``.npy`` 的稠密 ``(L+1,L+1)`` / ``(L+1,L+1,ntime)`` / 三角 ``(2*NC[,ntime])``;
* ``.npz`` 的 ``C``/``S`` / ``flat_cs`` / ``triangle`` + ``meta``(JSON);
* 纯文本稠密矩阵(``nmax+1`` 行 × ``nmax+1`` 列,MATLAB 直接 ``dlmwrite`` 那种);
* gz 压缩的文本与 gfc。

所有路径按 ``os.fspath`` 处理,文本 I/O 显式 UTF-8(写出时带 BOM,方便 Excel
打开中文表头),因此中文路径、中文表头都能正常往返。
"""

from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .coeffs import SHCoeffs, triangle_order

__all__ = [
    "COEFF_EXTENSIONS",
    "COEFF_LAYOUTS",
    "FORMAT_SPECS",
    "detect_coeff_layout",
    "read_coeffs",
    "write_coeffs",
    "coeff_info",
    "infer_nmax_from_rows",
    # v2.0
    "read_coeffs_series",
    "scan_gfc_dir",
    "read_series_nc",
    "write_series_nc",
    "read_series_dat",
    "write_series_dat",
]

#: 支持的系数文件扩展名(与 SHKit ``shkit.io.COEFF_EXTENSIONS`` 一致)。
#: v2.0 追加 ``.nc``(多时间序列的标准落盘格式,见 §12.1 契约)。
COEFF_EXTENSIONS = (".sh", ".txt", ".csv", ".dat", ".tsv", ".gfc", ".npy",
                    ".npz", ".nc")

#: ``read_coeffs(layout=...)`` 的合法取值。
#: ``series_nc`` / ``series_dat`` 是 v2.0 的多时间序列布局。
COEFF_LAYOUTS = ("auto", "triangle", "matrix", "gmfcsv", "gfc", "npy", "npz",
                 "series_nc", "series_dat")

#: 供 ``shsynth formats`` 与界面帮助使用的格式说明表。
FORMAT_SPECS = (
    {
        "layout": "triangle",
        "name": "三角堆叠 [C; S]",
        "ext": (".sh", ".txt", ".csv", ".dat", ".tsv", ".gz"),
        "source": "SHKit 默认(m2py / gridSHconvert 的 out_SHCS 兼容)",
        "detail": "2*NC 行 × ntime 列,NC=(L+1)(L+2)/2;排序 m 外层 / n 内层;"
                  "带 # 头(nmax/ntime/field_unit/高斯半径等)。",
    },
    {
        "layout": "gmfcsv",
        "name": "逐行 n,m,C,S 表",
        "ext": (".csv", ".txt", ".dat", ".tsv"),
        "source": "SHKit write_coeffs(layout='gmfcsv')",
        "detail": "每行一个系数;多时次写成 C,S,C_t1,S_t1,...;表头 n,m,C,S。",
    },
    {
        "layout": "gfc",
        "name": "ICGEM / GFZ 文本模型",
        "ext": (".gfc", ".gz"),
        "source": "SHKit write_coeffs(layout='gfc');ICGEM/GFZ 官方格式",
        "detail": "key value 头 + end_of_head,数据行 'gfc n m C S [sigmaC sigmaS]',"
                  "以 end_of_data 结束。单文件只装一个时次。",
    },
    {
        "layout": "npy",
        "name": "numpy 稠密数组",
        "ext": (".npy",),
        "source": "SHKit write_coeffs(layout='npy')",
        "detail": "(2, L+1, L+1[, ntime])(C/S 各一层);也接受 "
                  "(L+1,L+1[,ntime]) 与三角 (2*NC[,ntime])。",
    },
    {
        "layout": "npz",
        "name": "numpy 压缩包",
        "ext": (".npz",),
        "source": "SHKit write_coeffs(layout='npz')",
        "detail": "C / S / meta(JSON) / nmax / ntime / layout;"
                  "也接受 flat_cs / triangle。",
    },
)


# ---------------------------------------------------------------------------
# 小工具


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

# ---------------------------------------------------------------------------
def _require_file(path) -> Path:
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"系数文件不存在: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{p} 是一个目录,不是系数文件")
    return p


def _suffix(p: Path) -> str:
    """扩展名(小写);``x.sh.gz`` 返回 ``.sh``。"""
    s = p.suffix.lower()
    if s == ".gz":
        s = Path(p.stem).suffix.lower()
    return s


def _text_read(path) -> str:
    """读文本:支持 .gz,优先 UTF-8(含 BOM),退回 GBK / latin-1。"""
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
    """写文本:UTF-8(默认带 BOM,方便 Excel);``.gz`` 自动压缩。"""
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


def infer_nmax_from_rows(nrows: int) -> int:
    """由三角布局行数解出整数 ``L``:``2*(L+1)(L+2)/2 == nrows``。"""
    L = int(round((-3.0 + np.sqrt(1.0 + 4.0 * nrows)) / 2.0))
    for cand in (L, L + 1, L - 1):
        if cand >= 0 and 2 * (cand + 1) * (cand + 2) // 2 == nrows:
            return cand
    raise ValueError(
        f"三角布局的行数 {nrows} 写不成 2*(L+1)(L+2)/2;请用 nmax= 明确指定")


def _infer_nmax_from_nc(nrows: int) -> int:
    """由 ``n,m,C,S`` 表行数解出 ``L``:``(L+1)(L+2)/2 == nrows``。"""
    L = int(round((-3.0 + np.sqrt(1.0 + 8.0 * nrows)) / 2.0))
    for cand in (L, L + 1, L - 1):
        if cand >= 0 and (cand + 1) * (cand + 2) // 2 == nrows:
            return cand
    raise ValueError(
        f"行数 {nrows} 写不成 (L+1)(L+2)/2;请用 nmax= 明确指定")


def _fmt_float(v, fmt: str = "%.16g") -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "nan"
    return fmt % fv if np.isfinite(fv) else "nan"


def _split(line: str, delim: Optional[str]) -> list:
    if delim is None:
        return [t for t in re.split(r"\s+", line.strip()) if t]
    return [t.strip() for t in line.split(delim) if t.strip() != ""]


def _sniff_delimiter(text: str) -> Optional[str]:
    """从第一行可解析数据里嗅探分隔符(逗号/分号/制表符/竖线/空白)。"""
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        for d in (",", ";", "\t", "|"):
            if d in s:
                return d
        return None
    return None


# ---------------------------------------------------------------------------
# 文本表读取
# ---------------------------------------------------------------------------
def _read_text_table(path) -> tuple:
    """通用文本表读取。

    Returns
    -------
    (comment_lines, colnames, data, n_dropped)
        ``comment_lines`` 为 ``#`` / ``//`` 开头行的原文;
        ``colnames`` 为表头列名(无表头则 ``None``);
        ``data`` 为 ``(nrow, ncol)`` 的 float 数组;
        ``n_dropped`` 是**没能解析成数值、被跳过**的数据行数(分隔符不统一时
        会出现;调用方应当把它报成警告,而不是悄悄丢掉)。
    """
    text = _text_read(path)
    delim = _sniff_delimiter(text)

    comments: list = []
    header: Optional[list] = None
    rows: list = []
    dropped = 0
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
        try:
            vals = [float(t) for t in toks]
            rows.append(vals)
        except ValueError:
            if header is None and not rows:
                header = toks
            elif header is not None and not rows:
                header = header + toks          # 折行的表头
            elif rows:
                dropped += 1                    # 数据行之后出现的非数值行
            continue
    if not rows:
        raise ValueError(f"{Path(os.fspath(path))}: 没有可解析的数值行")
    ncol = max(len(r) for r in rows)
    clean = [r if len(r) == ncol else r + [np.nan] * (ncol - len(r))
             for r in rows]
    return comments, header, np.asarray(clean, dtype=float), dropped


def _header_field(comment_lines: Sequence[str], key: str):
    """从 ``# key = value`` / ``# key: value`` 注释里取一个字段。"""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*[:=]\s*(.+)$", re.IGNORECASE)
    for ln in comment_lines:
        m = pat.match(ln)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# 布局识别
# ---------------------------------------------------------------------------
def _detect_npy_layout(arr: np.ndarray, nmax: Optional[int]) -> tuple:
    """返回 ``(layout, nmax_hint)``。"""
    if arr.ndim == 3 and (arr.shape[0] == 2 or arr.shape[1] == arr.shape[2]):
        return "matrix", arr.shape[1] - 1
    if arr.ndim == 4 and (arr.shape[0] == 2 or arr.shape[1] == 2):
        return "matrix", arr.shape[1] - 1
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return "matrix", arr.shape[0] - 1
    if arr.ndim == 2:
        for cand in range(0, 4097):
            if 2 * (cand + 1) * (cand + 2) // 2 == arr.shape[0]:
                return "triangle", cand
        for cand in range(0, 4097):
            if 2 * (cand + 1) * (cand + 2) // 2 == arr.shape[1]:
                return "triangle", cand
        return "matrix", (arr.shape[1] - 1 if arr.shape[1] > 0 else None)
    if arr.ndim == 1 and arr.shape[0] % 2 == 0:
        try:
            return "triangle", infer_nmax_from_rows(arr.shape[0])
        except ValueError:
            pass
    return "matrix", None


def detect_coeff_layout(path, nmax: Optional[int] = None) -> str:
    """判断系数文件的布局(``'triangle'`` / ``'gmfcsv'`` / ``'gfc'`` /
    ``'npy'`` / ``'npz'``)。

    适合在调用 :func:`read_coeffs` 之前先看一眼启发式给出了什么。
    读 :func:`read_coeffs` 的 ``meta['warnings']`` 能看到歧义说明。
    """
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in COEFF_EXTENSIONS:
        raise ValueError(
            f"detect_coeff_layout: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}")
    return _detect_layout(p, ext, nmax)[0]


def _detect_layout(p: Path, ext: str, nmax: Optional[int]) -> tuple:
    """返回 ``(layout, nmax_hint, note)``。

    ``note`` 是需要转达给用户的歧义说明(可能为空字符串)。
    判据顺序:容器/扩展名 → 文件头标记 → (n,m) 列内容 → 方阵 → 行数。
    """
    if ext == ".npy":
        layout, hint = _detect_npy_layout(arr := np.asarray(
            np.load(p, allow_pickle=False), dtype=float), nmax)
        note = ""
        if layout == "matrix" and arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            note = (f"{arr.shape[0]}×{arr.shape[1]} 的 .npy 按稠密矩阵读取"
                    "(只有一层系数,S 置 0);若它其实是三角形布局,请用 --layout 指定")
        return layout, hint, note

    if ext == ".npz":
        with np.load(p, allow_pickle=False) as z:
            files = list(z.files)
            if "C" in files and "S" in files:
                return "npz", np.asarray(z["C"]).shape[0] - 1, ""
            if "flat_cs" in files or "triangle" in files:
                key = "flat_cs" if "flat_cs" in files else "triangle"
                arr = np.asarray(z[key], dtype=float)
                L = _tri_degree_from_rows(int(arr.shape[0]))
                if L is None:
                    L = _tri_degree_from_rows(int(arr.shape[1]))
                return "triangle", L, ""
        raise ValueError(
            f"{p}: .npz 必须含 'C'/'S'(或 'flat_cs'/'triangle'),"
            f"实际内容: {files}")

    if ext == ".gfc":
        return "gfc", None, ""

    if ext == ".nc":
        return "series_nc", None, ""

    text = _text_read(p)
    first = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        first = s
        break
    if first is None:
        raise ValueError(f"{p}: 文件没有有效数据行")

    low = first.lower()
    if low.startswith(("gfc", "gfct", "end_of_data", "end")):
        return "gfc", None, ""

    # --- legacy 多历元序列:首行是纯整数"历元编号"表头,行数 = 2*NC + 1 ---
    if ext in (".dat", ".txt", ".csv", ".tsv", ".sh") and _looks_series_dat(p):
        rows = _count_numeric_rows(p)
        L = _tri_degree_from_rows(rows - 1)
        return "series_dat", L, (f"首行是 {rows - 1} 行数据的历元表头"
                                 if L is not None else "首行疑似表头")

    comments, header, data, dropped = _read_text_table(p)
    joined = "\n".join(comments).lower()
    notes: list = []
    if dropped:
        notes.append(f"有 {dropped} 行无法解析为数值,已跳过(请核对分隔符是否统一)")

    # --- 文件头标记(比行数更可靠,优先) ---
    if "ncoef_triangle" in joined or "triangle layout" in joined \
            or "[c; s]" in joined:
        # 阶数以**行数**为准;文件头只作交叉核对(头部写错的文件真实存在)
        L = _tri_degree_from_rows(int(data.shape[0]))
        if L is None:
            L = _tri_degree_from_rows(int(data.shape[1]))
        if L is None:
            v = _header_field(comments, "nmax")
            if v is not None:
                try:
                    L = int(float(v))
                except ValueError:
                    L = None
        return "triangle", L, "; ".join(notes)
    if header is not None:
        hn = [str(h).strip().lower() for h in header[:4]]
        if len(hn) >= 4 and hn[0] in ("n", "degree", "l") and \
                hn[1] in ("m", "order") and hn[2] in ("c", "clm", "cos") \
                and hn[3] in ("s", "slm", "sin"):
            return "gmfcsv", None, "; ".join(notes)
    if "layout n,m,c,s" in joined:
        return "gmfcsv", None, "; ".join(notes)

    # --- 行数/列数判据 ---
    nrow, ncol = int(data.shape[0]), int(data.shape[1])
    if nrow == 0 or ncol == 0:
        raise ValueError(f"{p}: 空文件")
    tri_L = _tri_degree_from_rows(nrow)
    nc_L = _infer_nmax_from_nc_opt(nrow)

    # 判据 A:**前两列是合法的 (n, m) 整数对** → 逐行 n,m,C,S 表。
    # 这一条很关键:例如「6 行 × 4 列」既可读成 triangle(L=1, ntime=4)也可读成
    # gmfcsv(L=2),只看行数会判错,而 (n,m) 列的内容能一锤定音。
    if ncol >= 4 and _looks_like_nm_table(data, nc_L):
        note = "; ".join(notes)
        if tri_L is not None:
            note = (note + "; " if note else "") + (
                f"行数同时符合三角形布局(L={tri_L})与 n,m,C,S 表(L={nc_L});"
                "已按前两列是 (n,m) 判为 gmfcsv,如需另一种请用 --layout 指定")
        return "gmfcsv", None, note

    # 判据 B:**方阵** → 稠密矩阵文本(MATLAB dlmwrite 风格)
    if nrow == ncol and ncol >= 2:
        note = "; ".join(notes)
        if tri_L is not None or nc_L is not None:
            note = (note + "; " if note else "") + (
                f"{nrow}×{ncol} 方阵:按稠密矩阵(nmax={nrow - 1})读取;"
                "若它其实是三角形或 n,m,C,S 表,请用 --layout 指定")
        return "matrix_text", nrow - 1, note

    if tri_L is not None and nc_L is not None:
        # 极罕见:两种都说得通,且看不出 (n,m) 结构。三角布局是 SHKit 的默认
        # 输出,优先它,但必须**明确告诉用户存在歧义**。
        return "triangle", tri_L, (
            "; ".join(notes) + "; " if notes else "") + (
            f"行数 {nrow} 同时符合三角形布局(L={tri_L})与 n,m,C,S 表(L={nc_L});"
            "已按三角形读取,如不对请用 --layout 指定")
    if tri_L is not None:
        return "triangle", tri_L, "; ".join(notes)
    if nc_L is not None:
        return "gmfcsv", None, "; ".join(notes)
    if nmax is not None and nrow == nmax + 1:
        return "matrix_text", nmax, "; ".join(notes)
    raise ValueError(
        f"{p}: 无法判断系数布局({nrow} 行 × {ncol} 列);"
        "请用 --layout 明确指定 triangle / gmfcsv / matrix")


def _tri_degree_from_rows(nrows: int) -> Optional[int]:
    """``2*(L+1)(L+2)/2 == nrows`` 的整数解;无解返回 ``None``。"""
    try:
        return infer_nmax_from_rows(nrows)
    except ValueError:
        return None


def _infer_nmax_from_nc_opt(nrows: int) -> Optional[int]:
    """``(L+1)(L+2)/2 == nrows`` 的整数解;无解返回 ``None``。"""
    try:
        return _infer_nmax_from_nc(nrows)
    except ValueError:
        return None


def _looks_like_nm_table(data: np.ndarray, nc_L: Optional[int]) -> bool:
    """前两列是不是一组合法的 ``(n, m)``:非负整数、``m <= n``、
    每一阶 ``n`` 恰好出现 ``n+1`` 次(即覆盖 0..n 的全部 m)。

    ``L`` 由行数反推(``(L+1)(L+2)/2 == nrow``)。行数不是三角数时这一条自动
    不成立,所以不会误伤三角形布局。
    """
    if nc_L is None or data.shape[1] < 4:
        return False
    n_col, m_col = data[:, 0], data[:, 1]
    if not np.all(np.isfinite(n_col)) or not np.all(np.isfinite(m_col)):
        return False
    if np.any(np.abs(n_col - np.round(n_col)) > 1e-9) or \
            np.any(np.abs(m_col - np.round(m_col)) > 1e-9):
        return False
    n_i = np.round(n_col).astype(int)
    m_i = np.round(m_col).astype(int)
    if np.any(n_i < 0) or np.any(m_i < 0) or np.any(m_i > n_i):
        return False
    if n_i.max() != nc_L:
        return False
    # 每一阶 n 必须恰好出现 n+1 行(0..n 每个 m 一行)——这才是"逐行 n,m,C,S"的
    # 充要结构。以前这里写的是 sum(n+1) == nrow,那是错的。
    counts = np.bincount(n_i, minlength=nc_L + 1)
    return np.array_equal(counts, np.arange(nc_L + 1) + 1)


# ---------------------------------------------------------------------------
# 各布局读取
# ---------------------------------------------------------------------------
_TRI_HEADER_KEYS = ("nmax", "ntime", "field_unit", "forward_from",
                    "inverse_target", "output_unit", "weight_rule", "method",
                    "weight_sum", "coverage", "fit_rmse_rel", "created",
                    "gaussian_radius_km", "gaussian_W0", "gaussian_Wnmax",
                    "n_points")


def _read_triangle_text(p: Path, nmax: Optional[int]):
    """读三角 ``[C; S]`` 文本表。

    **阶数以行数为准,永远不做"前缀切行"。** 三角排序是 m 外层 / n 内层、每段
    长度不等,取前 ``2*NC(L)`` 行并不是低阶子三角 —— 那会把 m≥1 的系数整体搬错
    位置(实测最大错 2.7),而且看起来一切正常。按请求降阶由
    :func:`read_coeffs` 在读出完整对象之后用 :meth:`SHCoeffs.truncate` 完成。

    Returns
    -------
    (data, L_file, comments, warnings)
        ``data`` 是**完整**的 ``(2*NC, ntime)`` 数组。
    """
    comments, header, data, dropped = _read_text_table(p)
    ncol = int(data.shape[1])
    nrow = int(data.shape[0])

    warnings: list = []
    if dropped:
        warnings.append(f"有 {dropped} 行无法解析为数值,已跳过"
                        "(请核对分隔符是否统一)")

    L = _tri_degree_from_rows(nrow)
    # 列数 > 行数:可能是 (ntime, 2*NC) 的行优先写法
    if L is None and ncol > nrow:
        L_col = _tri_degree_from_rows(ncol)
        if L_col is not None:
            data = data.T
            warnings.append("系数表是 (ntime, 2*NC) 布局,已转置成 (2*NC, ntime)")
            nrow, ncol = int(data.shape[0]), int(data.shape[1])
            L = L_col
    if L is None:
        raise ValueError(
            f"{p}: 三角表有 {nrow} 行,写不成 2*(L+1)(L+2)/2;"
            "请确认这是三角布局,或用 --layout 指定其它布局")

    # 文件头的 nmax 只作交叉核对:行数才是事实(头部写错的文件真实存在)
    v = _header_field(comments, "nmax")
    if v is not None:
        try:
            L_hdr = int(float(v))
        except ValueError:
            L_hdr = None
        if L_hdr is not None and L_hdr != L:
            warnings.append(
                f"文件头写 nmax={L_hdr},但 {nrow} 行对应 {L} 阶;"
                f"**以行数为准**(按 {L} 阶读取)。若文件确实只有 {L_hdr} 阶,"
                f"请用 nmax={L_hdr} 或 --layout 重新指定")
    return data, L, comments, warnings


def _apply_triangle_header(meta: dict, comments: Sequence[str]) -> None:
    """把 SHKit 三角文件头里的元数据推回 ``meta``(物理量标签等必须回来,
    否则单位换算会静默退化成 ``unknown`` 而开始报错)。"""
    hdr = {}
    for key in _TRI_HEADER_KEYS:
        v = _header_field(comments, key)
        if v is not None:
            hdr[key] = v
    if hdr:
        meta["header_fields"] = hdr
    if hdr.get("field_unit"):
        meta["field_unit"] = str(hdr["field_unit"])
    for key in ("output_unit", "inverse_target", "forward_from", "method",
                "weight_rule"):
        if hdr.get(key):
            meta.setdefault(key, str(hdr[key]))
    if hdr.get("gaussian_radius_km") is not None:
        try:
            meta["gaussian_radius_km"] = float(hdr["gaussian_radius_km"])
        except (TypeError, ValueError):
            pass
    for key, cast in (("coverage", float), ("n_points", int)):
        if hdr.get(key) is not None:
            try:
                meta[key] = cast(hdr[key])
            except (TypeError, ValueError):
                pass
    if comments:
        meta["comment_header"] = list(comments)


def _read_gfc(p: Path, nmax: Optional[int]) -> tuple:
    """解析 ICGEM/GFZ ``.gfc``;返回 ``(C, S, meta, warnings)``。"""
    text = _text_read(p)
    warnings: list = []
    meta: dict = {}
    triples = []
    max_deg = 0
    n_sigma = 0
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        toks = s.split()
        low = toks[0].lower()
        if low in ("begin", "end", "begin_model", "end_model", "begin_of_head",
                   "end_of_head", "begin_of_data", "end_of_data"):
            continue
        if low in ("gfc", "gfct"):
            if len(toks) < 5:
                warnings.append(f"忽略字段不足的 gfc 行: {s[:60]!r}")
                continue
            try:
                n_ = int(toks[1])
                m_ = int(toks[2])
                c_ = float(toks[3])
                s_ = float(toks[4])
            except ValueError:
                warnings.append(f"忽略无法解析的 gfc 行: {s[:60]!r}")
                continue
            sig = None
            if len(toks) >= 7:
                try:
                    sig = (float(toks[5]), float(toks[6]))
                    n_sigma += 1
                except ValueError:
                    sig = None
            triples.append((n_, m_, c_, s_, sig))
            max_deg = max(max_deg, n_)
            continue
        if len(toks) >= 2:
            key = toks[0].rstrip(":").lower()
            if re.fullmatch(r"[a-z_]{3,}", key):
                meta.setdefault(key, " ".join(toks[1:]))
    if not triples:
        raise ValueError(
            f"{p}: 没有解析到任何 'gfc n m C S' 数据行;"
            "确认这是 ICGEM/GFZ gfc 格式(首列为 gfc 或 gfct)")

    if nmax is None:
        if meta.get("max_degree"):
            try:
                nmax = int(float(str(meta["max_degree"]).split()[0]))
            except ValueError:
                nmax = max_deg
        else:
            nmax = max_deg
    if max_deg > nmax:
        warnings.append(
            f"文件含到 {max_deg} 阶的系数,但 nmax={nmax};更高阶被丢弃。"
            f"传 nmax={max_deg} 可全部读入")

    C = np.zeros((nmax + 1, nmax + 1))
    S = np.zeros((nmax + 1, nmax + 1))
    sigC = np.zeros((nmax + 1, nmax + 1)) if n_sigma else None
    sigS = np.zeros((nmax + 1, nmax + 1)) if n_sigma else None
    used = 0
    seen: dict = {}
    for idx, (n_, m_, c_, s_, sig) in enumerate(triples, start=1):
        if n_ > nmax or m_ > nmax:
            continue
        if m_ > n_:
            warnings.append(f"忽略不合法的 (n={n_}, m={m_})(m > n)")
            continue
        key = (n_, m_)
        if key in seen:
            # 重复的 (n,m):后者覆盖前者。被拼接过的文件里真的会出现,
            # 必须说出来 —— 否则你看到的系数到底来自哪一行就无从知道。
            warnings.append(
                f"文件里 (n={n_}, m={m_}) 出现了多次:已取后出现的第 {idx} 个"
                f"数据行(C={c_:.6g}),第 {seen[key]} 个数据行被覆盖")
        seen[key] = idx
        C[n_, m_] = c_
        S[n_, m_] = s_
        if sig is not None and sigC is not None:
            sigC[n_, m_], sigS[n_, m_] = sig
        used += 1

    out_meta: dict[str, Any] = {
        "format": "gfc",
        "gfc_header": dict(meta),
        "gfc_lines_used": int(used),
        "max_degree_in_file": int(max_deg),
        "has_sigmas": bool(n_sigma),
    }
    for k in ("modelname", "earth_gravity_constant", "radius", "norm",
              "tide_system", "product_type", "value_type", "institution",
              "field_unit", "gaussian_km", "shkit_layout"):
        if k in meta:
            out_meta[k] = meta[k]
    if sigC is not None:
        out_meta["sigmaC"] = sigC
        out_meta["sigmaS"] = sigS
    # 只承认 4π(完全)归一化。SHKit 的清单里把 'unnormalized' 也算通过,那其实是
    # 错的:未归一化和 4π 归一化差着 sqrt(2n+1)·... 的因子,必须提醒。
    norm = str(meta.get("norm", "4pi")).lower().replace("-", "").replace(" ", "")
    if norm not in ("4pi", "fully_normalized", "fully_normalised",
                    "fullynormalized", "4pi_normalized"):
        warnings.append(
            f"文件声明的归一化是 {meta.get('norm')!r};本软件只按 4π 归一化"
            "(SHTOOLS norm=1, csphase=1)解释,使用前请核对")
    return C, S, out_meta, warnings


def _split_matrix_array(arr: np.ndarray, nmax: Optional[int]) -> tuple:
    """``(2,L+1,L+1[,ntime])`` 等 → ``(C, S, warnings)``。"""
    warnings: list = []
    if arr.ndim == 4 and arr.shape[0] == 2:
        C, S = arr[0], arr[1]
    elif arr.ndim == 4 and arr.shape[1] == 2:
        C, S = arr[:, 0], arr[:, 1]
        warnings.append("按 (L+1, L+1, 2, ntime) 解释")
    elif arr.ndim == 3 and arr.shape[0] == 2:
        C, S = arr[0], arr[1]
    elif arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
        C, S = arr, np.zeros_like(arr)
        warnings.append(
            "(L+1, L+1, ntime) 布局:只有一层系数,已按 C 读取、S 置 0;"
            "若该数组其实是 S 请改用 .npz(含 C/S 键)")
    elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        C, S = arr, np.zeros_like(arr)
        warnings.append("(L+1, L+1) 稠密矩阵:按 C 读取、S 置 0")
    else:
        raise ValueError(f"稠密数组 shape {arr.shape} 无法解释为系数")
    if C.shape != S.shape or C.shape[0] != C.shape[1]:
        raise ValueError(f"C{C.shape} 与 S{S.shape} 必须是同形方阵")
    if nmax is not None and C.shape[0] != nmax + 1:
        warnings.append(f"文件里是 nmax={C.shape[0] - 1},与 nmax={nmax} 不一致;"
                        "以文件为准(需要时请调用 truncate())")
    if C.ndim == 3 and C.shape[2] == 1:
        C, S = C[:, :, 0], S[:, :, 0]
    return C, S, warnings


def _apply_npz_meta(z, files, meta: dict, warnings: list) -> None:
    """把 ``.npz`` 里的 ``meta``(JSON 字符串)并进 ``meta``。

    不是 JSON、或 JSON 不是"字典",都**如实报警告**而不是悄悄丢掉 ——
    里面可能就装着 ``field_unit`` 标签,丢掉它会让单位换算静默退化。
    """
    if "meta" not in files:
        return
    try:
        raw = str(z["meta"].item()) if z["meta"].ndim == 0 else str(z["meta"])
    except Exception:                                    # pragma: no cover
        warnings.append(".npz 里的 meta 读不出来,已忽略")
        return
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        warnings.append(f".npz 里的 meta 不是合法 JSON({exc}),已忽略")
        return
    if not isinstance(parsed, dict):
        warnings.append(
            f".npz 里的 meta 是 {type(parsed).__name__} 而不是字典,已忽略"
            "(里面的 field_unit 等标签都不会生效)")
        return
    meta.update(parsed)
    # 顺手认一下 "meta 内容是 [[k, v], …]" 这种写法(有些工具会这么写)
    for k, v in list(meta.items()):
        if isinstance(v, list) and v and all(
                isinstance(pair, (list, tuple)) and len(pair) == 2
                for pair in v):
            try:
                meta.update({str(a): b for a, b in v})
                meta.pop(k, None)
            except Exception:                            # pragma: no cover
                pass


def _read_gmfcsv(p: Path, nmax: Optional[int]) -> tuple:
    """读 ``n,m,C,S`` 逐行表。"""
    comments, header, data, dropped = _read_text_table(p)
    warnings: list = []
    if dropped:
        warnings.append(f"有 {dropped} 行无法解析为数值,已跳过"
                        "(请核对分隔符是否统一)")
    cols = [str(h).strip() for h in header] if header else \
        [f"col{i + 1}" for i in range(data.shape[1])]

    def find(names):
        for i, c in enumerate(cols):
            if c.strip().lower() in names:
                return i
        return None

    i_n = find(("n", "degree", "l"))
    i_m = find(("m", "order"))
    i_c = find(("c", "clm", "cos", "c_nm"))
    i_s = find(("s", "slm", "sin", "s_nm"))
    if None in (i_n, i_m, i_c, i_s):
        if data.shape[1] >= 4:
            i_n, i_m, i_c, i_s = 0, 1, 2, 3
            warnings.append("未识别到 n,m,C,S 表头,按前 4 列解释")
        else:
            raise ValueError(
                f"{p}: 该布局需要 n,m,C,S 四列,文件只有 {data.shape[1]} 列: {cols}")

    nn = data[:, i_n]
    mm = data[:, i_m]
    if not np.all(np.isfinite(nn)) or not np.all(np.isfinite(mm)):
        raise ValueError(f"{p}: n/m 列含缺失值")

    extra = []
    low = [c.strip().lower() for c in cols]
    t = 1
    while f"c_t{t}" in low and f"s_t{t}" in low:
        extra.append((low.index(f"c_t{t}"), low.index(f"s_t{t}")))
        t += 1
    ntime = 1 + len(extra)
    cval = [data[:, i_c]] + [data[:, ic] for ic, _ in extra]
    sval = [data[:, i_s]] + [data[:, isx] for _, isx in extra]

    L_file = int(np.nanmax(nn))
    if nmax is None:
        nmax = L_file
    elif L_file > nmax:
        warnings.append(f"文件含到 {L_file} 阶的系数,但 nmax={nmax};更高阶被丢弃。"
                        f"传 nmax={L_file} 可全部读入")

    shape = (nmax + 1, nmax + 1) if ntime == 1 else \
        (nmax + 1, nmax + 1, ntime)
    C = np.zeros(shape)
    S = np.zeros(shape)
    if ntime > 1:
        warnings.append(f"表里含 {ntime} 个时次(C,S 及其后的 C_t1/S_t1 ...)")
    keep = (nn <= nmax) & (mm <= nmax)
    for row in np.nonzero(keep)[0]:
        i, j = int(round(nn[row])), int(round(mm[row]))
        if j > i:
            warnings.append(f"忽略不合法的 (n={i}, m={j})(m > n)")
            continue
        for k in range(ntime):
            if ntime == 1:
                C[i, j] = cval[0][row]
                S[i, j] = sval[0][row]
            else:
                C[i, j, k] = cval[k][row]
                S[i, j, k] = sval[k][row]
    if comments:
        pass
    meta = {"format": "gmfcsv", "max_degree_in_file": L_file,
            "comment_header": list(comments) if comments else []}
    return C, S, meta, warnings


# ---------------------------------------------------------------------------
# 主入口:读
# ---------------------------------------------------------------------------
def read_coeffs(path, nmax: Optional[int] = None, layout: str = "auto",
                time: Optional[int] = None) -> SHCoeffs:
    """读取**任意 SHKit 输出格式**的球谐系数。

    Parameters
    ----------
    path : str | Path
        支持 ``.sh .txt .csv .dat .tsv .gfc .npy .npz``(文本与 gfc 允许 ``.gz``)。
    nmax : int, optional
        最高阶。文件自证不了阶数时(例如稠密 ``.npy``)必须给。
        **比文件阶数小的时候是"截断",等于先完整读入再
        :meth:`SHCoeffs.truncate`** —— 对任何布局都一样,不会出现"切行导致 (n,m)
        错位"。比文件阶数大时为零填充。
    layout : str
        ``'auto'``(默认,见 :func:`detect_coeff_layout`);也可强制
        ``'triangle'`` / ``'gmfcsv'`` / ``'matrix'`` / ``'npy'`` / ``'npz'`` / ``'gfc'``。
    time : int, optional
        ``.gfc`` 只有单时次,这里的参数保留给未来多时次容器使用。

    Returns
    -------
    SHCoeffs
        ``meta`` 里一定有 ``source_file``、``layout``、``nmax``、``ntime``、
        ``warnings``;文件头里的 ``field_unit`` / 高斯半径 / 覆盖率会被推上来。
        布局判定有歧义时,``warnings`` 里会写清楚选了哪一种、怎么改。

    Raises
    ------
    FileNotFoundError, ValueError
        文件缺失 / 扩展名或布局不认识 / 内容无法解析。
    """
    out = _read_coeffs_impl(path, nmax=nmax, layout=layout, time=time)
    # 统一在这里做"按请求降阶":不管是哪种布局、有没有被自动推断填过 nmax,
    # 只要调用方给的值**小于**文件阶数,就走 truncate(数学上正确的子三角),
    # 而不是任何形式的前缀切行。
    if nmax is not None and int(nmax) < out.nmax:
        w = list(out.meta.get("warnings", []) or [])
        w.append(f"按请求截断到 nmax={nmax}(文件是 {out.nmax} 阶);"
                 "已先完整读入再截断,不会发生 (n,m) 错位")
        out = out.truncate(int(nmax))
        out.meta["warnings"] = w
    return out


def _read_coeffs_impl(path, nmax: Optional[int] = None, layout: str = "auto",
                      time: Optional[int] = None) -> SHCoeffs:
    """实际的分支实现;``read_coeffs`` 在其上补统一的截断与警告处理。"""
    p = _require_file(path)
    ext = _suffix(p)
    if ext not in COEFF_EXTENSIONS:
        raise ValueError(
            f"read_coeffs: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}")
    if layout not in COEFF_LAYOUTS:
        raise ValueError(
            f"layout 必须是 {'|'.join(COEFF_LAYOUTS)},实际 {layout!r}")

    warnings: list = []
    if layout == "auto":
        layout, hint, note = _detect_layout(p, ext, nmax)
        if note:
            warnings.append(note)
        if hint is not None and nmax is None:
            nmax = hint
    if layout == "npz" and ext != ".npz":
        layout = "matrix"
    if layout == "npy" and ext not in (".npy", ".npz"):
        layout = None                      # 交给下面的扩展名分支

    meta: dict = {"source_file": str(p), "layout": layout}

    # ------------------------------------------------- series_nc / series_dat
    if layout == "series_nc" or (ext == ".nc" and layout not in ("npy", "npz")):
        out = read_series_nc(p, nmax=nmax)
        out.meta["layout"] = "series_nc"
        return out
    if layout == "series_dat":
        out = read_series_dat(p, nmax=nmax)
        out.meta["layout"] = "series_dat"
        return out

    # ------------------------------------------------------------- gfc
    if layout == "gfc" or ext == ".gfc":
        C, S, gmeta, w = _read_gfc(p, nmax)
        warnings.extend(w)
        meta.update(gmeta)
        meta["layout"] = "gfc"
        # .gfc 是 ICGEM/GFZ 重力模型:它的 C_nm/S_nm 是无量纲位系数,
        # 不是水高。贴上标签后 synthesis(target_unit='ewh') 才会自动乘 Aₙ,
        # 而不是要求用户自己声明。
        fu = meta.get("field_unit")
        meta["field_unit"] = fu if fu else "geopotential"
        out = SHCoeffs(C, S, meta)
        meta.update({"nmax": out.nmax, "ntime": out.ntime,
                     "warnings": warnings})
        out.meta = meta
        # v2.0:一个 .gfc 就是一个历元 —— 顺手把它的日期挂上(头优先、文件名兜底)
        try:
            from .timeaxis import TimeAxis
            out.times = TimeAxis.from_gfc_headers([meta.get("gfc_header", {})],
                                                  paths=[p])
        except Exception as exc:                             # noqa: BLE001
            warnings.append(f"时间轴构造失败,已按无日期处理: {exc}")
        return out

    # ------------------------------------------- npz: 含 C/S 的矩阵布局
    # 注意:只处理"C/S 矩阵"这一种 npz;含 flat_cs/triangle 的 npz 会被识别成
    # triangle 布局,交给下面的三角分支(那里认识容器文件)。
    if ext == ".npz" and layout in ("matrix", "npz"):
        with np.load(p, allow_pickle=False) as z:
            files = list(z.files)
            if "C" not in files or "S" not in files:
                raise ValueError(
                    f"{p}: 矩阵布局的 .npz 需要 'C' 与 'S' 数组,"
                    f"实际内容: {files}。若装的是 flat_cs/triangle,"
                    "请把 layout 设为 'triangle' 或 'auto'")
            C = np.asarray(z["C"], dtype=float)
            S = np.asarray(z["S"], dtype=float)
            _apply_npz_meta(z, files, meta, warnings)
            if "nmax" in files:
                meta.setdefault("nmax", int(np.asarray(z["nmax"]).ravel()[0]))
        if C.ndim == 3 and C.shape[2] == 1:
            C, S = C[:, :, 0], S[:, :, 0]
        if nmax is not None and C.shape[0] != nmax + 1:
            warnings.append(
                f"文件里是 nmax={C.shape[0] - 1},与请求的 nmax={nmax} 不一致;"
                + ("按请求截断" if nmax < C.shape[0] - 1 else "以文件为准"))
        out = SHCoeffs(C, S, meta)
        meta.update({"layout": "npz", "nmax": out.nmax,
                     "ntime": out.ntime, "warnings": warnings})
        out.meta = meta
        return out

    # ------------------------------------------------------------ npy
    # triangle 布局的 .npy 由下面的三角分支处理;这里只处理稠密矩阵。
    if ext == ".npy" and layout != "triangle":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
        C, S, w = _split_matrix_array(arr, nmax)
        warnings.extend(w)
        out = SHCoeffs(C, S, meta)
        meta.update({"layout": "matrix", "nmax": out.nmax,
                     "ntime": out.ntime, "warnings": warnings})
        out.meta = meta
        return out

    # -------------------------------------------------------- 文本 —— 矩阵
    if layout in ("matrix", "matrix_text"):
        _comments, _header, data, dropped = _read_text_table(p)
        if dropped:
            warnings.append(f"有 {dropped} 行无法解析为数值,已跳过"
                            "(请核对分隔符是否统一)")
        arr = np.asarray(data, dtype=float)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"{p}: 稠密矩阵布局要求方阵,实际 {arr.shape[0]} 行 × "
                f"{arr.shape[1]} 列")
        C, S, w = _split_matrix_array(arr, nmax)
        warnings.extend(w)
        out = SHCoeffs(C, S, meta)
        meta.update({"layout": "matrix", "nmax": out.nmax,
                     "ntime": out.ntime, "warnings": warnings})
        out.meta = meta
        return out

    # -------------------------------------------------------- 文本 —— gmfcsv
    if layout == "gmfcsv":
        C, S, gmeta, w = _read_gmfcsv(p, nmax)
        warnings.extend(w)
        meta.update(gmeta)
        out = SHCoeffs(C, S, meta)
        meta.update({"layout": "gmfcsv", "nmax": out.nmax,
                     "ntime": out.ntime, "warnings": warnings})
        out.meta = meta
        return out

    # ------------------------------------------------------- triangle 布局
    # 三处来源都可能:文本(.sh/.txt/.csv/.dat/.tsv[.gz])、.npy、.npz(flat_cs /
    # triangle)。容器里的三角数组没有文件头,所以只认数值。
    if layout == "triangle":
        comments: list = []
        L_hint = nmax
        if ext == ".npz":
            with np.load(p, allow_pickle=False) as z:
                files = list(z.files)
                key = "flat_cs" if "flat_cs" in files else (
                    "triangle" if "triangle" in files else None)
                if key is None:
                    raise ValueError(
                        f"{p}: 三角布局的 .npz 需要 'flat_cs' 或 'triangle' 数组,"
                        f"实际内容: {files}")
                arr = np.asarray(z[key], dtype=float)
                _apply_npz_meta(z, files, meta, warnings)
            out, tmeta = _read_triangle_array(arr, L_hint, warnings)
            meta.update(tmeta)
            meta.update({"layout": "triangle", "nmax": out.nmax,
                         "ntime": out.ntime, "warnings": warnings})
            out.meta = meta
            return out
        if ext == ".npy":
            arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
            out, tmeta = _read_triangle_array(arr, L_hint, warnings)
            meta.update(tmeta)
            meta.update({"layout": "triangle", "nmax": out.nmax,
                         "ntime": out.ntime, "warnings": warnings})
            out.meta = meta
            return out

        arr, L, comments, w = _read_triangle_text(p, nmax)
        warnings.extend(w)
        if arr.ndim == 1:
            arr = arr[:, None]
        coeffs = SHCoeffs.from_triangle(arr, int(L), meta)
        _apply_triangle_header(meta, comments)
        meta.update({"layout": "triangle", "nmax": coeffs.nmax,
                     "ntime": coeffs.ntime, "warnings": warnings})
        coeffs.meta = meta
        return coeffs

    raise ValueError(f"{p}: 无法用 layout={layout!r} 读取")


def _read_triangle_array(arr: np.ndarray, L: Optional[int], warnings: list):
    """由三角数组构造 ``(SHCoeffs, meta)``。

    同样**不做前缀切行**:阶数以行数(或列数)为准,按请求降阶交给
    :meth:`SHCoeffs.truncate`。``L`` 参数只作为行数无解时的兜底。
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim == 2 and arr.shape[1] > 1 and arr.shape[0] != arr.shape[1]:
        need_row = _tri_degree_from_rows(int(arr.shape[0]))
        need_col = _tri_degree_from_rows(int(arr.shape[1]))
        if need_row is None and need_col is not None:
            arr = arr.T
            warnings.append("三角数组是 (ntime, 2*NC),已转置")
    L_rows = _tri_degree_from_rows(int(arr.shape[0]))
    if L_rows is None:
        if L is None:
            raise ValueError(
                f"三角数组有 {arr.shape[0]} 行,写不成 2*(L+1)(L+2)/2;"
                "请用 nmax= 明确指定阶数")
        need = 2 * (L + 1) * (L + 2) // 2
        if arr.shape[0] != need:
            raise ValueError(f"三角数组需要 {need} 行,实际 {arr.shape[0]} 行")
        L_rows = L
    elif L is not None and int(L) != L_rows:
        warnings.append(f"数组行数对应 {L_rows} 阶,与给定的 nmax={L} 不一致;"
                        "以行数为准(按请求降阶请用 truncate)")
    coeffs = SHCoeffs.from_triangle(arr, int(L_rows), {})
    return coeffs, {"layout": "triangle", "nmax": coeffs.nmax,
                    "ntime": coeffs.ntime}


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def write_coeffs(coeffs: SHCoeffs, path, layout: str = "auto",
                 comment: Optional[str] = None, fmt: str = "%.17g",
                 header: bool = True, with_sigma: bool = False,
                 meta_extra: Optional[Mapping[str, Any]] = None,
                 time: Optional[int] = None) -> Path:
    """把系数写成 SHKit 的任意一种输出布局(保证 SHKit 也能读回)。

    Parameters
    ----------
    coeffs : SHCoeffs
        要写的对象;``ntime = 1`` 就写成一列。
    path : str | Path
        ``.sh .txt .csv .dat .tsv``(文本)、``.gfc``(ICGEM)、``.npy``、``.npz``;
        文本与 gfc 支持 ``.gz``。
    layout : {'auto','triangle','gmfcsv','gfc','npy','npz'}
        ``'auto'`` 按扩展名决定。``'triangle'`` 是 SHKit 的默认布局,也是能交回
        旧软件(``m2py`` / ``gridSHconvert``)的那一种。
    comment : str, optional
        写进 ``#`` 头(仅文本布局)的自由文本。
    fmt : str
        数字格式,默认 ``%.17g`` —— 17 位有效数字才能让**任意** double 十进制
        往返无损(实测 ``%.16g`` 在 20 万个随机 double 里有约 43% 会差 1 ulp,
        虽然通常无害,但"可精确往返"这个说法只有 ``%.17g`` 才成立)。
    header : bool
        是否写 ``#`` 头(仅文本布局)。
    with_sigma : bool
        ``'gfc'`` 专用:有 ``meta['sigmaC']/['sigmaS']`` 时写出形式上误差列
        (``gfc n m C S sigmaC sigmaS`` 七列)。此时头部才写
        ``errors formal``;否则写 ``errors no`` —— ICGEM 规定 ``formal``
        必须带那两列中误差,头部与列数不一致会让合规读者解析失败。
    meta_extra : mapping, optional
        额外并入头/``meta`` 的键值。
    time : int, optional
        ``'gfc'`` 专用:一个 .gfc 文件只装一个时次,多时次必须选一个,
        否则报错而不是悄悄拼成读不回来的文件。

    Returns
    -------
    pathlib.Path
        实际写出的路径。
    """
    if not isinstance(coeffs, SHCoeffs):
        raise TypeError(f"coeffs 必须是 SHCoeffs,实际 {type(coeffs).__name__}")
    p = Path(os.fspath(path))
    ext = _suffix(p)
    if layout not in ("auto", "triangle", "gmfcsv", "gfc", "npy", "npz",
                      "series_nc", "series_dat"):
        raise ValueError(
            "layout 必须是 'auto'|'triangle'|'gmfcsv'|'gfc'|'npy'|'npz'|"
            f"'series_nc'|'series_dat',实际 {layout!r}")
    if layout == "auto":
        # 注意:_suffix() 返回的是带点的扩展名(如 '.npy')
        layout = {".npy": "npy", ".npz": "npz", ".gfc": "gfc",
                  ".nc": "series_nc"}.get(ext, "triangle")
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)

    # v2.0:多时间序列布局
    if layout == "series_nc" or ext == ".nc":
        return write_series_nc(coeffs, p, comment=comment,
                               meta_extra=meta_extra)
    if layout == "series_dat":
        return write_series_dat(coeffs, p)

    L, ntime = coeffs.nmax, coeffs.ntime
    m_vec, n_vec = triangle_order(L)
    C3 = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S3 = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    meta = dict(coeffs.meta)
    meta.update(dict(meta_extra or {}))

    # ------------------------------------------------------------- npz / npy
    if layout in ("npz", "npy") or ext in (".npz", ".npy"):
        if ext == ".npy" or layout == "npy":
            if str(p).lower().endswith(".gz"):
                raise ValueError("write_coeffs: .npy 不支持 .gz")
            np.save(p, np.stack([coeffs.C, coeffs.S]))
            return p
        np.savez(p,
                 C=np.asarray(coeffs.C, dtype=float),
                 S=np.asarray(coeffs.S, dtype=float),
                 meta=json.dumps(_jsonable_meta(meta), ensure_ascii=False),
                 nmax=L, ntime=ntime, layout="matrix",
                 writer="SHSynth")
        return p

    # ----------------------------------------------------------------- gfc
    if ext == ".gfc" or layout == "gfc":
        sigC = meta.get("sigmaC")
        sigS = meta.get("sigmaS")
        # ICGEM 规定 errors=formal 时每个 gfc 行必须带 sigmaC sigmaS 两列。
        # 所以"头里写 formal"与"行里真写 7 列"必须同时成立,否则合规读者会解析
        # 失败 —— 两者取同一个判据。
        write_sigmas = bool(with_sigma and sigC is not None and sigS is not None)
        out = ["# SHSynth %s  ICGEM/GFZ formatted spherical harmonic model"
               % _pkg_version(),
               "# Written by shsynth.coeffio.write_coeffs(layout='gfc')"]
        if comment:
            out.extend("# " + ln for ln in comment.splitlines())
        out.append(f"modelname            {meta.get('modelname') or p.stem}")
        out.append("earth_gravity_constant "
                   f"{meta.get('earth_gravity_constant', '0.0')}")
        out.append(f"radius                {meta.get('radius', '6378137.0')}")
        out.append(f"max_degree            {L}")
        out.append(f"errors                {'formal' if write_sigmas else 'no'}")
        out.append("norm                  4pi")
        out.append(f"tide_system           {meta.get('tide_system', 'unknown')}")
        out.append("format                icgem")
        out.append("shkit_layout          triangle-source")
        for k, v in _jsonable_meta(meta).items():
            if k in _GFC_HEADER_KEYS or k in ("sigmaC", "sigmaS"):
                continue
            try:
                out.append(f"{k:<22}{v if isinstance(v, (int, float)) else str(v)}")
            except Exception:
                pass
        out.append("end_of_head")
        if ntime > 1 and time is None:
            raise ValueError(
                f"write_coeffs(layout='gfc'):一个 ICGEM/GFZ .gfc 文件只装一个"
                f"时次,但该对象 ntime={ntime}。整体写入会得到一个读回来就坏掉的"
                "文件。可选:(a) 传 time=<k> 只导出一个时次;(b) 用 "
                "layout='triangle' 或 'npz'(保留全部时次);(c) 循环每个时次"
                "各写一个 .gfc。")
        t_sel = 0 if time is None else int(time)
        if not 0 <= t_sel < ntime:
            raise ValueError(f"time={time} 超出 ntime={ntime} 的范围")
        for k in range(len(m_vec)):
            n_, m_ = int(n_vec[k]), int(m_vec[k])
            c_ = float(C3[n_, m_, t_sel])
            s_ = float(S3[n_, m_, t_sel])
            if write_sigmas:
                arr_c = np.asarray(sigC)
                arr_s = np.asarray(sigS)
                sc = float(arr_c[n_, m_]) if arr_c.ndim >= 2 else 0.0
                ss = float(arr_s[n_, m_]) if arr_s.ndim >= 2 else 0.0
                out.append("gfc %3d %3d %s %s %s %s" % (
                    n_, m_, _fmt_float(c_, fmt), _fmt_float(s_, fmt),
                    _fmt_float(sc, fmt), _fmt_float(ss, fmt)))
            else:
                out.append("gfc %3d %3d %s %s" % (
                    n_, m_, _fmt_float(c_, fmt), _fmt_float(s_, fmt)))
        out.append("end_of_data")
        _text_write(p, "\n".join(out) + "\n")
        return p

    if ext not in (".sh", ".txt", ".csv", ".dat", ".tsv") and not \
            str(p).lower().endswith(".gz"):
        raise ValueError(
            f"write_coeffs: 不支持的扩展名 {p.suffix!r}(文件 {p})。"
            f"支持: {', '.join(COEFF_EXTENSIONS)}, .gz")
    sep = "," if ext == ".csv" else ("\t" if ext == ".tsv" else " ")

    # ------------------------------------------------------------ triangle
    if layout == "triangle":
        tri = coeffs.to_triangle()
        lines = []
        if header:
            lines.append(f"# SHSynth {_pkg_version()} "
                         "spherical harmonic coefficients "
                         "(triangle layout: [C; S], m outer, n inner)")
            lines.append(f"# nmax = {L}")
            lines.append(f"# ntime = {ntime}")
            lines.append(f"# ncoef_triangle = {tri.shape[0] // 2}")
            lines.append(f"# rows = 2*ncoef_triangle = {tri.shape[0]}  "
                         "(C rows first)")
            lines.append("# norm = 4pi (SHTOOLS norm=1, csphase=1), S[:,0] == 0")
            lines.append("# units = user field units, lat/lon in degrees")
            for k in ("field_unit", "forward_from", "inverse_target",
                      "output_unit", "weight_rule", "gaussian_km",
                      "gaussian_radius_km", "gaussian_W0", "gaussian_Wnmax",
                      "method", "weight_sum", "coverage", "n_points",
                      "fit_rmse_rel", "lmax_recommended", "resolution_km",
                      "truncated_from", "converted_from"):
                if k in meta and meta[k] is not None:
                    lines.append(f"# {k} = {meta[k]}")
            if comment:
                lines.extend("# " + ln for ln in comment.splitlines())
            lines.append("# columns: " + sep.join(
                [f"t{t}" for t in range(ntime)]))
        for r in range(tri.shape[0]):
            lines.append(sep.join(_fmt_float(v, fmt) for v in tri[r]))
        _text_write(p, "\n".join(lines) + "\n")
        return p

    # -------------------------------------------------------------- gmfcsv
    lines = []
    if header:
        lines.append(f"# SHSynth {_pkg_version()} "
                     "coefficients, layout n,m,C,S")
        lines.append(f"# nmax = {L}   ntime = {ntime}   "
                     "(C,S are the first time step; extra columns hold later steps)")
        if comment:
            lines.extend("# " + ln for ln in comment.splitlines())
        names = ["n", "m", "C", "S"]
        for t in range(1, ntime):
            names.extend([f"C_t{t}", f"S_t{t}"])
        lines.append(sep.join(names))
    for k in range(len(m_vec)):
        n_, m_ = int(n_vec[k]), int(m_vec[k])
        row = [str(n_), str(m_), _fmt_float(C3[n_, m_, 0], fmt),
               _fmt_float(S3[n_, m_, 0], fmt)]
        for t in range(1, ntime):
            row.extend([_fmt_float(C3[n_, m_, t], fmt),
                        _fmt_float(S3[n_, m_, t], fmt)])
        lines.append(sep.join(row))
    _text_write(p, "\n".join(lines) + "\n")
    return p


_GFC_HEADER_KEYS = ("modelname", "earth_gravity_constant", "radius",
                    "max_degree", "errors", "norm", "tide_system", "format",
                    "ref_frame", "institution", "product_type", "value_type",
                    "gapvalue", "descriptor", "unit")


def _jsonable_meta(meta: Mapping[str, Any]) -> dict:
    """把 meta 变成可 JSON 化的字典(数组转 list,不可序列化的转 str)。"""
    out: dict = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist() if v.size <= 64 else f"<array{v.shape}>"
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# v2.0:多时间序列布局(series_nc / series_dat)与目录级读取
# ---------------------------------------------------------------------------
def _count_numeric_rows(p: Path) -> int:
    """数一下非空、非注释的数据行数。"""
    n = 0
    for ln in _text_read(p).splitlines():
        s = ln.strip()
        if s and not s.startswith("#") and not s.startswith("//"):
            n += 1
    return n


def _looks_series_dat(p: Path) -> bool:
    """首行是不是"历元表头",且其余行数正好等于 ``2*NC``。

    这是 legacy ``CSR_rawSH60_total_216.dat`` 的形状:3783 行 × 216 列,
    首行是历元编号,主体 3782 = ``2*NC(nmax=60)``。

    首行**既可能**是历元序号(整数),**也可能**是十进制年(SHSynth 自己写的就是
    小数年),所以判据只看"数值合法 + 行数正好是 2*NC",不要求一定是整数。
    """
    try:
        rows = _count_numeric_rows(p)
        if rows < 3:
            return False
        if _tri_degree_from_rows(rows - 1) is None:
            return False
        for ln in _text_read(p).splitlines():
            s = ln.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            toks = s.replace(",", " ").replace("\t", " ").split()
            if len(toks) < 2:
                return False
            try:
                vals = [float(t) for t in toks]
            except ValueError:
                return False
            if not all(np.isfinite(vals)):
                return False
            # 历元表头:要么全是整数序号,要么落在合理年份区间(十进制年)
            ints = all(abs(v - round(v)) < 1e-9 for v in vals)
            years = all(1900.0 <= v <= 2200.0 for v in vals)
            return bool(ints or years)
    except Exception:                                        # noqa: BLE001
        return False
    return False


def _series_timeinfo_path(p: Path) -> Optional[Path]:
    """legacy 序列的配套时间表 ``*_TimeInfo.dat``。"""
    cand = p.with_name(p.stem + "_TimeInfo.dat")
    return cand if cand.exists() else None


def read_series_dat(path, nmax: Optional[int] = None) -> SHCoeffs:
    """读 legacy 多历元序列:首行历元表头 + ``(2*NC, ntime)`` 主体。

    自动配对同目录的 ``*_TimeInfo.dat``(有就当时间轴)。
    """
    p = _require_file(path)
    raw = np.loadtxt(p)
    if raw.ndim != 2:
        raise ValueError(f"{p}: legacy 序列应当是二维表")
    body = raw[1:, :]
    L = _tri_degree_from_rows(int(body.shape[0]))
    if L is None:
        raise ValueError(
            f"{p}: 去掉首行后有 {body.shape[0]} 行,不是 2*NC 的三角形长度;"
            "若这不是 legacy 多历元序列,请显式指定 layout")
    if nmax is not None and int(nmax) < L:
        L = int(nmax)
        body = body[:2 * ((L + 1) * (L + 2) // 2), :]
    meta = {"source_file": str(p), "layout": "series_dat",
            "epoch_header": [repr(v) for v in np.asarray(raw[0]).ravel()[:8]]}
    out = SHCoeffs.from_triangle(body, L, meta)
    ti = _series_timeinfo_path(p)
    if ti is not None:
        from .timeaxis import TimeAxis
        ax = TimeAxis.from_legacy_timeinfo(ti)
        if len(ax) == out.ntime:
            out.times = ax
            meta["time_source"] = "legacy_timeinfo"
        else:
            meta.setdefault("warnings", []).append(
                f"配套 {ti.name} 有 {len(ax)} 个历元,与系数 {out.ntime} 个不一致,已忽略")
    meta["nmax"] = out.nmax
    meta["ntime"] = out.ntime
    meta.setdefault("warnings", [])
    if out.times is None and out.ntime > 1:
        meta["warnings"].append(
            "这份 legacy 序列没有配套 *_TimeInfo.dat,日期不可恢复(按序号当时间)")
    return out


def read_series_nc(path, nmax: Optional[int] = None) -> SHCoeffs:
    """读 ``series_nc`` 契约的系数序列(见方案 §12.1)。

    **宽容读**:变量名 ``c/s`` 或 ``C/S``;维度顺序任意;``n``/``m`` 坐标可缺;
    时间坐标可缺(缺则 ``kind='index'`` 并警告)。
    """
    from .fieldio import _netcdf_open
    from .timeaxis import TimeAxis

    p = _require_path_ok(path)
    warnings: list = []
    with _netcdf_open(p, warnings) as ds:
        names = {str(k).lower(): str(k) for k in ds.variables}
        cname = names.get("c") or names.get("clm") or names.get("c_nm")
        sname = names.get("s") or names.get("slm") or names.get("s_nm")
        if cname is None and "C" in ds.variables:
            cname = "C"
        if sname is None and "S" in ds.variables:
            sname = "S"
        if cname is None or sname is None:
            raise ValueError(
                f"{p}: 不是系数序列文件(找不到 c/C 与 s/S 变量;"
                f"实际变量 {list(ds.variables)})。\n"
                "若这是一个**场**网格文件,请用 shsynth.fieldio.read_grid() "
                "或界面上的『借用网格文件』。")
        da_c, da_s = ds[cname], ds[sname]
        # 归一到 (time, n, m)
        def _norm(da):
            dims = list(da.dims)
            tname = next((d for d in dims if d.lower() in
                          ("time", "t", "nt", "epoch", "month")), None)
            nname = next((d for d in dims if d.lower() in ("n", "degree", "l")), None)
            mname = next((d for d in dims if d.lower() in ("m", "order")), None)
            rest = [d for d in dims if d not in (tname, nname, mname)]
            if nname is None or mname is None:
                # 没有 n/m 坐标:把除 time 外的两维按"方阵"解释,较大的当 n
                twod = rest or [d for d in dims if d != tname]
                if len(twod) != 2:
                    raise ValueError(f"{p}: 变量 {da.name} 的维度 {dims} 认不出 (n,m)")
                a, b = twod
                if da.sizes[a] >= da.sizes[b]:
                    nname, mname = a, b
                else:
                    nname, mname = b, a
            order = [d for d in (tname, nname, mname) if d is not None]
            arr = np.asarray(da.transpose(*order).values, dtype=float) \
                if order else np.asarray(da.values, dtype=float)
            return arr, tname
        C, tname = _norm(da_c)
        S, _ = _norm(da_s)
        if C.ndim == 2:                      # 没有 time 维 → 补一维
            C, S = C[:, :, None], S[:, :, None]
        if tname is not None and tname in ds.coords:
            try:
                ax = TimeAxis.from_netcdf_coord(ds[tname])
            except Exception:                                # noqa: BLE001
                ax = TimeAxis.from_index(C.shape[0])
                warnings.append("时间坐标无法解析,已退化为按序号当时间")
        else:
            ax = TimeAxis.from_index(C.shape[0])
            if C.shape[0] > 1:
                warnings.append("文件里没有时间坐标,日期不可恢复(按序号当时间)")
        if ax.kind == "index" and C.shape[0] > 1:
            warnings.append("时间坐标只是历元序号(无真实日期),日期不可恢复")
        attrs = {str(k): v for k, v in ds.attrs.items()}
        if hasattr(da_c, "attrs"):
            attrs.update({f"var_{k}": v for k, v in da_c.attrs.items()})
    Lfile = C.shape[1] - 1
    if C.shape[1] != C.shape[2]:
        raise ValueError(f"{p}: 系数矩阵不是方阵 ({C.shape[1]}×{C.shape[2]})")
    L = Lfile if nmax is None else min(int(nmax), Lfile)
    meta = {"source_file": str(p), "layout": "series_nc", "nmax": L,
            "ntime": int(C.shape[0]), "warnings": warnings,
            "series_attrs": {k: v for k, v in attrs.items()
                             if not k.startswith("shsynth_")}}
    for k in ("field_unit", "norm", "gaussian_km", "tide_system", "center",
              "mission", "product", "rl", "modelname"):
        v = attrs.get(k) or attrs.get(f"shsynth_{k}")
        if v is not None:
            meta[k] = v
    out = SHCoeffs(np.transpose(C[:, :L + 1, :L + 1], (1, 2, 0)).copy(),
                   np.transpose(S[:, :L + 1, :L + 1], (1, 2, 0)).copy(),
                   meta, ax)
    if L < Lfile:
        meta["warnings"] = list(meta.get("warnings", [])) + [
            f"按请求截断到 nmax={L}(文件是 {Lfile} 阶)"]
    return out


def write_series_nc(coeffs: SHCoeffs, path, *, comment: Optional[str] = None,
                    meta_extra: Optional[Mapping[str, Any]] = None) -> Path:
    """写 ``series_nc`` 契约的系数序列(推荐的多时间落盘格式)。"""
    import xarray as xr

    from .fieldio import _netcdf_save
    from .timeaxis import TimeAxis

    p = Path(os.fspath(path))
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    L, ntime = coeffs.nmax, coeffs.ntime
    C3 = coeffs.C[:, :, None] if coeffs.C.ndim == 2 else coeffs.C
    S3 = coeffs.S[:, :, None] if coeffs.S.ndim == 2 else coeffs.S
    # 契约是 (time, n, m)
    Ct = np.ascontiguousarray(np.transpose(C3, (2, 0, 1)))
    St = np.ascontiguousarray(np.transpose(S3, (2, 0, 1)))
    idx = np.arange(L + 1)
    ds = xr.Dataset(
        data_vars={
            "c": (("time", "n", "m"), Ct),
            "s": (("time", "n", "m"), St),
        },
        coords={"time": _series_time_values(coeffs, ntime),
                "n": idx.astype("int32"), "m": idx.astype("int32")},
    )
    ds["c"].attrs = {"long_name": "cosine spherical harmonic coefficients",
                     "units": "1"}
    ds["s"].attrs = {"long_name": "sine spherical harmonic coefficients",
                     "units": "1"}
    ax = coeffs.times
    if ax is not None and ax.kind != "index":
        ds["time"].attrs = {"long_name": "time", "standard_name": "time"}
        try:
            ys = ax.decimal_years
            ds["time"].attrs["shsynth_decimal_year"] = (
                f"{ys[0]:.6f} .. {ys[-1]:.6f}")
        except Exception:                                    # pragma: no cover
            pass
        if ax.start is not None and ax.end is not None:
            ds.attrs["time_coverage_start"] = str(ax.start[0])[:19]
            ds.attrs["time_coverage_end"] = str(ax.end[-1])[:19]
        ds.attrs["time_source"] = str(ax.meta.get("time_source", "unknown"))
        if ax.sources:
            ds.attrs["source_files"] = json.dumps(
                [os.path.basename(s) if s else "" for s in ax.sources],
                ensure_ascii=False)
    else:
        ds["time"].attrs = {"long_name": "epoch index", "units": "1",
                            "comment": "无真实日期"}
    m = {k: v for k, v in coeffs.meta.items()
         if k in ("field_unit", "norm", "gaussian_km", "tide_system", "center",
                  "mission", "product", "rl", "modelname") and v is not None}
    m.update(dict(meta_extra or {}))
    m.setdefault("norm", "4pi")
    m.setdefault("csphase", 1)
    m["nmax"] = L
    m["ntime"] = ntime
    for k, v in m.items():
        try:
            ds.attrs[str(k)] = _jsonable_attr(v)
        except Exception:                                    # noqa: BLE001
            ds.attrs[str(k)] = str(v)
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["title"] = "SHSynth spherical harmonic coefficient series"
    ds.attrs["producer"] = f"SHSynth {_pkg_version()}"
    ds.attrs["created"] = _now_iso()
    ds.attrs["history"] = comment or "written by shsynth.coeffio.write_series_nc"
    _netcdf_save(ds, p)
    return p


def _series_time_values(coeffs: SHCoeffs, ntime: int) -> np.ndarray:
    """序列的时间坐标值(``datetime64``;无日期时退化为序号)。"""
    from .timeaxis import TimeAxis
    ax = coeffs.times
    if ax is None:
        ax = TimeAxis.from_index(ntime)
    if ax.kind == "index" or not ax.has_dates:
        return np.arange(ntime, dtype="int32")
    return ax.to_datetime64()


def write_series_dat(coeffs: SHCoeffs, path, *,
                     header_times: bool = True,
                     write_timeinfo: bool = True,
                     fmt: str = "%.17g") -> Path:
    """写 legacy 多历元序列:首行历元表头 + ``(2*NC, ntime)`` 主体。

    ``fmt`` 默认 ``%.17g``(**可精确往返**)。注意用户手上的老文件多数是
    ``%.10g`` 写的,与 gfc 原值相对差可达 ``5e-7`` —— 那是**文件本身的十进制
    精度**,不是算法差;对表时判据要按"该文件的位数"给。

    ``write_timeinfo=True`` 且有时间轴时,顺带写配套 ``*_TimeInfo.dat``。
    """
    p = Path(os.fspath(path))
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    tri = coeffs.to_triangle()                       # (2*NC, ntime)
    header = np.arange(1, coeffs.ntime + 1, dtype=float)
    if header_times and coeffs.times is not None and coeffs.times.has_dates:
        header = coeffs.times.decimal_years
    with open(p, "w", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6f}" for v in header) + "\n")
        np.savetxt(f, tri, fmt=fmt)
    if write_timeinfo and coeffs.times is not None and coeffs.times.has_dates:
        coeffs.times.to_legacy_timeinfo(p.with_name(p.stem + "_TimeInfo.dat"))
    return p


def _require_path_ok(path) -> Path:
    p = Path(os.fspath(path))
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    return p


def _jsonable_attr(v):
    """netCDF 属性只接受标量/字符串/数组;numpy 类型转成 Python 标量。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 64 else str(v)
    if isinstance(v, (list, tuple)):
        return [ _jsonable_attr(x) for x in v ]
    return str(v)


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.now().replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# v2.0:一个目录 / 一份清单 → 系数序列
# ---------------------------------------------------------------------------
def scan_gfc_dir(source, *, pattern: Optional[str] = None,
                 product: Iterable[str] = ("GSM",), recursive: bool = True,
                 mission: Optional[str] = None, center: Optional[str] = None,
                 rl: Optional[str] = None) -> list:
    """扫出目录/清单里的 gfc 路径并按名字排序。

    ``product`` 默认只收 ``GSM``—— **混入 GAC/GAD 会静默改变物理含义**,
    所以这里是白名单而不是黑名单。
    """
    import glob as _glob
    src = os.fspath(source)
    if os.path.isdir(src):
        pat = pattern or "*.gfc"
        files = _glob.glob(os.path.join(src, "**", pat), recursive=recursive) \
            if recursive else _glob.glob(os.path.join(src, pat))
    elif src.lower().endswith(".txt") and os.path.exists(src):
        files = []
        with open(src, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if s and not s.startswith("#"):
                    files.append(os.path.join(os.path.dirname(src), s)
                                 if not os.path.isabs(s) else s)
    elif any(ch in src for ch in "*?["):
        files = _glob.glob(src, recursive=recursive)
    else:
        files = [src]
    files = [f for f in files if f.lower().endswith((".gfc", ".gfc.gz"))]
    prod = tuple(p.upper() for p in product)
    kept, rejected = [], []
    for f in files:
        base = os.path.basename(f).upper()
        if prod and not any(base.startswith(p) for p in prod):
            rejected.append(f)
            continue
        if mission and f"_{mission.upper()}" not in base:
            continue
        if center and f"_{center.upper()}" not in base:
            continue
        if rl and not base.rsplit(".", 2)[0].endswith(rl):
            continue
        kept.append(f)
    kept.sort()
    return kept, rejected


def read_coeffs_series(source, *, nmax: Optional[int] = None,
                       pattern: Optional[str] = None,
                       product: Iterable[str] = ("GSM",),
                       mission: Optional[str] = None,
                       center: Optional[str] = None,
                       rl: Optional[str] = None,
                       recursive: bool = True,
                       epoch_from: str = "auto",
                       sort: str = "time",
                       on_duplicate: str = "error",
                       on_missing: str = "report",
                       strict_nmax: bool = True) -> SHCoeffs:
    """把"一个目录 / 一份清单 / 一个多时间文件"读成**系数序列**。

    Parameters
    ----------
    source : str | Path | sequence
        目录、``glob``、路径列表、``.txt`` 清单,或单个多时间文件
        (``.nc`` / legacy ``.dat`` / ``.npz`` / ``.npy``)。
    nmax : int, optional
        截断或零填充到该阶。
    pattern : str, optional
        目录内的 glob(默认 ``*.gfc``)。
    product : sequence of str
        只收这些前缀(**默认只收 GSM**);混入 GAC/GAD 会被拒绝并**报告**。
    mission, center, rl : str, optional
        按 ``_GRAC``/``_UTCSR``/``0600`` 之类的片段过滤。
    recursive : bool
        递归子目录(ITSG 日解按年份分目录,必须递归)。
    epoch_from : {'auto','header','filename','index'}
        日期来源。``'auto'`` = 头优先、文件名兜底。
    sort : {'time','name','none'}
        排序方式(默认按时间)。
    on_duplicate : {'error','first','last','report'}
        重复历元的处理(**绝不静默**)。
    on_missing : {'report','ignore'}
        缺测只报告,不插补。
    strict_nmax : bool
        ``True``:各文件 nmax 不一致就报错;``False``:按最大阶零填充并警告。

    Returns
    -------
    SHCoeffs
        ``ntime = N``,``times`` 为 :class:`~shsynth.timeaxis.TimeAxis`,
        ``meta['source_files']`` 记下每个历元来自哪个文件。
    """
    from .timeaxis import TimeAxis

    # ---- 单个多时间文件:直接转交 -------------------------------------
    if isinstance(source, (str, os.PathLike)):
        s = os.fspath(source)
        if os.path.isfile(s):
            ext = _suffix(Path(s))
            if ext == ".nc":
                return read_series_nc(s, nmax=nmax)
            if ext in (".npz", ".npy"):
                return read_coeffs(s, nmax=nmax)
            # legacy .dat 若像序列就走序列读法
            if ext in (".dat", ".txt") and _looks_series_dat(Path(s)):
                out = read_series_dat(s, nmax=nmax)
                return out.truncate(int(nmax)) if (
                    nmax is not None and int(nmax) < out.nmax) else out
    elif isinstance(source, (list, tuple)) and source and \
            all(isinstance(x, (str, os.PathLike)) for x in source):
        pass
    else:
        raise ValueError(f"read_coeffs_series: 认不出的 source {source!r}")

    files, rejected = scan_gfc_dir(source, pattern=pattern, product=product,
                                   recursive=recursive, mission=mission,
                                   center=center, rl=rl)
    if not files:
        raise FileNotFoundError(
            f"在 {source!r} 里没有找到符合条件的 gfc 文件"
            + (f"(被 product={tuple(product)} 排除的有 {len(rejected)} 个)" 
               if rejected else ""))

    warns: list = []
    if rejected:
        warns.append(
            f"按 product={tuple(product)} 排除了 {len(rejected)} 个非目标产品文件"
            f"(例如 {os.path.basename(rejected[0])});若确实要用 GAC/GAD,"
            "请显式传 product=('GSM','GAC',...)")

    # ---- 逐个读文件 ---------------------------------------------------
    heads, cs = [], []
    nmaxs = {}
    for f in files:
        c = read_coeffs(f)
        heads.append(c.meta.get("gfc_header", {}))
        cs.append(c)
        nmaxs.setdefault(c.nmax, []).append(os.path.basename(f))
    if len(nmaxs) > 1:
        detail = "; ".join(f"{k} 阶: {len(v)} 个" for k, v in sorted(nmaxs.items()))
        if strict_nmax:
            raise ValueError(
                f"这些文件的最高阶不一致({detail})。混阶序列会静默改变物理含义。\n"
                "若确实要一起用,请传 strict_nmax=False(会按最大阶零填充并警告),"
                "或先用 nmax= 截断到较小的那个阶。")
        warns.append(f"各文件最高阶不一致({detail});已按最大阶零填充")
    L = max(nmaxs)

    # ---- 时间轴:头优先、文件名兜底 -----------------------------------
    if epoch_from in ("auto", "header"):
        ax = TimeAxis.from_gfc_headers(heads, paths=files)
    elif epoch_from == "filename":
        ax = TimeAxis.from_filenames(files)
    elif epoch_from == "index":
        ax = TimeAxis.from_index(len(files), label="文件名顺序")
        warns.append("按 epoch_from='index' 读入:没有真实日期")
    else:
        raise ValueError(
            f"epoch_from 必须是 'auto'|'header'|'filename'|'index',实际 {epoch_from!r}")
    if ax.meta.get("fallback_used"):
        warns.append(
            f"{ax.meta['fallback_used']} 个文件的头里没有时间键,已回退到文件名解析")
    for c in ax.meta.get("conflicts", [])[:5]:
        warns.append(c)

    # ---- 重复历元(绝不静默)------------------------------------------
    dup = ax.duplicates(tol_days=1.0)
    if dup:
        msg = ("发现重复历元(日期相同/相差 ≤1 天):"
               + "; ".join(f"下标 {g}({ax.values[g[0]]})" for g in dup[:5]))
        if on_duplicate == "error":
            raise ValueError(
                msg + "\n重复历元会让序列出现「同一时刻两套系数」的假象。"
                "请用 on_duplicate='first'|'last' 明确取舍,或先清理目录。")
        if on_duplicate == "report":
            warns.append(msg + "(已全部保留)")
        else:
            keep = []
            for g in dup:
                keep.append(g[0] if on_duplicate == "first" else g[-1])
            drop = sorted({i for g in dup for i in g} - set(keep))
            warns.append(msg + f";已按 on_duplicate={on_duplicate!r} 去掉 {len(drop)} 个")
            sel = np.array([i for i in range(len(files)) if i not in set(drop)])
            files = [files[i] for i in sel]
            cs = [cs[i] for i in sel]
            ax = ax.select(sel)

    # ---- 排序 ---------------------------------------------------------
    if sort == "time" and ax.has_dates:
        order = np.argsort(ax.values.astype("datetime64[D]").astype(int))
        cs = [cs[i] for i in order]
        files = [files[i] for i in order]
        ax = ax.select(order)
    elif sort == "name":
        order = np.argsort([os.path.basename(f) for f in files])
        cs = [cs[i] for i in order]
        files = [files[i] for i in order]
        ax = ax.select(order)
    elif sort not in ("time", "name", "none"):
        raise ValueError(f"sort 必须是 'time'|'name'|'none',实际 {sort!r}")

    # ---- 缺测:只报告,不插补 -----------------------------------------
    if on_missing == "report" and ax.has_dates:
        miss = ax.missing()
        if miss:
            warns.append(
                f"序列里有 {len(miss)} 处缺测(最大 {max(g for _, _, g in miss):.0f} 天);"
                "**没有插补**。需要插补请显式处理,不要让它悄悄影响趋势拟合")

    # ---- 拼接 ---------------------------------------------------------
    nt = len(cs)
    C = np.zeros((L + 1, L + 1, nt))
    S = np.zeros((L + 1, L + 1, nt))
    for k, c in enumerate(cs):
        n0 = c.nmax
        C[:n0 + 1, :n0 + 1, k] = c.C
        S[:n0 + 1, :n0 + 1, k] = c.S
    meta = {
        "source_file": os.fspath(source) if isinstance(source, (str, os.PathLike))
        else f"{len(files)} 个文件",
        "layout": "gfc_series",
        "nmax": L, "ntime": nt,
        "source_files": [os.path.basename(f) for f in files],
        "warnings": warns,
    }
    h0 = heads[0] if heads else {}
    for k_src, k_dst in (("modelname", "modelname"), ("tide_system", "tide_system"),
                         ("norm", "norm"), ("product_type", "product_type")):
        v = h0.get(k_src)
        if v is not None:
            meta[k_dst] = v
    fu = cs[0].meta.get("field_unit") if cs else None
    meta["field_unit"] = fu or "geopotential"
    out = SHCoeffs(C, S, meta, ax)
    if nmax is not None and int(nmax) < L:
        out = out.truncate(int(nmax))
        out.meta["warnings"] = warns + [f"按请求截断到 nmax={nmax}(文件是 {L} 阶)"]
    return out



# ---------------------------------------------------------------------------
# 信息汇总(CLI / GUI 共用)
# ---------------------------------------------------------------------------
def coeff_info(coeffs: SHCoeffs) -> dict:
    """把系数的关键信息整理成字典(供命令行与界面显示)。"""
    meta = coeffs.meta
    # 多时次用**时间平均**谱做代表性一行(逐历元谱请直接调 coeffs.degree_rms())
    rms = coeffs.degree_rms(time="mean")
    nz = np.nonzero(rms)[0]
    info = {
        "source_file": meta.get("source_file"),
        "layout": meta.get("layout"),
        "nmax": coeffs.nmax,
        "ntime": coeffs.ntime,
        "ncoef": coeffs.ncoef,
        "field_unit": coeffs.field_unit,
        "degree_rms_first": (int(nz[0]), float(rms[nz[0]])) if len(nz) else None,
        "degree_rms_last": (int(nz[-1]), float(rms[nz[-1]])) if len(nz) else None,
        "warnings": list(meta.get("warnings", []) or []),
        "meta": meta,
    }
    if coeffs.times is not None:
        info["times"] = coeffs.times
        info["has_times"] = True
        try:
            info["time_span"] = tuple(str(v) for v in coeffs.times.span)
        except Exception:                                  # pragma: no cover
            info["time_span"] = None
    for k in ("gaussian_radius_km", "gaussian_km", "truncated_from",
              "converted_from", "modelname", "norm"):
        if meta.get(k) is not None:
            info[k] = meta[k]
    return info
