# -*- coding: utf-8 -*-
"""
shsynth.lovenumbers
===================

载荷勒夫数 ``k'_n`` / ``h'_n`` / ``l'_n`` 的读取与缓存。

默认使用随包提供的表(由 SHKit 的 ``data/`` 目录复制而来,来源为
``PREM-LLNs.dat``(Wang et al. 2012),其中 ``k'_1`` 已按 ``pz_LLN.m`` 做
CE → CF 改正 ``k'_1 = -(h'_1 + 2 l'_1)/3``)。也可以传入
``PREM-LLNs.dat`` 风格的文本表(列 ``n h l k [nl nk]``,``n = inf`` 行为渐近值),
缺失的整阶用线性插值补齐(与 ``pz_LLN.m`` / SHKit 的做法一致)。

物理量换算(geoid / 面密度 / EWH / 径向形变)依赖这张表,所以它是
:mod:`shsynth.units` 的依赖,而不是反过来。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

__all__ = [
    "DATA_DIR",
    "load_love_numbers",
    "load_lln",
    "love_number",
    "describe_love_numbers",
]

#: 随包数据目录(可用环境变量 ``SHSYNTH_DATA_DIR`` 覆盖)。
DATA_DIR = os.environ.get(
    "SHSYNTH_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))

_CACHE: dict = {}


def _default_k_path() -> str:
    return os.path.join(DATA_DIR, "love_numbers.npy")


def _default_lln_path() -> str:
    return os.path.join(DATA_DIR, "load_love_numbers.npz")


def load_love_numbers(path: Optional[str] = None) -> np.ndarray:
    """读取逐阶索引的载荷勒夫数 ``k'_n``(``k'_0 = 0``)。

    默认表:``data/love_numbers.npy``(PREM / Wang 2012,与参考软件
    ``m2py`` 的表逐值一致)。也可给 ``.npz``(取其中的 ``k``)或
    ``.txt`` / ``.dat``(``PREM-LLNs.dat`` 风格)。
    """
    p = path or _default_k_path()
    key = ("k", str(p))
    if key in _CACHE:
        return _CACHE[key]

    lp = str(p).lower()
    if lp.endswith(".npz"):
        with np.load(p, allow_pickle=False) as z:
            if "k" not in z.files:
                raise ValueError(
                    f"{p}: .npz 中没有 'k' 数组(实际: {list(z.files)})")
            arr = np.asarray(z["k"], dtype=float).ravel()
    elif lp.endswith(".npy"):
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float).ravel()
    else:
        arr = load_lln(p)["k"]

    _CACHE[key] = arr
    return arr


def load_lln(path: Optional[str] = None) -> dict:
    """读取完整载荷勒夫数集 ``h'`` / ``l'`` / ``k'``。

    Returns
    -------
    dict
        键 ``n``、``h``、``l``、``k``,外加 ``model`` / ``source``,
        文本插值时为 ``interpolated`` / ``n_given``。
    """
    p = path or _default_lln_path()
    key = ("lln", str(p))
    if key in _CACHE:
        return _CACHE[key]

    lp = str(p).lower()
    if lp.endswith(".npz"):
        with np.load(p, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files}
        for k in ("h", "l", "k"):
            if k in out:
                out[k] = np.asarray(out[k], dtype=float).ravel()
        out.setdefault("model", os.path.basename(p))
        out.setdefault("source", p)
    elif lp.endswith(".npy"):
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=float).ravel()
        out = {"n": np.arange(arr.size), "h": np.zeros(arr.size),
               "l": np.zeros(arr.size), "k": arr,
               "model": os.path.basename(p), "source": p}
    else:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"载荷勒夫数文件不存在: {p}\n"
                "请给出 .npy / .npz / PREM-LLNs.dat 风格的表。")
        out = _read_lln_text(p)
    _CACHE[key] = out
    return out


def _read_lln_text(path: str) -> dict:
    """读取 ``PREM-LLNs.dat`` 风格的表(列 ``n h l k [nl nk]``)。"""
    rows, asym = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vals = [float(x) for x in parts[:6]]
            except ValueError:
                continue                                   # 表头
            if not np.isfinite(vals[0]):
                asym = vals
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"{path}: 没有解析到任何 'n h l k' 数据行")

    a = np.asarray(rows, dtype=float)
    n = a[:, 0].astype(int)
    nmax = int(n.max())
    k = a[:, 3].copy()
    k[0] = -(a[0, 1] + 2.0 * a[0, 2]) / 3.0        # 1 阶:CE -> CF

    interpolated = not np.array_equal(n, np.arange(1, nmax + 1))
    if interpolated:
        # 短表只列到 10 阶再加若干节点(18, 32, 56, 100, 180 ...);
        # pz_LLN.m 用 interp1 补全,这里同样做线性插值并如实记录。
        grid = np.arange(nmax + 1, dtype=float)
        hh = np.interp(grid, n, a[:, 1])
        ll = np.interp(grid, n, a[:, 2])
        kk = np.interp(grid, n, k)
        hh[0] = ll[0] = kk[0] = 0.0
        kk[1] = -(hh[1] + 2.0 * ll[1]) / 3.0       # 保持 CE→CF 精确
        hh[1], ll[1] = a[0, 1], a[0, 2]
    else:
        hh = np.concatenate(([0.0], a[:, 1]))
        ll = np.concatenate(([0.0], a[:, 2]))
        kk = np.concatenate(([0.0], k))

    return {
        "n": np.arange(nmax + 1),
        "h": hh, "l": ll, "k": kk,
        "asymptote": np.asarray(asym[1:4]) if asym else np.zeros(3),
        "model": os.path.basename(path),
        "source": path,
        "interpolated": interpolated,
        "n_given": int(n.size),
    }


def love_number(n: int, table: Optional[np.ndarray] = None,
                kind: str = "k") -> float:
    """第 ``n`` 阶载荷勒夫数;超出表长返回 0。

    ``kind``:``'k'``(位,默认)、``'h'``(径向)、``'l'``(水平)。
    """
    if table is not None:
        arr = np.asarray(table, dtype=float).ravel()
    else:
        arr = np.asarray(load_lln()[kind], dtype=float)
    n = int(n)
    return float(arr[n]) if 0 <= n < arr.size else 0.0


def describe_love_numbers(nmax: int = 8, lln: Optional[dict] = None) -> str:
    """打印前 ``nmax`` 阶的 ``h' / l' / k'``(用于文档与自检)。"""
    d = load_lln() if lln is None else lln
    lines = [f"载荷勒夫数表: {d.get('model', '?')}   "
             f"(阶数上限 {len(d['k']) - 1})",
             f"{'n':>3} {'h′':>16} {'l′':>16} {'k′':>16}"]
    for n in range(min(nmax, len(d["k"]) - 1) + 1):
        lines.append(f"{n:>3} {d['h'][n]:>16.8e} {d['l'][n]:>16.8e} "
                     f"{d['k'][n]:>16.8e}")
    return "\n".join(lines)
