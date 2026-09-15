# -*- coding: utf-8 -*-
"""
shsynth.coeffs
==============

球谐系数容器。

约定与 SHKit / ``m2py`` / ``gridSHconvert`` / SHTOOLS(``norm=1, csphase=1``)
**逐位一致**:

.. math::

    f(\\theta,\\lambda)=\\sum_{n=0}^{N}\\sum_{m=0}^{n}
        \\bar P_{nm}(\\cos\\theta)\\,
        \\bigl[C_{nm}\\cos m\\lambda + S_{nm}\\sin m\\lambda\\bigr]

* 4π 归一化连带勒让德函数;
* **无 Condon–Shortley 相位**;
* ``S[:, 0] ≡ 0``(因为 ``sin(0·λ) = 0``)。

系数按 ``(nmax+1, nmax+1)`` (或带时间轴 ``(nmax+1, nmax+1, ntime)``) 的稠密
矩阵存放,``m > n`` 的元素恒为 0。互换用的两种扁平布局:

``matrix``(本软件原生)
    ``C[n, m]``、``S[n, m]``。

``triangle``(m2py / gridSHconvert 的 ``out_SHCS`` 兼容)
    ``[C; S]`` 上下堆叠,每段长度 ``(nmax+1)(nmax+2)/2``,排序为 **m 外层 /
    n 内层**:``(0,0), (0,1), ..., (0,N), (1,1), (1,2), ...``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:                                    # 仅类型标注,避免与 timeaxis 环导入
    from .timeaxis import TimeAxis

__all__ = [
    "SHCoeffs",
    "triangle_order",
    "triangle_index",
    "mn_index",
    "ncoef_triangle",
    "ncoef_real",
]


# ---------------------------------------------------------------------------
# 三角形(顺序)索引
# ---------------------------------------------------------------------------
def triangle_order(nmax: int):
    """返回 m2py 三角形顺序的 ``(m_vec, n_vec)``(m 外层 / n 内层)。"""
    m_list, n_list = [], []
    for m in range(nmax + 1):
        for n in range(m, nmax + 1):
            m_list.append(m)
            n_list.append(n)
    return np.asarray(m_list, dtype=int), np.asarray(n_list, dtype=int)


def triangle_index(nmax: int):
    """返回每个三角元素落在 ``C``/``S`` 矩阵里的 ``(rows, cols)``。"""
    m_vec, n_vec = triangle_order(nmax)
    return n_vec, m_vec


def mn_index(nmax: int):
    """``triangle_order`` 的别名(m 外层 / n 内层)。"""
    return triangle_order(nmax)


def ncoef_triangle(nmax: int) -> int:
    """三角形布局单段长度 ``(nmax+1)(nmax+2)/2``。"""
    return (nmax + 1) * (nmax + 2) // 2


def ncoef_real(nmax: int) -> int:
    """实数独立系数个数 ``(nmax+1)**2``(``S[:, 0]`` 不算)。"""
    return (nmax + 1) ** 2


@dataclass
class SHCoeffs:
    """球谐系数(4π 归一化、无 CS 相位)。

    Attributes
    ----------
    C, S : ndarray
        ``(nmax+1, nmax+1)`` 或 ``(nmax+1, nmax+1, ntime)``;``S[:, 0] == 0``。
    meta : dict
        自由格式的来源信息(源文件、布局、物理量标签 ``field_unit``、
        高斯半径、警告列表……)。
    times : TimeAxis, optional
        时间轴(每个历元对应哪一天)。默认 ``None`` ⇒ 与 v1.0 行为**逐位一致**。
        见 :mod:`shsynth.timeaxis` 与 ``docs/多时间数据批量处理方案.md`` §4.2。
    """

    C: np.ndarray
    S: np.ndarray
    meta: dict = field(default_factory=dict)
    times: Optional["TimeAxis"] = None

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        self.C = np.asarray(self.C, dtype=float)
        self.S = np.asarray(self.S, dtype=float)
        if self.C.shape != self.S.shape:
            raise ValueError(
                f"C{self.C.shape} 与 S{self.S.shape} 形状必须一致")
        if self.C.ndim not in (2, 3):
            raise ValueError("C/S 必须是 2 维 (nmax+1, nmax+1) 或 3 维(带时间轴)")
        if self.C.shape[0] != self.C.shape[1]:
            raise ValueError("C/S 必须在 (阶, 次) 两维上是方阵")
        # S[:, 0] 结构上恒为 0:显式索引"次"这一维 —— 对
        # (nmax+1, nmax+1, ntime) 而言最后才是时间轴。
        if self.S.ndim == 2:
            self.S[:, 0] = 0.0
        else:
            self.S[:, 0, :] = 0.0
        if self.times is not None and len(self.times) != self.ntime:
            raise ValueError(
                f"times 有 {len(self.times)} 个历元,但系数有 ntime={self.ntime} 个时次")

    # -------------------------------------------------------------- 几何量
    @property
    def nmax(self) -> int:
        """最高阶 n(文件里实际写到的阶)。"""
        return self.C.shape[0] - 1

    @property
    def ntime(self) -> int:
        return 1 if self.C.ndim == 2 else self.C.shape[2]

    @property
    def ncoef(self) -> int:
        """独立系数个数 ``(nmax+1)**2``。"""
        return ncoef_real(self.nmax)

    @property
    def shape(self):
        return self.C.shape

    def copy(self) -> "SHCoeffs":
        return SHCoeffs(self.C.copy(), self.S.copy(), dict(self.meta),
                        self.times)

    # ----------------------------------------------------------- 时间轴
    @property
    def has_time(self) -> bool:
        """有没有时间轴(与 ``ntime > 1`` 不是一回事)。"""
        return self.times is not None

    def with_times(self, times: "TimeAxis") -> "SHCoeffs":
        """挂上时间轴(返回新对象;不改数值)。"""
        return SHCoeffs(self.C, self.S, dict(self.meta), times)

    def time_slice(self, idx: int) -> "SHCoeffs":
        """取单个时次 → **2 维**对象(``times=None``,因为它只有一个历元)。"""
        C, S = self.matrix(idx)
        meta = dict(self.meta)
        meta["time_index"] = int(idx)
        if self.times is not None:
            meta["time_value"] = str(self.times.values[int(idx)])
        return SHCoeffs(C.copy(), S.copy(), meta, None)

    def select_times(self, t0=None, t1=None, idx=None) -> "SHCoeffs":
        """按日期范围或下标集合取子序列(**保留**时间轴)。

        与 :meth:`time_slice` 的区别:这里可能还剩多个历元,所以时间轴跟着走。
        """
        if idx is not None:
            sel = self.times.select(idx) if self.times is not None else None
            ii = np.atleast_1d(np.asarray(idx))
            if ii.dtype == bool:
                ii = np.nonzero(ii)[0]
            sub = self.C[:, :, ii], self.S[:, :, ii]
        elif t0 is not None or t1 is not None:
            if self.times is None:
                raise ValueError(
                    "这个对象没有时间轴,无法按日期切片;"
                    "请先用 read_coeffs_series() 读入序列,或改用 idx= 按下标取")
            sel, mask = self.times.slice(t0, t1)
            sub = self.C[:, :, mask], self.S[:, :, mask]
        else:
            return self.copy()
        C, S = sub
        if C.shape[2] == 1:                      # 退化成单历元 → 2 维
            meta = dict(self.meta)
            return SHCoeffs(C[:, :, 0].copy(), S[:, :, 0].copy(), meta, None)
        return SHCoeffs(C.copy(), S.copy(), dict(self.meta), sel)

    def _require_same_times(self, other: "SHCoeffs", op: str = "运算") -> None:
        """双操作数运算前校验时间轴一致 —— **绝不静默广播**。"""
        a, b = self.times, other.times
        if a is None and b is None:
            return
        if a is None or b is None:
            raise ValueError(
                f"{op}的两个操作数时间轴不一致:一个有 {len(a) if a else 0} "
                f"个历元、另一个没有时间轴。请先统一(例如都 times=None,"
                "或用 select_times() 取同一段)。")
        if len(a) != len(b):
            raise ValueError(
                f"{op}的两个操作数时间轴长度不一致:"
                f"{len(a)} vs {len(b)}。若本意是「整段减参考历元」,"
                "请先用 coeffs.time_slice(k) 取出单历元再相减。")
        if a.kind == "datetime" and b.kind == "datetime":
            same = np.array_equal(a.values, b.values)
        else:
            same = np.array_equal(a.decimal_years, b.decimal_years)
        if not same:
            raise ValueError(
                f"{op}的两个操作数时间轴**日期不同**(长度都是 {len(a)},但历元对不上)。"
                "请先用 TimeAxis.match() 按日期配对,或只取交集。")

    # --------------------------------------------------------------- 取值
    def matrix(self, time: int = 0):
        """返回单个时次的 ``(C, S)``,形状 ``(nmax+1, nmax+1)``。"""
        if self.C.ndim == 2:
            if time not in (0, -1):
                raise IndexError(f"该对象只有单个时次(请求 time={time})")
            return self.C, self.S
        return self.C[:, :, time], self.S[:, :, time]

    def flat_cs(self, time: int = 0):
        """返回 ``(C_flat, S_flat)``:余弦段含全部 ``m<=n``,正弦段只含 ``m>=1``。"""
        C, S = self.matrix(time)
        m_vec, n_vec = triangle_order(self.nmax)
        cf = C[n_vec, m_vec]
        sel = m_vec >= 1
        sf = S[n_vec[sel], m_vec[sel]]
        return cf, sf

    @classmethod
    def from_flat_cs(cls, cf: np.ndarray, sf: np.ndarray, nmax: int,
                     meta: Optional[dict] = None) -> "SHCoeffs":
        m_vec, n_vec = triangle_order(nmax)
        sel = m_vec >= 1
        C = np.zeros((nmax + 1, nmax + 1))
        S = np.zeros((nmax + 1, nmax + 1))
        C[n_vec, m_vec] = np.asarray(cf, dtype=float)
        S[n_vec[sel], m_vec[sel]] = np.asarray(sf, dtype=float)
        return cls(C, S, dict(meta or {}))

    def to_triangle(self) -> np.ndarray:
        """``[C; S]`` 堆叠的三角向量(m2py ``out_SHCS`` 布局)。

        形状 ``(2*NC, ntime)``,``NC = (nmax+1)(nmax+2)/2``。
        """
        C, S = self.C, self.S
        ntime = self.ntime
        if C.ndim == 2:
            C = C[:, :, None]
            S = S[:, :, None]
        m_vec, n_vec = triangle_order(self.nmax)
        NC = len(m_vec)
        out = np.zeros((2 * NC, ntime))
        for t in range(ntime):
            out[:NC, t] = C[n_vec, m_vec, t]
            out[NC:, t] = S[n_vec, m_vec, t]
        return out

    @classmethod
    def from_triangle(cls, arr: np.ndarray, nmax: int,
                      meta: Optional[dict] = None) -> "SHCoeffs":
        """``to_triangle`` 的逆运算。"""
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        NC = ncoef_triangle(nmax)
        if arr.shape[0] != 2 * NC and arr.shape[1] == 2 * NC:
            arr = arr.T                      # 也接受 (ntime, 2*NC)
        if arr.shape[0] != 2 * NC:
            raise ValueError(
                f"三角布局需要 {2 * NC} 行(2*NC),实际 {arr.shape[0]} 行")
        m_vec, n_vec = triangle_order(nmax)
        ntime = arr.shape[1]
        C = np.zeros((nmax + 1, nmax + 1, ntime))
        S = np.zeros((nmax + 1, nmax + 1, ntime))
        for t in range(ntime):
            C[n_vec, m_vec, t] = arr[:NC, t]
            S[n_vec, m_vec, t] = arr[NC:, t]
        if ntime == 1:
            C = C[:, :, 0]
            S = S[:, :, 0]
        return cls(C, S, dict(meta or {}))

    # --------------------------------------------------------------- 统计
    def _degree_reduce(self, values: np.ndarray, time) -> np.ndarray:
        """把逐阶统计量按 ``time`` 归约。

        ``time=None``:2 维 → ``(L+1,)``;3 维 → **逐历元** ``(ntime, L+1)``。
        ``time=int``:该历元 → ``(L+1,)``。
        ``time='mean'``:跨时间取平均 → ``(L+1,)``(画谱图用)。
        """
        if values.ndim == 1:
            return values
        if time is None:
            return values                       # (ntime, L+1) 逐历元
        if isinstance(time, str):
            if time in ("mean", "avg"):
                return values.mean(axis=0)
            if time in ("rms",):
                return np.sqrt((values ** 2).mean(axis=0))
            if time in ("min",):
                return values.min(axis=0)
            if time in ("max",):
                return values.max(axis=0)
            raise ValueError(
                f"degree_rms(time=...) 只接受 None / 整数 / 'mean' / 'rms' / "
                f"'min' / 'max',收到 {time!r}")
        t = int(time)
        if not -values.shape[0] <= t < values.shape[0]:
            raise IndexError(f"time={time} 超出 ntime={values.shape[0]}")
        return values[t]

    def degree_rms(self, time=None) -> np.ndarray:
        """逐阶 RMS 幅值。

        Parameters
        ----------
        time : None | int | str
            ``None``(默认):2 维对象返回 ``(nmax+1,)``(与 v1.0 逐位相同);
            **3 维对象返回逐历元矩阵 ``(ntime, nmax+1)``**。
            ``int``:只要那个历元,返回 ``(nmax+1,)``。
            ``'mean'`` / ``'rms'`` / ``'min'`` / ``'max'``:跨时间聚合,返回 ``(nmax+1,)``。

        .. note::
           v1.0 对 3 维对象把 **C/S/时间三者一起平均**(分母是 ``(n+1)·2·ntime``),
           而 :meth:`power` 却是逐时次**求和** —— 两者时间语义不一致且名字看不出来。
           现在两者都由 ``time=`` 显式控制,3 维的默认值统一为"逐历元"。
        """
        C, S = self.C, self.S
        if C.ndim == 2:
            C3, S3 = C[:, :, None], S[:, :, None]
        else:
            C3, S3 = C, S
        ntime = C3.shape[2]
        out = np.zeros((self.nmax + 1, ntime))
        for n in range(self.nmax + 1):
            c = C3[n, :n + 1, :]                 # (n+1, ntime)
            s = S3[n, :n + 1, :]
            out[n] = np.sqrt(np.mean(np.concatenate([c, s], axis=0) ** 2, axis=0))
        if C.ndim == 2:
            return out[:, 0]
        return self._degree_reduce(out.T, time)   # (ntime, L+1) → 按 time 归约

    def power(self, time=None) -> np.ndarray:
        """逐阶总功率(系数平方和);``time`` 语义与 :meth:`degree_rms` 相同。

        2 维:``(nmax+1,)``(与 v1.0 相同);3 维且 ``time=None``:``(ntime, nmax+1)``。
        """
        C, S = self.C, self.S
        if C.ndim == 2:
            C3, S3 = C[:, :, None], S[:, :, None]
        else:
            C3, S3 = C, S
        ntime = C3.shape[2]
        out = np.zeros((self.nmax + 1, ntime))
        for n in range(self.nmax + 1):
            out[n] = (np.sum(C3[n, :n + 1, :] ** 2, axis=0)
                      + np.sum(S3[n, :n + 1, :] ** 2, axis=0))
        if C.ndim == 2:
            return out[:, 0]
        return self._degree_reduce(out.T, time)

    def truncate(self, lmax: int) -> "SHCoeffs":
        """截断(或零填充)到 ``lmax`` 阶的副本。"""
        L = max(int(lmax), 0)
        shape = (L + 1, L + 1) if self.C.ndim == 2 else (L + 1, L + 1, self.ntime)
        C = np.zeros(shape)
        S = np.zeros(shape)
        n = min(L, self.nmax)
        if self.C.ndim == 2:
            C[:n + 1, :n + 1] = self.C[:n + 1, :n + 1]
            S[:n + 1, :n + 1] = self.S[:n + 1, :n + 1]
        else:
            C[:n + 1, :n + 1, :] = self.C[:n + 1, :n + 1, :]
            S[:n + 1, :n + 1, :] = self.S[:n + 1, :n + 1, :]
        meta = dict(self.meta)
        if L < self.nmax:
            meta["truncated_from"] = self.nmax
        return SHCoeffs(C, S, meta, self.times)

    # ------------------------------------------------------------- 物理量
    @property
    def field_unit(self) -> str:
        """这套系数声明的物理含义(见 :mod:`shsynth.units`)。"""
        from .units import field_unit as _fu
        return _fu(self)

    def with_unit(self, unit: str, **meta) -> "SHCoeffs":
        """打上物理量标签(只改元数据,不改数值)。"""
        from .units import with_field_unit
        return with_field_unit(self, unit, **meta)

    # --------------------------------------------------------------- 运算
    def _binary_meta(self, other: "SHCoeffs", op: str) -> dict:
        """双操作数运算的 meta:保留物理量标签,不一致就报错。

        v1.0 直接返回 ``{"op": "add"}`` —— ``field_unit`` 被丢掉,
        之后任何单位换算都会拒绝或错换。这里改成显式合并/校验。
        """
        from .units import field_unit as _fu
        a, b = _fu(self), _fu(other)
        vague = ("unknown", "scalar")
        if a not in vague and b not in vague and a != b:
            raise ValueError(
                f"{op}的两个操作数物理量不一致:{a} vs {b}。\n"
                "把两套不同物理量的系数相加没有意义;请先用 units.convert() "
                "换算到同一个量(例如都换成 geopotential)。")
        meta = dict(self.meta)
        unit = a if a not in vague else b
        if unit not in vague:
            meta["field_unit"] = unit
        meta.pop("converted_from", None)
        meta["op"] = op
        return meta

    def __add__(self, other: "SHCoeffs") -> "SHCoeffs":
        self._require_same_times(other, "相加")
        meta = self._binary_meta(other, "add")
        L = max(self.nmax, other.nmax)
        a, b = self.truncate(L), other.truncate(L)
        return SHCoeffs(a.C + b.C, a.S + b.S, meta,
                        self.times if self.times is not None else other.times)

    def __sub__(self, other: "SHCoeffs") -> "SHCoeffs":
        self._require_same_times(other, "相减")
        meta = self._binary_meta(other, "sub")
        L = max(self.nmax, other.nmax)
        a, b = self.truncate(L), other.truncate(L)
        return SHCoeffs(a.C - b.C, a.S - b.S, meta,
                        self.times if self.times is not None else other.times)

    def __mul__(self, scalar: float) -> "SHCoeffs":
        return SHCoeffs(self.C * scalar, self.S * scalar, dict(self.meta),
                        self.times)

    __rmul__ = __mul__

    # --------------------------------------------------------------- 显示
    def __repr__(self) -> str:
        t = "" if self.times is None else f", {len(self.times)} 个日期"
        return (f"SHCoeffs(nmax={self.nmax}, ntime={self.ntime}, "
                f"ncoef={self.ncoef}{t})")

    def summary(self) -> str:
        lines = [repr(self)]
        rms = self.degree_rms(time="mean")        # 展示用:跨时间平均
        nz = np.nonzero(rms)[0]
        if len(nz):
            lines.append(f"  逐阶 RMS: n={nz[0]} -> {rms[nz[0]]:.6e} ... "
                         f"n={nz[-1]} -> {rms[nz[-1]]:.6e}"
                         + ("(时间平均)" if self.ntime > 1 else ""))
        for k in ("layout", "field_unit", "gaussian_km", "truncated_from"):
            if k in self.meta and self.meta[k] is not None:
                lines.append(f"  {k}: {self.meta[k]}")
        if self.times is not None:
            lines.append("  " + self.times.summary().replace("\n", "\n  "))
        return "\n".join(lines)
