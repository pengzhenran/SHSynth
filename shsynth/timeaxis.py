# -*- coding: utf-8 -*-
"""
shsynth.timeaxis
================

球谐/场**序列的时间轴**:一个历元对应哪一天,以及这些日期是从哪儿推出来的。

设计要点(与 ``docs/多时间数据批量处理方案.md`` §4.1 一致):

* 内部一律用 ``datetime64[ns]``;十进制年**只是标签与输出格式**,不作内部时间;
* 日期来源的权威顺序:``gfc 头 time_coverage_start/end`` >
  ``time_period_of_data`` 的 ``(mid: YYYYMMDD)`` > 文件名 > 用户显式给定;
* ``decimal_years`` 的默认口径与用户的 legacy ``*_TimeInfo.dat`` **逐位对齐**:

  .. math::

      y = \\mathrm{year} + \\frac{DOY-1}{D_{year}},\\qquad
      y_{mid} = \\frac{y_{start} + y_{end}}{2}

  其中 :math:`D_{year}` 是**该年的实际天数**(闰年 366)。实测这是唯一能让
  203 个真实 CSR 文件在 ``1e-6`` 内对上 legacy 时间表的口径(分母恒 365 只有
  71.9% 通过);详见方案 §1.3 缺陷 C。
"""

from __future__ import annotations

import calendar
import datetime as _dt
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "TimeAxis",
    "parse_gfc_time_header",
    "parse_filename_epoch",
    "dec_year",
]

#: 单位换算:datetime64 → 天
_D = np.timedelta64(1, "D")

# ---------------------------------------------------------------------------
# 日期字符串 / 文件名解析
# ---------------------------------------------------------------------------
_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DOY_RANGE = re.compile(r"_(\d{7})-(\d{7})_")          # GSM-2_2002095-2002120_...
_YMD = re.compile(r"_(\d{4})-(\d{2})-(\d{2})\.gfc")     # ITSG 日解
_YM = re.compile(r"_(\d{4})-(\d{2})\.gfc")              # ITSG 月解
_MID_IN_PERIOD = re.compile(r"\(mid:\s*(\d{8})\)")


def dec_year(d, *, denominator: str = "actual") -> float:
    """日期 → 十进制年 ``year + (DOY-1)/D``。

    ``denominator``:
      * ``'actual'``(默认):``D`` = 该年的实际天数(闰年 366)—— 与用户 legacy
        ``*_TimeInfo.dat`` 一致(实测 100% 通过 1e-6);
      * ``'365'``:分母恒 365(流传的写法;实测只有 71.9% 能到 1e-6)。

    接受 ``datetime.date`` 或 ``datetime64``;**保留小数天**(带时刻时用真实小数天数,
    不截断 —— 否则 legacy 的 ``y_mid`` 会差到 1 天 = 2.7e-3 年)。
    """
    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
        d = np.datetime64(d.isoformat())
    if not isinstance(d, np.datetime64):
        d = np.datetime64(d, "ns")
    if np.isnat(d):
        return float("nan")
    year = int(d.astype("datetime64[Y]").astype(int)) + 1970
    jan1 = np.datetime64(f"{year:04d}-01-01", "ns")
    frac_days = float((d - jan1) / np.timedelta64(1, "D"))      # DOY-1(可含小数)
    if denominator == "365":
        den = 365.0
    elif denominator == "actual":
        den = 366.0 if calendar.isleap(year) else 365.0
    else:
        raise ValueError(f"denominator 只能是 'actual' 或 '365',收到 {denominator!r}")
    return year + frac_days / den


def _as_date(text) -> Optional[_dt.date]:
    """从字符串里抠出日期。

    只取前 10 个字符的 ``YYYY-MM-DD``,因此对下面这些真实写法都成立:

    * ``': 2002-04-05T00:00:00.00'``(带 ``': '`` 前缀,GRACE gfc 头)
    * ``'2020-03-01T00.00.00.00'``(**点号**而不是冒号 —— 实测真实文件里就有)
    """
    if text is None:
        return None
    m = _ISO_DATE.search(str(text))
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _as_date_from_doys(s: str) -> Optional[_dt.date]:
    """``2002095`` → 2002 年第 95 天(**年+积日**,YYYYDOY)。"""
    try:
        y, doy = int(s[:4]), int(s[4:])
        return _dt.date(y, 1, 1) + _dt.timedelta(days=doy - 1)
    except (ValueError, TypeError):
        return None


def _as_date_from_yyyymmdd(s: str) -> Optional[_dt.date]:
    """``20020418`` → 2002-04-18(**日历日期**,YYYYMMDD)。

    ⚠️ 与 :func:`_as_date_from_doys` 是**两种不同的写法**,不能混用:
    ``time_period_of_data`` 里的 ``(mid: 20020418)`` 是日历日期;
    而文件名里的 ``2002095`` 是"年+积日"。实测把前者当后者会偏一整年。
    """
    try:
        return _dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        return None


def parse_gfc_time_header(header, path: Optional[str] = None) -> dict:
    """从 gfc 头(``meta['gfc_header']``)里抽时间信息。

    返回 ``dict(start=None|date, end=None|date, mid=None|date,
    unused_days=[date...], source='gfc_header')``。
    """
    h = dict(header or {})
    start = _as_date(h.get("time_coverage_start"))
    end = _as_date(h.get("time_coverage_end"))
    mid = None
    tpd = h.get("time_period_of_data")
    if tpd:
        m = _MID_IN_PERIOD.search(str(tpd))
        if m:
            mid = _as_date_from_yyyymmdd(m.group(1))
        if start is None or end is None:
            rng = re.findall(r"(\d{8})", str(tpd))
            if len(rng) >= 2:
                start = start or _as_date_from_yyyymmdd(rng[0])
                end = end or _as_date_from_yyyymmdd(rng[1])
    unused: list = []
    raw_unused = h.get("unused_days")
    if raw_unused:
        for tok in re.findall(r"(\d{4}-\d{2}-\d{2})", str(raw_unused)):
            d = _as_date(tok)
            if d is not None:
                unused.append(d)
    return {"start": start, "end": end, "mid": mid, "unused_days": unused,
            "source": "gfc_header" if (start or end or mid) else "none",
            "source_file": os.fspath(path) if path is not None else None}


def parse_filename_epoch(name: str) -> dict:
    """从文件名推历元(三条规则)。

    * 规则1 ``GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600.gfc`` → 开始/结束年积日;
    * 规则2 ``ITSG-Grace2018_n60_2002-04.gfc`` → 年-月(取当月 15 日);
    * 规则3 ``ITSG-Grace2018_Kalman_n40_2002-04-01.gfc`` → 年-月-日。

    返回 ``dict(start, end, mid, rule)``,解析不出时各项为 ``None``、``rule='none'``。
    """
    base = os.path.basename(str(name))
    m = _DOY_RANGE.search(base)
    if m:
        a, b = _as_date_from_doys(m.group(1)), _as_date_from_doys(m.group(2))
        if a and b:
            return {"start": a, "end": b,
                    "mid": a + (b - a) / 2, "rule": "doy_range"}
    m = _YMD.search(base)
    if m:
        try:
            d = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return {"start": d, "end": d, "mid": d, "rule": "ymd"}
        except ValueError:
            pass
    m = _YM.search(base)
    if m:
        try:
            y, mo = int(m.group(1)), int(m.group(2))
            last = calendar.monthrange(y, mo)[1]
            a = _dt.date(y, mo, 1)
            b = _dt.date(y, mo, last)
            return {"start": a, "end": b,
                    "mid": a + (b - a) / 2, "rule": "ym"}
        except ValueError:
            pass
    return {"start": None, "end": None, "mid": None, "rule": "none"}


# ---------------------------------------------------------------------------
def _to_ns(dates: Sequence) -> np.ndarray:
    """``date`` / ``datetime64`` / 字符串 / ``None`` 的列表 → ``datetime64[ns]``(None → NaT)。"""
    out = np.full(len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for i, d in enumerate(dates):
        if d is None:
            continue
        if isinstance(d, np.datetime64):
            out[i] = d.astype("datetime64[ns]") if not np.isnat(d) \
                else np.datetime64("NaT", "ns")
        elif isinstance(d, str):
            out[i] = np.datetime64(d, "ns")
        else:
            out[i] = np.datetime64(d.isoformat(), "ns")
    return out


def _to_ns_opt(dates) -> Optional[np.ndarray]:
    if dates is None:
        return None
    return _to_ns(list(dates))


@dataclass
class TimeAxis:
    """一条时间轴:历元代表日期 + 可选的时段起止 + 来源记录。

    Attributes
    ----------
    values : ndarray
        ``datetime64[ns]``,每个历元的**代表日期**(默认取时段日历中点);``NaT`` 表示缺。
    start, end : ndarray, optional
        时段的起止(coverage start/end),有则用于 legacy 十进制年口径与诊断。
    kind : str
        ``'datetime'`` | ``'decimal_year'`` | ``'index'``(后两者表示"推不出真日期")。
    labels : list[str], optional
        原样保留的标签(``modelname`` 或文件名)。
    sources : list[str], optional
        每个历元来自哪个文件。
    """

    values: np.ndarray
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    kind: str = "datetime"
    labels: Optional[list] = None
    sources: Optional[list] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.atleast_1d(np.asarray(self.values, dtype="datetime64[ns]"))
        for name in ("start", "end"):
            v = getattr(self, name)
            if v is not None:
                setattr(self, name,
                        np.atleast_1d(np.asarray(v, dtype="datetime64[ns]")))

    # ------------------------------------------------------------ 基本量
    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "TimeAxis(n=0)"
        if self.kind == "index":
            return f"TimeAxis(n={len(self)}, kind='index' 无真实日期)"
        return (f"TimeAxis(n={len(self)}, {self.values[0]} .. {self.values[-1]}, "
                f"kind={self.kind!r})")

    @property
    def has_dates(self) -> bool:
        """有没有真实日期(``kind='index'`` 时为 False)。"""
        return self.kind != "index"

    @property
    def span(self) -> tuple:
        """``(起, 止)``;空轴返回 ``(None, None)``。"""
        if len(self) == 0:
            return (None, None)
        return (self.values[0], self.values[-1])

    @property
    def dt_days(self) -> np.ndarray:
        """相邻历元间隔(天),首元素 ``NaN``。"""
        if len(self) < 2:
            return np.full(max(len(self), 0), np.nan)
        d = (self.values[1:] - self.values[:-1]) / _D
        return np.concatenate([[np.nan], d.astype(float)])

    @property
    def decimal_years(self) -> np.ndarray:
        """十进制年,**默认与 legacy ``*_TimeInfo.dat`` 同口径**。

        有 ``start``/``end`` 时用 ``(dec(start)+dec(end))/2``;否则用 ``dec(values)``。
        """
        return self.decimal_years_with("actual")

    def decimal_years_with(self, denominator: str = "actual") -> np.ndarray:
        if self.start is not None and self.end is not None:
            s, e = self.start, self.end
            ys = np.array([dec_year(x, denominator=denominator) for x in s])
            ye = np.array([dec_year(x, denominator=denominator) for x in e])
            out = (ys + ye) / 2.0
            bad = ~np.isfinite(out)
            if bad.any():                    # 起止缺失(NaT)时退回代表日期
                out[bad] = [dec_year(x, denominator=denominator)
                            for x in self.values[bad]]
            return out
        return np.array([dec_year(x, denominator=denominator)
                         for x in self.values])

    def is_regular(self, tol_days: float = 3.0) -> bool:
        """间隔是否近似等间隔(相对中位间隔的偏差 ≤ ``tol_days``)。"""
        d = self.dt_days
        d = d[np.isfinite(d)]
        if d.size == 0:
            return True
        med = float(np.median(d))
        return bool(np.all(np.abs(d - med) <= tol_days))

    # ------------------------------------------------------------ 构造
    @classmethod
    def from_index(cls, n: int, *, label: str = "历元序号") -> "TimeAxis":
        """没有日期时退化而成(``kind='index'``)—— **必须**在调用方给出警告。"""
        n = int(n)
        return cls(np.arange(n).astype("datetime64[ns]") if n == 0
                   else np.arange(n).astype("timedelta64[D]").astype("datetime64[ns]"),
                   kind="index", meta={"note": f"无真实日期,按{label}当时间"})

    @classmethod
    def from_datetimes(cls, values, *, start=None, end=None,
                       labels=None, sources=None, source: str = "user") -> "TimeAxis":
        return cls(_to_ns(list(values)), start=_to_ns_opt(start),
                   end=_to_ns_opt(end), labels=labels, sources=sources,
                   meta={"time_source": source})

    @classmethod
    def from_decimal_years(cls, years, *, denominator: str = "actual",
                           labels=None, sources=None,
                           round_to_day: bool = False) -> "TimeAxis":
        """十进制年 → 时间轴(反解成 1 月 1 日 + 小数部分的天数)。

        ``round_to_day=True``:把结果取整到**最近的整天**。legacy
        ``*_TimeInfo.dat`` 本身就是天分辨率的,而十进制年只有 ~6 位小数
        (≈3 秒),不取整会让 ``2003-01-16T00:00:00`` 漂成 ``T23:59:58``,
        再截断到日期就变成前一天 —— 取整后逐日一致。
        """
        ys = np.atleast_1d(np.asarray(years, dtype=float))
        dates = []
        for y in ys:
            if not np.isfinite(y):
                dates.append(None)
                continue
            yy = int(np.floor(y))
            frac = y - yy
            den = (366.0 if (denominator == "actual" and calendar.isleap(yy))
                   else 365.0)
            # 用 datetime(不是 date)以**保留小数天**:legacy 的 y_mid 常常落在半天上
            dt = _dt.datetime(yy, 1, 1) + _dt.timedelta(days=frac * den)
            if round_to_day:
                dt = _dt.datetime(dt.year, dt.month, dt.day) + \
                    _dt.timedelta(days=1 if dt.hour >= 12 else 0)
            dates.append(dt)
        return cls(_to_ns(dates), labels=labels, sources=sources,
                   meta={"time_source": "decimal_year",
                         "legacy_denominator": denominator})

    @classmethod
    def from_filenames(cls, paths: Iterable[str]) -> "TimeAxis":
        paths = [os.fspath(p) for p in paths]
        info = [parse_filename_epoch(p) for p in paths]
        rules = {i["rule"] for i in info}
        return cls(_to_ns([i["mid"] for i in info]),
                   start=_to_ns_opt([i["start"] for i in info]),
                   end=_to_ns_opt([i["end"] for i in info]),
                   labels=[os.path.basename(p) for p in paths],
                   sources=list(paths),
                   meta={"time_source": "filename",
                         "filename_rules": sorted(rules)})

    @classmethod
    def from_gfc_headers(cls, headers, *, paths=None,
                         on_conflict: str = "report") -> "TimeAxis":
        """**权威**来源:一组 gfc 头(``meta['gfc_header']``)。

        头里没有时自动退回文件名(并记在 ``meta['fallback_used']``)。
        """
        headers = list(headers)
        paths = list(paths) if paths is not None else [None] * len(headers)
        parsed = [parse_gfc_time_header(h, p) for h, p in zip(headers, paths)]
        fb = [parse_filename_epoch(p) if p else {"start": None, "end": None,
                                                 "mid": None, "rule": "none"}
              for p in paths]
        starts, ends, mids, notes = [], [], [], []
        n_fb = 0
        for k, (p, f) in enumerate(zip(parsed, fb)):
            s = p["start"] if p["start"] is not None else f["start"]
            e = p["end"] if p["end"] is not None else f["end"]
            m = p["mid"] if p["mid"] is not None else f["mid"]
            if p["source"] == "none":
                n_fb += 1
            starts.append(s)
            ends.append(e)
            mids.append(m)
            # 头里声明的 (mid:) 与"两个十进制年的算术平均"最多差 1 天;
            # 差得更多说明文件有问题,报告出来(不当场改数)。
            if (on_conflict == "report" and p["mid"] and s and e
                    and abs(dec_year(p["mid"])
                            - (dec_year(s) + dec_year(e)) / 2.0) * 365.0 > 1.5):
                notes.append(f"第 {k} 个历元的 (mid:) 与 coverage 起止不自洽")
        return cls(_to_ns(mids), start=_to_ns_opt(starts), end=_to_ns_opt(ends),
                   labels=[os.path.basename(p) if p else None for p in paths],
                   sources=[p for p in paths],
                   meta={"time_source": "gfc_header",
                         "fallback_used": n_fb,
                         "conflicts": notes})

    @classmethod
    def from_legacy_timeinfo(cls, path) -> "TimeAxis":
        """读 legacy ``*_TimeInfo.dat``:``[idx, y_start, y_end, y_mid]``(N×4)。

        十进制年按**最近整天**取整(该格式本来就是天分辨率)。
        """
        arr = np.loadtxt(os.fspath(path))
        arr = np.atleast_2d(arr)
        if arr.shape[1] < 4:
            raise ValueError(
                f"{path}: 期望 4 列 [idx, y_start, y_end, y_mid],实际 {arr.shape[1]} 列")
        ax = cls.from_decimal_years(arr[:, 3], round_to_day=True)
        # ⚠️ start/end **不取整**:decimal_years 用的是
        # (dec(start)+dec(end))/2,取整会把误差放大到 ~0.5 天(1.4e-3 年),
        # 破坏与 legacy ≤1e-6 的对表;取整只用于"代表日期" values。
        ax.start = cls.from_decimal_years(arr[:, 1]).values
        ax.end = cls.from_decimal_years(arr[:, 2]).values
        ax.meta = {"time_source": "legacy_timeinfo", "source_file": os.fspath(path)}
        return ax

    @classmethod
    def from_netcdf_coord(cls, coord) -> "TimeAxis":
        """xarray 的 time 坐标(``datetime64`` 或 ``数值 + units``)→ 时间轴。"""
        import xarray as xr
        if not isinstance(coord, xr.DataArray):
            coord = xr.DataArray(np.asarray(coord))
        vals = coord.values
        if np.issubdtype(np.asarray(vals).dtype, np.datetime64):
            a = cls(np.asarray(vals, dtype="datetime64[ns]"),
                    meta={"time_source": "netcdf_datetime64"})
            a.meta["units"] = str(coord.attrs.get("units", ""))
            a.meta["calendar"] = str(coord.attrs.get("calendar", "standard"))
            return a
        units = str(coord.attrs.get("units", ""))
        m = re.search(r"(days|hours|seconds|minutes)\s+since\s+(.+)", units,
                      flags=re.IGNORECASE)
        if not m:
            # 纯数值坐标:只能当序号,明确说明
            return cls.from_index(np.asarray(vals).size,
                                  label="数值坐标(无 units)")
        unit, origin = m.group(1).lower(), m.group(2).strip()
        origin_d = _as_date(origin)
        if origin_d is None:
            return cls.from_index(np.asarray(vals).size, label="无法解析的 units")
        per_day = {"days": 1.0, "hours": 24.0, "minutes": 1440.0,
                   "seconds": 86400.0}[unit]
        secs = np.asarray(vals, dtype=float) / per_day * 86400.0
        base = np.datetime64(origin_d.isoformat(), "s")
        return cls((base + (secs * 1e3).astype("timedelta64[ms]")
                    ).astype("datetime64[ns]"),
                   meta={"time_source": "netcdf_cf_units", "units": units})

    # ------------------------------------------------------------ 查询
    def select(self, idx) -> "TimeAxis":
        """按索引/布尔掩码取子轴。"""
        idx = np.atleast_1d(np.asarray(idx))
        if idx.dtype == bool:
            if idx.size != len(self):
                raise ValueError(f"布尔掩码长度 {idx.size} 与时间轴 {len(self)} 不符")
            sel = idx
        else:
            sel = np.zeros(len(self), dtype=bool)
            sel[idx.astype(int) % max(len(self), 1)] = True
        return TimeAxis(self.values[sel],
                        None if self.start is None else self.start[sel],
                        None if self.end is None else self.end[sel],
                        kind=self.kind,
                        labels=None if self.labels is None else
                        [l for l, k in zip(self.labels, sel) if k],
                        sources=None if self.sources is None else
                        [s for s, k in zip(self.sources, sel) if k],
                        meta=dict(self.meta))

    def slice(self, t0=None, t1=None) -> tuple:
        """按日期范围取子轴,返回 ``(TimeAxis, 布尔掩码)``(两端都含)。"""
        if len(self) == 0:
            return self.select(np.zeros(0, dtype=bool)), np.zeros(0, dtype=bool)
        mask = np.ones(len(self), dtype=bool)
        if t0 is not None:
            mask &= self.values >= np.datetime64(t0, "ns")
        if t1 is not None:
            mask &= self.values <= np.datetime64(t1, "ns")
        return self.select(mask), mask

    def nearest(self, when, tol_days: float = 15.0) -> int:
        """最接近 ``when`` 的历元下标;超过 ``tol_days`` 抛错。"""
        if len(self) == 0:
            raise ValueError("空时间轴没有最近历元")
        target = np.datetime64(when, "ns")
        d = np.abs((self.values - target) / _D).astype(float)
        j = int(np.nanargmin(d))
        if tol_days is not None and d[j] > tol_days:
            raise ValueError(
                f"{when} 与最近的历元 {self.values[j]} 相差 {d[j]:.1f} 天,"
                f"超过容差 {tol_days} 天")
        return j

    def match(self, other: "TimeAxis", tol_days: float = 1.0) -> np.ndarray:
        """把自己对齐到 ``other``:返回 ``self`` 每个历元在 ``other`` 里的下标(``-1`` 表示配不上)。

        **按日期配对,不按下标** —— 两侧的历元集合本来就可能不同
        (实测:203 个 gfc vs 216 行 TimeInfo)。
        """
        out = np.full(len(self), -1, dtype=int)
        if len(other) == 0:
            return out
        for i, v in enumerate(self.values):
            d = np.abs((other.values - v) / _D).astype(float)
            j = int(np.nanargmin(d))
            if d[j] <= tol_days:
                out[i] = j
        return out

    def missing(self, cadence_days: float = 31.0,
                tol_days: float = 6.0) -> list:
        """缺测区间:相邻间隔超过 ``cadence_days + tol_days`` 的地方。

        返回 ``[(左下标, 右下标, 间隔天数), ...]``;只在有真实日期的轴上做。
        """
        if self.kind == "index" or len(self) < 2:
            return []
        d = self.dt_days
        out = []
        for i in range(1, len(self)):
            if np.isfinite(d[i]) and d[i] > cadence_days + tol_days:
                out.append((i - 1, i, float(d[i])))
        return out

    def duplicates(self, tol_days: float = 1.0) -> list:
        """重复历元:日期靠得比 ``tol_days`` 还近的历元组(返回下标列表的列表)。"""
        if len(self) < 2:
            return []
        order = np.argsort(self.values.astype("datetime64[D]").astype(int))
        groups, cur = [], [int(order[0])]
        for k in range(1, len(order)):
            i, j = int(order[k]), int(order[k - 1])
            gap = abs(float((self.values[i] - self.values[j]) / _D))
            if gap <= tol_days:
                cur.append(i)
            else:
                if len(cur) > 1:
                    groups.append(sorted(cur))
                cur = [i]
        if len(cur) > 1:
            groups.append(sorted(cur))
        return groups

    # ------------------------------------------------------------ 输出
    def summary(self, *, max_list: int = 6) -> str:
        """中文摘要:历元数 / 跨度 / 间隔统计 / 缺测 / 重复 / 来源。"""
        lines = [f"时间轴: {len(self)} 个历元" + ("(无真实日期)" if not self.has_dates else "")]
        if len(self) and self.has_dates:
            lines.append(f"  跨度      : {self.values[0]} .. {self.values[-1]}")
            d = self.dt_days
            d = d[np.isfinite(d)]
            if d.size:
                lines.append(f"  相邻间隔  : min {d.min():.0f} / 中位 "
                             f"{np.median(d):.0f} / max {d.max():.0f} 天"
                             + ("(不规则)" if not self.is_regular() else "(近似规则)"))
            miss = self.missing()
            if miss:
                head = ", ".join(f"{a}→{b} 差 {g:.0f} 天"
                                 for a, b, g in miss[:max_list])
                lines.append(f"  缺测      : {len(miss)} 处  {head}"
                             + (" …" if len(miss) > max_list else ""))
            else:
                lines.append("  缺测      : 无")
            dup = self.duplicates()
            if dup:
                lines.append(f"  重复历元  : {len(dup)} 组 "
                             + ", ".join(str(g) for g in dup[:max_list]))
        src = self.meta.get("time_source")
        if src:
            extra = ""
            if self.meta.get("fallback_used"):
                extra = f"(其中 {self.meta['fallback_used']} 个回退到文件名)"
            lines.append(f"  日期来源  : {src}{extra}")
        for c in (self.meta.get("conflicts") or [])[:max_list]:
            lines.append(f"  ! {c}")
        return "\n".join(lines)

    def to_legacy_timeinfo(self, path) -> str:
        """写成 legacy ``*_TimeInfo.dat``:``[idx, y_start, y_end, y_mid]``。"""
        from . import __version__
        d = self.decimal_years
        if self.start is not None and self.end is not None:
            s = self.decimal_years_with_bounds("start")
            e = self.decimal_years_with_bounds("end")
        else:
            s = e = d
        arr = np.column_stack([np.arange(1, len(self) + 1), s, e, d])
        p = os.fspath(path)
        header = (f"# SHSynth {__version__} legacy TimeInfo "
                  f"[idx, y_start, y_end, y_mid]\n"
                  f"# 口径: y = year + (DOY-1)/该年实际天数;y_mid = (y_start+y_end)/2\n")
        with open(p, "w", encoding="utf-8") as f:
            f.write(header)
            np.savetxt(f, arr, fmt="%.6f")
        return p

    def decimal_years_with_bounds(self, which: str = "start") -> np.ndarray:
        arr = self.start if which == "start" else self.end
        if arr is None:
            return self.decimal_years
        return np.array([dec_year(d) for d in arr], dtype=float)

    def to_datetime64(self, *, index_as_days: bool = False) -> np.ndarray:
        """给 netCDF 坐标用:返回 ``datetime64[ns]``;``kind='index'`` 时可退化成天序号。"""
        if self.kind == "index" and index_as_days:
            return np.arange(len(self), dtype="int32")
        return self.values
