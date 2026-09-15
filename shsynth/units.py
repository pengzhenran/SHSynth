# -*- coding: utf-8 -*-
"""
shsynth.units
=============

一组球谐系数**描述的是哪个物理量**,以及如何互相换算。

同一张网格上的数字可以是水准面高、等效水高、面密度、径向形变,或者一个
无量纲场(例如区域平均核),而它们的球谐系数是**不同的几套数**。从无量纲重力位
系数(GRACE Level-2 公布的 ``C_nm, S_nm``)换到其它量,因子是**逐阶**的:

=======================  ==========================================  ==============
目标 ``field_unit``      由 ``C_nm`` 出发的逐阶因子                   需要
=======================  ==========================================  ==============
``geopotential``         ``1``                                       —
``geoid``                ``R``(常数,不逐阶)                        —
``surface_density``      ``R·ρ̄/3 · (2n+1)/(1+k′ₙ)``                 ``k′``
``ewh``                  ``R·ρ̄/(3ρ_w) · (2n+1)/(1+k′ₙ)`` = ``Aₙ``   ``k′``
``radial_displacement``  ``R·h′ₙ/(1+k′ₙ)``                           ``k′`` 与 ``h′``
=======================  ==========================================  ==============

``scalar`` / ``unknown`` 是**终点标签**:「这就是一个场,单位由数字本身决定」,
不存在已知的物理换算,所以软件宁可明确拒绝,也不给一个差 ``1e7`` 倍的结果。

``Aₙ`` **不是常数**:实测 ``A₀ = 1.17e7``、``A₆ = 1.68e8``,仅 0→6 阶就差 14.3 倍。
所以两套系数既不能相加,也不能"乘一个常数"互换;重复乘一次 ``Aₙ`` 会放大
1e7~1e8 倍。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .coeffs import SHCoeffs
from .filters import EARTH_RADIUS_M, RHO_AVE, RHO_WATER
from .lovenumbers import load_lln, load_love_numbers

__all__ = [
    "CANONICAL",
    "FIELD_UNITS",
    "FIELD_UNIT_LABELS",
    "UNITS_NEEDING_LOVE_K",
    "UNITS_NEEDING_LOVE_H",
    "field_unit",
    "with_field_unit",
    "degree_factors",
    "forward_factors",
    "convert",
    "require_convertible",
    "formula_text",
    "describe_conversion",
    "normalise_unit",
]

#: 规范中间量:所有换算都经由**无量纲重力位系数**中转。
CANONICAL = "geopotential"

#: ``SHCoeffs.meta['field_unit']`` 的全部合法取值。
FIELD_UNITS = (
    "unknown",              # 未声明(旧文件) —— 按标量处理
    "scalar",               # 就是一个场;单位由数字本身携带
    CANONICAL,              # 无量纲位系数 C_nm/S_nm(GRACE L2)
    "geoid",                # 水准面高 ΔN(米)
    "surface_density",      # 面密度 σ(kg/m²)
    "ewh",                  # 等效水高
    "radial_displacement",  # 弹性径向形变(米)
    "horizontal_displacement",  # v2.0:水平形变 u_h(矢量:北/东,米)
)

FIELD_UNIT_LABELS = {
    "unknown": "未声明",
    "scalar": "普通标量(无量纲或不作换算的场)",
    "geopotential": "无量纲重力位系数 C_nm/S_nm",
    "geoid": "水准面高 ΔN",
    "surface_density": "面密度 σ",
    "ewh": "等效水高 EWH",
    "radial_displacement": "径向形变 u_r",
    "horizontal_displacement": "水平形变 u_h(北/东)",
}

UNITS_NEEDING_LOVE_K = ("geoid", "surface_density", "ewh", "radial_displacement",
                        "horizontal_displacement")
UNITS_NEEDING_LOVE_H = ("radial_displacement",)
#: v2.0:需要载荷勒夫数 ``l′`` 的物理量(水平形变)。
UNITS_NEEDING_LOVE_L = ("horizontal_displacement",)
#: 矢量物理量:不能和标量互转(丢分量不可逆)。
VECTOR_UNITS = ("horizontal_displacement",)

_SHORT = {
    "geopotential": "无量纲位系数",
    "geoid": "水准面",
    "surface_density": "面密度",
    "ewh": "EWH",
    "radial_displacement": "径向形变",
    "horizontal_displacement": "水平形变",
    "scalar": "普通标量",
    "unknown": "未声明",
}

_FORMULAS = {
    "geopotential": "1(本身就是无量纲位系数场)",
    "scalar": "1(无物理公式,就是该场自身的系数)",
    "unknown": "1(未声明物理量,就是该场自身的系数)",
    "geoid": "R = 6378136.46 m(常数)",
    "surface_density": "R·ρ̄/3 · (2n+1)/(1+k′ₙ)",
    "ewh": "Aₙ = R·ρ̄/(3ρ_w) · (2n+1)/(1+k′ₙ)",
    "radial_displacement": "R·h′ₙ/(1+k′ₙ)",
    "horizontal_displacement": "R·l′ₙ/(1+k′ₙ)(北/东分量由球面梯度给出)",
}

#: 中文/常用别名 → 规范键(命令行与界面里都接受)。
_ALIASES = {
    "重力位": CANONICAL, "位系数": CANONICAL, "geopot": CANONICAL,
    "geopotential": CANONICAL, "potential": CANONICAL,
    "水准面": "geoid", "geoid": "geoid", "geoid_height": "geoid",
    "面密度": "surface_density", "sigma": "surface_density",
    "surface_density": "surface_density", "density": "surface_density",
    "等效水高": "ewh", "ewh": "ewh", "water": "ewh", "equivalent_water_height": "ewh",
    "径向形变": "radial_displacement", "位移": "radial_displacement",
    "radial_displacement": "radial_displacement", "ur": "radial_displacement",
    "u_r": "radial_displacement", "displacement": "radial_displacement",
    # v2.0:水平形变
    "水平形变": "horizontal_displacement", "水平位移": "horizontal_displacement",
    "horizontal_displacement": "horizontal_displacement",
    "horizontal": "horizontal_displacement", "uh": "horizontal_displacement",
    "u_h": "horizontal_displacement", "u_horizontal": "horizontal_displacement",
    "标量": "scalar", "scalar": "scalar", "普通标量": "scalar",
    "未声明": "unknown", "unknown": "unknown",
}

#: 「不换算」的各种写法。**不能**把它们映射到 geopotential ——
#: 对"未声明物理量"的系数要 geopotential 会被守卫拦住并报错,而用户的本意
#: 恰恰就是"什么都别换算";报错信息里还建议"把目标设为不换算",自相矛盾。
NO_CONVERSION_ALIASES = frozenset({
    "不换算", "不转换", "不进行换算", "原样", "none", "null", "no", "off",
    "asis", "as_is", "keep", "pass", "-", "", "无",
})


def is_no_conversion(text) -> bool:
    """``text`` 是不是"不换算"的意思(``None`` 也算)。"""
    if text is None:
        return True
    if isinstance(text, str):
        return text.strip().lower().replace(" ", "") in NO_CONVERSION_ALIASES
    return False


def normalise_unit(text: str) -> str:
    """把用户写的物理量名(英文键 / 中文 / 常用别名)规范成 :data:`FIELD_UNITS` 之一。

    "不换算"这类写法**不在**这里处理:它表示"不做任何换算",应当由调用方
    用 :func:`is_no_conversion` 判掉(换算函数里没有"不换算"这一个物理量)。
    """
    if text is None:
        return CANONICAL
    key = str(text).strip()
    if key in FIELD_UNITS:
        return key
    low = key.lower().replace("-", "_").replace(" ", "_")
    if low in _ALIASES:
        return _ALIASES[low]
    if low in FIELD_UNITS:
        return low
    extra = ""
    if is_no_conversion(text):
        extra = ("\n(『不换算』不是一种物理量:想表达「什么都不换算」时,"
                 "请把 target_unit 留空 / 传 None,而不要传字符串。)")
    raise ValueError(
        f"无法识别的物理量 {text!r};可选: "
        + "、".join(f"{k}({FIELD_UNIT_LABELS[k]})" for k in FIELD_UNITS) + extra)


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------
def field_unit(coeffs: SHCoeffs) -> str:
    """返回声明的 ``field_unit``,缺省 ``'unknown'``。

    大小写不敏感:文件头里写成 ``EWH`` / ``Geoid`` 也能认出来
    (以前会静默退化成 ``unknown``,于是"本来已经是 EWH 却要换算到 EWH"
    反而报错)。
    """
    raw = coeffs.meta.get("field_unit", "unknown")
    if raw is None:
        return "unknown"
    u = str(raw).strip()
    if u in FIELD_UNITS:
        return u
    low = u.lower().replace("-", "_").replace(" ", "_")
    if low in FIELD_UNITS:
        return low
    if low in _ALIASES:
        return _ALIASES[low]
    return "unknown"


def with_field_unit(coeffs: SHCoeffs, unit: str, **meta) -> SHCoeffs:
    """给系数打物理量标签(只改元数据)。"""
    unit = normalise_unit(unit)
    out = coeffs.copy()
    out.meta["field_unit"] = unit
    out.meta.update(meta)
    return out


def formula_text(unit: str) -> str:
    """该物理量正/反变换用的因子的人可读写法。"""
    return _FORMULAS.get(unit, f"(未知物理量 {unit!r})")


# ---------------------------------------------------------------------------
# 逐阶因子
# ---------------------------------------------------------------------------
def degree_factors(target: str, nmax: int, *,
                   love_numbers: Optional[Sequence[float]] = None,
                   love_numbers_h: Optional[Sequence[float]] = None,
                   love_numbers_l: Optional[Sequence[float]] = None,
                   radius_m: float = EARTH_RADIUS_M,
                   rho_ave: float = RHO_AVE,
                   rho_water: float = RHO_WATER) -> np.ndarray:
    """由 ``geopotential`` **换到** ``target`` 的逐阶因子 ``f_t``。

    Parameters
    ----------
    target : str
        :data:`FIELD_UNITS` 中除 ``'scalar'`` / ``'unknown'`` 之外的键。
    nmax : int
        最高阶。
    love_numbers : sequence, optional
        载荷勒夫数 ``k'_n``;默认用随包表(PREM / Wang 2012)。
    love_numbers_h : sequence, optional
        载荷勒夫数 ``h'_n``;默认用随包表。只有 ``radial_displacement`` 需要。
    love_numbers_l : sequence, optional
        载荷勒夫数 ``l'_n``;默认用随包表(v2.0)。只有
        ``horizontal_displacement`` 需要。
    """
    target = normalise_unit(target)
    n = np.arange(nmax + 1, dtype=float)

    if target == CANONICAL:
        return np.ones(nmax + 1)
    if target == "geoid":
        return np.full(nmax + 1, float(radius_m))
    if target not in FIELD_UNITS or target in ("scalar", "unknown"):
        raise ValueError(
            f"无法为 field_unit={target!r} 定义逐阶因子;可用目标为 "
            "'geopotential'、'geoid'、'surface_density'、'ewh'、"
            "'radial_displacement'")

    kl = load_love_numbers() if love_numbers is None else \
        np.asarray(love_numbers, dtype=float)
    kl = np.asarray(kl, dtype=float).ravel()
    if love_numbers is not None and kl.size < nmax + 1:
        # 表比阶数短时**不能**悄悄按 k'=0 算:那等于不做载荷改正,
        # EWH/面密度的因子会系统性偏差,却看不出任何异常。
        raise ValueError(
            f"你给的载荷勒夫数 k′ 表只有 {kl.size} 阶,而要算到 {nmax} 阶。"
            "缺失的阶按 k′=0 处理等于**不做载荷改正**,结果会系统性偏差;"
            "请给完整的表(0..nmax),或先用 shsynth.lovenumbers.load_lln() "
            "看随包表的阶数上限")
    kn = np.array([kl[i] if 0 <= i < kl.size else 0.0
                   for i in range(nmax + 1)])

    if target == "surface_density":
        # sigma_n = rho_ave * R * (2n+1) / (3 (1+k'_n)) * C_n     [kg/m^2]
        return radius_m * rho_ave / 3.0 * (2 * n + 1.0) / (1.0 + kn)

    if target == "ewh":
        # EWH_n = sigma_n / rho_w = R rho_ave/(3 rho_w) * (2n+1)/(1+k'_n) * C_n
        return (radius_m * rho_ave / (3.0 * rho_water)
                * (2 * n + 1.0) / (1.0 + kn))

    if target == "radial_displacement":
        if love_numbers_h is None:
            love_numbers_h = load_lln()["h"]
        hl = np.asarray(love_numbers_h, dtype=float).ravel()
        hn = np.array([hl[i] if 0 <= i < hl.size else 0.0
                       for i in range(nmax + 1)])
        # u_r,n = R h'_n/(1+k'_n) C_n ;向外为正,h'<0 时正载荷下沉
        return radius_m * hn / (1.0 + kn)

    if target == "horizontal_displacement":
        if love_numbers_l is None:
            love_numbers_l = load_lln()["l"]
        ll = np.asarray(love_numbers_l, dtype=float).ravel()
        if love_numbers_l is not None or nmax >= ll.size:      # 缺阶必须报错
            pass
        ln = np.array([ll[i] if 0 <= i < ll.size else 0.0
                       for i in range(nmax + 1)])
        # u_h,n = R l'_n/(1+k'_n) C_n —— 与径向同一结构,只换 h'→l'。
        # ⚠️ 不含 (2n+1)/3 或 ρ̄/ρ_w(那是 EWH/面密度的因子,乘错差 1e7 倍)。
        return radius_m * ln / (1.0 + kn)

    raise AssertionError("unreachable")


def forward_factors(src: str, nmax: int, **kw) -> np.ndarray:
    """**正变换**因子 ``f_u``:``C_nm = a_nm / f_u``(见 SHKit 方案说明)。"""
    src = normalise_unit(src)
    if src in ("scalar", "unknown"):
        return np.ones(nmax + 1)
    return degree_factors(src, nmax, **kw)


# ---------------------------------------------------------------------------
# 换算
# ---------------------------------------------------------------------------
def require_convertible(src: str, dst: str) -> Optional[str]:
    """可换算返回 ``None``,否则返回中文说明(为什么不能猜)。"""
    src = normalise_unit(src)
    dst = normalise_unit(dst)
    if src == dst:
        return None
    if src in ("scalar", "unknown"):
        extra = ""
        if dst == "ewh":
            extra = (
                "\n\n为什么不能替你猜:换算到 EWH 的因子取决于起点 —— "
                "位系数 → EWH 要乘 Aₙ(n=2 时 8.44e7),"
                "而 geoid → EWH 只要乘约 13.2,两者差整整一个地球半径 R。"
                "同一串数字,起点不同结果差 1e7 倍,所以必须由你说明。")
        return (
            f"源系数的物理含义是「{FIELD_UNIT_LABELS.get(src, src)}」,"
            f"没有定义到「{FIELD_UNIT_LABELS.get(dst, dst)}」的换算。\n"
            "「无物理含义的标量场」(例如区域平均核、任意无量纲网格)"
            "只代表它自己,没有已知的物理换算。"
            "要换算,请先用 field_unit 声明它到底是位系数、geoid 还是 EWH。"
            + extra +
            "\n\n如果你只是要做「网格 → 系数 → 网格」,"
            "那本来就不需要换算:把目标设为「不换算」即可。")
    if dst in ("scalar", "unknown"):
        return (
            f"不能把「{FIELD_UNIT_LABELS.get(src, src)}」换算成"
            f"「{FIELD_UNIT_LABELS.get(dst, dst)}」——"
            "「无物理含义的标量场」不是一个可换算的目标。")
    return None


ZERO_FACTOR_TOL = 1e-8


def _divide_by(coeffs: SHCoeffs, factors, what: str) -> tuple:
    """``coeffs / factors``,并处理**因子为 0** 的阶(目前只有径向形变的 0 阶)。

    因子为 0 的阶物理上不可反推,只能取 0;若该阶系数明显不为 0,说明它不符合
    该物理模型,报错而不是悄悄给出 inf/NaN。
    """
    f = np.asarray(factors, dtype=float).ravel()[: coeffs.nmax + 1]
    if not np.all(np.isfinite(f)):
        raise ValueError(
            f"「{what}」的换算因子含非有限值,无法换算"
            "(通常是载荷勒夫数表在该阶缺失)。")
    C = np.array(coeffs.C, dtype=float, copy=True)
    S = np.array(coeffs.S, dtype=float, copy=True)
    zero = np.flatnonzero(f == 0.0)
    bad = []
    if zero.size:
        scale = float(np.sqrt(np.mean(coeffs.C ** 2 + coeffs.S ** 2))) or 1.0
        for d in zero:
            d = int(d)
            if d >= C.shape[0]:
                continue
            content = max(float(np.max(np.abs(C[d, :]))),
                          float(np.max(np.abs(S[d, :]))))
            if content <= ZERO_FACTOR_TOL * scale:
                if C.ndim == 2:
                    C[d, :] = 0.0
                    S[d, :] = 0.0
                else:
                    C[d, :, :] = 0.0
                    S[d, :, :] = 0.0
            else:
                bad.append((d, content))
    if bad:
        lst = "、".join(f"第 {d} 阶(该阶系数已达 {v:.3g})" for d, v in bad)
        raise ValueError(
            f"无法完成「{what}」:{lst} 的换算因子为 0,信息在该阶丢失。\n"
            + ("径向形变的 0 阶载荷勒夫数 h′₀ = 0 —— 全球均匀载荷不产生"
               "(相对参考系定义的)径向位移,所以 u_r 的均值与载荷无关,"
               "不能由它反推位系数的 0 阶。\n"
               if "径向形变" in what else "")
            + "请改从位系数 / EWH / geoid 出发,或先用其它信息确定该阶后再合并。")
    finv = np.zeros_like(f)
    ok = f != 0.0
    finv[ok] = 1.0 / f[ok]
    if C.ndim == 2:
        C = C * finv[:, None]
        S = S * finv[:, None]
    else:
        C = C * finv[:, None, None]
        S = S * finv[:, None, None]
    return SHCoeffs(C, S, dict(coeffs.meta)), [int(d) for d in zero]


def _scale(coeffs: SHCoeffs, factors: np.ndarray) -> SHCoeffs:
    f = np.asarray(factors, dtype=float).ravel()[: coeffs.nmax + 1]
    if coeffs.C.ndim == 2:
        C = coeffs.C * f[:, None]
        S = coeffs.S * f[:, None]
    else:
        C = coeffs.C * f[:, None, None]
        S = coeffs.S * f[:, None, None]
    return SHCoeffs(C, S, dict(coeffs.meta))


def to_canonical(coeffs: SHCoeffs, src: Optional[str] = None, **kw) -> SHCoeffs:
    """**正变换**:「某物理量的场」的系数 → 经典无量纲位系数(``C_nm = a_nm/f_u``)。"""
    if src is None:
        src = field_unit(coeffs)
    src = normalise_unit(src)
    out, zeroed = _divide_by(
        coeffs, forward_factors(src, coeffs.nmax, **kw),
        f"从「{FIELD_UNIT_LABELS.get(src, src)}」正变换为位系数")
    out.meta = dict(out.meta)
    out.meta["field_unit"] = (src if src in ("scalar", "unknown") else CANONICAL)
    out.meta["forward_from"] = src
    if zeroed:
        out.meta["zero_degrees"] = zeroed
    return out


def convert(coeffs: SHCoeffs, target: str, *,
            love_numbers: Optional[Sequence[float]] = None,
            love_numbers_h: Optional[Sequence[float]] = None,
            radius_m: float = EARTH_RADIUS_M,
            rho_ave: float = RHO_AVE,
            rho_water: float = RHO_WATER,
            allow_unit_mismatch: bool = False) -> SHCoeffs:
    """把系数从它声明的物理量换算到 ``target``。

    所有换算都经由规范中间量(无量纲位系数),所以 ``ewh → geoid``、
    ``surface_density → ewh`` 也能算,尽管教科书只给了位系数方向的公式。

    ``allow_unit_mismatch=True`` 时,``unknown``/``scalar`` → ``geopotential``
    只**改标签、不动数值**(本来就是"假定这些数就是位系数")。以前这条放行分支
    其实是死代码——后面还会去调 ``degree_factors('unknown')`` 而抛错。
    """
    target = normalise_unit(target)
    src = field_unit(coeffs)

    problem = require_convertible(src, target)
    if problem is not None:
        if not (allow_unit_mismatch and src == "unknown" and target == CANONICAL):
            raise ValueError(problem)

    if src == target:
        return coeffs.copy()

    L = coeffs.nmax
    kw = dict(love_numbers=love_numbers, love_numbers_h=love_numbers_h,
              radius_m=radius_m, rho_ave=rho_ave, rho_water=rho_water)

    if src in ("scalar", "unknown") and target == CANONICAL:
        # 强行放行:数字原样保留,只把标签改成"无量纲位系数"
        out = coeffs.copy()
        out.meta = dict(out.meta)
        out.meta["field_unit"] = CANONICAL
        out.meta["converted_from"] = src
        out.meta["assumed_geopotential"] = True
        return out

    if src == CANONICAL:
        x = coeffs
    else:
        x, _zeroed = _divide_by(
            coeffs, degree_factors(src, L, **kw),
            f"从「{FIELD_UNIT_LABELS.get(src, src)}」反推")

    if target == CANONICAL:
        out = x
    else:
        out = _scale(x, degree_factors(target, L, **kw))

    out.meta = dict(out.meta)
    out.meta["field_unit"] = target
    out.meta["converted_from"] = src
    out.meta.pop("degree_filter", None)
    return out


def describe_conversion(nmax: int = 8, **kw) -> str:
    """逐阶因子的人可读表格(文档与自检用)。"""
    lines = [f"{'n':>3}  " + "".join(f"{_SHORT[t]:>16}" for t in
                                      ("geoid", "surface_density", "ewh"))]
    cols = {t: degree_factors(t, nmax, **kw)
            for t in ("geoid", "surface_density", "ewh")}
    for n in range(nmax + 1):
        lines.append(f"{n:>3}  " + "".join(
            f"{cols[t][n]:>16.6e}"
            for t in ("geoid", "surface_density", "ewh")))
    return "\n".join(lines)
