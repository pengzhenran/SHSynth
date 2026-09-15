# -*- coding: utf-8 -*-
"""
tools/make_icon.py
==================

生成 SHSynth 的图标(``SHSynth.ico`` 多尺寸 + ``docs/SHSynth_icon_preview.png``)。

设计:一个球面 + 球谐波(带谐 Y₂₀ 与扇谐 Y₄₄ 的叠加)的"经线投影"图案 ——
一眼能看出"球面 + 波动",而且和软件做的事(把球谐系数画到球面上)对得上。
配色沿用界面里那张图的色系(深蓝 → 白 → 暗红),保证和软件风格一致。

**不依赖 Pillow**:用 matplotlib 画好各尺寸的 PNG,再按 ICO 容器格式手工打包
(Vista 以后 .ico 允许直接内嵌 PNG 帧)。这样打包环境里不需要额外装 Pillow。

    python tools/make_icon.py                    # 默认写到项目根 + docs/
    python tools/make_icon.py --out D:\\SHKit_build\\SHSynth.ico
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

#: .ico 里包含的尺寸(Windows 会按需挑;256 给大图标视图)
SIZES = (256, 128, 64, 48, 32, 24, 16)


def _harmonic_field(n: int = 260) -> np.ndarray:
    """造一个"球谐样"的标量场(只用于取色)。

    刻意只用**两三个**低阶谐波:图标在 16×16 下也要认得出,所以宁可结构清楚
    (两三个大瓣 + 一条平滑零线),不要真实重力场那种细碎纹理。
    这里用 ``cosφ·cosλ`` 的分瓣项 + 低阶带谐项,组合出 2 红 2 蓝的分布。
    """
    lat = np.linspace(-90.0, 90.0, n)
    lon = np.linspace(-180.0, 180.0, 2 * n)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    phi = np.deg2rad(LA)
    lam = np.deg2rad(LO)
    c, s = np.cos(phi), np.sin(phi)
    # 扇谐(左右分瓣) + 带谐(上下分带) + m=2 项 —— 组合出三个大瓣,
    # 零线是弯的,一眼能看出"球谐"而不是普通球面贴图
    f = (0.95 * c * np.cos(lam)
         + 0.80 * (1.5 * s ** 2 - 0.5)
         + 0.45 * c ** 2 * np.cos(2 * lam))
    f -= f.mean()
    return f / (np.abs(f).max() + 1e-30)


def _project(phi_deg, lam_deg):
    """正射投影:返回 ``(X, Y)`` 平面坐标(球半径为 1,视线沿 +x)。"""
    phi = np.deg2rad(phi_deg)
    lam = np.deg2rad(lam_deg)
    return np.cos(phi) * np.sin(lam), np.sin(phi)


def _draw(size: int, cmap: str = "RdBu_r") -> np.ndarray:
    """画一张 ``size×size`` 的 RGBA 图标,返回 ``(size, size, 4)`` uint8。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(-1.10, 1.10)
    ax.set_ylim(-1.10, 1.10)
    ax.set_aspect("equal")

    # ---- 球面上的异常场(正射投影) --------------------------------------
    f = _harmonic_field(n=320)
    lat = np.linspace(-90.0, 90.0, f.shape[0])
    lon = np.linspace(-180.0, 180.0, f.shape[1])
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    X, Y = _project(LA, LO)
    R = np.hypot(X, Y)
    F = np.where(R <= 1.0, f, np.nan)                       # 圆外留白
    lim = float(np.nanmax(np.abs(F))) or 1.0
    ax.pcolormesh(X, Y, F, cmap=cmap, vmin=-lim, vmax=lim,
                  shading="auto", rasterized=True, zorder=1)

    # ---- 经纬网(点明"这是球") -----------------------------------------
    if size >= 32:
        style = dict(color="#12304d", linewidth=max(size / 340.0, 0.4),
                     alpha=0.38, zorder=2)
        for lon0 in (-120.0, -60.0, 0.0, 60.0, 120.0):
            la = np.linspace(-90.0, 90.0, 181)
            x, y = _project(la, np.full_like(la, lon0))
            ax.plot(x, y, **style)
        for lat0 in (-60.0, -30.0, 0.0, 30.0, 60.0):
            lo = np.linspace(-180.0, 180.0, 361)
            x, y = _project(np.full_like(lo, lat0), lo)
            ax.plot(x, y, **style)
    elif size >= 20:
        th = np.linspace(0, 2 * np.pi, 181)
        ax.plot(np.cos(th), np.zeros_like(th), color="#12304d",
                linewidth=0.5, alpha=0.45, zorder=2)

    # ---- 球面边缘:一圈深色描边,任何背景上都立得住 ---------------------
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(np.cos(th), np.sin(th), color="#12233a",
            linewidth=max(size / 26.0, 1.2), zorder=3, solid_capstyle="round")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return buf


def write_ico(path: str, frames: list) -> None:
    """把若干 PNG 帧写成一个 .ico(Vista+ 允许 PNG 帧,无需 BMP 转换)。"""
    n = len(frames)
    header = struct.pack("<HHH", 0, 1, n)          # reserved, type=icon, count
    entries, blobs = b"", []
    offset = 6 + 16 * n
    for size, png in frames:
        w = 0 if size >= 256 else size             # 256 在 ICO 里记作 0
        h = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32,
                               len(png), offset)
        blobs.append(png)
        offset += len(png)
    with open(path, "wb") as fh:
        fh.write(header + entries + b"".join(blobs))


def _png_bytes(rgba: np.ndarray) -> bytes:
    """把 RGBA 数组编码成 PNG 字节(matplotlib 自带编码器,不需要 Pillow)。"""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format="png")
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 SHSynth 图标")
    ap.add_argument("--out", default=None, metavar="FILE",
                    help="输出 .ico 路径(默认 <项目>/SHSynth.ico)")
    ap.add_argument("--preview", default=None, metavar="PNG",
                    help="预览 PNG 路径(默认 <项目>/docs/SHSynth_icon_preview.png)")
    ap.add_argument("--cmap", default="RdBu_r")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(root, "SHSynth.ico")
    preview = args.preview or os.path.join(root, "docs",
                                           "SHSynth_icon_preview.png")
    for d in (os.path.dirname(os.path.abspath(out)),
              os.path.dirname(os.path.abspath(preview))):
        if d:
            os.makedirs(d, exist_ok=True)

    frames = []
    for size in SIZES:
        rgba = _draw(size, cmap=args.cmap)
        frames.append((size, _png_bytes(rgba)))
        print(f"  已绘制 {size}×{size}")
    write_ico(out, frames)
    print(f"图标已写出: {out}({os.path.getsize(out):,} 字节,"
          f"{len(frames)} 个尺寸)")

    # 预览图:256 图标 + 小尺寸并排,方便肉眼确认小尺寸下还看得清
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    big = _draw(256, cmap=args.cmap)
    fig = plt.figure(figsize=(6.4, 2.4), dpi=100, facecolor="white")
    ax0 = fig.add_axes([0.02, 0.05, 0.30, 0.9])
    ax0.imshow(big)
    ax0.set_axis_off()
    ax0.set_title("256 px", fontsize=9)
    ax1 = fig.add_axes([0.35, 0.05, 0.63, 0.9])
    xs = 0.0
    for s in (64, 48, 32, 24, 16):
        ax1.imshow(_draw(s, cmap=args.cmap),
                   extent=(xs, xs + s, 0, s))
        ax1.text(xs + s / 2, -6, f"{s}", ha="center", fontsize=8)
        xs += s + 8
    ax1.set_xlim(-4, xs)
    ax1.set_ylim(-14, 70)
    ax1.set_aspect("equal")
    ax1.set_axis_off()
    ax1.set_title("readability at small sizes", fontsize=9)
    fig.savefig(preview, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"预览图已写出: {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
