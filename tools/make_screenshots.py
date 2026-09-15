# -*- coding: utf-8 -*-
"""
tools/make_screenshots.py
=========================

生成说明书与 README 里用的界面截图。

为什么单独做这个工具(而不是在测试里顺手截)
--------------------------------------------
用 ``QT_QPA_PLATFORM=offscreen`` 跑界面时,Qt 的离屏插件**不带字体库**
(启动时会警告 ``QFontDatabase: Cannot find font directory …/PySide6/lib/fonts``),
于是所有控件文字都渲染成**方框** —— 截出来的图不能用。所以本工具:

1. 优先用**真实平台**(Windows 的 ``windows`` 平台),字体来自系统 → 中文正常;
2. 如果真实平台起不来(无桌面会话 / CI),退回 offscreen,但**显式把系统中文
   字体加进 Qt 字体库**(``QFontDatabase.addApplicationFont``),并设成应用字体;
3. 截完图会打印每张图的路径与尺寸,并用"文字是否渲染成方框"的**像素检查**兜底:
   对方框图形来说,同一行文字的黑色像素分布高度规则;对真实汉字则不规则。
   这一项不通过就报错,避免把烂图交出去。

    python tools/make_screenshots.py                 # 截全部
    python tools/make_screenshots.py --allow-offscreen   # 允许离屏兜底
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "使用说明_img")

#: 系统里常见的中文字体文件(第一条存在的就用)
_CJK_FONT_FILES = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体
    r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    r"C:\Windows\Fonts\Deng.ttf",      # 等线
)


def _pick_coeffs() -> str:
    """优先用 SHKit 的样例系数;没有就用 examples 里的。"""
    base = os.path.dirname(ROOT)
    for cand in (os.path.join(base, "SHKit", "shkit_coeffs.sh"),
                 os.path.join(ROOT, "examples", "out", "demo_coeffs.sh")):
        if os.path.exists(cand):
            return cand
    return ""


def _force_cjk_font(app) -> str | None:
    """把系统中文装进 Qt 字体库并设为应用字体(offscreen 下必需)。"""
    from PySide6.QtGui import QFont, QFontDatabase
    added = None
    for path in _CJK_FONT_FILES:
        if os.path.exists(path):
            fid = QFontDatabase.addApplicationFont(path)
            if fid != -1:
                fams = QFontDatabase.applicationFontFamilies(fid)
                if fams:
                    added = fams[0]
                    f = QFont(added)
                    f.setPointSize(9)
                    app.setFont(f)
                    break
    if added is None:
        # 退一步:直接问 Qt 有哪些中文字体族
        from shsynth.gui.app import _CJK_FONTS
        fams = set(QFontDatabase.families())
        for name in _CJK_FONTS:
            if name in fams:
                f = QFont(name)
                f.setPointSize(9)
                app.setFont(f)
                added = name
                break
    return added


def _looks_like_boxes(pixmap, box=None) -> bool:
    """粗略判断一小块区域里的文字是不是"方框":方框是等宽等高的实心矩形。

    做法:统计该区域的黑色像素行轮廓。真实汉字的笔画分布不均匀(行与行之间
    差异大);方框字符每行的黑像素数几乎相同。
    """
    import numpy as np
    from PySide6.QtCore import QRect
    img = pixmap.toImage()
    w, h = img.width(), img.height()
    if w < 8 or h < 8:
        return False
    r = box or QRect(int(w * 0.03), int(h * 0.05), int(w * 0.22), int(h * 0.30))
    rows = []
    for y in range(r.top(), min(r.bottom(), h)):
        dark = 0
        for x in range(r.left(), min(r.right(), w)):
            c = img.pixelColor(x, y)
            if c.lightness() < 128:
                dark += 1
        rows.append(dark)
    arr = np.asarray(rows, dtype=float)
    ink = arr[arr > 0]
    if ink.size < 4:
        return False                      # 没有文字 → 不算方框
    # 方框:有墨的行里,黑像素数几乎一致(变异系数极小)
    cv = float(ink.std() / max(ink.mean(), 1e-9))
    return cv < 0.06


def main(argv=None) -> int:
    sys.path.insert(0, ROOT)          # 直接跑脚本时也能 import shsynth
    ap = argparse.ArgumentParser(description="生成界面截图")
    ap.add_argument("--allow-offscreen", action="store_true",
                    help="没有桌面会话时允许用离屏平台(并显式加载中文字体)")
    ap.add_argument("--coeffs", default=None, help="用于截图的系数文件")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    coeffs = args.coeffs or _pick_coeffs()
    if not coeffs:
        print("找不到可用的系数样例文件(需要 ../SHKit/shkit_coeffs.sh)")
        return 1
    print(f"系数文件: {coeffs}")

    # ---- 选平台 -----------------------------------------------------------
    platform_note = "真实平台(windows)"
    os.environ.pop("QT_QPA_PLATFORM", None)
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    if app.platformName() == "offscreen":
        platform_note = "离屏平台"
    real_ok = app.platformName() != "offscreen"
    if not real_ok and not args.allow_offscreen:
        print("当前平台是 offscreen(截出来的中文会是方框)。"
              "请在桌面会话里运行,或加 --allow-offscreen 强制继续。")
        return 1
    font = _force_cjk_font(app)
    print(f"平台: {platform_note}(Qt 报 {app.platformName()});"
          f" 中文字体: {font or '未显式指定'}")

    from shsynth.gui.app import MainWindow, AboutDialog
    from shsynth.gui.docs_window import GuideDialog

    win = MainWindow()
    win.resize(1440, 900)
    win.coeffs_edit.setText(coeffs)
    win.load_coeffs_info()
    _wait(app, lambda: win._info_thread is None and win._coeffs is not None, 60)
    win.show()
    app.processEvents()

    saved = []

    def shot(name: str, widget=None, check_box: bool = True) -> str:
        app.processEvents()
        time.sleep(0.35)
        app.processEvents()
        w = widget or win
        pm = w.grab()
        path = os.path.join(OUT_DIR, name)
        pm.save(path, "PNG")
        note = ""
        if check_box and _looks_like_boxes(pm):
            note = "  ⚠️ 疑似文字方框(请检查字体)"
        print(f"  {name}  {pm.width()}x{pm.height()}  "
              f"{os.path.getsize(path) // 1024} KB{note}")
        saved.append((path, note))
        return path

    # 1) 主界面(空参数,干净)
    shot("screenshot_gui.png")

    # 2) 跑一次区域解算 → 报告图页签
    win.mode_radios["range"].setChecked(True)
    win.lat_min.setValue(20.0)
    win.lat_max.setValue(50.0)
    win.lon_min.setValue(100.0)
    win.lon_max.setValue(140.0)
    win.lat_step.setValue(0.5)
    win.lon_step.setValue(0.5)
    win.figkind_combo.setCurrentIndex(0)              # 报告图
    win.out_edit.setText("")
    win.figfile_edit.setText("")
    win.start_run()
    _wait(app, lambda: win._thread is None, 300)
    app.processEvents()
    shot("screenshot_report.png")
    win.tabs.setCurrentWidget(win.map_canvas)
    shot("screenshot_map.png")
    win.tabs.setCurrentWidget(win.spectrum_canvas)
    shot("screenshot_spectrum.png")

    # 3) 说明书窗口
    guide = GuideDialog(win)
    guide.resize(1080, 780)
    guide.show()
    app.processEvents()
    time.sleep(0.4)
    app.processEvents()
    shot("screenshot_guide.png", widget=guide)
    guide.close()

    # 4) 关于 / 作者信息
    about = AboutDialog(win)
    about.resize(920, 840)
    about.show()
    app.processEvents()
    time.sleep(0.4)
    app.processEvents()
    shot("screenshot_about.png", widget=about)
    about.close()

    win.close()
    bad = [p for p, n in saved if n]
    print(f"\n共写出 {len(saved)} 张到 {OUT_DIR}")
    if bad:
        print("有截图疑似文字渲染成方框,请检查字体后重跑:")
        for p in bad:
            print("  " + os.path.basename(p))
        return 2
    return 0


def _wait(app, cond, timeout_s: float):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return bool(cond())


if __name__ == "__main__":
    sys.exit(main())
