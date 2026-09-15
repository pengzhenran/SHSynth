# -*- coding: utf-8 -*-
"""
tools/check_licensing.py
========================

闭源分发前的许可自检(在**正式打包用的那个环境**里跑一次)。

检查三件事:

1. 环境里有没有 GPL-only 的 Qt 模块(Qt Charts / Qt Data Visualization /
   Qt Graphs / Qt Virtual Keyboard …)—— 有的话闭源分发就不可能了;
2. 可用的是不是那套 LGPLv3 模块(QtCore / QtGui / QtWidgets …);
3. 发行包里该带的许可文本在不在(``licenses/LGPL-3.0.txt``、``GPL-3.0.txt``、
   ``NOTICE.txt``),以及 PySide6 的 wheel 是否**没有**自带 LGPL 文本
   (实测没有,所以必须自己提供)。

    python tools/check_licensing.py
"""

from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GPL_ONLY = ["QtCharts", "QtDataVisualization", "QtGraphs", "QtVirtualKeyboard",
            "QtQuick3D", "QtWebEngineCore", "QtWebEngineWidgets"]
LGPL_OK = ["QtCore", "QtGui", "QtWidgets", "QtSvg", "QtNetwork",
           "QtPrintSupport", "QtSql", "QtXml", "QtConcurrent", "QtOpenGL",
           "QtUiTools", "QtTest", "QtQml", "QtQuick", "QtHelp"]

REQUIRED_FILES = [
    "LICENSE.txt",
    "licenses/LGPL-3.0.txt",
    "licenses/GPL-3.0.txt",
    "licenses/NOTICE.txt",
]

#: 我们自己代码里绝不能出现的 GPL-only 导入
FORBIDDEN_SOURCE = ["PySide6.QtCharts", "PySide6.QtDataVisualization",
                    "PySide6.QtGraphs", "PySide6.QtVirtualKeyboard"]


def check_qt() -> int:
    print("=" * 68)
    print("1. Qt 模块许可检查")
    print("=" * 68)
    try:
        import PySide6
    except ImportError:
        print("  PySide6 未安装(只跑命令行不需要它)—— 跳过")
        return 0
    print(f"  PySide6 版本: {PySide6.__version__}")

    bad = []
    for m in GPL_ONLY:
        try:
            importlib.import_module(f"PySide6.{m}")
            bad.append(m)
        except ImportError:
            pass
    ok = 0
    for m in LGPL_OK:
        try:
            importlib.import_module(f"PySide6.{m}")
            ok += 1
        except ImportError:
            pass
    print(f"  可用的 LGPL 模块: {ok}/{len(LGPL_OK)}")
    if bad:
        print(f"  [失败] 检测到 GPL-only 模块: {bad}")
        print("         请执行:pip uninstall PySide6 PySide6-Addons")
        print("                 pip install PySide6-Essentials")
        return 1
    print("  [通过] 未检测到 GPL-only 模块 -> 当前环境可用于闭源分发")

    # PySide6 的 wheel 是否自带 LGPL 文本
    dist = os.path.join(os.path.dirname(PySide6.__file__),
                        "..", "PySide6_Essentials-%s.dist-info"
                        % PySide6.__version__)
    if os.path.isdir(dist):
        names = os.listdir(dist)
        has_lgpl = any("LGPL" in n.upper() for n in names)
        print(f"  PySide6 的 dist-info 内容: {names}")
        print("  [提示] wheel 自带 LGPL 文本:" , has_lgpl,
              "(实测为 False,所以必须由本项目 licenses/ 提供)")
    return 0


def check_source() -> int:
    print()
    print("=" * 68)
    print("2. 源码中是否出现 GPL-only 导入")
    print("=" * 68)
    hits = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "shsynth")):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            for pat in FORBIDDEN_SOURCE:
                if pat in text:
                    hits.append((os.path.relpath(p, ROOT), pat))
    if hits:
        print("  [失败] 发现 GPL-only 导入:")
        for rel, pat in hits:
            print(f"         {rel}: {pat}")
        return 1
    print("  [通过] shsynth/ 下没有出现任何 GPL-only 导入")
    return 0


def check_files() -> int:
    print()
    print("=" * 68)
    print("3. 必备许可文本")
    print("=" * 68)
    rc = 0
    for rel in REQUIRED_FILES:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if os.path.exists(p) and os.path.getsize(p) > 200:
            print(f"  [通过] {rel}({os.path.getsize(p)} 字节)")
        else:
            print(f"  [失败] 缺少 {rel}")
            rc = 1
    return rc


def main() -> int:
    print(f"SHSynth 许可自检   (项目目录 {ROOT})")
    print()
    rc = check_qt()
    rc |= check_source()
    rc |= check_files()
    print()
    print("=" * 68)
    if rc:
        print("结果:有问题,请看上面的 [失败] 条目")
    else:
        print("结果:全部通过 —— 当前状态可用于闭源分发")
        print("      别忘了发行包里要带上 licenses/ 整个目录,以及")
        print("      用 PyInstaller --onedir(不是 --onefile)打包。")
    print("=" * 68)
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
