# -*- coding: utf-8 -*-
"""
shsynth.gui
===========

PySide6 桌面界面(启动方式见 :func:`shsynth.gui.app.run_gui`)。

依赖 **PySide6-Essentials**(LGPLv3,动态链接即可闭源商用)+ **matplotlib**
(BSD 风格)。**刻意不使用** GPL-only 的 Qt Charts / Qt Data Visualization。
"""

from __future__ import annotations

__all__ = ["run_gui", "MainWindow"]


def run_gui(argv=None, preload=None):
    """启动图形界面(延迟导入,命令行不装 PySide6 也能用)。"""
    from .app import run_gui as _run
    return _run(argv, preload=preload)


def __getattr__(name):                                   # PEP 562 懒加载
    if name == "MainWindow":
        from .app import MainWindow
        return MainWindow
    raise AttributeError(name)
