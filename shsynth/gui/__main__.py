# -*- coding: utf-8 -*-
"""
``python -m shsynth.gui [系数文件]`` 的入口。
"""

from __future__ import annotations

import sys

from .app import run_gui

if __name__ == "__main__":
    preload = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(run_gui(preload=preload))
