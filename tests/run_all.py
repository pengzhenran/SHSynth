# -*- coding: utf-8 -*-
"""
tests/run_all.py
================

一次跑完全部测试并汇总。

::

    python tests/run_all.py            # 全部 8 套
    python tests/run_all.py --fast     # 跳过界面与跨软件比对(最快)
    python tests/run_all.py --only test_engine.py test_formats.py

每套测试都是独立脚本:自己打印 **PASS/FAIL** 明细,最后返回 0/1。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: (脚本, 说明, 是否属于"慢"套件)
SUITES = [
    ("test_formats.py", "SHKit 全部系数格式的读写与识别", False),
    ("test_engine.py", "综合引擎 + 解析物理校验", False),
    ("test_units_filters.py", "物理量换算与高斯平滑", False),
    ("test_gridfiles.py", "网格文件形式 / 聚焦 / 列识别", False),
    ("test_workflow_cli.py", "流程 / 输出格式 / 命令行", False),
    ("test_multitime.py", "v2.0:时间轴/序列/FFT/水平形变", False),
    ("test_packaging.py", "打包脚本:BOM / Inno 引号 / 含空格路径", False),
    ("test_vs_shkit.py", "与 SHKit 逐位比对", True),
    ("test_gui_smoke.py", "图形界面离屏冒烟", True),
]


def _python() -> str:
    return sys.executable or "python"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    fast = "--fast" in argv
    only = None
    if "--only" in argv:
        i = argv.index("--only")
        only = [a for a in argv[i + 1:] if not a.startswith("-")]

    print("=" * 74)
    print(f"SHSynth 测试总入口   ({'快速模式' if fast else '完整模式'})")
    print(f"Python: {_python()}")
    print("=" * 74)

    results = []
    t_all = time.time()
    for name, desc, slow in SUITES:
        if only and name not in only:
            continue
        if fast and slow:
            print(f"\n>>> 跳过 {name}({desc};--fast)")
            results.append((name, desc, None, 0.0))
            continue
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            print(f"\n>>> 缺失 {name}")
            results.append((name, desc, 127, 0.0))
            continue
        print(f"\n>>> 运行 {name} —— {desc}")
        t0 = time.time()
        proc = subprocess.run([_python(), path], cwd=ROOT)
        dt = time.time() - t0
        results.append((name, desc, proc.returncode, dt))

    # ------------------------------------------------------------- 汇总
    print("\n" + "=" * 74)
    print("汇总")
    print("=" * 74)
    n_fail = 0
    for name, desc, code, dt in results:
        if code is None:
            tag = "跳过"
        elif code == 0:
            tag = "通过"
        else:
            tag = "失败"
            n_fail += 1
        print(f"  [{tag}] {name:<24} {desc:<34} {dt:6.1f}s")
    print("-" * 74)
    print(f"总耗时 {time.time() - t_all:.1f}s")
    if n_fail:
        print(f"结果: {n_fail} 套失败")
        return 1
    print("结果: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
