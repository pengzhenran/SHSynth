# -*- coding: utf-8 -*-
"""
test_packaging.py —— 打包脚本的"静默陷阱"自检(第 9 套)
==========================================================================

盯的是这几条**破坏了也不报错**的约定(详见 ``tools/check_packaging.py``):

* ``build_installer.ps1`` / ``installer_shsynth.iss`` 必须是 UTF-8 **with BOM**
  —— Windows PowerShell 5.1 与 ISCC 都会按系统 ANSI 读无 BOM 文件,中文变乱码后
  报的是"看起来毫不相干"的语法错误;
* ``[Run]`` / ``[Icons]`` 的 ``Filename`` 不许自带引号(Inno 直接拒绝编译);
* 安装校验必须用**含空格**的路径,并且真的从那儿跑一次 ``--cli``。

跑法:``python tests/test_packaging.py``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import Checker                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_packaging as CP                                   # noqa: E402


def test_packaging_scripts(check: Checker):
    check.section("打包脚本(BOM / Inno 引号 / 含空格校验路径)")
    results = CP.collect()
    check.ok(bool(results), f"检查项跑出来了({len(results)} 项)")
    for good, msg in results:
        check.ok(good, msg)

    # 这几条是"最容易被无声破坏"的,单独点名断言一遍,信息更直白
    check.ok(CP.has_bom(CP.PS1),
             "build_installer.ps1 带 BOM(编辑它之后别忘了补回来)")
    check.ok(CP.has_bom(CP.ISS), "installer_shsynth.iss 带 BOM")
    check.ok(not CP._filename_with_own_quotes(CP.read_text(CP.ISS)),
             "Inno 的 Filename 没有多余引号")
    txt = CP.read_text(CP.PS1)
    check.ok("SHSynth verify" in txt,
             "安装校验路径确实是含空格的那个(不是 %TEMP%\\SHSynth_verify)")


def main() -> int:
    check = Checker("test_packaging —— 打包脚本静默陷阱")
    try:
        test_packaging_scripts(check)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        check.ok(False, f"test_packaging_scripts 抛异常: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
