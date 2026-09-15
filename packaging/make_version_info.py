# -*- coding: utf-8 -*-
"""
packaging/make_version_info.py
==============================

生成 PyInstaller 用的版本资源 ``packaging/version_info.txt``。

版本号、作者、单位都从 :mod:`shsynth.author` 与 :mod:`shsynth` 的 ``__version__``
取,避免"exe 属性里的版本"和代码里的版本对不上。

    python packaging/make_version_info.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TEMPLATE = """\
# UTF-8
#
# PyInstaller 版本资源 —— 决定 SHSynth.exe 的「属性 → 详细信息」里显示什么,
# 也是杀软/用户判断"这是谁发的软件"的依据。pyinstaller 用 --version-file 读它。
#
# **本文件由 packaging/make_version_info.py 自动生成,别手改**:
#   改版本号/作者请改 shsynth/__init__.py 与 shsynth/author.py,再重新生成。

VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={vers4},
    prodvers={vers4},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        '080404B0',
        [StringStruct('CompanyName', '{company}'),
        StringStruct('FileDescription', '{app} —— {app_cn}'),
        StringStruct('FileVersion', '{vers4d}'),
        StringStruct('InternalName', '{app}'),
        StringStruct('LegalCopyright', '© {year} {author_cn} {author_en}. 本软件代码 MIT 许可;界面 PySide6(LGPLv3)。'),
        StringStruct('OriginalFilename', '{app}.exe'),
        StringStruct('ProductName', '{app} {app_cn}'),
        StringStruct('ProductVersion', '{vers3}'),
        StringStruct('Comments', '{tagline};适配 SHKit 全部系数输出格式'),
        StringStruct('LegalTrademarks', 'Qt 及相关商标归 The Qt Company Ltd. 所有'),
        StringStruct('Author', '{author_cn} {author_en}'),
        StringStruct('Contact', '{email}')])
      ]),
    VarFileInfo([VarStruct('Translation', [0x0804, 1200])])
  ]
)
"""


def main() -> int:
    import datetime

    import shsynth
    from shsynth import author

    vers = shsynth.__version__
    parts = [int(p) for p in vers.split(".") if p.isdigit()]
    while len(parts) < 4:
        parts.append(0)
    f = dict(
        vers4=tuple(parts[:4]),
        vers4d=".".join(str(p) for p in parts[:4]),
        vers3=".".join(str(p) for p in parts[:3]),
        company=f"{author.AUTHOR_AFFILIATION_CN} {author.AUTHOR_AFFILIATION_EN}",
        app=author.APP_NAME,
        app_cn=author.APP_NAME_CN,
        tagline=author.APP_TAGLINE,
        author_cn=author.AUTHOR_NAME_CN,
        author_en=author.AUTHOR_NAME_EN,
        email=author.AUTHOR_EMAIL,
        year=datetime.date.today().year,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "version_info.txt")
    text = TEMPLATE.format(**f)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"已生成 {out}(版本 {f['vers4d']},作者 {f['author_cn']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
