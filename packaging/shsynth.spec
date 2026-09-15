# -*- mode: python ; coding: utf-8 -*-
"""
packaging/shsynth.spec
======================

PyInstaller 打包配置(闭源分发合规版)。

    pyinstaller --noconfirm --clean packaging/shsynth.spec

刻意做的几个选择(与同组 SHKit 的 packaging/shkit.spec 一致):

* **onedir,绝不用 onefile。** PySide6/Qt 是 LGPLv3:许可要求终端用户能替换该库。
  ``--onedir`` 下 Qt 的 DLL 作为独立文件躺在 exe 旁边,可以替换;``--onefile``
  会把它们压进归档再解到临时目录,属于社区公认的 relinking 风险。
* **主动排除 GPL-only 的 Qt 模块**,即使打包环境里误装了完整 PySide6
  (会带上 PySide6-Addons → Qt Charts / Qt Data Visualization / Qt Graphs)
  也不会漏进发行包。``tools/check_licensing.py`` 会复核这件事。
* ``licenses/`` 与 ``docs/`` 必须随包:LGPLv3 要求随附许可全文与显著声明;
  ``docs/`` 里还有帮助菜单运行时读的 ``使用说明.html``、配图与公众号二维码,
  丢了就会出现"帮助菜单打开是空的"。
* 入口是 ``packaging/shsynth_launcher.py``,**不是** ``shsynth/gui/app.py`` ——
  PyInstaller 把入口当顶层脚本执行,后者里的相对导入会 ImportError。
"""

import os

PROJECT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# 需要看启动期报错时:先 set SHSYNTH_CONSOLE=1 再打包 → 会出带控制台窗口的 exe
CONSOLE = os.environ.get("SHSYNTH_CONSOLE", "") not in ("", "0", "false")

# --- 绝不进包的模块 ---------------------------------------------------------
# 1) 其它 Qt 绑定:PyInstaller 一旦同时发现 PySide6 与 PyQt5/PyQt6 会直接中止;
#    发行包只用 PySide6,这几条是保险。
# 2) 用不上的重型/无关依赖:让目录包小一点、启动快一点。
ALWAYS_EXCLUDE = [
    "PyQt5", "PyQt5.sip", "PyQt6", "PyQt6.sip", "qtpy", "PySide2",
    "IPython", "jupyter", "notebook", "nbformat", "nbconvert",
    "pytest", "_pytest", "sphinx", "docutils",
    "tkinter", "_tkinter", "matplotlib.backends._tkagg",
    "matplotlib.tests", "numpy.tests", "scipy.tests", "pandas.tests",
    "h5py", "pyarrow", "dask", "numba", "sqlalchemy",
]

# --- GPL-only 的 Qt 模块:任何情况下都不发行 --------------------------------
GPL_ONLY = [
    "PySide6.QtCharts", "PySide6.QtChartsQml",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs", "PySide6.QtGraphsWidgets",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtQuick3D", "PySide6.QtQuick3DUtils",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtSpatialAudio",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtTest", "PySide6.QtDesigner", "PySide6.QtHelp",
]

# --- 随包数据 ---------------------------------------------------------------
# (源路径, 包内目标目录)
#
# ⚠️ 只放**使用者需要**的东西:说明书、发行说明、许可文本、运行数据。
#    开发者文档(构建方法说明、许可与第三方组件、命令与参数、物理量与换算)、
#    测试、打包脚本一律**不进安装包** —— 安装包是给第三方使用者用的。
_USER_DOCS = ("使用说明.html", "使用说明.md", "发行说明.md")

datas = [
    (os.path.join(PROJECT, "shsynth", "data"), "shsynth/data"),  # 勒夫数表+海岸线+二维码
    (os.path.join(PROJECT, "licenses"), "licenses"),             # LGPL/GPL/NOTICE(必需)
]
for name in _USER_DOCS:
    p = os.path.join(PROJECT, "docs", name)
    if os.path.exists(p):
        datas.append((p, "docs"))
_img_dir = os.path.join(PROJECT, "docs", "使用说明_img")
if os.path.isdir(_img_dir):
    datas.append((_img_dir, "docs/使用说明_img"))

# --- 图标与版本资源 ---------------------------------------------------------
icon = os.path.join(PROJECT, "SHSynth.ico")
if not os.path.exists(icon):
    icon = None

version_file = os.path.join(SPECPATH, "version_info.txt")
if not os.path.exists(version_file):
    version_file = None

a = Analysis(
    [os.path.join(SPECPATH, "shsynth_launcher.py")],
    pathex=[PROJECT],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "shsynth", "shsynth.cli", "shsynth.selftest", "shsynth.plotting",
        "shsynth.gui", "shsynth.gui.app", "shsynth.gui.canvases",
        "shsynth.gui.workers", "shsynth.author",
        # 可选 IO 后端:这些是延迟导入,PyInstaller 的静态扫描看不到
        "xarray", "netCDF4", "pandas", "openpyxl",
        "scipy.special", "scipy.spatial",
        # v2.0 动画:GIF 用 Pillow 编码(函数内延迟导入)。imageio 故意**不列** ——
        # 本机没装,MP4 会走"明确报错并建议改用 .gif"那条路,不静默降级。
        "PIL", "PIL.Image",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=ALWAYS_EXCLUDE + GPL_ONLY,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SHSynth",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # UPX 会改 DLL、杀软误报多;体积够小就不压
    console=CONSOLE,                # 默认无控制台;排错时 set SHSYNTH_CONSOLE=1
    disable_windowed_traceback=False,
    icon=icon,
    version=version_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SHSynth",
)
