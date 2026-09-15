# -*- coding: utf-8 -*-
"""SHSynth 打包入口(PyInstaller 用)。

为什么要单独一个入口文件
------------------------
PyInstaller 会把**入口脚本**当顶层脚本执行(``__name__ == "__main__"``、没有父包)。
如果直接拿 ``shsynth/gui/app.py`` 当入口,它里面的相对导入(``from .. import ...``)
就会炸::

    ImportError: attempted relative import with no known parent package

所以入口放在包外,先 ``import shsynth.*``(包被正常导入,相对导入全部成立)再调用。
用户直接 ``python packaging/shsynth_launcher.py`` 也能用。

支持的命令行
------------
``SHSynth.exe``                     打开图形界面
``SHSynth.exe <系数文件>``           打开界面并预载该系数文件
``SHSynth.exe --self-test``         跑冻结版自检并以退出码表态(打包脚本用它)
``SHSynth.exe --version``           打印版本与作者信息
``SHSynth.exe --guide``             在**窗口里**打开使用说明
``SHSynth.exe --guide-browser``     用系统浏览器打开同一份 HTML 说明书
``SHSynth.exe --cli <子命令…>``      转给命令行接口(synth/info/convert/…)

另外这里把 matplotlib 的配置目录指到临时目录:冻结版若去写
``%USERPROFILE%\\.matplotlib``,在只读或受管环境里会报错,而本软件并不需要保存
matplotlib 配置。
"""

from __future__ import annotations

import os
import sys
import tempfile

__all__ = ["main"]

_HELP = """\
SHSynth —— 球谐系数解算(综合)

用法:
  SHSynth.exe                        打开图形界面
  SHSynth.exe 系数文件               打开界面并预载该系数文件
  SHSynth.exe --self-test            自检(冻结版关键路径验证)
  SHSynth.exe --version              显示版本与作者
  SHSynth.exe --guide                在窗口里打开使用说明
  SHSynth.exe --guide-browser        用系统浏览器打开同一份说明书
  SHSynth.exe --cli <子命令> [参数]   命令行模式,例如:
        SHSynth.exe --cli synth --coeffs model.sh --global-grid 1 --out out.nc
        SHSynth.exe --cli info  --coeffs model.sh --spectrum
"""


def _isolate_matplotlib_config() -> None:
    """把 matplotlib 配置目录指到临时目录(在它被导入之前设置)。"""
    tmp = os.path.join(tempfile.gettempdir(), "shsynth-mplconfig")
    try:
        os.makedirs(tmp, exist_ok=True)
    except OSError:
        return
    os.environ.setdefault("MPLCONFIGDIR", tmp)


#: 控制台方式启动的命令行开关(这些模式必须有可用的 stdout)
_CONSOLE_MODES = ("--cli", "--self-test", "--version", "-V", "-h", "--help", "/?")


def _stdout_is_usable() -> bool:
    """Windows 上判断标准输出句柄是不是真的能写。

    ⚠️ 发行版是 **windowed**(``console=False``)exe:双击开 GUI。但用户也会在
    cmd/PowerShell 里跑 ``SHSynth.exe --cli …`` —— 那时进程**没有**接上父控制台,
    ``sys.stdout`` 是一个非 None 但**无效**的句柄,``print()`` 会抛

        OSError: [Errno 22] Invalid argument

    把整条命令带走(计算还没开始/结果还没落盘就崩了)。这里先用
    ``GetFileType`` 判一下:无效句柄返回 ``FILE_TYPE_UNKNOWN``(0)。
    """
    if os.name != "nt":
        return True
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)                      # STD_OUTPUT_HANDLE
        if not h or h == ctypes.c_void_p(-1).value:
            return False
        return bool(k.GetFileType(ctypes.c_void_p(h)))
    except Exception:                                # noqa: BLE001
        return True                                  # 判断不了就不动它


def _reopen_console_streams() -> None:
    """把 ``sys.stdout``/``sys.stderr`` 重绑到 ``CONOUT$``。

    接上父控制台之后**必须重开**:原来的 ``sys.stdout`` 还绑在接之前的
    (无效)句柄上,继续用照样 EINVAL。
    编码按**控制台当前代码页**选(中文 Windows 是 cp936),不擅自改用户的
    代码页 —— 否则把 UTF-8 字节塞进 GBK 控制台会变成乱码。
    """
    import ctypes
    try:
        cp = int(ctypes.windll.kernel32.GetConsoleOutputCP())
    except Exception:                                # noqa: BLE001
        cp = 0
    enc = "utf-8" if cp in (0, 65001) else f"cp{cp}"
    for name in ("stdout", "stderr"):
        try:
            setattr(sys, name, open("CONOUT$", "w", encoding=enc,
                                    errors="replace", buffering=1))
        except OSError:
            pass


def _setup_console() -> None:
    """让 ``--cli``/``--self-test`` 这类模式在命令行里能用。

    顺序:能用就直接用;不能就 ``AttachConsole(父进程)``;再不行就放着 ——
    真正的兜底在 :func:`shsynth.cli._out`(写不出去也**不能让计算失败**)。
    """
    if os.name != "nt":
        return
    if _stdout_is_usable():
        return
    try:
        import ctypes
        if not ctypes.windll.kernel32.AttachConsole(-1):   # ATTACH_PARENT_PROCESS
            return                                   # 真没有父控制台(双击启动)
    except Exception:                                # noqa: BLE001
        return
    _reopen_console_streams()


def _ensure_importable() -> None:
    """源码运行时把项目根加进 sys.path;冻结后这一步无害(只是兜底)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (here, os.path.dirname(here)):
        if cand not in sys.path and os.path.isdir(os.path.join(cand, "shsynth")):
            sys.path.insert(0, cand)
            break


def main(argv=None) -> int:
    _isolate_matplotlib_config()
    _ensure_importable()

    argv = list(sys.argv[1:] if argv is None else argv)

    # windowed exe 从命令行跑 --cli 时先把父控制台接回来,否则 print() 会
    # 直接 OSError: [Errno 22] Invalid argument(见 _stdout_is_usable 的说明)。
    if argv and argv[0] in _CONSOLE_MODES:
        _setup_console()

    if argv and argv[0] in ("-h", "--help", "/?"):
        print(_HELP)
        return 0
    if argv and argv[0] in ("-V", "--version"):
        from shsynth import author
        print(author.about_text())
        return 0
    if argv and argv[0] == "--self-test":
        from shsynth.selftest import run_self_test
        return run_self_test(verbose=True, keep_files="--keep" in argv)
    if argv and argv[0] == "--guide":
        # 在**窗口里**看说明书(与「帮助 → 使用说明」同一个窗口),不丢给浏览器
        from PySide6.QtWidgets import QApplication
        from shsynth.gui.app import _apply_cjk_font
        from shsynth.gui.docs_window import open_guide
        app = QApplication.instance() or QApplication(sys.argv)
        _apply_cjk_font(app)
        return 0 if open_guide() else 1
    if argv and argv[0] == "--guide-browser":
        # 备用:直接用系统默认浏览器打开 HTML(受管机器上偶尔需要)
        from shsynth import author
        path = author.guide_html_path()
        if not path:
            print("未找到随包的说明书文件。")
            return 1
        os.startfile(path)                               # noqa: S606 (Windows)
        return 0
    if argv and argv[0] == "--cli":
        from shsynth.cli import main as cli_main
        return int(cli_main(argv[1:]))

    # 默认:图形界面(可带一个系数文件作为预载)
    preload = argv[0] if argv and not argv[0].startswith("-") else None
    if preload and not os.path.exists(preload):
        print(f"警告:找不到系数文件 {preload};将只打开界面。")
        preload = None
    from shsynth.gui.app import run_gui
    return int(run_gui(argv, preload=preload))


if __name__ == "__main__":
    sys.exit(main())
