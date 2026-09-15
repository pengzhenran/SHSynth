# SHSynth 打包环境(出安装程序用)

> 详细版见 [`../docs/构建方法说明.md`](../docs/构建方法说明.md)。
> 这一页只回答"在这台机器上怎么一键出安装程序"。

终端用户装 `SHSynth_Setup_v1.0.exe` 即可,**不需要 Python**。
下面是**构建机**的一次性环境与打包步骤。

## 1. 极简环境(uv + Python 3.12)

```powershell
uv venv --python 3.12 D:\SHSynth_build\venv
uv pip install --python D:\SHSynth_build\venv\Scripts\python.exe `
    numpy scipy matplotlib PySide6-Essentials pandas xarray netCDF4 openpyxl pyinstaller
```

* 用 **uv** 而不是 conda:秒级建好、默认不带 pip/测试框架,天然"极简";
* NumPy/SciPy 走 **PyPI(OpenBLAS)**:Anaconda 的 MKL 版会把目录包从
  几百 MB 撑到 1.3 GB;
* 界面只装 **PySide6-Essentials**,避免 GPL-only 的 Qt Charts / Qt Data
  Visualization 混进来(闭源分发红线)。

装完确认:

```powershell
D:\SHSynth_build\venv\Scripts\python.exe ..\tools\check_licensing.py
```

实测环境:Python 3.12.13 / numpy 2.5.3 / scipy 1.18.1 / matplotlib 3.11.2 /
PySide6-Essentials 6.11.2 / PyInstaller 6.22.3 / Inno Setup 6。

## 2. 一键打包

```powershell
cd <项目>\SHSynth
powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -VerifyInstall
```

产物:

```
D:\SHSynth_build\dist\SHSynth\                 目录包(绿色版)
D:\SHSynth_build\dist\SHSynth_Setup_v1.0.exe   安装程序
D:\SHSynth_installer\                          交付目录(安装程序 + 绿色版 zip)
D:\SHSynth_build\logs\selftest.log             冻结版自检日志
```

## 3. 环境 / 工具要求

| 项目 | 版本 / 位置 |
| --- | --- |
| Python | 3.12(`D:\SHSynth_build\venv`,uv 管理) |
| PySide6 | 6.11.2(PySide6-Essentials,pip/uv) |
| PyInstaller | 6.22.x |
| Inno Setup | 6,`D:\Inno Setup 6\ISCC.exe`(脚本会自动找几个常见位置) |
| 中文语言文件 | `<Inno Setup>\Languages\ChineseSimplified.isl`;仓库自带精简版,脚本会自动装过去 |
| 图标 | `SHSynth.ico`(由 `tools/make_icon.py` 生成,不需要 Pillow) |

## 4. 常见参数

| 参数 | 作用 |
| --- | --- |
| `-SkipBuild` | 复用已有目录包,只重编安装程序 |
| `-SkipInstaller` | 只出目录包(绿色版) |
| `-VerifyInstall` | 额外安装-校验-卸载一遍 |
| `-KeepConsole` | 出带控制台窗口的 exe(排错) |
| `-Python` / `-BuildRoot` / `-FinalDir` / `-Iscc` | 换解释器 / 构建根 / 交付目录 / ISCC 路径 |

## 5. 为什么中间产物放在项目外

项目在网盘同步目录里:同步驱动会锁文件(PyInstaller 清理时报
`PermissionError: WinError 32`),而且 HDF5 在该网络盘上写不出 netCDF
(`Errno 13`)。所以 `build/`、`dist/`、`logs/` 一律放 `D:\SHSynth_build\`。
