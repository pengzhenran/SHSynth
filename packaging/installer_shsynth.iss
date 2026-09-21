; =============================================================================
;  Inno Setup script —— SHSynth v2.0(球谐系数解算)
; =============================================================================
;  用法(推荐直接跑一键脚本,它会把路径按实际构建根目录注入):
;
;      powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
;
;  或者手工两步:
;    1) 先出 PyInstaller 目录包(在 SHSynth\ 下执行):
;         pyinstaller --clean --noconfirm packaging\shsynth.spec
;       → D:\SHSynth_build\dist\SHSynth\SHSynth.exe + _internal\
;    2) 再编译本脚本:
;         ISCC.exe /DProjDir=<项目目录> /DBuildRoot=<构建根> installer_shsynth.iss
;       → <构建根>\dist\SHSynth_Setup_v2.0.exe
;
;  发行包内容:
;    SHSynth.exe + _internal\   运行所需的全部组件(Python 运行时、Qt、绘图库…)
;    _internal\docs\            使用说明.html + 配图 + 公众号二维码 + 截图
;    _internal\licenses\        LGPLv3 / GPLv3 / NOTICE
;    source\shsynth\            软件运行代码(Python 包),便于用户核对与复现
;    source\tools\ tests\ examples\ packaging\ + pyproject/requirements
; =============================================================================

; 路径与版本:允许命令行用 /D 覆盖(避免中文路径 + 相对路径解析的坑)
#ifndef ProjDir
  #define ProjDir "D:\SHSynth_build\project"
#endif
#ifndef BuildRoot
  #define BuildRoot "D:\SHSynth_build"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "2.0.1"
#endif

#define MyAppName       "SHSynth"
#define MyAppNameCN     "球谐系数解算"
#define MyAppPublisher  "彭桢燃  Zhenran Peng  (China University of Geosciences, Wuhan)"
#define MyAppURL        "https://www.cug.edu.cn"
#define MyAppExeName    "SHSynth.exe"
#define SourceDir       BuildRoot + "\dist\SHSynth"
#define OutputDir       BuildRoot + "\dist"
#define IconFile        ProjDir + "\SHSynth.ico"

[Setup]
; 每次发布换一个 GUID:Inno Setup 菜单 Tools -> Generate GUID
AppId={{B3D7A621-58C4-4E9F-A1D2-6F0C9E4B7A55}
AppName={#MyAppName} {#MyAppNameCN}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppNameCN} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; 默认按用户安装(不弹管理员):装到 %LOCALAPPDATA%\Programs\SHSynth
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SHSynth_Setup_v{#MyAppVersion}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
ShowLanguageDialog=auto
; 安装前请用户关掉正在运行的 SHSynth,避免文件占用装不上
CloseApplications=yes
RestartApplications=no

[Languages]
; Inno Setup 官方安装包**不带**简体中文语言文件,所以:
;   · 构建脚本会把 packaging\languages\ChineseSimplified.isl 复制到
;     <Inno Setup>\Languages\(仓库里带了一份精简版,版本兼容、可复现);
;   · 只有真找到该文件时才加中文,否则退回英文 —— 用 /DHaveChinese=1 控制。
#ifdef HaveChinese
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#endif
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式 / Create a &desktop shortcut"; GroupDescription: "附加任务 / Additional shortcuts:"; Flags: checkedonce

[Files]
; --- 程序本体(PyInstaller onedir 输出) -------------------------------------
Source: "{#SourceDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; --- 只装**使用者需要**的东西 -------------------------------------------------
; 说明书(HTML + 配图 + Markdown 版)、发行说明都在 _internal\docs\(由 spec 带进去);
; 许可文本在 _internal\licenses\。
; **不装**开发者内容:源码、测试、打包脚本、构建说明、许可合规文档 ——
; 安装包是给第三方使用者用的,不是给开发者的。
Source: "{#ProjDir}\LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName} {#MyAppNameCN}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\使用说明"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--guide"
; 快捷方式名里不能有 "/":Inno 会把 "/" 当子目录分隔符去建目录(建出带尾随
; 空格的 ``作者信息 ``),却把 "/" 原样留在文件名里,于是
; ``IPersistFile::Save failed; code 0x80070003``(系统找不到指定的路径),
; 安装到"创建快捷方式"这一步就失败回滚。要子目录请用 "\"。
Name: "{group}\作者信息与关于"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--version"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName} {#MyAppNameCN}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\{#MyAppExeName}"; Parameters: "--guide"; Description: "打开使用说明"; Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
; 用户自己产生的输出目录不删;只清掉可能残留的日志/缓存
Type: filesandordirs; Name: "{app}\_internal\__pycache__"
