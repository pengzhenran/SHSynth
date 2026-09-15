; *** Inno Setup 简体中文语言文件(精简版)***
;
; 为什么是"精简版":Inno Setup 的语言文件是**覆盖**在 Default.isl 之上的 ——
; 只列出需要的条目即可,其余自动用英文默认值。这样既不依赖 Inno 的具体版本
; (完整版会引用新版本才有的消息名,在旧编译器上会报警告),又好维护。
;
; 覆盖范围:安装/卸载向导的主流程(欢迎、选目录、附加任务、准备安装、安装中、
; 完成、卸载确认、按钮、开始菜单/桌面快捷方式、磁盘空间、错误提示)。
;
; 编码:UTF-8 with BOM;LanguageCodePage=0 表示按 Unicode 处理。
; 安装位置:<Inno Setup>\Languages\ChineseSimplified.isl
; 本仓库副本:packaging\languages\ChineseSimplified.isl(构建脚本会自动装过去)

[LangOptions]
LanguageName=简体中文
LanguageID=$0804
LanguageCodePage=0
DialogFontName=Microsoft YaHei
DialogFontSize=9

[Messages]

; *** 程序标题
SetupAppTitle=安装
SetupWindowTitle=安装 - %1
UninstallAppTitle=卸载
UninstallAppFullTitle=%1 卸载

; *** 通用
InformationTitle=信息
ConfirmTitle=确认
ErrorTitle=错误

; *** 启动相关
SetupLdrStartupMessage=现在将安装 %1。是否继续？
SetupAlreadyRunning=安装程序已在运行。
WindowsVersionNotSupported=本程序不支持当前 Windows 版本。
AdminPrivilegesRequired=安装本程序需要以管理员身份登录。
SetupAppRunningError=安装程序检测到 %1 正在运行。%n%n请先关闭它,然后点击「确定」继续,或点击「取消」退出。
UninstallAppRunningError=卸载程序检测到 %1 正在运行。%n%n请先关闭它,然后点击「确定」继续,或点击「取消」退出。

; *** 退出
ExitSetupTitle=退出安装
ExitSetupMessage=安装尚未完成。如果现在退出,程序不会被安装。%n%n之后可以重新运行安装程序。%n%n现在退出吗?

; *** 按钮
ButtonBack=< 上一步(&B)
ButtonNext=下一步(&N) >
ButtonInstall=安装(&I)
ButtonOK=确定
ButtonCancel=取消
ButtonYes=是(&Y)
ButtonNo=否(&N)
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&B)...
ButtonWizardBrowse=浏览(&R)...

; *** 选择语言
SelectLanguageTitle=选择安装语言
SelectLanguageLabel=选择安装过程中使用的语言。

; *** 欢迎页
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=即将在您的计算机上安装 [name/ver]。%n%n建议在继续之前关闭其它正在运行的程序。

; *** 许可页
WizardLicense=许可协议
LicenseLabel=请在继续安装前阅读以下重要信息。
LicenseLabel3=请阅读下列许可协议。继续安装前必须接受这些条款。
LicenseAccepted=我接受此协议(&A)
LicenseNotAccepted=我不接受(&D)

; *** 选择目标位置
WizardSelectDir=选择安装位置
SelectDirDesc=要将 [name] 安装到哪里?
SelectDirLabel3=安装程序将把 [name] 安装到下面的文件夹。
SelectDirBrowseLabel=点击「下一步」继续;要换位置请点击「浏览」。
DiskSpaceGBLabel=至少需要 [gb] GB 可用磁盘空间。
DiskSpaceMBLabel=至少需要 [mb] MB 可用磁盘空间。
CannotInstallToNetworkDrive=无法安装到网络驱动器。
InvalidPath=请输入带盘符的完整路径,例如:%n%nC:\App
InvalidDrive=所选驱动器或 UNC 共享不存在或无法访问,请另选。
DiskSpaceWarningTitle=磁盘空间不足
DiskSpaceWarning=至少需要 %1 KB 可用空间,但所选驱动器只有 %2 KB。%n%n仍要继续吗?
DirExistsTitle=文件夹已存在
DirExists=文件夹:%n%n%1%n%n已经存在。要安装到这个文件夹吗?
DirDoesntExistTitle=文件夹不存在
DirDoesntExist=文件夹:%n%n%1%n%n不存在。要创建它吗?

; *** 附加任务
WizardSelectTasks=选择附加任务
SelectTasksDesc=要执行哪些附加任务?
SelectTasksLabel2=选择安装 [name] 时要执行的附加任务,然后点击「下一步」。

; *** 开始菜单
WizardSelectProgramGroup=选择开始菜单文件夹
SelectStartMenuFolderDesc=要在哪里放置程序快捷方式?
SelectStartMenuFolderLabel3=安装程序将在下列「开始」菜单文件夹中创建快捷方式。
SelectStartMenuFolderBrowseLabel=点击「下一步」继续;要换文件夹请点击「浏览」。
NoProgramGroupCheck2=不创建开始菜单文件夹(&D)

; *** 准备安装
WizardReady=准备安装
ReadyLabel1=安装程序已准备好,现在开始安装 [name]。
ReadyLabel2a=点击「安装」继续;想复查或修改设置就点击「上一步」。
ReadyMemoDir=安装位置:
ReadyMemoType=安装类型:
ReadyMemoComponents=已选组件:
ReadyMemoGroup=开始菜单文件夹:
ReadyMemoTasks=附加任务:

; *** 正在准备 / 安装中
WizardPreparing=正在准备安装
PreparingDesc=安装程序正在准备安装 [name]。
CannotContinue=安装程序无法继续,请点击「取消」退出。
CloseApplications=自动关闭这些程序(&A)
DontCloseApplications=不关闭这些程序(&D)
WizardInstalling=正在安装
InstallingLabel=正在把 [name] 安装到您的计算机,请稍候。
StatusCreateDirs=正在创建目录...
StatusExtractFiles=正在解压文件...
StatusCreateIcons=正在创建快捷方式...
StatusCreateRegistryEntries=正在写入注册表...
StatusSavingUninstall=正在保存卸载信息...
StatusRunProgram=正在完成安装...
StatusRollback=正在撤销更改...

; *** 安装完成
FinishedHeadingLabel=完成 [name] 安装向导
FinishedLabelNoIcons=安装程序已在您的计算机上安装 [name]。
FinishedLabel=安装程序已在您的计算机上安装 [name],可以通过创建的快捷方式运行它。
ClickFinish=点击「完成」退出安装程序。
RunEntryExec=运行 %1
RunEntryShellExec=打开 %1

; *** 错误
SetupAborted=安装未完成。%n%n请修正问题后重新运行安装程序。
ErrorCreatingDir=安装程序无法创建目录「%1」
ErrorInternal2=内部错误:%1。
ErrorExecutingProgram=无法执行文件:%n%1

; *** 卸载
ConfirmUninstall=确定要完全移除 %1 及其所有组件吗?
UninstallStatusLabel=正在从您的计算机上移除 %1,请稍候。
UninstalledAll=已从您的计算机上成功移除 %1。
UninstalledMost=%1 已卸载。%n%n有少量内容未能删除,可以手动清理。
UninstalledAndNeedsRestart=完成 %1 的卸载需要重启计算机。%n%n现在重启吗?
WizardUninstalling=卸载状态
StatusUninstalling=正在卸载 %1...
UninstallDataCorrupted=文件「%1」已损坏,无法卸载。

; *** 卸载显示名
UninstallDisplayNameMark=%1 (%2)
UninstallDisplayNameMark32Bit=32 位
UninstallDisplayNameMark64Bit=64 位

; *** 自定义消息(我们自己 .iss 里用到的)
[CustomMessages]
NameAndVersion=%1 版本 %2
AdditionalIcons=附加快捷方式:
CreateDesktopIcon=创建桌面快捷方式(&D)
UninstallProgram=卸载 %1
LaunchProgram=运行 %1
