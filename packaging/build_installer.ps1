# =============================================================================
#  SHSynth 一键出安装程序
# =============================================================================
#  做四件事:
#    1. 用极简虚拟环境里的 PyInstaller 按 packaging\shsynth.spec 打 onedir 目录包
#       → <BuildRoot>\dist\SHSynth\(入口 packaging\shsynth_launcher.py,不是 gui/app.py)
#    2. 校验目录包:说明书/配图/二维码/许可是否都在;启动一次 exe;
#       再跑一遍**冻结版自检** SHSynth.exe --self-test
#    3. Inno Setup 编译 packaging\installer_shsynth.iss
#       → <BuildRoot>\dist\SHSynth_Setup_v2.0.exe
#    4. (-VerifyInstall)静默安装到临时目录 → 校验文件 → 跑自检 → 启动一次 → 卸载
#
#  用法:
#    powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
#    powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -SkipBuild
#    powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -VerifyInstall
#
#  注意:本脚本必须存成 **UTF-8 with BOM**(Windows PowerShell 5.1 会把无 BOM 的
#  UTF-8 当 GBK 读,中文注释会乱码并报解析错误)。
# =============================================================================
[CmdletBinding()]
param(
    [string]$Python    = "D:\SHSynth_build\venv\Scripts\python.exe",
    [string]$BuildRoot = "D:\SHSynth_build",
    [string]$Iscc      = "",
    [string]$FinalDir  = "D:\SHSynth_installer",
    [switch]$SkipBuild,
    [switch]$SkipInstaller,
    [switch]$VerifyInstall,
    [switch]$KeepConsole
)

$ErrorActionPreference = "Stop"
$Project   = Split-Path -Parent $PSScriptRoot          # …\SHSynth
$DistRoot  = Join-Path $BuildRoot "dist"
$Dist      = Join-Path $DistRoot "SHSynth"
$Work      = Join-Path $BuildRoot "build"
$LogDir    = Join-Path $BuildRoot "logs"
$Version   = "2.0.1"

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-OK($msg)   { Write-Host "    [OK]  " -NoNewline -ForegroundColor Green; Write-Host $msg }
function Write-Warn2($msg){ Write-Host "    [!]   " -NoNewline -ForegroundColor Yellow; Write-Host $msg }
function Fail($msg)       { Write-Host "    [FAIL] " -NoNewline -ForegroundColor Red; Write-Host $msg; exit 1 }

function Find-Iscc {
    param([string]$Hint)
    $cands = @()
    if ($Hint) { $cands += $Hint }
    $cands += @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "D:\Inno Setup 6\ISCC.exe",
        "E:\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Invoke-Quiet([string]$Exe, [string[]]$Args) {
    # 启动一个 GUI exe 并等它退出;返回退出码
    $p = Start-Process -FilePath $Exe -ArgumentList $Args -PassThru -Wait
    return $p.ExitCode
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SHSynth —— 打包出安装程序" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  项目目录 : $Project"
Write-Host "  构建根   : $BuildRoot"
Write-Host "  最终目录 : $FinalDir"
Write-Host ""

New-Item -ItemType Directory -Force -Path $DistRoot, $Work, $LogDir, $FinalDir | Out-Null

# ── 0. 环境与版本资源 ────────────────────────────────────────────────────────
Write-Step "[0/4]  检查打包环境 …"
if (-not (Test-Path $Python)) { Fail "找不到打包用的 Python: $Python(先按 packaging\BUILD_ENV.md 建极简环境)" }
$pyVersion = & $Python -c "import sys;print(sys.version.split()[0])"
Write-OK "Python $pyVersion"
foreach ($mod in @("PyInstaller", "PySide6", "matplotlib", "numpy", "scipy", "xarray", "netCDF4", "pandas")) {
    & $Python -c "import $mod" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "缺少 $mod(相关功能不可用)" } 
}
& $Python (Join-Path $Project "packaging\make_version_info.py") | Out-Null
Write-OK "版本资源已刷新(packaging\version_info.txt)"

# GPL-only 模块自检:装了完整 PySide6 就必须拦下来
$lic = & $Python (Join-Path $Project "tools\check_licensing.py") 2>&1
if ($LASTEXITCODE -ne 0) {
    $lic | Select-Object -Last 12 | ForEach-Object { Write-Host "      $_" }
    Fail "许可自检未通过(见上);闭源分发前必须修掉"
}
Write-OK "许可自检通过(无 GPL-only Qt 模块)"

# 说明书一致性:md 与 html 必须同步、配图不能多也不能少。
# (改完 .md 忘了跑 tools\make_help_html.py 就打包,包里的 HTML 会是旧的)
& $Python (Join-Path $Project "tools\check_docs_ready.py")
if ($LASTEXITCODE -ne 0) { Fail "说明书不一致:先跑 python tools\make_help_html.py" }
Write-OK "说明书 md/html 一致、配图齐全"

# 打包脚本自检:本 .ps1 与 .iss 的 **BOM**、PowerShell 语法、Inno 的 Filename
# 引号约定、以及"安装校验路径必须含空格"。这几条破坏后都不会报错,只会**静默**
# 出问题(实测:无 BOM → Windows PowerShell 把中文当 GBK 读,报假语法错误;
# 校验路径没空格 → 从来没验过含空格路径)。
& $Python (Join-Path $Project "tools\check_packaging.py")
if ($LASTEXITCODE -ne 0) { Fail "打包脚本自检未通过(见上)" }
Write-OK "打包脚本自检通过(BOM / 语法 / 含空格校验路径)"

# ── 1. PyInstaller ───────────────────────────────────────────────────────────
if ($SkipBuild) {
    Write-Step "[1/4]  跳过打包(-SkipBuild),复用已有目录包"
} else {
    Write-Step "[1/4]  PyInstaller 打目录包 …"
    if (Test-Path $Dist) { Remove-Item -Recurse -Force $Dist -ErrorAction SilentlyContinue }
    Push-Location $Project
    try {
        if ($KeepConsole) { $env:SHSYNTH_CONSOLE = "1" } else { Remove-Item Env:\SHSYNTH_CONSOLE -ErrorAction SilentlyContinue }
        & $Python -m PyInstaller --noconfirm --clean --distpath $DistRoot `
            --workpath $Work --log-level WARN `
            (Join-Path $Project "packaging\shsynth.spec")
        if ($LASTEXITCODE -ne 0) { Fail "PyInstaller 失败(退出码 $LASTEXITCODE)" }
    } finally { Pop-Location }
}
if (-not (Test-Path (Join-Path $Dist "SHSynth.exe"))) { Fail "没有生成 $Dist\SHSynth.exe" }
$sz = [math]::Round((Get-ChildItem $Dist -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-OK "目录包 $Dist($sz MB)"

# ── 2. 校验目录包 + 冻结版自检 ───────────────────────────────────────────────
Write-Step "[2/4]  校验目录包 …"
$must = @(
    "SHSynth.exe",
    "_internal\shsynth\data\load_love_numbers.npz",
    "_internal\shsynth\data\coastline_110m.npz",
    "_internal\shsynth\data\地球重力与人类生活TVGG.jpg",
    "_internal\docs\使用说明.html",
    "_internal\docs\使用说明.md",
    "_internal\docs\发行说明.md",
    "_internal\docs\使用说明_img\screenshot_gui.png",
    "_internal\licenses\LGPL-3.0.txt",
    "_internal\licenses\GPL-3.0.txt",
    "_internal\licenses\NOTICE.txt"
)
$bad = 0
foreach ($rel in $must) {
    if (Test-Path (Join-Path $Dist $rel)) { Write-OK $rel } else { Write-Warn2 "缺少 $rel"; $bad++ }
}
if ($bad -gt 0) { Fail "$bad 个必需文件缺失,拒绝出包" }

# 安装包里**不应有**开发者内容:说明书只放给第三方使用者看的东西
$forbidden = @(
    "_internal\docs\构建方法说明.md",
    "_internal\docs\许可与第三方组件.md",
    "_internal\docs\命令与参数.md",
    "_internal\docs\物理量与换算.md",
    "source",
    "README.md"
)
foreach ($rel in $forbidden) {
    if (Test-Path (Join-Path $Dist $rel)) {
        Fail "安装包里混入了开发者内容:$rel(说明书只应包含使用者需要的文档)"
    }
}
Write-OK "未混入开发者文档 / 源码(安装包面向第三方使用者)"

# 说明书里引用的配图必须真的在包里(否则帮助菜单打开是空图)
$htmlPath = Join-Path $Dist "_internal\docs\使用说明.html"
if (Test-Path $htmlPath) {
    $html = Get-Content $htmlPath -Raw -Encoding UTF8
    $refs = [regex]::Matches($html, 'src="([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
    $miss = @($refs | Where-Object { $_ -notmatch '^https?://' -and
               -not (Test-Path (Join-Path (Join-Path $Dist "_internal\docs") $_)) })
    if ($miss.Count -gt 0) {
        $miss | Select-Object -First 5 | ForEach-Object { Write-Warn2 "说明书缺图: $_" }
        Fail "说明书引用了 $($miss.Count) 个不存在的图片"
    }
    Write-OK "说明书配图完整($($refs.Count) 个引用)"
}

# 冻结版自检:进程活着不等于没崩,必须让它自己走关键路径并表态
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$stOut = Join-Path $LogDir "selftest.log"
$stErr = Join-Path $LogDir "selftest.err"
$exe = Join-Path $Dist "SHSynth.exe"
$p = Start-Process -FilePath $exe -ArgumentList "--self-test" -PassThru -Wait `
        -RedirectStandardOutput $stOut -RedirectStandardError $stErr
$lines = @(Get-Content $stOut -Encoding UTF8 -ErrorAction SilentlyContinue)
$passed = @($lines | Where-Object { $_ -match '^\[PASS\]' }).Count
$failed = @($lines | Where-Object { $_ -match '^\[FAIL\]' })
Write-Host "      自检:$passed 项通过,$($failed.Count) 项失败(退出码 $($p.ExitCode))"
if ($failed.Count -gt 0) {
    $failed | Select-Object -First 10 | ForEach-Object { Write-Warn2 $_ }
    Fail "冻结版自检未通过;完整日志:$stOut"
}
if ($passed -lt 20) {
    Fail "自检日志里只有 $passed 条 [PASS](应当 50 项上下)—— 说明自检没真正跑完,见 $stOut"
}
if ($p.ExitCode -ne 0) { Fail "自检退出码 $($p.ExitCode)(见 $stOut)" }
Write-OK "冻结版自检通过($passed 项)"

# 双击式启动(不等它退出,2 秒后关掉)——
# 只有 windowed 的 exe 才需要这一步:确认它能起来而不是一闪就没
Write-Step "      启动一次 exe(3 秒后关闭)…"
$p2 = Start-Process -FilePath $exe -PassThru
Start-Sleep -Seconds 3
if ($p2.HasExited) { Fail "exe 启动后立刻退出(退出码 $($p2.ExitCode))—— 多半是导入期崩了" }
Stop-Process -Id $p2.Id -Force -ErrorAction SilentlyContinue
Write-OK "exe 能正常启动并保持运行"

# ── 3. Inno Setup ────────────────────────────────────────────────────────────
if ($SkipInstaller) {
    Write-Step "[3/4]  跳过安装程序(-SkipInstaller);绿色版在 $Dist"
} else {
    Write-Step "[3/4]  编译安装程序 …"
    $iscc = Find-Iscc -Hint $Iscc
    if (-not $iscc) { Fail "找不到 ISCC.exe;装一个 Inno Setup 6 或用 -Iscc 指定路径" }
    Write-OK "ISCC: $iscc"
    # 中文语言文件:Inno 官方包不带,仓库里带了一份精简版,缺就装过去
    $langDir = Join-Path (Split-Path -Parent $iscc) "Languages"
    $langTarget = Join-Path $langDir "ChineseSimplified.isl"
    $langRepo = Join-Path $Project "packaging\languages\ChineseSimplified.isl"
    if (-not (Test-Path $langTarget) -and (Test-Path $langRepo)) {
        New-Item -ItemType Directory -Force -Path $langDir | Out-Null
        Copy-Item $langRepo $langTarget -Force
        Write-OK "已装中文语言文件 → $langTarget"
    }
    if (Test-Path $langTarget) {
        $haveChinese = "/DHaveChinese=1"
        Write-OK "中文语言文件已就位(安装向导会显示中文)"
    } else {
        $haveChinese = "/DHaveChinese=0"
        Write-Warn2 "没有中文语言文件,安装界面只有英文"
    }
    # .iss 里用 /D 注入绝对路径:避免相对路径按 .iss 所在目录解析 + 中文路径的坑
    & $iscc $haveChinese "/DProjDir=$Project" "/DBuildRoot=$BuildRoot" "/DMyAppVersion=$Version" `
        (Join-Path $Project "packaging\installer_shsynth.iss") | ForEach-Object { Write-Host "      $_" }
    if ($LASTEXITCODE -ne 0) { Fail "ISCC 编译失败(退出码 $LASTEXITCODE)" }
    $setup = Join-Path $DistRoot "SHSynth_Setup_v$Version.exe"
    if (-not (Test-Path $setup)) { Fail "没有生成 $setup" }
    $ssz = [math]::Round((Get-Item $setup).Length / 1MB, 1)
    Write-OK "安装程序 $setup($ssz MB)"
}

# ── 4. 安装-校验-卸载 ────────────────────────────────────────────────────────
if ($VerifyInstall) {
    Write-Step "[4/4]  安装 → 校验 → 卸载 …"
    $setup = Join-Path $DistRoot "SHSynth_Setup_v$Version.exe"
    if (-not (Test-Path $setup)) { Fail "没有安装程序可验证" }
    # ⚠️ 故意装到一个**含空格**的路径!用户机器上太容易出现了:
    #   · Windows 用户名带空格 → %LOCALAPPDATA% 就有空格 → 默认安装路径有空格;
    #   · 用户选"为所有用户安装" → C:\Program Files\SHSynth;
    #   · 用户自己把路径改成 "D:\我的 软件\SHSynth"。
    # 以前这里装到 %TEMP%\SHSynth_verify(没有空格),等于**从来没验过这种情况**。
    # 顺带带上一个非 ASCII 目录名(项目本身就在中文路径下,一并压住)。
    #
    # ⚠️ 根目录用 %LOCALAPPDATA% 而**不是** %TEMP%:某些环境下 %TEMP% 是以 8.3
    # 短名给的(实测 C:\Users\PENGZH~1\...),拿它拼出的路径跟快捷方式里存的
    # 长路径字符串不相等,下面"快捷方式指向"的校验会**误报**(实测踩过一次)。
    # %LOCALAPPDATA% 是长名形式,而且这里是"按用户安装"的默认落点,更贴近真实。
    $fso = New-Object -ComObject Scripting.FileSystemObject
    $target = Join-Path $env:LOCALAPPDATA "SHSynth verify 验证目录"
    if (Test-Path $target) { Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue }
    if ($target -notmatch ' ') { Fail "校验路径必须含空格,否则测不出空格问题:$target" }
    Write-Host "      校验用安装路径(含空格):$target"

    # ── 先把自己上一次验证留下的东西清干净 ──────────────────────────────────
    # 为什么必须清:Inno 的 [Tasks] desktopicon 带 checkedonce,检测到"已安装过"
    # 时不会重新勾上 → 桌面图标不会被重建 → 上一次**失败构建**留下的旧快捷方式
    # 会一直指着旧的验证目录,让校验一直失败(实测就这么卡过一次)。
    # 只清"目标指向某个 SHSynth verify* 目录"的快捷方式 —— 绝不碰用户自己装在
    # 别处的那一份。
    $shClean = New-Object -ComObject WScript.Shell
    $stale = 0
    foreach ($dir in @((Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\SHSynth"),
                       (Join-Path $env:USERPROFILE "Desktop"),
                       (Join-Path $env:USERPROFILE "Desktop\SHSynth"))) {
        foreach ($lnk in @(Get-ChildItem $dir -Filter "*.lnk" -ErrorAction SilentlyContinue |
                           Where-Object { $_.Name -match "SHSynth|使用说明|关于|卸载" })) {
            $tp = ""
            try { $tp = $shClean.CreateShortcut($lnk.FullName).TargetPath } catch { }
            if ($tp -match "SHSynth verify") {
                Remove-Item -Force $lnk.FullName -ErrorAction SilentlyContinue
                $stale++
            }
        }
    }
    foreach ($old in @(Get-ChildItem $env:TEMP, $env:LOCALAPPDATA -Directory -ErrorAction SilentlyContinue |
                       Where-Object { $_.Name -like "SHSynth verify*" })) {
        if ($old.FullName -ne $target) {
            Remove-Item -Recurse -Force $old.FullName -ErrorAction SilentlyContinue
            $stale++
        }
    }
    if ($stale) { Write-Host "      已清掉上次验证的残留 $stale 处(快捷方式 / 目录)" }

    # /MERGETASKS=desktopicon 明确要求建桌面图标 —— 否则 [Tasks] 的 checkedonce
    # 会因为"检测到已安装过"而不勾它,桌面图标就永远不建(校验也就无从谈起)。
    $ip = Start-Process -FilePath $setup -PassThru -Wait -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
        "/MERGETASKS=desktopicon",
        "/DIR=`"$target`"", "/LOG=`"$(Join-Path $LogDir 'install.log')`"")
    if ($ip.ExitCode -ne 0) { Fail "静默安装失败(退出码 $($ip.ExitCode))" }
    Write-OK "静默安装完成 → $target"

    $instExe = Join-Path $target "SHSynth.exe"
    if (-not (Test-Path $instExe)) { Fail "安装目录里没有 SHSynth.exe" }
    foreach ($rel in @("_internal\docs\使用说明.html",
                       "_internal\docs\使用说明_img\screenshot_gui.png",
                       "_internal\docs\发行说明.md",
                       "_internal\licenses\NOTICE.txt",
                       "LICENSE.txt")) {
        if (Test-Path (Join-Path $target $rel)) { Write-OK "已安装 $rel" } else { Fail "安装目录缺少 $rel" }
    }
    foreach ($rel in @("source", "_internal\docs\构建方法说明.md")) {
        if (Test-Path (Join-Path $target $rel)) { Fail "安装目录里混入了开发者内容:$rel" }
    }
    Write-OK "已安装内容干净(无源码 / 无开发者文档)"

    # ── 空格路径专项:快捷方式的指向必须真的指到那个带空格的安装目录 ────────
    # 每类快捷方式的"正确目标"不一样,分开判:
    #   启动/使用说明/关于 → {app}\SHSynth.exe(后两个还各带一个参数)
    #   卸载 SHSynth        → {app}\unins000.exe(不是主程序!)
    # 目标比较一律先归一到 **8.3 短路径**:同一个文件写成短名/长名时字符串并不
    # 相等,而 %TEMP% 之类的环境变量在某些环境下就是短名形式(实测 PENGZH~1)。
    $sh = New-Object -ComObject WScript.Shell
    $uninsExe = (Get-ChildItem $target -Filter "unins*.exe" -ErrorAction SilentlyContinue |
                 Select-Object -First 1)
    $checked = 0
    $lnkDirs = @((Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\SHSynth"),
                 (Join-Path $env:USERPROFILE "Desktop"))
    foreach ($dir in $lnkDirs) {
        foreach ($lnk in @(Get-ChildItem $dir -Filter "*.lnk" -ErrorAction SilentlyContinue |
                           Where-Object { $_.Name -match "SHSynth|使用说明|关于|卸载" })) {
            $s = $sh.CreateShortcut($lnk.FullName)
            if (-not (Test-Path $s.TargetPath)) {
                Fail "快捷方式指向了不存在的目标:$($lnk.Name) → $($s.TargetPath)"
            }
            $isUninstall = $lnk.Name -match "卸载|uninstall"
            $want = if ($isUninstall) { if ($uninsExe) { $uninsExe.FullName } else { $null } }
                    else { $instExe }
            if ($null -eq $want) { Fail "找不到卸载程序,但存在卸载快捷方式:$($lnk.Name)"; continue }
            $got = $s.TargetPath
            try {
                if (Test-Path $got) { $got = $fso.GetFile($got).ShortPath }
                if (Test-Path $want) { $want = $fso.GetFile($want).ShortPath }
            } catch { }
            if ($got -ine $want) {
                Fail "快捷方式目标不对:$($lnk.Name)`n        实际:$got`n        期望:$want"
            }
            # 参数也要对:使用说明 = --guide,关于 = --version
            if ($lnk.Name -match "使用说明" -and $s.Arguments -notmatch "--guide") {
                Fail "『使用说明』快捷方式少了 --guide(实际参数 '$($s.Arguments)')"
            }
            if ($lnk.Name -match "关于" -and $s.Arguments -notmatch "--version") {
                Fail "『关于』快捷方式少了 --version(实际参数 '$($s.Arguments)')"
            }
            $checked++
        }
    }
    if ($checked -eq 0) { Write-Warn2 "没找到快捷方式可校验(可能是 /MERGETASKS 没建)" }
    else { Write-OK "$checked 个快捷方式的目标与参数都正确(路径含空格)" }

    $stOut2 = Join-Path $LogDir "selftest_installed.log"
    $p3 = Start-Process -FilePath $instExe -ArgumentList "--self-test" -PassThru -Wait `
            -RedirectStandardOutput $stOut2 -RedirectStandardError (Join-Path $LogDir "selftest_installed.err")
    $ok2 = @(Get-Content $stOut2 -Encoding UTF8 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\[PASS\]' }).Count
    Write-Host "      已安装版本自检:$ok2 项通过(退出码 $($p3.ExitCode))"
    if ($p3.ExitCode -ne 0) { Fail "已安装版本自检未通过(见 $stOut2)" }

    # ── 空格路径专项:真的从那个目录跑一次命令行,并确认**结果落了盘** ──────
    # 只跑 --self-test 不够:它主要在临时目录里干活。这里让冻结版从带空格的
    # 安装目录出一次结果文件,把"路径里有空格导致取不到资源/写不出去"压住。
    $cliOut = Join-Path $target "cli_smoke_out.nc"
    $cliLog = Join-Path $LogDir "cli_smoke.log"
    # SHKit 是 SHSynth 的**同级**目录(取它的样例系数来跑一次真实解算)。
    # 注意变量名:$Project 是 SHSynth 目录;$ProjDir 是 Inno 那边的 define,
    # PowerShell 里没有 —— 曾经在这儿误用 $ProjDir,Join-Path 收到 $null 直接报
    # "Cannot bind argument to parameter 'Path' because it is null"。
    $coeffSample = Join-Path (Split-Path -Parent $Project) "SHKit\shkit_coeffs.sh"
    if (-not (Test-Path $coeffSample)) {
        $coeffSample = Join-Path $Project "sample_data\shkit_coeffs.sh"
    }
    if (Test-Path $coeffSample) {
        $p4 = Start-Process -FilePath $instExe -PassThru -Wait `
                -RedirectStandardOutput $cliLog `
                -RedirectStandardError (Join-Path $LogDir "cli_smoke.err") -ArgumentList @(
            "--cli", "synth", "--coeffs", "`"$coeffSample`"",
            "--global-grid", "5", "--out", "`"$cliOut`"")
        if ($p4.ExitCode -ne 0) { Fail "从含空格安装目录跑 --cli 失败(退出码 $($p4.ExitCode),见 $cliLog)" }
        if (-not (Test-Path $cliOut)) { Fail "从含空格安装目录跑 --cli 没有写出结果文件" }
        Write-OK "从含空格安装目录跑 --cli 成功并写出结果"
        Remove-Item -Force $cliOut -ErrorAction SilentlyContinue
    } else {
        Write-Warn2 "跳过 --cli 冒烟(没找到 SHKit 样例系数)"
    }
    Write-OK "已安装版本可用"

    $un = Get-ChildItem $target -Filter "unins*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($un) {
        Start-Process -FilePath $un.FullName -PassThru -Wait -ArgumentList @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") | Out-Null
        Start-Sleep -Seconds 2
        if (Test-Path $instExe) { Write-Warn2 "卸载后 $instExe 仍存在" } else { Write-OK "卸载完成" }
    }
    Remove-Item -Recurse -Force $target -ErrorAction SilentlyContinue
}

# ── 交付:复制到最终目录 + 绿色版 zip ────────────────────────────────────────
Write-Step "整理交付物 …"
$setup = Join-Path $DistRoot "SHSynth_Setup_v$Version.exe"
if (Test-Path $setup) { Copy-Item $setup $FinalDir -Force; Write-OK "安装程序 → $FinalDir" }

# MIT 要求"版权声明与许可正文随所有副本分发"。安装版的 LICENSE.txt 由 .iss 单独放;
# **绿色版**是直接压缩目录包,所以这里必须先把它放进目录包,否则 zip 里没有许可正文。
$lic = Join-Path $Project "LICENSE.txt"
if (-not (Test-Path $lic)) { Fail "找不到 LICENSE.txt" }
Copy-Item $lic (Join-Path $Dist "LICENSE.txt") -Force
Write-OK "LICENSE.txt 已放进目录包(绿色版会一起带上)"

$zip = Join-Path $FinalDir "SHSynth_v$Version.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $Dist "*") -DestinationPath $zip -CompressionLevel Optimal -Force
$zsz = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zr = [System.IO.Compression.ZipFile]::OpenRead($zip)
$zNames = @($zr.Entries | ForEach-Object { $_.FullName })
$zr.Dispose()
foreach ($need in @("LICENSE.txt", "_internal\docs\使用说明.html",
                    "_internal\licenses\NOTICE.txt")) {
    if ($zNames -contains $need) { Write-OK "绿色版含 $need" } else { Fail "绿色版缺少 $need" }
}
if (@($zNames | Where-Object { $_ -match "(^|\\)(source|tests|tools|packaging)\\" }).Count -gt 0) {
    Fail "绿色版里混入了源码 / 开发内容"
}
Write-OK "绿色版内容干净(只有程序、说明书、许可)"
Write-OK "绿色版(免安装) → $zip($zsz MB)"
Copy-Item (Join-Path $Project "SHSynth.ico") $FinalDir -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  打包完成" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  目录包   : $Dist"
if (Test-Path $setup) { Write-Host "  安装程序 : $setup" }
Write-Host "  交付目录 : $FinalDir"
Write-Host "  自检日志 : $LogDir\selftest.log"
Write-Host ""
