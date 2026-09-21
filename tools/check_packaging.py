# -*- coding: utf-8 -*-
"""打包脚本自检 —— 抓那些"一旦破坏就**静默**出问题"的约定。

跑法:``python tools/check_packaging.py``(退出码 0 = 通过)
也可以用 :func:`collect` 拿结果给测试套件复用(``tests/test_packaging.py``)。

为什么要单独有这个检查
----------------------
这几个约定一旦破坏,表现都不是"报错",而是**打包出问题或装出来的东西不对**:

1. **`build_installer.ps1` 必须是 UTF-8 with BOM**。Windows PowerShell 5.1
   读无 BOM 的 UTF-8 会当 GBK,中文注释直接把它读成乱码,然后报
   "字符串缺少终止符"之类**看起来毫不相干**的语法错误。构建脚本自己开头就写了
   这条,但只要有人用不带 BOM 的编辑器/脚本改一下这个文件,BOM 就没了 ——
   实测踩过两次(最近一次就是加"含空格路径校验"的时候)。
2. **`installer_shsynth.iss` 同样要 BOM**(ISCC 按 ANSI 读无 BOM 文件,中文会乱)。
3. **[Run]/[Icons] 的 Filename 里不许自己加引号**。Inno 编译器会直接报
   ``Parameter "Filename" cannot include quotes (")`` —— 也就是说**含空格路径
   由 Inno 内部处理**,不需要(也不能)手写引号。有人"好心"加上反而编不过。
4. **安装校验必须用含空格的路径**。用户机器上太容易出现(用户名带空格 →
   ``%LOCALAPPDATA%`` 就有空格;或选"为所有用户安装" → ``C:\\Program Files``)。
   校验路径没空格 = 等于从来没验过这种情况。
5. **安装校验里要有 `--cli` 冒烟**:证明冻结版能从含空格的安装目录里真跑一次
   并写出结果,而不是只在临时目录里自检通过。
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGING = os.path.join(ROOT, "packaging")
PS1 = os.path.join(PACKAGING, "build_installer.ps1")
ISS = os.path.join(PACKAGING, "installer_shsynth.iss")

BOM = b"\xef\xbb\xbf"
_RESULTS: list = []
_QUIET = False


def _say(msg: str = "") -> None:
    if not _QUIET:
        print(msg)


def ok(cond: bool, msg: str) -> bool:
    _RESULTS.append((bool(cond), msg))
    _say(("  [PASS] " if cond else "  [FAIL] ") + msg)
    return bool(cond)


def note(msg: str) -> None:
    _say("  [NOTE] " + msg)


def has_bom(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(3) == BOM


def read_text(path: str) -> str:
    with open(path, encoding="utf-8-sig") as fh:
        return fh.read()


def check_bom(path: str, why: str) -> None:
    name = os.path.basename(path)
    if not os.path.exists(path):
        ok(False, f"{name} 存在")
        return
    got = has_bom(path)
    ok(got, f"{name} 是 UTF-8 with BOM({why})")
    if not got:
        _say("         修法:用 UTF-8 **with BOM** 重写,例如")
        _say("           python -c \"import io;"
             f"p=r'{path}';s=io.open(p,encoding='utf-8').read();"
             "io.open(p,'w',encoding='utf-8-sig',newline='').write(s)\"")


def check_ps1_parses() -> None:
    """能调 PowerShell 就顺手真解析一遍(比只看 BOM 更能兜住)。"""
    import shutil
    import subprocess
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        note("没找到 PowerShell,跳过 .ps1 真解析")
        return
    code = ("$e=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            f"'{PS1}',[ref]$null,[ref]$e)|Out-Null;"
            "if($e.Count){$e|%{$_.Message};exit 1}else{exit 0}")
    try:
        r = subprocess.run([ps, "-NoProfile", "-Command", code],
                           capture_output=True, text=True, timeout=120)
    except Exception as exc:                              # noqa: BLE001
        note(f"调用 PowerShell 失败:{type(exc).__name__}: {exc}")
        return
    out = (r.stdout or "").strip()
    ok(r.returncode == 0,
       "build_installer.ps1 能被 Windows PowerShell 真正解析"
       + (f"(报错:{out[:120]})" if r.returncode else ""))


def _filename_with_own_quotes(txt: str) -> list:
    """找出 [Run]/[Icons] 里自带引号的 Filename(Inno 会拒绝编译)。"""
    bad = []
    for sec in ("[Run]", "[Icons]"):
        body = re.search(re.escape(sec) + r"(.*?)(?=\n\[|\Z)", txt, re.S)
        if not body:
            continue
        for line in body.group(1).splitlines():
            line = line.strip()
            if line.startswith(";") or "Filename:" not in line:
                continue
            frag = line.split("Filename:", 1)[1].split(";")[0].strip()
            if frag.startswith('"') and frag.count('"') > 2:
                bad.append(f"{sec} {line[:70]}")
    return bad


def _icon_names_with_slash(txt: str) -> list:
    """找出 [Icons] 里 Name 含 ``/`` 的条目。

    Inno 把 ``/`` 当子目录分隔符建目录,
    却把 ``/`` 留在文件名里 ⇒
    ``IPersistFile::Save failed; code 0x80070003``。
    要子目录必须写反斜杠。
    """
    bad = []
    body = re.search(r"\[Icons\](.*?)(?=\n\[|\Z)", txt, re.S)
    if not body:
        return bad
    for line in body.group(1).splitlines():
        line = line.strip()
        if line.startswith(";") or "Name:" not in line:
            continue
        frag = line.split("Name:", 1)[1].split(";")[0].strip()
        if "/" in frag:
            bad.append(line[:80])
    return bad


def check_iss_icons() -> None:
    """[Icons] 的 Name 里不许有 '/' —— 实测会让安装直接失败回滚的坑。"""
    if not os.path.exists(ISS):
        ok(False, "installer_shsynth.iss 存在(检查 [Icons])")
        return
    bad = _icon_names_with_slash(read_text(ISS))
    ok(not bad,
       "installer_shsynth.iss 的 [Icons] Name 里没有 '/'"
       "(Inno 把 '/' 当子目录建目录、却把 '/' 留在文件名里 ⇒ "
       "IPersistFile::Save failed 0x80070003;要子目录请用反斜杠)")
    for b in bad:
        _say(f"         {b}")


def check_iss() -> None:
    if not os.path.exists(ISS):
        ok(False, "installer_shsynth.iss 存在")
        return
    txt = read_text(ISS)
    bad = _filename_with_own_quotes(txt)
    ok(not bad,
       "installer_shsynth.iss 的 Filename 没有自带引号"
       "(Inno 会报 'cannot include quotes';含空格路径由 Inno 内部处理)")
    for b in bad:
        _say(f"         {b}")


def check_verify_install() -> None:
    if not os.path.exists(PS1):
        ok(False, "build_installer.ps1 存在")
        return
    txt = read_text(PS1)
    # $target 用 Join-Path 拼,根目录应是**长名**的 %LOCALAPPDATA%
    # (不能用 %TEMP%:某些环境下它是 8.3 短名,会让快捷方式校验误报)。
    m = re.search(r'\$target\s*=\s*Join-Path\s+\$(?:env:)?(\w+)\s+"([^"]*)"', txt)
    if ok(m is not None, "build_installer.ps1 里能找到校验用安装路径"):
        root, leaf = m.group(1), m.group(2)
        ok(" " in leaf,
           f"安装校验路径含空格({leaf!r})—— 否则测不出空格问题")
        ok(root == "LOCALAPPDATA",
           f"校验路径根目录用 ${root}(长名;%TEMP% 可能是 8.3 短名 PENGZH~1)")
    ok("cli_smoke" in txt or "--cli" in txt,
       "安装校验里有 --cli 冒烟(证明能从含空格目录真跑一次)")
    ok("-notmatch" in txt and "校验路径必须含空格" in txt,
       "安装校验里有『路径必须含空格』的断言(防止以后又被改回无空格)")
    ok("ShortPath" in txt,
       "快捷方式比较用 .ShortPath 归一(短名/长名字符串不等会误报)")


def collect() -> list:
    """跑全部检查,返回 ``[(是否通过, 说明), …]``(不打印)。"""
    global _QUIET, _RESULTS
    _QUIET, _RESULTS = True, []
    check_bom(PS1, "无 BOM 会被当 GBK 读,中文注释导致假语法错误")
    check_bom(ISS, "无 BOM ISCC 会按 ANSI 读,中文乱码")
    check_ps1_parses()
    check_iss()
    check_iss_icons()
    check_verify_install()
    out = list(_RESULTS)
    _QUIET, _RESULTS = False, []
    return out


def main() -> int:
    print("=" * 70)
    print("打包脚本自检")
    print("=" * 70)
    print("\n-- 编码(Windows PowerShell 5.1 / ISCC 都按系统 ANSI 读无 BOM 文件)--")
    check_bom(PS1, "无 BOM 会被当 GBK 读,中文注释导致假语法错误")
    check_bom(ISS, "无 BOM ISCC 会按 ANSI 读,中文乱码")
    print("\n-- PowerShell 语法 --")
    check_ps1_parses()
    print("\n-- Inno 脚本约定 --")
    check_iss()
    check_iss_icons()
    note('实测结论:Inno 编译器对 [Run] 的 Filename 一旦出现引号就报')
    note("  \"Parameter 'Filename' cannot include quotes\" ⇒ 含空格路径")
    note("  由 Inno 内部加引号,脚本里**不要**自己加。")
    print("\n-- 含空格路径的安装校验覆盖 --")
    check_verify_install()
    bad = [m for good, m in _RESULTS if not good]
    print("\n" + "=" * 70)
    if bad:
        print(f"结果:有 {len(bad)} 项失败(共 {len(_RESULTS)} 项)")
        for f in bad:
            print("  - " + f)
        print("=" * 70)
        return 1
    print(f"结果:全部通过(共 {len(_RESULTS)} 项)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
