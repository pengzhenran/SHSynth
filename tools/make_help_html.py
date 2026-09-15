# -*- coding: utf-8 -*-
"""
tools/make_help_html.py
=======================

把**面向使用者**的说明书(`docs/使用说明.md`)转成一份自包含 HTML:
``docs/使用说明.html`` + ``docs/使用说明_img/``。

两点刻意设计:

1. **只包含用户文档**。构建方法、测试、许可合规、API 细节这些开发者内容一律不进
   —— 说明书是给第三方使用者看的,不是给开发者看的。
2. **只用 Qt 认得的排版**。界面里的说明书是 QTextBrowser 渲染的(窗口内查看),
   它的富文本引擎只支持 CSS 2.1 的一个子集(**没有 flex / sticky / grid**),
   所以这里生成的 HTML 只有:标题、段落、列表、表格、代码块、引用、图片、
   以及**文档顶部的目录锚点**。同一份文件丢进浏览器也照样好看。

不引入 markdown 依赖,只实现本文档实际用到的 Markdown 子集。

    python tools/make_help_html.py
"""

from __future__ import annotations

import html
import os
import re
import shutil
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_MD = os.path.join(ROOT, "docs", "使用说明.md")
OUT_HTML = os.path.join(ROOT, "docs", "使用说明.html")
IMG_DIR_NAME = "使用说明_img"

#: 说明书里图片的默认显示宽度。QTextBrowser **不会**自适应图片宽度,所以先在
#: HTML 里给一个保守值;界面窗口打开时会按窗口宽度重算(见
#: shsynth/gui/docs_window.py 的 _fit_images)。只给 width、**不给 height**,
#: 让 Qt 按比例自己算高度 —— 真正会造成"图下方一大片空隙"的是 line-height
#: (见下面 _CSS 里的 p.pic),不是 height 属性。
DEFAULT_IMG_WIDTH = 760

#: "整段只有图片"的判定:用于给该段加 class="pic"(见 _CSS 里的 p.pic)。
_ONLY_IMG = re.compile(r"^(?:<img\b[^>]*>\s*)+$")

#: Qt(QTextBrowser)与浏览器都能接受的样式:不用 flex / sticky / grid
_CSS = """
body { font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif;
       font-size:11pt; color:#1b2733; line-height:1.5; }
h1 { font-size:19pt; color:#14304d; }
h2 { font-size:15pt; color:#14304d; border-bottom:1px solid #cfd8e3;
     padding-bottom:4px; margin-top:18px; }
h3 { font-size:12.5pt; color:#1b2733; margin-top:14px; }
h4 { font-size:11.5pt; color:#1b2733; }
p { margin:5px 0; }
/* 只有一张图、没有文字的段落:必须把 line-height 压回 100%。
   Qt 的 QTextBrowser 会把块级 line-height(1.5)也乘到行内图片的高度上,
   图片占 475px 时行框会被撑到约 715px —— 图下面就多出半张图高的空隙。 */
p.pic { margin:8px 0; line-height:100%; }
a { color:#1f5fa9; text-decoration:none; }
code { font-family:Consolas,"Courier New",monospace; font-size:10pt;
       background-color:#f2f5f8; white-space:pre-wrap; }
pre { font-family:Consolas,"Courier New",monospace; font-size:9.5pt;
      background-color:#f7f9fb; padding:8px; white-space:pre-wrap; }
table { border-collapse:collapse; margin:6px 0; }
th { background-color:#eef3f8; border:1px solid #c8d3e0; padding:5px 8px;
     font-size:10pt; }
td { border:1px solid #d7dfe8; padding:5px 8px; font-size:10pt;
     white-space:normal; }
blockquote { margin:6px 0; padding:8px 12px; background-color:#fff8e6; }
/* vertical-align:bottom 去掉行内图片基线下方那条"看不见的空隙",
   否则每张图下面都会多出半行高度,读起来很松散 */
img { vertical-align:bottom; }
.toc { background-color:#f7fafd; padding:10px 14px; }
.byline { font-size:10pt; color:#5b6b7c; }
.foot { font-size:9.5pt; color:#5b6b7c; margin-top:16px; }
"""


def _image_size(path: str) -> tuple:
    """读 PNG / JPEG 的像素尺寸(不依赖 Pillow);读不出返回 ``(0, 0)``。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":                  # JPEG:扫 SOF 段
                fh.seek(2)
                for _ in range(64):
                    b = fh.read(1)
                    while b and b != b"\xff":
                        b = fh.read(1)
                    if not b:
                        break
                    marker = fh.read(1)
                    while marker == b"\xff":
                        marker = fh.read(1)
                    if not marker:
                        break
                    if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6,
                                     0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        fh.read(3)
                        h, w = struct.unpack(">HH", fh.read(4))
                        return int(w), int(h)
                    seg = fh.read(2)
                    if len(seg) < 2:
                        break
                    fh.seek(struct.unpack(">H", seg)[0] - 2, 1)
    except Exception:                                    # noqa: BLE001
        pass
    return 0, 0


def _slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", s).strip("-").lower()
    return s or "sec"


def _inline(text: str) -> str:
    """行内标记 → HTML。"""
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)",
               lambda m: _image(m.group(1), m.group(2)), t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
               lambda m: _link(m.group(1), m.group(2)), t)
    return t


def _image(alt: str, src: str) -> str:
    """图片:拷进 使用说明_img/,并带上显示宽度与原始宽度。

    ``width`` 给一个保守的显示宽度(避免在 QTextBrowser 里横向滚动),
    ``data-full`` 记原始像素宽度 —— 界面窗口据此**按窗口宽度**重算
    (见 :func:`shsynth.gui.docs_window.GuideDialog._fit_images`)。
    """
    src = src.strip()
    if re.match(r"^https?://", src):
        return f'<img src="{src}" alt="{alt}">'
    raw = src.replace("/", os.sep)
    for c in (os.path.join(ROOT, "docs", raw), os.path.join(ROOT, raw)):
        if os.path.exists(c):
            os.makedirs(os.path.join(ROOT, "docs", IMG_DIR_NAME), exist_ok=True)
            base = os.path.basename(c)
            dst = os.path.join(ROOT, "docs", IMG_DIR_NAME, base)
            try:
                shutil.copyfile(c, dst)
            except OSError:
                pass
            w, _h = _image_size(dst)
            full = w or DEFAULT_IMG_WIDTH
            disp = min(DEFAULT_IMG_WIDTH, full)
            return (f'<img src="{IMG_DIR_NAME}/{base}" alt="{alt}" '
                    f'width="{disp}" data-full="{full}">')
    return f'<img src="{src}" alt="{alt}">'


def _link(text: str, url: str) -> str:
    url = url.strip()
    if url.startswith("#"):
        return f'<a href="{url}">{text}</a>'
    if url.endswith(".md") or ".md#" in url:
        anchor = url.split("#", 1)[1] if "#" in url else ""
        return f'<a href="#{anchor}">{text}</a>' if anchor else f"<b>{text}</b>"
    return f'<a href="{url}">{text}</a>'


def md_to_html(md: str, toc: list) -> str:
    """Markdown → HTML(只用 Qt 认得的标签),同时收集标题做目录。"""
    out: list = []
    lines = md.splitlines()
    i = 0
    in_code = False
    code_buf: list = []
    in_table = False
    table_buf: list = []
    list_kind = None
    list_buf: list = []
    para_buf: list = []

    def flush_para():
        if para_buf:
            inner = _inline(" ".join(para_buf))
            # 整段只有图片 → 加 class="pic",让 CSS 把行高压回 100%
            cls = ' class="pic"' if _ONLY_IMG.match(inner) else ""
            out.append(f"<p{cls}>{inner}</p>")
            para_buf.clear()

    def flush_list():
        nonlocal list_kind
        if list_buf:
            tag = "ol" if list_kind == "ol" else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{x}</li>" for x in list_buf)
                       + f"</{tag}>")
            list_buf.clear()
        list_kind = None

    def flush_table():
        nonlocal in_table
        if table_buf:
            rows = [r for r in table_buf if r.strip()]
            head, body = None, rows
            if len(rows) >= 2 and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", rows[1]):
                head, body = rows[0], rows[2:]
            html_rows = []
            for r in body:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                html_rows.append("<tr>" + "".join(
                    f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
            thead = ""
            if head:
                cells = [c.strip() for c in head.strip().strip("|").split("|")]
                thead = "<tr>" + "".join(
                    f"<th>{_inline(c)}</th>" for c in cells) + "</tr>"
            out.append('<table width="100%">' + thead
                       + "".join(html_rows) + "</table>")
            table_buf.clear()
        in_table = False

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith("```"):
            if in_code:
                out.append("<pre>" + html.escape("\n".join(code_buf)) + "</pre>")
                code_buf.clear()
                in_code = False
            else:
                flush_para(); flush_list(); flush_table()
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        if s.startswith("|") and s.endswith("|"):
            flush_para(); flush_list()
            in_table = True
            table_buf.append(s)
            i += 1
            continue
        if in_table:
            flush_table()

        if not s:
            flush_para(); flush_list()
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            flush_para(); flush_list(); flush_table()
            lvl = min(len(m.group(1)), 4)
            text = m.group(2).strip()
            if lvl == 1:
                out.append(f'<h1 id="top">{_inline(text)}</h1>')
            else:
                sid = _slug(text) + f"-{len(toc)}"
                toc.append((lvl, text, sid))
                out.append(f'<h{lvl} id="{sid}">{_inline(text)}</h{lvl}>')
            i += 1
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", s):
            flush_para(); flush_list()
            out.append("<hr>")
            i += 1
            continue

        if s.startswith(">"):
            flush_para(); flush_list()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(buf))
                       + "</blockquote>")
            continue

        m = re.match(r"^([-*+])\s+(.*)$", s)
        if m:
            flush_para()
            if list_kind not in (None, "ul"):
                flush_list()
            list_kind = "ul"
            list_buf.append(_inline(m.group(2)))
            i += 1
            continue
        m = re.match(r"^\d+[.)]\s+(.*)$", s)
        if m:
            flush_para()
            if list_kind not in (None, "ol"):
                flush_list()
            list_kind = "ol"
            list_buf.append(_inline(m.group(1)))
            i += 1
            continue

        if list_kind and ln.startswith(("  ", "\t")):
            if list_buf:
                list_buf[-1] += " " + _inline(s)
            i += 1
            continue

        flush_list()
        para_buf.append(s)
        i += 1

    flush_para(); flush_list(); flush_table()
    if in_code and code_buf:
        out.append("<pre>" + html.escape("\n".join(code_buf)) + "</pre>")
    return "\n".join(out)


def _author_block() -> str:
    sys.path.insert(0, ROOT)
    from shsynth import __version__, author
    qr = author.qr_image_path()
    qr_tag = ""
    if qr:
        os.makedirs(os.path.join(ROOT, "docs", IMG_DIR_NAME), exist_ok=True)
        dst = os.path.join(ROOT, "docs", IMG_DIR_NAME, os.path.basename(qr))
        try:
            shutil.copyfile(qr, dst)
            qr_tag = (f'<img src="{IMG_DIR_NAME}/'
                      f'{os.path.basename(qr)}" width="140" data-full="140" '
                      f'alt="公众号二维码">')
        except OSError:
            pass
    return (
        '<table width="100%"><tr><td width="64%">'
        f'<b>{author.APP_NAME} {__version__} —— {author.APP_NAME_CN}</b><br>'
        f'作者:{author.AUTHOR_NAME_CN}({author.AUTHOR_NAME_EN})<br>'
        f'单位:{author.AUTHOR_AFFILIATION_CN}<br>'
        f'{author.AUTHOR_AFFILIATION_EN}<br>'
        f'邮箱:<a href="mailto:{author.AUTHOR_EMAIL}">{author.AUTHOR_EMAIL}</a><br>'
        f'电话:{author.AUTHOR_PHONE}<br>'
        f'公众号:{author.WECHAT_ACCOUNT}({author.WECHAT_ACCOUNT_EN})'
        # line-height:100% 同 p.pic:否则 140px 的二维码会被 1.5 倍行高撑成 210px;
        # valign=middle 让它跟左侧的作者信息竖直居中
        '</td><td align="center" valign="middle" style="line-height:100%">'
        + qr_tag + '</td></tr></table>')


def main(argv=None) -> int:
    sys.path.insert(0, ROOT)
    from shsynth import __version__, author

    if not os.path.exists(SRC_MD):
        print(f"找不到 {SRC_MD}")
        return 1
    with open(SRC_MD, encoding="utf-8") as fh:
        md = fh.read()

    toc: list = []
    body = md_to_html(md, toc)

    nav = ['<div class="toc"><b>目录</b><br>']
    for lvl, text, sid in toc:
        if lvl == 2:
            nav.append(f'　<a href="#{sid}">{html.escape(text)}</a><br>')
        elif lvl == 3:
            nav.append(f'　　<a href="#{sid}">{html.escape(text)}</a><br>')
    nav.append("</div><hr>")

    doc = (
        '<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n'
        f"<title>{author.APP_NAME} 使用说明</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f'<p class="byline">{author.APP_NAME} v{__version__} · '
        f"{author.AUTHOR_NAME_CN} · {author.AUTHOR_AFFILIATION_CN}</p>\n"
        + _author_block() + "<hr>\n"
        + "\n".join(nav) + "\n"
        + body + "\n"
        f'<p class="foot">本说明书随 {author.APP_NAME} v{__version__} 一起提供;'
        "更新请见同目录的 <code>发行说明.md</code>。反馈请发 "
        f'<a href="mailto:{author.AUTHOR_EMAIL}">{author.AUTHOR_EMAIL}</a>。</p>\n'
        "</body></html>\n")

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    # 使用说明_img\ 完全由本脚本与 make_screenshots.py 生成;只留正文真正引用到的,
    # 免得安装包里躺着几张没人看的截图
    img_dir = os.path.join(ROOT, "docs", IMG_DIR_NAME)
    if os.path.isdir(img_dir):
        used = set(re.findall(rf'src="{re.escape(IMG_DIR_NAME)}/([^"]+)"', doc))
        for name in sorted(os.listdir(img_dir)):
            p = os.path.join(img_dir, name)
            if name in used or not os.path.isfile(p):
                continue
            try:
                os.remove(p)
                print(f"  清掉没被引用的配图:{name}")
            except OSError:
                pass
    # 数**正文真正用到**的图,而不是目录里的文件数(目录里可能还有别的截图)
    n_img = doc.count("<img ") - _author_block().count("<img ")
    n_h2 = sum(1 for lvl, _t, _s in toc if lvl == 2)
    print(f"已生成 {OUT_HTML}({os.path.getsize(OUT_HTML):,} 字节,"
          f"{n_h2} 个章节,配图 {n_img} 张)")
    # 检查正文里有没有漏掉未转换的 Markdown 标记
    leftovers = [p for p in ("**", "![", "](") if p in body]
    if leftovers:
        print("警告:正文里可能残留未转换的 Markdown 标记:" + "、".join(leftovers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
