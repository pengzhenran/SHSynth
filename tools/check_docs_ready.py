# -*- coding: utf-8 -*-
"""打包前的一致性自查:md 与 html 同步、图片引用都在、行高修复还在。"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = os.path.join(ROOT, "docs")
md = open(os.path.join(docs, "使用说明.md"), encoding="utf-8").read()
html = open(os.path.join(docs, "使用说明.html"), encoding="utf-8").read()
rel = open(os.path.join(docs, "发行说明.md"), encoding="utf-8").read()
img_dir = os.path.join(docs, "使用说明_img")

bad = []
for name, text in (("使用说明.md", md), ("使用说明.html", html)):
    if "跟着窗口宽度自动缩放" not in text:
        bad.append(f"{name} 缺少『配图随窗口自适应』说明")

# 行高修复必须还在(否则图片下面又会多出半张图高的空隙)
if "p.pic" not in html:
    bad.append("HTML 里没有 p.pic(图片下方空隙的修复丢了)")
if 'style="line-height:100%"' not in html:
    bad.append("HTML 里表格图片没有压回 100% 行高")

used = set(re.findall(r'src="使用说明_img/([^"]+)"', html))
have = set(os.listdir(img_dir))
if used - have:
    bad.append("HTML 引用了不存在的图:" + "、".join(sorted(used - have)))
if have - used:
    bad.append("目录里有没被引用的图:" + "、".join(sorted(have - used)))

# 用户手册里不该再出现打包前的旧说法
for stale in ("source\\", "500+ 项", "test_gui_smoke`(94)"):
    if stale in rel:
        bad.append(f"发行说明.md 还留着旧说法:{stale}")

print(f"使用说明.md {len(md):,} 字符 / 使用说明.html {len(html):,} 字符")
print(f"配图 {len(used)} 张,目录 {len(have)} 个文件")
for line in bad:
    print("[FAIL] " + line)
print("一致" if not bad else f"{len(bad)} 项不一致")
sys.exit(1 if bad else 0)
