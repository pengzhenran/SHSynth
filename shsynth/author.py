# -*- coding: utf-8 -*-
"""
shsynth.author
==============

作者 / 单位 / 联系方式 / 公众号二维码 / 随包说明书的**单一定义处**。

界面「关于」对话框、HTML 使用说明、安装程序(.iss)、pyproject 元数据都从这里取,
避免同一个邮箱在三处写得不一样。与同课题组的 GRACE Downloader / SHKit 系列
保持一致。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

__all__ = [
    "APP_NAME", "APP_NAME_CN", "APP_TAGLINE",
    "AUTHOR_NAME_CN", "AUTHOR_NAME_EN", "AUTHOR_EMAIL", "AUTHOR_PHONE",
    "AUTHOR_AFFILIATION_CN", "AUTHOR_AFFILIATION_EN",
    "AUTHOR_PUBLISHER", "WECHAT_ACCOUNT", "WECHAT_ACCOUNT_EN",
    "WECHAT_QR_FILENAME", "WECHAT_QR_CAPTION",
    "GUIDE_HTML_NAME", "GUIDE_MD_NAME",
    "resource_dirs", "qr_image_path", "resource_path",
    "guide_html_path", "about_text",
]

APP_NAME = "SHSynth"
APP_NAME_CN = "球谐系数解算"
APP_TAGLINE = "输入球谐系数 + 网格/散点位置 → 输出网格/散点值并绘图"

AUTHOR_NAME_CN = "彭桢燃"
AUTHOR_NAME_EN = "Zhenran Peng"
AUTHOR_EMAIL = "zhenran.peng@cug.edu.cn"
AUTHOR_PHONE = "15927402265"
AUTHOR_AFFILIATION_CN = "中国地质大学(武汉)"
AUTHOR_AFFILIATION_EN = "China University of Geosciences (Wuhan)"

#: 安装程序里的发行者字段
AUTHOR_PUBLISHER = f"{AUTHOR_NAME_CN} {AUTHOR_NAME_EN}  ({AUTHOR_AFFILIATION_EN})"

WECHAT_ACCOUNT = "地球重力与人类生活"
WECHAT_ACCOUNT_EN = "TVGG"
WECHAT_QR_FILENAME = "地球重力与人类生活TVGG.jpg"
WECHAT_QR_CAPTION = f"课题组公众号:{WECHAT_ACCOUNT}({WECHAT_ACCOUNT_EN})"

GUIDE_HTML_NAME = "使用说明.html"
GUIDE_MD_NAME = "使用说明.md"


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _app_dir() -> str:
    """程序所在目录(冻结版是 exe 所在目录,源码运行时是项目根目录)。"""
    if _frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _resource_dirs() -> list:
    """按优先级列出可能放"资源文件"的目录。

    冻结版里 PyInstaller 把 ``docs/`` 与 ``licenses/`` 放在 ``_internal`` 下,
    源码运行时它们在项目根目录 —— 两种都找,找不到就返回空列表(界面会给出
    "把文件放哪里"的提示,而不是崩)。
    """
    app = _app_dir()
    cands = [
        os.path.join(app, "docs"),
        os.path.join(app, "_internal", "docs"),
        os.path.join(app, "_internal", "shsynth", "data"),
        _data_dir(),
        os.path.join(app, "shsynth", "data"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs"),
    ]
    out, seen = [], set()
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


#: 供界面/工具遍历的资源目录(调用时求值,便于测试里改环境)
def resource_dirs() -> list:
    """按优先级列出候选资源目录(每次调用重新算,便于冻结/源码两种情形)。"""
    return _resource_dirs()


def resource_path(filename: str) -> Optional[str]:
    """在若干候选目录里找 ``filename``;找不到返回 ``None``。"""
    for d in _resource_dirs():
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def qr_image_path() -> Optional[str]:
    """公众号二维码图片路径(随包提供;找不到返回 ``None``)。"""
    return resource_path(WECHAT_QR_FILENAME)


def guide_html_path() -> Optional[str]:
    """HTML 使用说明路径(安装包里带;源码树里可能没有)。"""
    return resource_path(GUIDE_HTML_NAME)


def about_text() -> str:
    """「关于」对话框里那段纯文字版本(命令行/日志也用得上)。"""
    from . import __version__
    return (
        f"{APP_NAME} {__version__} —— {APP_NAME_CN}\n"
        f"{APP_TAGLINE}\n\n"
        f"作者    : {AUTHOR_NAME_CN}({AUTHOR_NAME_EN})\n"
        f"单位    : {AUTHOR_AFFILIATION_CN} / {AUTHOR_AFFILIATION_EN}\n"
        f"邮箱    : {AUTHOR_EMAIL}\n"
        f"电话    : {AUTHOR_PHONE}\n"
        f"公众号  : {WECHAT_ACCOUNT}({WECHAT_ACCOUNT_EN})\n\n"
        "约定    : 4π 归一化、无 Condon–Shortley 相位"
        "(等同 SHTOOLS norm=1, csphase=1),与 SHKit 逐位一致\n"
        "许可    : 本软件 MIT;界面 PySide6(LGPLv3,动态链接、未修改);"
        "绘图 matplotlib(BSD 风格);海岸线 Natural Earth(公有领域)\n"
    )
