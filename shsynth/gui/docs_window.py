# -*- coding: utf-8 -*-
"""
shsynth.gui.docs_window
=======================

**在窗口里看说明书**:左侧目录、右侧正文,支持缩放与"用浏览器打开"。

为什么不用 QDesktopServices 直接丢给浏览器:

* 安装到受管/离线机器上时不一定有默认浏览器,或者会打开一个用户没预期的窗口;
* 用户想看某一段时,窗口里能直接点目录跳转、调字号,体验更接近"帮助”该有的样子;
* 说明书是随包的 ``docs/使用说明.html``,窗口渲染不依赖外网。

Qt 的富文本引擎(QTextBrowser)只支持 CSS 2.1 的一个子集(**没有 flex/sticky**),
所以 ``tools/make_help_html.py`` 生成的 HTML 刻意只用 Qt 认得的排版
(标题 + 段落 + 表格 + 代码块 + 顶部目录锚点),同一份文件在浏览器里也好看。
"""

from __future__ import annotations

import os
import re

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton,
                               QSplitter, QTextBrowser, QVBoxLayout)

from .. import author

__all__ = ["GuideDialog", "open_guide"]


class _FittingBrowser(QTextBrowser):
    """会**按窗口宽度自适应图片**的说明书正文区。

    QTextBrowser 不会自己缩放图片:说明书里的界面截图有 2000+ 像素宽,直接塞进去
    就要左右拖动才能看全。这里在**窗口尺寸变化时**把每张图的 ``width`` 重算成
    "视口宽度减去一点边距",并只重排一次(带防抖),同时保持原来的滚动位置,
    所以缩放窗口时阅读位置不会跳。

    规则:
    * 不放大:图比窗口窄就按原始宽度显示(避免糊);
    * 不设 ``height``:让 Qt 按比例算高度,图下方不会留一大片空白;
    * 宽度只在小幅变化(>24 px)时才重排,避免拖动窗口时反复刷新。
    """

    _IMG_W = re.compile(r'(<img\b[^>]*?)\bwidth="(\d+)"', re.I)
    _IMG_FULL = re.compile(r'data-full="(\d+)"', re.I)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_html = ""
        self._applied_width = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._reflow)

    # ------------------------------------------------------------ 对外接口
    def set_source_html(self, html: str):
        """设置原始 HTML(不带缩放后的宽度),立刻按当前视口渲染一次。"""
        self._raw_html = html
        self._applied_width = 0
        self._reflow(force=True)

    def resizeEvent(self, event):                        # noqa: N802
        super().resizeEvent(event)
        if self._raw_html:
            self._timer.start()

    # ---------------------------------------------------------------- 内部
    def _viewport_width(self) -> int:
        return max(self.viewport().width() - 26, 320)

    def _fit_images(self, html: str, avail: int) -> str:
        """把每张图的 width 重算成 ``min(原始宽度, 可用宽度)``。

        没有 ``data-full`` 的图(例如作者信息里的二维码)以它**声明**的宽度为
        原始宽度 —— 这样小图不会被拉大成模糊的一整行。
        """
        def repl(m):
            head, cur = m.group(1), int(m.group(2))
            full_m = self._IMG_FULL.search(head)
            full = int(full_m.group(1)) if full_m else cur
            new_w = max(80, min(full, avail))
            return f'{head}width="{new_w}"'

        fitted = self._IMG_W.sub(repl, html)
        # 完全没有 width 的图:补一个宽度(以可用宽度为上限)
        def repl2(m):
            tag = m.group(0)
            if "width=" in tag:
                return tag
            return tag[:-1] + f' width="{avail}">'
        return re.sub(r"<img\b[^>]*>", repl2, fitted)

    def _reflow(self, force: bool = False):
        avail = self._viewport_width()
        if not force and abs(avail - self._applied_width) < 24:
            return
        bar = self.verticalScrollBar()
        frac = (bar.value() / bar.maximum()) if bar.maximum() else 0.0
        anchor = self.anchorAt(self.viewport().rect().topLeft())
        self._applied_width = avail
        # 保留基地址,图片才能按相对路径找到
        base = self.document().baseUrl()
        self.setHtml(self._fit_images(self._raw_html, avail))
        self.document().setBaseUrl(base)
        if bar.maximum():
            bar.setValue(int(frac * bar.maximum()))
        if anchor:
            self.scrollToAnchor(anchor)


class GuideDialog(QDialog):
    """窗口内说明书(目录 + 正文 + 缩放)。"""

    def __init__(self, parent=None, html_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"{author.APP_NAME} 使用说明")
        self.resize(1040, 760)
        self.setMinimumSize(680, 460)

        self._html_path = html_path or author.guide_html_path()
        self._zoom = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---------------------------------------------------------- 顶部工具条
        bar = QHBoxLayout()
        title = QLabel(
            f"<b style='font-size:14pt;'>{author.APP_NAME} 使用说明</b>"
            f"<span style='color:#666;'>　{author.APP_NAME_CN}</span>")
        bar.addWidget(title)
        bar.addStretch(1)
        self.btn_bigger = QPushButton("A+")
        self.btn_bigger.setFixedWidth(44)
        self.btn_bigger.clicked.connect(lambda: self._zoom_by(1))
        self.btn_smaller = QPushButton("A−")
        self.btn_smaller.setFixedWidth(44)
        self.btn_smaller.clicked.connect(lambda: self._zoom_by(-1))
        self.btn_reset = QPushButton("默认字号")
        self.btn_reset.clicked.connect(self._zoom_reset)
        self.btn_browser = QPushButton("用浏览器打开")
        self.btn_browser.setToolTip("在系统默认浏览器里打开同一份说明书")
        self.btn_browser.clicked.connect(self._open_in_browser)
        for b in (self.btn_smaller, self.btn_bigger, self.btn_reset,
                  self.btn_browser):
            b.setMinimumHeight(28)
            bar.addWidget(b)
        root.addLayout(bar)

        # ---------------------------------------------------------- 目录 + 正文
        split = QSplitter(Qt.Horizontal)
        self.toc = QListWidget()
        self.toc.setMaximumWidth(300)
        self.toc.setMinimumWidth(160)
        self.toc.itemClicked.connect(self._goto_item)
        split.addWidget(self.toc)

        self.view = _FittingBrowser()
        self.view.setOpenExternalLinks(True)
        self.view.setOpenLinks(False)          # 内部锚点自己处理,外部链接交给系统
        self.view.anchorClicked.connect(self._on_anchor)
        split.addWidget(self.view)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([240, 800])
        split.splitterMoved.connect(lambda *_: self.view._reflow())
        root.addWidget(split, 1)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(self.reject)
        root.addWidget(box)

        self._load()

    # ------------------------------------------------------------------ 加载
    def _load(self):
        path = self._html_path
        if not path or not os.path.exists(path):
            self.view.setHtml(
                "<h2>没有找到说明书文件</h2>"
                f"<p>预期位置:<code>{path or 'docs/使用说明.html'}</code></p>"
                "<p>源码运行时请先执行 <code>python tools/make_help_html.py</code> "
                "生成它;安装版应位于程序目录的 "
                "<code>_internal\\docs\\使用说明.html</code>。</p>")
            return
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
        # QTextBrowser 不会去磁盘找相对路径图片 —— 给它一个基地址
        base = QUrl.fromLocalFile(os.path.dirname(os.path.abspath(path)) + os.sep)
        self.view.document().setBaseUrl(base)
        # 交给自适应浏览器:它会按当前窗口宽度决定每张图显示多大
        self.view.set_source_html(html)
        self._build_toc(html)

    def _build_toc(self, html: str):
        """从 HTML 的 h2/h3 生成目录(带层级缩进;过长的标题截断)。"""
        self.toc.clear()
        for m in re.finditer(r"<h([23])\s+id=\"([^\"]+)\"[^>]*>(.*?)</h\1>",
                             html, re.S | re.I):
            lvl, sid, raw = m.group(1), m.group(2), m.group(3)
            text = re.sub(r"<[^>]+>", "", raw).replace("↑", "").strip()
            label = text if len(text) <= 26 else text[:25] + "…"
            item = QListWidgetItem(("　" if lvl == "3" else "") + label)
            item.setData(Qt.UserRole, sid)
            item.setToolTip(text)
            if lvl == "2":
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.toc.addItem(item)
        if self.toc.count():
            self.toc.setCurrentRow(0)

    # ------------------------------------------------------------------ 动作
    def _goto_item(self, item: QListWidgetItem):
        sid = item.data(Qt.UserRole)
        if sid:
            self.view.scrollToAnchor(str(sid))

    def _on_anchor(self, url: QUrl):
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
        elif url.hasFragment() or (url.toString() or "").startswith("#"):
            self.view.scrollToAnchor(url.fragment())
        else:
            QDesktopServices.openUrl(url)

    def _zoom_by(self, step: int):
        self._zoom = max(-4, min(8, self._zoom + step))
        self.view.zoomIn(1 if step > 0 else -1)

    def _zoom_reset(self):
        while self._zoom > 0:
            self.view.zoomOut(1)
            self._zoom -= 1
        while self._zoom < 0:
            self.view.zoomIn(1)
            self._zoom += 1

    def _open_in_browser(self):
        if self._html_path and os.path.exists(self._html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._html_path))


def open_guide(parent=None) -> bool:
    """打开说明书窗口;成功返回 ``True``。"""
    path = author.guide_html_path()
    dlg = GuideDialog(parent, html_path=path)
    dlg.exec()
    return bool(path)
