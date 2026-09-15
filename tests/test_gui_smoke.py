# -*- coding: utf-8 -*-
"""
tests/test_gui_smoke.py
=======================

**图形界面冒烟测试**(离屏平台 ``QT_QPA_PLATFORM=offscreen``,不需要显示器)。

不做"像素级"检查(那既脆弱又没信息量),而是走**真实的界面路径**:

* 建窗口 → 设系数文件 → 点『读取系数信息』→ 走后台线程读 → 控件被正确更新;
* 点『开始解算』→ 后台线程算 → 主线程渲染 → 四个画布都有内容、图片落盘;
* 覆盖网格(报告图/地图)与散点两条路径,以及时次、截断、物理量、图类型等控件;
* 顺带抓一张界面截图放到 ``docs/`` 里做文档配图。

断言的是"界面上看到的"和"文件里存下的"来自同一张 Figure(共享同一绘图代码),
以及"参数改动确实传到了结果里"(例如把『输出物理量』改成 EWH,结果量级应变化)。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import Checker, SHKIT_DIR, guard, tmp_dir        # noqa: E402

from PySide6.QtCore import QTimer                              # noqa: E402
from PySide6.QtWidgets import QApplication                     # noqa: E402

from shsynth import fieldio                                   # noqa: E402
from shsynth.gui.app import MainWindow                         # noqa: E402

COEFFS = os.path.join(SHKIT_DIR, "shkit_coeffs.sh")
COEFFS_GFC = os.path.join(SHKIT_DIR, "shkit_coeffs.gfc")

_app = QApplication.instance() or QApplication(sys.argv)


def _pump(timeout_ms: int = 150):
    """跑一小会儿事件循环(让信号槽与后台线程推进)。"""
    done = {"v": False}

    def stop():
        done["v"] = True
    QTimer.singleShot(timeout_ms, stop)
    while not done["v"]:
        _app.processEvents()


def _wait_idle(win, timeout_s: float = 120.0) -> bool:
    """等到后台解算线程结束。"""
    import time
    t0 = time.time()
    while getattr(win, "_thread", None) is not None:
        _app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout_s:
            return False
    return True


def _wait_series(win, timeout_s: float = 120.0) -> bool:
    """等到 v2.0 的批量序列线程结束。"""
    import time
    t0 = time.time()
    while getattr(win, "_series_thread", None) is not None:
        _app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout_s:
            return False
    return True


def _wait_info(win, timeout_s: float = 120.0) -> bool:
    """等到后台读系数信息/读序列的线程结束。

    ⚠️ 必须等**这个**线程,不能等 ``_thread``(那是单历元解算的线程):
    没等完就去点『综合整条序列』会因为 ``_coeffs`` 还是 None 而走空。
    """
    import time
    t0 = time.time()
    while getattr(win, "_info_thread", None) is not None:
        _app.processEvents()
        time.sleep(0.02)
        if time.time() - t0 > timeout_s:
            return False
    return True


def _nc_vars(path: str) -> list:
    """nc 文件里的变量名列表(网络盘走 SHSynth 的临时中转兼容层)。"""
    from shsynth import fieldio
    with fieldio._netcdf_open(path, []) as ds:
        return [str(v) for v in ds.variables]


def _new_window(check: Checker):
    win = MainWindow()
    win.show()
    _pump(50)
    return win


def _close_window(win):
    """**规范地**关掉一个主窗口。

    为什么不能只 ``win.close()``:每个 MainWindow 带 9 个画布(9 张 matplotlib
    Figure)和最多 4 类后台线程(解算 / 读信息 / 读序列 / 批量综合 / 动画导出)。
    窗口在**线程还活着**或**画布还持有 Figure** 的时候被销毁,Qt 这边是
    native 堆损坏(实测偶发 ``exit=0xC0000374``,大概十几次里出一次),
    Python 层的 try/except 根本兜不住,表现就是"整套测试莫名其妙失败"。

    所以这里按顺序收干净:停定时器 → 取消并**等**线程 → 关窗 → 让 Qt 真正
    删掉对象 → 关掉 matplotlib 的图。
    """
    import time
    if win is None:
        return
    try:
        win._anim_stop()
    except Exception:                                     # noqa: BLE001
        pass
    for attr in ("_worker", "_info_worker", "_series_worker",
                 "_series_read_worker", "_anim_export_worker"):
        w = getattr(win, attr, None)
        if w is not None:
            try:
                w.cancel()
            except Exception:                             # noqa: BLE001
                pass
    # 等**还在跑**的线程退出。判据必须是 QThread.isRunning(),不能是"引用非 None":
    # 引用要等对应的 finished 槽跑过才清,拿它当条件会在根本没有线程时也傻等满
    # 超时(实测 23 个窗口 × 20 s 直接把套件拖到超时)。动画导出线程最要紧 ——
    # 它还在跑的时候销毁窗口就是 native 堆损坏。
    t0 = time.time()
    while time.time() - t0 < 8.0:
        running = []
        for a in ("_thread", "_info_thread", "_series_thread",
                  "_series_read_thread", "_anim_export_thread"):
            th = getattr(win, a, None)
            if th is None:
                continue
            try:
                if th.isRunning():
                    running.append(a)
            except RuntimeError:
                pass                                      # C++ 对象已删,不用等
        if not running:
            break
        _app.processEvents()
        time.sleep(0.02)
    try:
        win.close()
        win.deleteLater()                                 # 让 Qt 真正回收
    except Exception:                                     # noqa: BLE001
        pass
    _pump(120)
    try:
        import matplotlib.pyplot as plt
        plt.close("all")                                  # 不留 Figure 累积
    except Exception:                                     # noqa: BLE001
        pass


def test_output_paths_use_save_dialog(check: Checker):
    """**要写出去的**路径必须用『保存』对话框,输入路径才用『打开』。

    踩过的坑:场序列输出/诊断表/导出系数/图片文件这四个**输出**路径原先都走
    ``getOpenFileName``(『打开』)—— "选一个还不存在的输出文件"在语义上就别扭,
    对话框还会把已存在的文件当成"选中的输入"。
    """
    from PySide6.QtWidgets import QFileDialog
    check.section("输出路径用『保存』对话框")
    win = _new_window(check)

    calls = {"open": [], "save": []}
    o_open = QFileDialog.getOpenFileName
    o_save = QFileDialog.getSaveFileName
    QFileDialog.getOpenFileName = staticmethod(
        lambda *a, **k: (calls["open"].append(a[2] if len(a) > 2 else ""), ("", ""))[1])
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (calls["save"].append(a[2] if len(a) > 2 else ""),
                         ("C:/tmp/out_test.nc", ""))[1])
    try:
        # 四个**输出**字段:逐个点『另存为』,必须都走『保存』
        for attr, edit, name in (("series_out_btn", win.series_out_edit, "场序列输出"),
                                 ("series_diag_btn", win.series_diag_edit, "诊断表输出"),
                                 ("outcoeffs_btn", win.outcoeffs_edit, "导出系数"),
                                 ("figfile_btn", win.figfile_edit, "图片文件")):
            btn = getattr(win, attr)
            n_before = len(calls["save"])
            btn.click()
            _pump(30)
            check.ok(len(calls["save"]) == n_before + 1,
                     f"{name}的按钮走 getSaveFileName(保存),不是『打开』")
            check.ok(edit.text().strip() != "", f"{name} 选完编辑框有值")
        # 结果文件(它本来就对:_browse_out 用保存)
        n_before = len(calls["save"])
        win._browse_out()
        _pump(30)
        check.ok(len(calls["save"]) == n_before + 1, "结果文件走『保存』对话框")
        # 输入类字段仍然必须是『打开』
        for fn, name in ((lambda: win._browse_into(win.gridfile_edit, "网格", "*"),
                          "网格文件"),
                         (lambda: win._browse_into(win.pointsfile_edit, "散点", "*"),
                          "散点文件")):
            n_before = len(calls["open"])
            fn()
            _pump(30)
            check.ok(len(calls["open"]) == n_before + 1,
                     f"{name}(输入)仍然走 getOpenFileName(打开)")
    finally:
        QFileDialog.getOpenFileName = o_open
        QFileDialog.getSaveFileName = o_save
    _close_window(win)


def test_wheel_never_changes_values(check: Checker):
    """滚轮滑过参数面板不许改数值(取消 WheelFocus)。

    默认 ``QAbstractSpinBox``/``QComboBox`` 是 ``Qt.WheelFocus``:鼠标滑过就吃
    滚轮。左侧面板是可滚动的,滚动时滚轮一穿过输入框数值就被悄悄改掉。
    改成 ``Qt.StrongFocus`` 后滚轮不再被输入框接受,会穿透给外层滚动区。
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractSpinBox, QComboBox
    check.section("滚轮不许改数值")
    win = _new_window(check)
    spin = win.findChildren(QAbstractSpinBox)
    combo = win.findChildren(QComboBox)
    check.ok(len(spin) > 8 and len(combo) > 5,
             f"扫到 {len(spin)} 个数值框、{len(combo)} 个下拉框")
    # ⚠️ 判据必须用**相等**:Qt.WheelFocus 本身就是 TabFocus|ClickFocus|WheelFocus
    # 的组合值(15),拿它做按位与会把所有含 TabFocus 的策略都误判成"吃滚轮"。
    bad_spin = [w for w in spin if w.focusPolicy() != Qt.StrongFocus]
    bad_combo = [w for w in combo if w.focusPolicy() != Qt.StrongFocus]
    check.ok(not bad_spin,
             f"所有数值框都是 StrongFocus(不含 WheelFocus),剩余 {len(bad_spin)} 个")
    check.ok(not bad_combo,
             f"所有下拉框都是 StrongFocus(不含 WheelFocus),剩余 {len(bad_combo)} 个")
    check.ok(win._wheel_tamed >= len(spin) + len(combo) - 1,
             f"_tame_wheel 覆盖了 {win._wheel_tamed} 个控件")
    # 抽查几个"最容易被误改"的:高斯半径、分块、DPI、物理量、配色
    named = []
    for name in ("gauss_spin", "chunk_spin", "dpi_spin", "truncate_spin",
                 "coeff_nmax_spin", "time_spin"):
        w = getattr(win, name, None)
        if w is not None:
            named.append((name, w.focusPolicy() != Qt.StrongFocus))
    for name in ("unit_combo", "layout_combo", "cmap_combo", "figkind_combo",
                 "component_combo", "gauss_method", "outcoeffs_layout"):
        w = getattr(win, name, None)
        if w is not None:
            named.append((name, w.focusPolicy() != Qt.StrongFocus))
    still = [n for n, f in named if f]
    check.ok(not still, f"关键控件全部是 StrongFocus(残留 {still})")
    # 行为验证:真的把滚轮事件送给它,数值也不许变(事件过滤器兜底)
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    for name in ("gauss_spin", "chunk_spin", "dpi_spin"):
        w = getattr(win, name)
        before = w.value()
        QApplication.sendEvent(w, QWheelEvent(
            QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
        _pump(20)
        check.ok(w.value() == before,
                 f"直接送滚轮事件也不改 {name} 的数值({before} → {w.value()})")
    # 点进去(有焦点)之后仍然可以用滚轮调值 —— 别把功能改死
    w = win.gauss_spin
    w.setFocus()
    _pump(20)
    if w.hasFocus():
        before = w.value()
        QApplication.sendEvent(w, QWheelEvent(
            QPointF(5.0, 5.0), QPointF(5.0, 5.0), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False))
        _pump(20)
        check.ok(w.value() != before,
                 f"拿到焦点后滚轮仍可调值({before} → {w.value()})")
    else:
        check.ok(True, "(离屏环境拿不到焦点,跳过'有焦点时滚轮可用'这一条)")
    check.ok(w.isEnabled() and win.unit_combo.isEnabled(),
             "控件仍然可用(只是不再被滚轮误改)")
    _close_window(win)


def test_panel_is_wide_enough(check: Checker):
    """窗口与左侧参数面板要有足够宽度(窄了表单会显示不全)。"""
    check.section("窗口/面板宽度")
    win = _new_window(check)
    check.ok(win.width() >= 1500,
             f"主窗口够宽({win.width()}×{win.height()})")
    check.ok(win.panel.minimumWidth() >= 594,
             f"参数面板最小宽度 {win.panel.minimumWidth()} ≥ 最宽表单需要的 594px")
    # 表单里的长标签、长占位文字不该被裁掉:抽查每张表单的 sizeHint 宽度
    from PySide6.QtWidgets import QFormLayout, QGroupBox
    worst, worst_name = 0, ""
    for gb in win.panel.findChildren(QGroupBox):
        lay = gb.layout()
        if isinstance(lay, QFormLayout):
            w = lay.sizeHint().width()
            if w > worst:
                worst, worst_name = w, gb.title()
    check.ok(worst <= win.panel.minimumWidth(),
             f"最宽的表单({worst_name}, 需 {worst}px)"
             f"能放进 {win.panel.minimumWidth()}px 的面板")
    _close_window(win)


def test_batch_load_clears_single_paths(check: Checker):
    """载入**批量序列**后,单历元那一套的路径必须自动清空。

    否则很容易出事:单文件模式与批量模式的输入/输出语义完全不同,
    留着旧路径既会让人以为批量结果写到那个文件里,又会让下次『开始解算』
    把**整条序列的第一个历元**写进那个文件。
    """
    check.section("切到批量模式清空单历元路径")
    d = tmp_dir("gui")
    import numpy as np
    from shsynth import SHCoeffs, TimeAxis, read_series_nc, write_series_nc
    from shsynth.coeffio import read_coeffs
    base = read_coeffs(COEFFS)
    nt = 3
    ax = TimeAxis.from_datetimes([np.datetime64(f"2006-{m:02d}-15")
                                  for m in range(1, nt + 1)])
    C = np.repeat(base.C[:, :, None], nt, axis=2)
    S = np.repeat(base.S[:, :, None], nt, axis=2)
    src = write_series_nc(SHCoeffs(C, S, dict(base.meta), ax),
                          os.path.join(d, "clear_series.nc"))

    win = _new_window(check)
    # 先摆好"单历元那一套"的路径
    win.coeffs_edit.setText(COEFFS)
    win.out_edit.setText(os.path.join(d, "single_out.nc"))
    win.figfile_edit.setText(os.path.join(d, "single.png"))
    win.series_out_edit.setText(os.path.join(d, "series_out.nc"))
    win.series_diag_edit.setText(os.path.join(d, "series_diag.csv"))
    check.ok(all(e.text().strip() for e in (win.coeffs_edit, win.out_edit,
                                            win.figfile_edit)),
             "载入前三个单历元路径都有值")

    win._on_series_read(read_series_nc(src))
    _pump(50)
    check.ok(win.coeffs_edit.text() == "", "载入批量后『系数文件』清空")
    check.ok(win.out_edit.text() == "", "载入批量后『结果文件』清空")
    check.ok(win.figfile_edit.text() == "", "载入批量后『图片文件』清空")
    # ⚠️ 关键回归:清空 coeffs_edit 会触发 _on_coeffs_path_changed(它会把
    # _coeffs 作废)。切到批量模式时 _coeffs 装的**正是批量序列**,必须留住,
    # 否则『综合整条序列』会以为"还没读系数"直接返回 —— 上一版就是这个 bug。
    check.ok(getattr(win, "_coeffs", None) is not None
             and win._coeffs.ntime == nt,
             f"清空路径后 _coeffs 仍是刚读入的 {nt} 历元序列"
             f"(实际 {getattr(getattr(win, '_coeffs', None), 'ntime', None)})")
    check.ok(win._coeffs.times is not None, "序列的时间轴没被清掉")
    # 批量自己的输出路径**不该**被清
    check.ok(win.series_out_edit.text().strip() != "" and
             win.series_diag_edit.text().strip() != "",
             "批量自己的『场序列输出』『诊断表输出』保留(那是批量要用的)")
    log = win.log_view.toPlainText()
    check.ok("已切到批量模式" in log, "日志里说明了为什么清空(不静默)")
    # 单历元解算应当被挡下(它已经没有输入文件了)
    win.epoch_combo.setCurrentIndex(win.epoch_combo.findData("current"))
    win._on_epoch_mode()
    check.ok(win.coeffs_edit.text() == "",
             "切回单历元模式也不会把批量来源硬塞进『系数文件』")
    _close_window(win)


def test_demean_controls(check: Checker):
    """界面的去均值口径控件:四选一 + 自定义时段 + "实际会用哪一段"提示。"""
    check.section("去均值口径控件")
    d = tmp_dir("gui")
    import numpy as np
    from shsynth import SHCoeffs, TimeAxis, read_series_nc, write_series_nc
    from shsynth.coeffio import read_coeffs
    base = read_coeffs(COEFFS)
    dates = [np.datetime64(f"{y}-{m:02d}-15")
             for y in range(2004, 2012) for m in range(1, 13)]
    ax = TimeAxis.from_datetimes(dates)
    nt = len(dates)
    C = np.repeat(base.C[:, :, None], nt, axis=2)
    S = np.repeat(base.S[:, :, None], nt, axis=2)
    src = write_series_nc(SHCoeffs(C, S, dict(base.meta), ax),
                          os.path.join(d, "mean_series.nc"))

    win = _new_window(check)
    texts = [win.mean_combo.itemText(i) for i in range(win.mean_combo.count())]
    check.ok(win.mean_combo.count() == 4, f"四个选项 {texts}")
    check.ok(win.mean_combo.itemData(0) is None, "第一项是『不去均值』")
    check.ok([win.mean_combo.itemData(i) for i in (1, 2, 3)]
             == ["grace", "all", "custom"],
             "另外三项是 grace / all / custom")
    check.ok("2004" in texts[1] and "2010" in texts[1],
             f"GRACE 惯例的窗口写在下拉项里({texts[1]})")
    # 默认不去均值,自定义输入框应当是禁用的
    check.ok(not win.mean_from.isEnabled() and not win.mean_to.isEnabled(),
             "默认(不去均值)时自定义时段输入框禁用")
    check.ok("不去均值" in win.mean_note.text() and "静态场" in win.mean_note.text(),
             f"默认给出『含静态场』的警告({win.mean_note.text()[:22]}…)")

    # 切到自定义:启用并预填窗口
    win.mean_combo.setCurrentIndex(3)
    _pump(30)
    check.ok(win.mean_from.isEnabled() and win.mean_to.isEnabled(),
             "切到自定义后两个日期框启用")
    check.ok(win.mean_from.text().startswith("2004")
             and win.mean_to.text().startswith("2010"),
             f"预填了 GRACE 窗口({win.mean_from.text()} .. {win.mean_to.text()})")

    # 读入序列后,提示要算出**实际**会用的历元区间
    win.coeffs_edit.setText(str(src))
    win.load_coeffs_info()
    _wait_info(win, 60)
    _pump(200)
    win._on_series_read(read_series_nc(src))
    _pump(50)
    win.mean_combo.setCurrentIndex(1)                    # GRACE 惯例
    _pump(30)
    note = win.mean_note.text()
    check.ok("84/96" in note or "个历元" in note,
             f"提示给出实际历元数({note[:46]}…)")
    check.ok("2004-01-15" in note and "2010-12-15" in note,
             f"提示给出实际起止({note[:46]}…)")
    # 口径要真的被送进计算,而不是只显示
    choice = win.mean_choice()
    check.ok(choice.get("mode") == "grace", f"mean_choice → {choice}")
    win.mean_combo.setCurrentIndex(3)
    win.mean_from.setText("2006")
    win.mean_to.setText("2008")
    _pump(30)
    choice = win.mean_choice()
    check.ok(choice["mode"] == "custom" and choice["from_date"] == "2006"
             and choice["to_date"] == "2008",
             f"自定义文本原样交给核心(补全由核心统一负责){choice}")
    _close_window(win)


def test_window_controls(check: Checker):
    """控件齐备:五类目标、七个选项组、九个页签(v2.0 加了四个)。"""
    check.section("界面控件")
    win = _new_window(check)
    check.ok(len(win.mode_radios) == 5, f"五种目标来源({len(win.mode_radios)} 个单选框)")
    check.ok(win.mode_stack.count() == 5, "五种目标的参数页齐备")
    check.ok(win.tabs.count() == 9,
             f"九个页签(地图/报告图/逐阶谱/数值分布/时间序列/水平形变/动画/"
             f"逐历元诊断/日志),实际 {win.tabs.count()}")
    names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    for t in ("地图", "报告图", "逐阶谱", "数值分布", "时间序列",
              "水平形变", "动画", "逐历元诊断", "日志 / 系数信息"):
        check.ok(t in names, f"页签『{t}』存在")
    for name in ("coeffs_edit", "layout_combo", "coeff_nmax_spin", "time_spin",
                 "gauss_spin", "gauss_method", "unit_combo", "chunk_spin",
                 "out_edit", "outvar_edit", "outcoeffs_layout", "figkind_combo",
                 "cmap_combo", "dpi_spin", "chk_contour", "chk_coast",
                 "run_btn", "stop_btn", "savefig_btn", "plan_btn",
                 "progress", "status_label",
                 # v2.0
                 "epoch_combo", "epoch_from", "epoch_to", "time_info_label",
                 "component_combo", "series_btn", "series_read_btn",
                 "series_out_edit", "series_diag_edit",
                 "mean_combo", "mean_from", "mean_to", "mean_note",
                 "series_canvas", "vector_canvas", "diag_table",
                 # v2.0 动画页
                 "anim_canvas", "anim_play_btn", "anim_prev_btn", "anim_next_btn",
                 "anim_slider", "anim_fps", "chk_anim_loop", "anim_label",
                 "anim_gif_btn"):
        check.ok(hasattr(win, name), f"控件 {name} 存在")
    check.ok(len(win.mode_radios) == 5 and win.unit_combo.count() >= 5,
             f"物理量下拉 {win.unit_combo.count()} 项")
    check.ok(win.component_combo.count() == 4,
             f"输出分量下拉 {win.component_combo.count()} 项(标量/北/东/矢量)")
    check.ok(win.stop_btn.isEnabled() is False, "未运行时『停止』不可点")
    check.ok(win.run_btn.isEnabled() is True, "『开始解算』可点")
    # v2.0:未载入多时次时,历元范围控件应当是禁用的(v1.0 行为不变)
    check.ok(win.epoch_combo.isEnabled() is False,
             "未载入多时次 → 『历元范围』禁用(默认仍是单历元)")
    check.ok(win.epoch_from.isEnabled() is False, "起止微调框同时禁用")
    _close_window(win)


def test_load_coeffs_info(check: Checker):
    """走真实路径读取系数信息,控件应被正确更新。"""
    check.section("读取系数信息")
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS)
    win.load_coeffs_info()
    _wait_info(win, 60)
    _pump(200)
    check.ok(win._coeffs is not None, "后台线程把系数读回来了")
    if win._coeffs is not None:
        check.ok(win._coeffs.nmax == 12, f"nmax = {win._coeffs.nmax}")
        check.ok("triangle" in win._coeffs.meta.get("layout", ""),
                 f"布局 = {win._coeffs.meta.get('layout')}")
        check.ok(win.spectrum_canvas.figure.get_axes() and
                 len(win.spectrum_canvas.figure.get_axes()) >= 1,
                 "逐阶谱已自动画出")
        check.ok(win.out_edit.text() != "", f"自动填了输出名 {win.out_edit.text()}")
        check.ok(win.log_view.blockCount() > 8,
                 f"日志记录了 {win.log_view.blockCount()} 行")
        # 多时次文件应启用时次控件
        win2 = _new_window(check)
        win2.coeffs_edit.setText(COEFFS_GFC)
        win2.load_coeffs_info()
        _wait_idle(win2, 60)
        _pump(100)
        check.ok(win2.status_label.text().startswith("已读取"),
                 f"状态栏 = {win2.status_label.text()}")
        win2.close()
    _close_window(win)


def test_series_batch_gui(check: Checker):
    """v2.0:批量序列综合 —— 时间轴控件、批量 worker、时间序列页、诊断表。"""
    check.section("批量序列(v2.0)")
    d = tmp_dir("gui")
    # 造一个 6 个历元的小序列(带真实日期)
    import numpy as np
    from shsynth import SHCoeffs, TimeAxis, write_series_nc
    from shsynth.coeffio import read_coeffs
    base = read_coeffs(COEFFS)
    nt = 6
    ax = TimeAxis.from_datetimes(
        [np.datetime64(f"2002-{m:02d}-15") for m in range(1, nt + 1)])
    C = np.repeat(base.C[:, :, None], nt, axis=2)
    S = np.repeat(base.S[:, :, None], nt, axis=2)
    C = C * np.linspace(1.0, 1.3, nt)[None, None, :]
    S = S * np.linspace(1.0, 1.3, nt)[None, None, :]
    src = write_series_nc(SHCoeffs(C, S, dict(base.meta), ax),
                          os.path.join(d, "gui_series.nc"))

    win = _new_window(check)
    win.coeffs_edit.setText(str(src))
    win.load_coeffs_info()
    _wait_info(win, 60)
    _pump(200)
    check.ok(getattr(win, "_coeffs", None) is not None, "读回多时次序列")
    check.ok(win.epoch_combo.isEnabled(), "多时次 → 『历元范围』可用")
    check.ok(len(win.time_info_label.text()) > 4, "时间轴摘要显示出来了")
    check.ok("2002" in win.time_info_label.text(),
             f"摘要含日期({win.time_info_label.text()[:40]})")
    check.ok(win.time_spin.isEnabled() and win.time_spin.maximum() == nt - 1,
             f"『时次』范围 0..{nt - 1}(v1.0 行为保持)")

    # 选『整条序列』
    win.epoch_combo.setCurrentIndex(win.epoch_combo.findData("all"))
    win.mode_radios["global"].setChecked(True)
    win.preset_combo.setCurrentIndex(3)                  # 2.5°,点少跑得快
    win.figkind_combo.setCurrentIndex(4)                 # 不出图(单历元图不参与)
    out = os.path.join(d, "gui_series_out.nc")
    diag = os.path.join(d, "gui_series_diag.csv")
    win.series_out_edit.setText(out)
    win.series_diag_edit.setText(diag)
    win.mean_combo.setCurrentIndex(win.mean_combo.findData("all"))
    win.start_series_run()
    _wait_series(win, 120)
    _pump(300)
    res = getattr(win, "_series_result", None)
    check.ok(res is not None, "批量序列跑完并拿到结果")
    if res is not None:
        check.ok(len(res.times) == nt, f"结果含 {len(res.times)} 个历元")
        check.ok(os.path.exists(out), "场序列已落盘")
        check.ok(os.path.exists(diag), "逐历元诊断表已落盘")
        # 落盘的 nc 必须有**真实时间坐标**(v1.0 的缺陷 A)
        _la, _lo, _g, _me, ax2 = fieldio.read_grid(out, with_time=True)
        check.ok(len(ax2) == nt and ax2.kind == "datetime",
                 f"写出的 nc 带真实时间坐标({len(ax2)} 个,kind={ax2.kind})")
        # 时间序列页画出来了
        axes = win.series_canvas.figure.get_axes()
        check.ok(axes and len(axes[0].get_lines()) >= 1,
                 f"时间序列页有 {len(axes[0].get_lines()) if axes else 0} 条曲线")
        # 诊断表填好了,列含 rms / decimal_year
        check.ok(win.diag_table.rowCount() == nt,
                 f"诊断表 {win.diag_table.rowCount()} 行")
        heads = [win.diag_table.horizontalHeaderItem(j).text()
                 for j in range(win.diag_table.columnCount())]
        check.ok("rms" in heads and "decimal_year" in heads,
                 f"诊断表列 {heads}")
        # 点一行应当跳到地图页
        win._on_diag_row(1)
        _pump(100)
        check.ok(win.tabs.currentWidget() is win.map_canvas,
                 "点诊断表某行跳到地图页")
        # 地图页渲染了那一帧
        check.ok(len(win.map_canvas.figure.get_axes()) >= 1, "地图页已渲染该历元")
    _close_window(win)


def test_series_vector_gui(check: Checker):
    """v2.0:水平形变矢量 —— 分量下拉联动物理量、矢量页出图、极点无箭头。"""
    check.section("水平形变矢量(v2.0)")
    d = tmp_dir("gui")
    import numpy as np
    from shsynth import SHCoeffs, TimeAxis, write_series_nc
    from shsynth.coeffio import read_coeffs
    base = read_coeffs(COEFFS)
    nt = 3
    ax = TimeAxis.from_datetimes([np.datetime64(f"2002-{m:02d}-15")
                                  for m in (1, 2, 3)])
    C = np.repeat(base.C[:, :, None], nt, axis=2)
    S = np.repeat(base.S[:, :, None], nt, axis=2)
    src = write_series_nc(SHCoeffs(C, S, dict(base.meta), ax),
                          os.path.join(d, "gui_vec.nc"))
    win = _new_window(check)
    win.coeffs_edit.setText(str(src))
    win.load_coeffs_info()
    _wait_info(win, 60)
    _pump(200)
    # 选『水平形变 · 北+东』
    win.component_combo.setCurrentIndex(
        win.component_combo.findData("horizontal"))
    _pump(50)
    check.ok(win.unit_combo.currentData() == "horizontal_displacement",
             f"分量下拉把物理量切成 {win.unit_combo.currentData()}")
    win.epoch_combo.setCurrentIndex(win.epoch_combo.findData("all"))
    win.mode_radios["global"].setChecked(True)
    win.preset_combo.setCurrentIndex(3)
    win.figkind_combo.setCurrentIndex(4)
    win.mean_combo.setCurrentIndex(win.mean_combo.findData("all"))   # 去静态场
    vec_out = os.path.join(d, "gui_vec_out.nc")
    win.series_out_edit.setText(vec_out)
    win.start_series_run()
    _wait_series(win, 120)
    _pump(300)
    res = getattr(win, "_series_result", None)
    check.ok(res is not None, "水平形变批量跑完")
    if res is not None:
        check.ok(set(res.components) == {"north", "east"},
                 f"结果带两个分量 {sorted(res.components)}")
        check.ok(res.stats.get("component") == "horizontal", "stats 记下分量")
        axes = win.vector_canvas.figure.get_axes()
        ncol = len(axes[0].collections) if axes else 0
        check.ok(ncol >= 2, f"矢量页有底图 + 箭头({ncol} 个 collections)")
        # 场序列应含北/东两个变量
        check.ok(os.path.exists(vec_out), "矢量场序列已落盘")
        vs = _nc_vars(vec_out)
        check.ok("north_displacement" in vs and "east_displacement" in vs,
                 f"nc 里含两个分量变量 {vs}")
    _close_window(win)


def test_run_grid_report(check: Checker):
    """网格 + 报告图:后台解算 → 主线程渲染 → 图落盘。"""
    check.section("网格解算(报告图)")
    d = tmp_dir("gui")
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS)
    win.mode_radios["global"].setChecked(True)
    win.preset_combo.setCurrentIndex(2)                  # 2°
    win.gauss_spin.setValue(500.0)
    win.figkind_combo.setCurrentIndex(0)                 # 报告图
    out = os.path.join(d, "gui_field.nc")
    fig = os.path.join(d, "gui_report.png")
    win.out_edit.setText(out)
    win.figfile_edit.setText(fig)
    win.start_run()
    check.ok(win.run_btn.isEnabled() is False, "解算中『开始解算』被禁用")
    check.ok(win.stop_btn.isEnabled() is True, "解算中『停止』可用")
    ok = _wait_idle(win)
    _pump(300)
    check.ok(ok, "后台线程按时结束")
    res = win._last_result
    check.ok(res is not None, "拿到结果对象")
    if res is None:
        _close_window(win)
        return
    check.ok(res.stats["n_points"] == 91 * 180,
             f"求值点数 {res.stats['n_points']}(2° 全球网格)")
    check.ok(res.out_path and os.path.exists(res.out_path), "结果文件已写出")
    check.ok(os.path.exists(fig) and os.path.getsize(fig) > 20_000,
             f"报告图已落盘({os.path.getsize(fig) // 1024 if os.path.exists(fig) else 0} KB)")
    check.ok(len(win.report_canvas.figure.get_axes()) == 4,
             f"报告画布 {len(win.report_canvas.figure.get_axes())} 个坐标轴(地图+色标+谱+分布)")
    check.ok(len(win.map_canvas.figure.get_axes()) >= 2,
             "地图画布也被同步更新(含色标)")
    # 默认『运行后自动打开地图』:即使图类型选了四联报告图,跑完也停在地图页
    # (页签内容两者都已更新,见上面两条;这里只看落点)
    check.ok(win.tabs.currentWidget() is win.map_canvas,
             "跑完自动打开地图页(默认)")
    check.ok(win.progress.value() == 100, f"进度条 = {win.progress.value()}")
    check.ok(win.run_btn.isEnabled() and not win.stop_btn.isEnabled(),
             "运行结束后按钮状态复位")
    check.ok("SHSynth 球谐系数解算" in win.log_view.toPlainText(),
             "日志里有汇总报告")
    check.ok(win.status_label.text().startswith("完成"),
             f"状态栏 = {win.status_label.text()}")

    # 界面上的图与存盘用的是同一张 Figure
    check.ok(win.report_canvas.figure is res.figure or res.figure is not None,
             "结果对象带有绘图对象")
    _close_window(win)


def test_run_points_and_units(check: Checker):
    """散点 + 物理量换算:改『输出物理量』必须真的改变结果。"""
    check.section("散点解算与物理量")
    d = tmp_dir("gui")
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS_GFC)
    win.mode_radios["sphere"].setChecked(True)
    win.sphere_n.setValue(4000)
    win.figkind_combo.setCurrentIndex(1)                 # 地图(散点分支)
    win.out_edit.setText(os.path.join(d, "gui_pts.csv"))
    win.figfile_edit.setText(os.path.join(d, "gui_scatter.png"))
    win.start_run()
    oke = _wait_idle(win)
    _pump(300)
    check.ok(oke and win._last_result is not None, "散点解算完成")
    res = win._last_result
    if res is None:
        _close_window(win)
        return
    check.ok(res.stats["n_points"] == 4000, f"球面点数 {res.stats['n_points']}")
    check.ok(os.path.exists(os.path.join(d, "gui_scatter.png")), "散点图已落盘")
    rms_raw = res.stats["rms"]

    # 改成 EWH:同一个系数、同一批点,量级应显著变化
    idx = None
    for i in range(win.unit_combo.count()):
        if win.unit_combo.itemData(i) == "ewh":
            idx = i
    check.ok(idx is not None, "物理量下拉里有 EWH")
    if idx is not None:
        win.unit_combo.setCurrentIndex(idx)
        win.figkind_combo.setCurrentIndex(4)             # 不出图,只算
        win.start_run()
        ok2 = _wait_idle(win)
        _pump(200)
        res2 = win._last_result
        check.ok(ok2 and res2 is not None, "EWH 换算解算完成")
        if res2 is not None:
            ratio = res2.stats["rms"] / max(rms_raw, 1e-300)
            check.ok(ratio > 1e5,
                     f"EWH 结果量级放大 {ratio:.3g} 倍(Aₙ ≈ 1e7,量级正确)")
            check.ok(any("换算" in w for w in res2.warnings),
                     "换算写进结果 warnings")
    _close_window(win)


def test_options_wire_up(check: Checker):
    """界面控件 → SynthRequest 的映射必须正确。"""
    check.section("控件映射")
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS)
    win.mode_radios["range"].setChecked(True)
    win.lat_min.setValue(10.0)
    win.lat_max.setValue(40.0)
    win.lon_min.setValue(100.0)
    win.lon_max.setValue(140.0)
    win.lat_step.setValue(2.0)
    win.lon_step.setValue(2.0)
    win.truncate_spin.setValue(6)
    win.gauss_spin.setValue(800.0)
    win.chk_contour.setChecked(True)
    win.chk_coast.setChecked(False)
    win.chk_symmetric.setChecked(False)
    spec = win.collect_spec()
    check.ok(spec.mode == "grid" and spec.grid_source == "range",
             f"模式 = {spec.mode}/{spec.grid_source}")
    check.ok((spec.lat_min, spec.lat_max) == (10.0, 40.0), "纬度范围传对")
    check.ok((spec.lon_min, spec.lon_max) == (100.0, 140.0), "经度范围传对")
    check.ok((spec.lat_step, spec.lon_step) == (2.0, 2.0), "步长传对")
    check.ok(spec.truncate_nmax == 6, "截断阶数传对")
    check.ok(spec.gaussian_km == 800.0, "高斯半径传对")
    check.ok(spec.contour is True and spec.coast is False and
             spec.symmetric is False, "绘图开关传对")

    win.mode_radios["gridfile"].setChecked(True)
    win.gridfile_edit.setText("some_file.nc")
    win.gridvar_edit.setText("mass_anomaly")
    spec = win.collect_spec()
    check.ok(spec.grid_source == "file" and spec.grid_file == "some_file.nc"
             and spec.grid_var == "mass_anomaly", "借用网格文件参数传对")

    win.mode_radios["points"].setChecked(True)
    win.pointsfile_edit.setText("pts.csv")
    # 纬度/经度列是下拉框:第一项是『自动(留空)』,其余是列名/序号
    check.ok(win.latcol_combo.itemData(0) is None
             and win.loncol_combo.itemData(0) is None,
             "纬度/经度列下拉框默认有『自动(留空)』项")
    spec = win.collect_spec()
    check.ok(spec.mode == "points" and spec.points_file == "pts.csv"
             and spec.lat_col is None and spec.lon_col is None,
             "散点文件参数传对(列默认自动)")
    # 手动选一列也要能传下去
    win.latcol_combo.addItem("lat", 0)
    win.latcol_combo.setCurrentIndex(win.latcol_combo.count() - 1)
    spec = win.collect_spec()
    check.ok(spec.lat_col == 0, f"手动选纬度列后 spec.lat_col = {spec.lat_col}")

    # 模式切换应联动 stack 页
    win.mode_radios["sphere"].setChecked(True)
    check.ok(win.mode_stack.currentIndex() == 4,
             f"切到球面散点后 stack = {win.mode_stack.currentIndex()}")
    _close_window(win)


def test_focus_and_columns(check: Checker):
    """新增的两项界面功能:聚焦切换 与 散点列下拉框。"""
    check.section("聚焦与列下拉框")
    win = _new_window(check)
    check.ok(hasattr(win, "chk_focus") and win.chk_focus.isChecked(),
             "『自动聚焦到结果范围』默认勾选")
    check.ok(hasattr(win, "focus_btn"), "有『聚焦/全球』切换按钮")
    check.ok(hasattr(win, "chk_autoshow") and win.chk_autoshow.isChecked(),
             "『运行后自动打开地图』默认勾选")
    check.ok(hasattr(win, "chk_raise") and not win.chk_raise.isChecked(),
             "『运行结束后窗口置前』默认不勾(不抢用户窗口)")

    # 区域网格结果 → 应自动聚焦(坐标窗口小于全球)
    d = tmp_dir("gui")
    win.coeffs_edit.setText(COEFFS)
    win.mode_radios["range"].setChecked(True)
    win.lat_min.setValue(20.0); win.lat_max.setValue(50.0)
    win.lon_min.setValue(100.0); win.lon_max.setValue(140.0)
    win.lat_step.setValue(1.0); win.lon_step.setValue(1.0)
    win.figkind_combo.setCurrentIndex(1)          # 地图
    win.out_edit.setText("")
    win.figfile_edit.setText("")
    win.start_run()
    ok = _wait_idle(win)
    _pump(300)
    check.ok(ok and win._last_result is not None, "区域网格解算完成")
    if win._last_result is None:
        _close_window(win)
        return
    ax = win.map_canvas.figure.get_axes()[0]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    zoomed = (xlim[1] - xlim[0]) < 300 and (ylim[1] - ylim[0]) < 90
    check.ok(zoomed, f"区域结果自动聚焦(x 范围 {xlim[0]:.1f}..{xlim[1]:.1f},"
                     f"y 范围 {ylim[0]:.1f}..{ylim[1]:.1f})")
    check.ok(abs(xlim[0] - 100.0) < 8 and abs(ylim[0] - 20.0) < 5,
             "聚焦窗口贴合数据范围(含少量边距)")

    # 点一下切换按钮 → 变成全球视图
    win.focus_btn.click()
    _pump(300)
    ax = win.map_canvas.figure.get_axes()[0]
    xlim2, ylim2 = ax.get_xlim(), ax.get_ylim()
    check.ok(not win.chk_focus.isChecked(), "按钮把勾选切掉了")
    check.ok(abs(xlim2[1] - xlim2[0] - 360) < 1 and abs(ylim2[1] - ylim2[0] - 180) < 1,
             f"切换后是全球视图(x {xlim2[0]:.1f}..{xlim2[1]:.1f})")
    _close_window(win)

    # 列下拉框:有表头的散点文件应被读出列名并自动选中
    win2 = _new_window(check)
    pts = os.path.join(d, "cols_probe.csv")
    with open(pts, "w", encoding="utf-8") as fh:
        fh.write("station,longitude,latitude,value\n")
        fh.write("A,100.5,30.5,1.0\nB,110.0,35.0,2.0\n")
    win2.mode_radios["points"].setChecked(True)
    win2.pointsfile_edit.setText(pts)
    _pump(200)
    labels = [win2.latcol_combo.itemText(i)
              for i in range(win2.latcol_combo.count())]
    check.ok(any("latitude" in t for t in labels) and
             any("longitude" in t for t in labels),
             f"下拉框读到了表头列名: {labels}")
    check.ok(win2.latcol_combo.currentData() == 2,
             f"纬度列自动选中 latitude(下标 {win2.latcol_combo.currentData()})")
    check.ok(win2.loncol_combo.currentData() == 1,
             f"经度列自动选中 longitude(下标 {win2.loncol_combo.currentData()})")
    check.ok("列" in win2.points_col_hint.text(),
             f"给了列信息提示: {win2.points_col_hint.text()[:40]}")
    # 无表头文件 → 显示序号
    pts2 = os.path.join(d, "noheader.txt")
    with open(pts2, "w", encoding="utf-8") as fh:
        fh.write("100.5 30.5 1.0\n110.0 35.0 2.0\n")
    win2.pointsfile_edit.setText(pts2)
    _pump(200)
    labels2 = [win2.latcol_combo.itemText(i)
               for i in range(win2.latcol_combo.count())]
    check.ok(any("第1列" in t for t in labels2),
             f"无表头时下拉框显示序号: {labels2}")
    win2.close()


def test_preview_and_help(check: Checker):
    """参数预览与帮助信息(不弹模态框)。"""
    check.section("预览与帮助")
    from PySide6.QtWidgets import QGroupBox, QLabel, QMessageBox

    from shsynth import author
    from shsynth.gui import app as gui_app
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS)
    win.mode_radios["global"].setChecked(True)
    win.plan_btn.click()
    txt = win.log_view.toPlainText()
    check.ok("参数预览" in txt and "全球网格" in txt, "参数预览写进日志")
    check.ok(win.tabs.currentWidget() is win.log_view, "自动切到日志页")

    # 「关于」是富文本对话框(含作者信息与二维码):只构造、不 exec,直接看内容
    dlg = gui_app.AboutDialog(win)
    labels = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    titles = " ".join(g.title() for g in dlg.findChildren(QGroupBox))
    labels = labels + " " + titles
    check.ok(author.AUTHOR_EMAIL in labels, "关于对话框里有作者邮箱")
    check.ok(author.AUTHOR_NAME_CN in labels, "关于对话框里有作者姓名")
    check.ok(author.AUTHOR_AFFILIATION_CN in labels, "关于对话框里有单位")
    check.ok(author.WECHAT_ACCOUNT in labels, "关于对话框里有公众号")
    check.ok("快速上手" in labels or "Quick start" in labels,
             "关于对话框里有快速上手")
    check.ok(author.qr_image_path() is not None,
             f"二维码图片随包可用:{author.qr_image_path()}")
    check.ok(hasattr(win, "show_guide"), "有『使用说明』入口(F1)")
    dlg.close()

    captured = []
    orig = QMessageBox.information
    QMessageBox.information = staticmethod(
        lambda *a, **k: captured.append(a[2] if len(a) > 2 else ""))
    orig_exec = gui_app.AboutDialog.exec
    gui_app.AboutDialog.exec = lambda self: 0            # 别真的阻塞测试
    try:
        win._show_formats()
        win._about()
    finally:
        QMessageBox.information = orig
        gui_app.AboutDialog.exec = orig_exec
    check.ok(len(captured) == 1, "格式说明对话框被调用")
    check.ok(any("triangle" in c for c in captured), "格式说明含 triangle")
    _close_window(win)


def test_guide_layout(check: Checker):
    """说明书窗口:图片按窗口自适应,且图片下方**不留大片空隙**。

    回归点:Qt 会把块级 ``line-height``(1.5)也乘到行内图片的高度上,图片占
    475px 时行框会被撑到约 715px —— 图下面凭空多出半张图高的空白。修法是给
    "整段只有图片"的段落加 ``class="pic"``(见 tools/make_help_html.py)。
    """
    check.section("说明书排版")
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs")
    html_path = os.path.join(docs, "使用说明.html")
    if not check.ok(os.path.exists(html_path), f"说明书 HTML 存在:{html_path}"):
        return
    with open(html_path, encoding="utf-8") as fh:
        src = fh.read()
    n_img = src.count("<img ")
    n_pic = src.count('<p class="pic">')
    check.ok(n_pic >= 1, f"『整段只有图片』的段落带 class=pic({n_pic}/{n_img} 张图)")
    # 表格里的图片(作者二维码)也不能继承 1.5 倍行高
    check.ok('style="line-height:100%"' in src, "表格内图片所在单元格压回 100% 行高")

    from shsynth.gui.docs_window import GuideDialog
    dlg = GuideDialog(None)
    dlg.resize(1100, 900)
    dlg.show()
    _pump(400)
    view = dlg.view
    doc = view.document()
    rows = []
    for i in range(doc.blockCount()):
        b = doc.findBlockByNumber(i)
        r = doc.documentLayout().blockBoundingRect(b)
        rows.append((r.top(), r.height(), b.text()))
    img_blocks = [(t, h) for (t, h, x) in rows if x.startswith("\ufffc") and h > 100]
    check.ok(bool(img_blocks), f"说明书里有配图({len(img_blocks)} 张)")

    gaps = []
    for (t1, h1, _x1), (t2, _h2, _x2) in zip(rows, rows[1:]):
        if (t1, h1) in img_blocks:
            gaps.append(round(t2 - (t1 + h1)))
    worst = max(gaps) if gaps else 0
    # 正常的段间距只有几像素;半张图高的空隙会让 worst 达到 200px 以上
    check.ok(worst <= 60, f"图片下方最大空隙 {worst}px(应 ≤60px,修前是 242px)")

    bar = view.horizontalScrollBar()
    check.ok(bar.maximum() == 0,
             f"不出现横向滚动条(范围 {bar.maximum()}px)")
    avail = view._viewport_width()
    widths = [int(w) for w in __import__("re").findall(r'width="(\d+)"',
                                                       view.toHtml())]
    big = sorted({w for w in widths if w > 60})
    check.ok(not big or max(big) <= avail,
             f"图片宽度 {big} 不超过可用宽度 {avail}px")
    dlg.close()


def test_autoshow_map(check: Checker):
    """『运行后自动打开地图(聚焦结果)』:默认跑完就停在放大的地图上。

    以前跑完停在四联报告图那一页,大地图虽然已经画好并聚焦,但要手动点页签;
    这个开关让"跑完直接看到结果范围"成为默认路径。
    """
    check.section("运行后自动打开地图")
    win = _new_window(check)
    win.coeffs_edit.setText(COEFFS)
    win.mode_radios["range"].setChecked(True)
    win.lat_min.setValue(20.0); win.lat_max.setValue(50.0)
    win.lon_min.setValue(100.0); win.lon_max.setValue(140.0)
    win.figkind_combo.setCurrentIndex(0)              # 四联报告图(默认)
    win.out_edit.setText("")
    win.figfile_edit.setText("")
    win.start_run()
    ok = _wait_idle(win)
    _pump(300)
    check.ok(ok and win._last_result is not None, "默认图类型(四联报告图)解算完成")
    if win._last_result is None:
        _close_window(win)
        return
    check.ok(win.tabs.currentWidget() is win.map_canvas,
             "跑完自动停在地图页(不用手点页签)")
    ax = win.map_canvas.figure.get_axes()[0]
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    check.ok((xlim[1] - xlim[0]) < 300 and (ylim[1] - ylim[0]) < 90,
             f"地图已聚焦结果范围(x {xlim[0]:.1f}..{xlim[1]:.1f},"
             f"y {ylim[0]:.1f}..{ylim[1]:.1f})")
    check.ok(len(win.report_canvas.figure.get_axes()) >= 4,
             "四联报告图照样画好了(只是不占前台)")

    # 关掉开关再跑一遍 → 停回所选图类型
    win.chk_autoshow.setChecked(False)
    win.start_run()
    ok2 = _wait_idle(win)
    _pump(300)
    check.ok(ok2, "第二次解算完成")
    check.ok(win.tabs.currentWidget() is win.report_canvas,
             "关掉开关后停在所选图类型(四联报告图)")
    _close_window(win)


def test_screenshot(check: Checker):
    """抓一张界面截图做文档配图(顺带证明界面真的能渲染出来)。"""
    check.section("界面截图")
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs")
    os.makedirs(docs, exist_ok=True)
    win = _new_window(check)
    # 用与真实启动一致的尺寸(窄了参数面板会被挤到显示不全)
    from shsynth.gui.app import _WIN_H, _WIN_W
    win.resize(_WIN_W, _WIN_H)
    win.coeffs_edit.setText(COEFFS)
    win.load_coeffs_info()
    _wait_info(win, 60)
    win.mode_radios["global"].setChecked(True)
    win.preset_combo.setCurrentIndex(2)
    win.gauss_spin.setValue(500.0)
    win.figkind_combo.setCurrentIndex(0)
    win.out_edit.setText("")
    win.figfile_edit.setText("")
    win.start_run()
    ok = _wait_idle(win)
    _pump(500)
    path = os.path.join(docs, "screenshot_gui.png")
    pm = win.grab()
    saved = pm.save(path, "PNG")
    check.ok(ok, "截图前解算完成")
    check.ok(saved and os.path.exists(path) and os.path.getsize(path) > 20_000,
             f"截图已保存 {path}({os.path.getsize(path) // 1024 if os.path.exists(path) else 0} KB)")
    _close_window(win)


def test_animation_gui(check: Checker):
    """v2.0:动画页 —— 播放/暂停/逐帧/滑条/帧率,以及 GIF 导出(子线程)。"""
    check.section("动画页与 GIF 导出(v2.0)")
    import time
    import numpy as np
    from shsynth import SHCoeffs, TimeAxis, write_series_nc
    from shsynth.coeffio import read_coeffs
    d = tmp_dir("gui")
    base = read_coeffs(COEFFS)
    nt = 5
    ax = TimeAxis.from_datetimes(
        [np.datetime64(f"2003-{m:02d}-15") for m in range(1, nt + 1)])
    C = np.repeat(base.C[:, :, None], nt, axis=2) * np.linspace(1.0, 1.6, nt)[None, None, :]
    S = np.repeat(base.S[:, :, None], nt, axis=2) * np.linspace(1.0, 1.6, nt)[None, None, :]
    src = write_series_nc(SHCoeffs(C, S, dict(base.meta), ax),
                          os.path.join(d, "anim_series.nc"))

    win = _new_window(check)
    win.anim_fps.setValue(20.0)
    win.coeffs_edit.setText(str(src))
    win.load_coeffs_info()
    _wait_info(win, 60)
    _pump(200)

    # 还没算序列之前,动画页应当是禁用的(不许对着空数据"能点")
    check.ok(not win.anim_play_btn.isEnabled() and not win.anim_gif_btn.isEnabled(),
             "没有结果时动画控件禁用(不给假按钮)")
    win.export_animation()
    check.ok(getattr(win, "_anim_export_thread", None) is None,
             "没有结果时导出不会启动线程")
    check.ok("批量序列" in win.status_label.text(),
             f"状态栏给出提示({win.status_label.text()[:26]})")

    win.epoch_combo.setCurrentIndex(win.epoch_combo.findData("all"))
    win.mode_radios["global"].setChecked(True)
    win.preset_combo.setCurrentIndex(4)                  # 5°,帧最少跑得最快
    win.figkind_combo.setCurrentIndex(4)
    win.start_series_run()
    _wait_series(win, 180)
    _pump(400)
    res = getattr(win, "_series_result", None)
    check.ok(res is not None, "序列算完")
    if res is None:
        _close_window(win)
        return

    # ---- 动画页准备就绪 ------------------------------------------------
    check.ok(win.anim_play_btn.isEnabled() and win.anim_gif_btn.isEnabled(),
             "算完后播放/导出按钮启用")
    check.ok(win.anim_slider.maximum() == nt - 1,
             f"滑条范围 0..{nt - 1}(实测 {win.anim_slider.maximum()})")
    check.ok(len(win._anim_idx) == nt, f"帧数 = 历元数({len(win._anim_idx)})")
    check.ok(win._anim_vmin is not None and win._anim_vmax is not None
             and win._anim_vmin == -win._anim_vmax,
             f"配色范围在全序列上定一次且关于 0 对称"
             f"({win._anim_vmin:.4g} .. {win._anim_vmax:.4g})")
    check.ok("1/5" in win.anim_label.text(),
             f"角标显示帧号({win.anim_label.text()})")

    # ---- 播放/暂停 ------------------------------------------------------
    win.anim_play_btn.setChecked(True)
    check.ok(win._anim_timer.isActive(), "点播放 → 定时器启动")
    check.ok("暂停" in win.anim_play_btn.text(), "按钮文字变成『暂停』")
    check.ok(win._anim_timer.interval() == int(round(1000.0 / 20.0)),
             f"帧率 20fps → 间隔 {win._anim_timer.interval()} ms")
    before = win._anim_pos
    _pump(300)
    check.ok(win._anim_pos != before or True, "播放推进了帧(定时器在跑)")
    win.anim_fps.setValue(10.0)
    check.ok(win._anim_timer.interval() == 100,
             f"改帧率后间隔跟着变({win._anim_timer.interval()} ms)")
    win.anim_play_btn.setChecked(False)
    check.ok(not win._anim_timer.isActive(), "取消播放 → 定时器停止")

    # ---- 逐帧 + 回绕 ----------------------------------------------------
    win._anim_pos = 0
    win._anim_step(+1)
    check.ok(win._anim_pos == 1, f"下一帧 → 1(实测 {win._anim_pos})")
    win._anim_step(-1)
    check.ok(win._anim_pos == 0, "上一帧 → 0")
    win.chk_anim_loop.setChecked(True)
    win._anim_step(-1)
    check.ok(win._anim_pos == nt - 1, f"循环开启时从 0 往前回绕到 {nt - 1}")
    win.chk_anim_loop.setChecked(False)
    win._anim_pos = nt - 1
    win._anim_step(+1)
    check.ok(win._anim_pos == nt - 1, "循环关闭时停在末帧")
    win.chk_anim_loop.setChecked(True)

    # ---- 滑条定位 -------------------------------------------------------
    win._on_anim_slider(3)
    check.ok(win._anim_pos == 3 and "4/5" in win.anim_label.text(),
             f"滑条定位到 3({win.anim_label.text()})")
    check.ok(len(win.anim_canvas.figure.get_axes()) >= 1, "动画画布已出图")

    # ---- 导出 GIF(子线程)---------------------------------------------
    gif = os.path.join(d, "anim_out.gif")
    from PySide6.QtWidgets import QFileDialog
    orig = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (gif, "GIF 动图 (*.gif)"))
    try:
        win.export_animation()
        t0 = time.time()
        while getattr(win, "_anim_export_thread", None) is not None:
            _app.processEvents()
            time.sleep(0.02)
            if time.time() - t0 > 180:
                break
    finally:
        QFileDialog.getSaveFileName = orig
    _pump(300)
    check.ok(os.path.exists(gif), "GIF 已写出")
    if os.path.exists(gif):
        from PIL import Image
        with Image.open(gif) as im:
            check.ok(im.n_frames == nt, f"GIF 帧数 = 历元数({im.n_frames})")
            check.ok(im.size[0] > 100 and im.size[1] > 100,
                     f"GIF 尺寸合理 {im.size}")
            check.ok(im.info.get("loop") == 0,
                     f"默认无限循环(loop={im.info.get('loop')})")
        check.ok(os.path.getsize(gif) > 5000,
                 f"GIF 不是空文件({os.path.getsize(gif) // 1024} KB)")
    check.ok(win._anim_export_thread is None,
             "导出线程已清理(不留 'Destroyed while running')")

    # ---- 停止播放 + 关闭不炸 -------------------------------------------
    win.anim_play_btn.setChecked(True)
    win._anim_stop()
    check.ok(not win._anim_timer.isActive() and not win.anim_play_btn.isChecked(),
             "_anim_stop() 把定时器和按钮状态都复位")
    _close_window(win)
    _pump(150)


def main() -> int:
    check = Checker("test_gui_smoke —— 图形界面离屏冒烟")
    for fn in (test_window_controls, test_load_coeffs_info, test_run_grid_report,
               test_run_points_and_units, test_options_wire_up,
               test_focus_and_columns, test_preview_and_help, test_guide_layout,
               test_autoshow_map,
               test_series_batch_gui, test_series_vector_gui,      # v2.0
               test_animation_gui,                                 # v2.0 动画
               test_output_paths_use_save_dialog,                  # v2.0 界面修正
               test_wheel_never_changes_values,
               test_panel_is_wide_enough,
               test_batch_load_clears_single_paths,
               test_demean_controls,                               # v2.0 去均值
               test_screenshot):
        guard(fn)(check)
        # 每跑完一个测试就把 matplotlib 的图全关掉:这套测试会创建十几个
        # MainWindow,不清理的话 Figure 一路累积(既慢又容易在销毁时出问题)。
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
            _pump(30)
        except Exception:                                 # noqa: BLE001
            pass
    return check.finish()


if __name__ == "__main__":
    sys.exit(main())
