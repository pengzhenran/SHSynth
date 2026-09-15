# -*- coding: utf-8 -*-
"""
shsynth.gui.workers
===================

把综合任务放到后台线程跑,界面不卡。

设计要点:worker **只算数值**(``figure_kind='none'``),绘图在主线程完成 ——
matplotlib 的 Figure 与 Qt 画布绑在主线程,跨线程创建再拿去显示容易出玄学问题;
分开以后既安全又快(计算才是耗时的那部分)。
"""

from __future__ import annotations

import traceback

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from .. import workflow

__all__ = ["SynthWorker", "CoeffInfoWorker", "SeriesSynthWorker",
           "SeriesReadWorker"]


class SeriesSynthWorker(QObject):
    """在后台线程里跑一次**批量序列综合**(v2.0)。

    与 :class:`SynthWorker` 的分工一样:worker 只算数值,绘图留给主线程。
    批量的取消走 ``cancel`` 回调,在历元块之间检查。
    """

    progress = Signal(str, float)
    finished = Signal(object)          # SeriesResult
    failed = Signal(str, str)
    cancelled = Signal()

    def __init__(self, coeffs, target, **kwargs):
        super().__init__()
        self.coeffs = coeffs
        self.target = target            # (lat_vec, lon_vec, points)
        self.kwargs = kwargs
        self._stop = False

    def cancel(self):
        self._stop = True

    @Slot()
    def run(self):
        from ..series import synth_series
        lat_vec, lon_vec, points = self.target
        try:
            res = synth_series(self.coeffs, lat_vec=lat_vec, lon_vec=lon_vec,
                               points=points,
                               progress=lambda m, f: self.progress.emit(m, f),
                               cancel=lambda: self._stop, **self.kwargs)
        except InterruptedError:
            self.cancelled.emit()
            return
        except MemoryError:
            self.failed.emit("内存不足:请把网格步长放粗、降低阶数,"
                             "或把分块点数调小", traceback.format_exc())
            return
        except Exception as exc:                         # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished.emit(res)


class SeriesReadWorker(QObject):
    """后台读"目录/清单 → 系数序列"(v2.0);200+ 个 gfc 不能卡界面。"""

    finished = Signal(object)          # SHCoeffs(带 times)
    failed = Signal(str, str)

    def __init__(self, source: str, **kwargs):
        super().__init__()
        self.source = source
        self.kwargs = kwargs

    @Slot()
    def run(self):
        from ..coeffio import read_coeffs_series
        try:
            c = read_coeffs_series(self.source, **self.kwargs)
        except Exception as exc:                         # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished.emit(c)


class SynthWorker(QObject):
    """在后台线程里执行一次 :func:`shsynth.workflow.run`。"""

    progress = Signal(str, float)      # (阶段文字, 0..1)
    finished = Signal(object)          # SynthResult
    failed = Signal(str, str)          # (简短说明, 完整堆栈)
    cancelled = Signal()

    def __init__(self, spec):
        super().__init__()
        # 计算阶段不画图:图回到主线程画,保证与界面上显示的是同一张。
        # 但**原始请求要留着** —— 结果里的 spec 决定主线程之后画哪种图、存到哪里。
        self.original = spec
        self.spec = _copy_spec(spec)
        self.spec.figure_kind = "none"
        self.spec.figure_path = ""
        self._stop = False

    def cancel(self):
        """请求中止(工作线程会在下一个检查点退出)。"""
        self._stop = True

    @Slot()
    def run(self):
        try:
            result = workflow.run(self.spec,
                                  progress=lambda m, f: self.progress.emit(m, f),
                                  cancel=lambda: self._stop)
        except InterruptedError:
            self.cancelled.emit()
            return
        except MemoryError:
            self.failed.emit("内存不足:请减小网格步长、降低阶数,或把『分块点数』调小",
                             traceback.format_exc())
            return
        except Exception as exc:                         # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        # 把请求换回原样,主线程据此绘图/存盘
        result.spec = self.original
        self.finished.emit(result)


class CoeffInfoWorker(QObject):
    """后台读取系数文件信息(文件很大时不阻塞界面)。"""

    finished = Signal(object)          # SHCoeffs
    failed = Signal(str, str)

    def __init__(self, path: str, layout: str = "auto", nmax=None):
        super().__init__()
        self.path = path
        self.layout = layout
        self.nmax = nmax

    @Slot()
    def run(self):
        try:
            from ..coeffio import read_coeffs
            coeffs = read_coeffs(self.path, nmax=self.nmax, layout=self.layout)
        except Exception as exc:                         # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished.emit(coeffs)


def _copy_spec(spec):
    """浅拷贝一份请求,避免后台线程与界面同时改同一个对象。"""
    import dataclasses

    if dataclasses.is_dataclass(spec):
        return dataclasses.replace(spec, extra=dict(getattr(spec, "extra", {}) or {}))
    return spec


class AnimationExportWorker(QObject):
    """把场序列导出成 GIF(v2.0)。

    渲染每一帧要走一遍 matplotlib(pcolormesh + 海岸线),203 帧要几十秒到几分钟,
    **不能放在界面线程里**;所以照 :class:`SeriesSynthWorker` 的老规矩搬到子线程,
    每帧发一次 ``progress``,``cancel`` 在帧之间检查。
    """

    progress = Signal(int, int)        # (已完成帧, 总帧数)
    finished = Signal(str, dict)       # (输出路径, info)
    failed = Signal(str, str)          # (简短说明, 完整堆栈)
    cancelled = Signal()

    def __init__(self, res, path, *, fps=5.0, stride=1, values=None,
                 vmin=None, vmax=None, focus="global", title="",
                 cb_label="", dpi=100):
        super().__init__()
        self.res = res
        self.path = path
        self.fps = float(fps)
        self.stride = max(1, int(stride))
        self.values = values
        self.vmin = vmin
        self.vmax = vmax
        self.focus = focus
        self.title = title
        self.cb_label = cb_label
        self.dpi = int(dpi)
        self._stop = False
        self.info = {}

    def cancel(self):
        self._stop = True

    def run(self):
        import traceback
        try:
            from ..plotting import make_series_frame, save_animation, \
                _symmetric_range
            v = np.asarray(self.values if self.values is not None
                           else self.res.values, dtype=float)
            idx = list(range(0, v.shape[2], self.stride))

            # 配色范围在**整段序列**上定一次:逐帧定标会让动画随历元极值闪。
            vmin, vmax = self.vmin, self.vmax
            if vmin is None or vmax is None:
                a, b = _symmetric_range(v[:, :, idx], symmetric=True)
                vmin = a if vmin is None else vmin
                vmax = b if vmax is None else vmax
            self.info.update(vmin=float(vmin), vmax=float(vmax),
                             n_frames=len(idx), stride=self.stride,
                             times=[str(self.res.times.values[k])[:10]
                                    for k in idx])

            def _frame(k):
                return make_series_frame(self.res.times, self.res.lat,
                                         self.res.lon, v, idx[k],
                                         title=self.title,
                                         cb_label=self.cb_label,
                                         vmin=vmin, vmax=vmax,
                                         focus=self.focus, dpi=self.dpi)

            p = save_animation(self.path, _frame, n_frames=len(idx),
                               fps=self.fps, info=self.info,
                               progress=lambda d, n: self.progress.emit(d, n),
                               cancel=lambda: self._stop, dpi=self.dpi)
        except KeyboardInterrupt:
            self.cancelled.emit()
            return
        except Exception as exc:                             # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
            return
        self.finished.emit(p, dict(self.info))


def make_thread(worker: QObject) -> QThread:
    """把 worker 搬到新线程并接好清理逻辑;返回(未启动的)线程。"""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for sig in ("finished", "failed", "cancelled"):
        s = getattr(worker, sig, None)
        if s is not None:
            s.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    return thread
