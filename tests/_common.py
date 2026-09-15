# -*- coding: utf-8 -*-
"""
tests/_common.py
================

测试用的小工具:统一的检查计数与输出(不依赖 pytest)。

每个测试脚本跑完调用 :func:`finish`,返回 0(全过)或 1(有失败),
``tests/run_all.py`` 据此汇总。
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHKIT_DIR = os.path.join(os.path.dirname(ROOT), "SHKit")
TMP_DIR = os.path.join(ROOT, "out", "_tests")

#: 本进程里已经清空过的临时子目录(见 :func:`tmp_dir`)
_CLEARED: set = set()

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Checker:
    """极简的检查器:统计通过/失败条数,失败立即打印原因。"""

    def __init__(self, title: str):
        self.title = title
        self.passed = 0
        self.failed = 0
        self.failures: list = []
        print("=" * 70)
        print(f"{title}")
        print("=" * 70)

    # -------------------------------------------------------------- 基本检查
    def ok(self, cond, label: str, detail: str = "") -> bool:
        cond = bool(cond)
        if cond:
            self.passed += 1
            print(f"  [PASS] {label}")
        else:
            self.failed += 1
            self.failures.append(label)
            print(f"  [FAIL] {label}" + (f"   {detail}" if detail else ""))
        return cond

    def close(self, a, b, tol: float, label: str) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return self.ok(False, label, f"形状 {a.shape} vs {b.shape}")
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        return self.ok(d <= tol, label, f"最大差 {d:.3e} > 容差 {tol:.3e}")

    def raises(self, fn, exc, label: str) -> bool:
        try:
            fn()
        except exc as caught:
            return self.ok(True, f"{label}（捕获 {type(caught).__name__}）")
        except Exception as other:                       # noqa: BLE001
            return self.ok(False, label, f"抛出了 {type(other).__name__}: {other}")
        return self.ok(False, label, "没有抛出异常")

    def section(self, name: str):
        print(f"\n-- {name} " + "-" * max(0, 64 - len(name)))

    def skip(self, why: str) -> None:
        """跳过一组检查(例如需要用户真实数据 / 隔壁 SHKit / 已编译的外部程序)。

        跳过**不算失败**,但要**打印原因**,不让"没跑"看起来像"跑过了"。
        """
        self.skipped = getattr(self, "skipped", 0) + 1
        print(f"  [SKIP] {why}")

    # ---------------------------------------------------------------- 收尾
    def finish(self) -> int:
        print("-" * 70)
        total = self.passed + self.failed
        if self.failed:
            print(f"{self.title}: {self.passed}/{total} 项通过,"
                  f"{self.failed} 项失败")
            for f in self.failures:
                print(f"   失败: {f}")
            return 1
        print(f"{self.title}: 全部 {total} 项通过")
        return 0


def _fs_retry(fn, tries: int = 4):
    """网络盘(本机是华为家庭存储的映射盘)偶尔会瞬时拒绝写入,重试几次。

    实测:同一个测试套件连跑两遍,第二遍偶尔在 ``np.save`` 上收到
    ``PermissionError``(SMB 上上一个进程的句柄还没释放),隔一会儿再跑就正常。
    这类失败跟代码无关,重试即可,免得红叉误导人。
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except PermissionError as exc:              # noqa: PERF203
            last = exc
            time.sleep(0.35 * (i + 1))
    raise last


def tmp_dir(name: str = "") -> str:
    """测试用的临时输出目录(项目内,便于人工查看)。

    每个名字在**本进程里第一次**用到时先清空:上次跑剩下的文件在网络盘上可能
    还锁着,直接覆盖会 PermissionError。
    """
    d = os.path.join(TMP_DIR, name) if name else TMP_DIR
    _fs_retry(lambda: os.makedirs(d, exist_ok=True))
    if name and name not in _CLEARED:
        _CLEARED.add(name)
        for entry in os.listdir(d):
            p = os.path.join(d, entry)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    _fs_retry(lambda p=p: os.remove(p))
            except OSError:
                pass
    return d


def shkit_samples() -> list:
    """SHKit 目录里可用的系数样例文件。"""
    cands = [
        os.path.join(SHKIT_DIR, "shkit_coeffs.sh"),
        os.path.join(SHKIT_DIR, "shkit_coeffs.gfc"),
        os.path.join(SHKIT_DIR, "yantze_shkit_coeffs.gfc"),
        os.path.join(SHKIT_DIR, "sample_data", "truth_coeffs_20.sh"),
        os.path.join(SHKIT_DIR, "shkit_coeffs.npy"),
    ]
    return [p for p in cands if os.path.exists(p)]


def have_shkit() -> bool:
    """SHKit 在隔壁目录且能 import 时返回 True(否则相关测试自动跳过)。"""
    if not os.path.isdir(SHKIT_DIR):
        return False
    p = os.path.dirname(SHKIT_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        import importlib
        importlib.import_module("shkit.io")
        return True
    except Exception:                                                # noqa: BLE001
        return False


def guard(fn):
    """把单个测试函数包一层,异常记为失败而不是让整脚本崩掉。"""
    def wrapper(checker: Checker):
        try:
            fn(checker)
        except Exception as exc:                         # noqa: BLE001
            checker.ok(False, f"{fn.__name__} 抛出异常",
                       f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
    wrapper.__name__ = fn.__name__
    return wrapper
