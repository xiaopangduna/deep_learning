"""CLI 集成测试共享 fixture。"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/experiments/image_classifiter_IMAGE_NETTE.yaml"
NUM_WORKERS_0 = ["--data.init_args.num_workers", "0"]


def _imagenette_data_ready(root: Path) -> bool:
    return (root / "datasets/IMAGENETTE/imagenette2-320").is_dir() and (
        root / "datasets/IMAGENETTE/train.csv"
    ).is_file()


def _run_cli(args: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    return result


def _pick_latest_ckpt_after_fit(*, not_before: float) -> Path:
    logs = REPO_ROOT / "logs"
    if not logs.is_dir():
        raise AssertionError(f"{logs} 不存在，无法查找 .ckpt")
    skew = 5.0
    fresh = [
        p
        for p in logs.rglob("*.ckpt")
        if p.is_file() and p.stat().st_mtime >= not_before - skew
    ]
    if not fresh:
        raise AssertionError(
            f"在 {logs} 下未找到 mtime>={not_before - skew} 的 .ckpt"
        )
    return max(fresh, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def imagenette_ckpt() -> str:
    """fit + max_epochs=1 产出 ckpt，供 prune / resume 等测试复用（缩短耗时）。"""
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE（imagenette2-320 与 train.csv）")

    fit_args = [
        "python",
        "scripts/train.py",
        "fit",
        "--config",
        CONFIG,
        "--trainer.max_epochs",
        "3",
        *NUM_WORKERS_0,
    ]
    not_before = time.time()
    result = _run_cli(fit_args, timeout=3600)
    if result.returncode != 0:
        pytest.fail(f"fit 失败:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

    try:
        ckpt = _pick_latest_ckpt_after_fit(not_before=not_before)
    except AssertionError as e:
        pytest.fail(
            "fit 后未找到本次运行产生的 .ckpt；stdout:\n"
            f"{result.stdout}\nstderr:\n{result.stderr}\n{e}"
        )
    return str(ckpt)
