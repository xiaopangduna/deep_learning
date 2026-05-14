"""ImageNette 分类：CLI 集成测试（LovelyLightningCLI）。

覆盖场景（需本地 ``datasets/IMAGENETTE``）：

一、``fit`` + ``--trainer.fast_dev_run`` — 仅验证能跑通（不要求产生 ``.ckpt``）。
python scripts/train.py fit \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --trainer.fast_dev_run true

二、``fit`` + ``--trainer.max_epochs 5`` — 短程完整训练并产出 ``.ckpt``（module fixture，供场景三～七）。
python scripts/train.py fit \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --trainer.max_epochs 3

三、``fit`` + ``--ckpt_path`` — 从已有 ckpt 断点续训。
python scripts/train.py fit \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt \
  --trainer.max_epochs 5

四、``validate`` + ``--ckpt_path``。
python scripts/train.py validate \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt 


五、``test`` + ``--ckpt_path``。
python scripts/train.py test \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt

六、``predict`` + ``--ckpt_path``。
python scripts/train.py predict \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt

七、``export`` + ``--ckpt_path``。
python scripts/train.py export \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --ckpt_path logs/image_classifiter_IMAGE_NETTE/version_0/checkpoints/last.ckpt \
  --export_format pt

八、``export``（不传 ckpt，依赖 YAML 初始权重）。
python scripts/train.py export \
  --config logs/image_classifiter_IMAGE_NETTE/version_0/config.yaml \
  --export_format pt

**产物位置**：均在仓库根下 ``.cli_test_imagenette/``（及配置里相对 cwd 的 ``logs/`` 等），不使用系统临时目录；可自行删除。
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/experiments/image_classifiter_IMAGE_NETTE.yaml"
NUM_WORKERS_0 = ["--data.init_args.num_workers", "0"]
# 固定写在项目内，便于查看与手动清理（见模块说明）。
CLI_TEST_ROOT = REPO_ROOT / ".cli_test_imagenette"
FIT_EP5_DIR = CLI_TEST_ROOT / "fit_ep5"
FASTDEV_DIR = CLI_TEST_ROOT / "fastdev"
RESUME_FIT_DIR = CLI_TEST_ROOT / "resume_fit"


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


def _pick_latest_ckpt_after_fit(*, run_root: Path, not_before: float) -> Path:
    """Lightning 常把 ckpt 写在 ``logs/``（相对 cwd），不一定在 ``default_root_dir`` 下。"""
    candidates: list[Path] = []
    if run_root.is_dir():
        candidates.extend(run_root.rglob("*.ckpt"))
    logs = REPO_ROOT / "logs"
    if logs.is_dir():
        candidates.extend(logs.rglob("*.ckpt"))
    skew = 5.0
    fresh = [
        p
        for p in candidates
        if p.is_file() and p.stat().st_mtime >= not_before - skew
    ]
    if not fresh:
        raise AssertionError(
            f"在 {run_root} 与 {logs} 下未找到 mtime>={not_before - skew} 的 .ckpt"
        )
    return max(fresh, key=lambda p: p.stat().st_mtime)


@pytest.fixture(scope="module")
def imagenette_ckpt() -> str:
    """场景二：``fit`` + ``max_epochs=5`` 产出 ckpt，供场景三～七复用。"""
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE（imagenette2-320 与 train.csv）")

    FIT_EP5_DIR.mkdir(parents=True, exist_ok=True)
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
        ckpt = _pick_latest_ckpt_after_fit(run_root=FIT_EP5_DIR, not_before=not_before)
    except AssertionError as e:
        pytest.fail(
            "fit 后未找到本次运行产生的 .ckpt；stdout:\n"
            f"{result.stdout}\nstderr:\n{result.stderr}\n{e}"
        )
    return str(ckpt)


@pytest.mark.cli
def test_cli_01_fit_fast_dev_run_exits_ok() -> None:
    """一：fit + fast_dev_run 仅验证进程正常结束（不检查是否写出 .ckpt）。"""
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE（imagenette2-320 与 train.csv）")

    FASTDEV_DIR.mkdir(parents=True, exist_ok=True)
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "fit",
            "--config",
            CONFIG,
            "--trainer.fast_dev_run",
            "true"
        ],
        timeout=180,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.cli
def test_cli_02_fit_max_epochs_five_produces_ckpt(imagenette_ckpt: str) -> None:
    """二：fit + max_epochs=5 产生 .ckpt（由 module fixture 执行一次 fit）。"""
    path = Path(imagenette_ckpt)
    assert path.is_file()
    assert path.suffix == ".ckpt"


@pytest.mark.cli
def test_cli_03_fit_resume_with_ckpt_path(imagenette_ckpt: str) -> None:
    """三：fit + --ckpt_path 断点续训（ckpt 来自 5 epoch；``max_epochs=7`` 再训至多 7 轮）。"""
    RESUME_FIT_DIR.mkdir(parents=True, exist_ok=True)
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "fit",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            "--trainer.max_epochs",
            "5",
            *NUM_WORKERS_0,
        ],
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.cli
def test_cli_04_validate_with_ckpt_path(imagenette_ckpt: str) -> None:
    """四：validate + --ckpt_path。"""
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "validate",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            "--trainer.limit_val_batches",
            "2",
            *NUM_WORKERS_0,
        ],
        timeout=120,
    )
    assert result.returncode == 0


@pytest.mark.cli
def test_cli_05_test_with_ckpt_path(imagenette_ckpt: str) -> None:
    """五：test + --ckpt_path。"""
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "test",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            *NUM_WORKERS_0,
        ],
        timeout=120,
    )
    assert result.returncode == 0


@pytest.mark.cli
def test_cli_06_predict_with_ckpt_path(imagenette_ckpt: str) -> None:
    """六：predict + --ckpt_path。"""
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "predict",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            *NUM_WORKERS_0,
        ],
        timeout=120,
    )
    assert result.returncode == 0


@pytest.mark.cli
def test_cli_07_export_with_ckpt_path(imagenette_ckpt: str) -> None:
    """七：export + --ckpt_path（pt 格式缩短耗时；校验「导出完成」路径）。"""
    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "export",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            "--export_format",
            "pt",
            *NUM_WORKERS_0,
        ],
        timeout=180,
    )
    assert result.returncode == 0
    combined = result.stdout + "\n" + result.stderr
    m = re.search(r"导出完成:\s*(.+)", combined)
    assert m, f"未找到「导出完成」日志:\n{combined}"
    out_path = Path(m.group(1).strip())
    assert out_path.is_file(), f"导出文件不存在: {out_path}"


@pytest.mark.cli
def test_cli_08_export_without_ckpt_path() -> None:
    """八：export 不传 ckpt，依赖 YAML 初始化权重。"""
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE")

    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "export",
            "--config",
            CONFIG,
            "--export_format",
            "pt",
            *NUM_WORKERS_0,
        ],
        timeout=300,
    )
    assert result.returncode == 0
    combined = result.stdout + "\n" + result.stderr
    m = re.search(r"导出完成:\s*(.+)", combined)
    assert m, f"未找到「导出完成」日志:\n{combined}"
    assert Path(m.group(1).strip()).is_file()
