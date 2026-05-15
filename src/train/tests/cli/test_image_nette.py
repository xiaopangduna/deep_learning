"""ImageNette 分类：CLI 集成测试（LovelyLightningCLI）。

覆盖场景（需本地 ``datasets/IMAGENETTE``）：

一、``fit`` + ``--trainer.fast_dev_run`` — 仅验证能跑通（不要求产生 ``.ckpt``）。
python scripts/train.py fit \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --trainer.fast_dev_run true

二、``fit`` + ``--trainer.max_epochs 1`` — 短程训练并产出 ``.ckpt``（module fixture，供场景三～七及剪枝测试）。
python scripts/train.py fit \
  --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml \
  --trainer.max_epochs 1

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

**产物位置**：训练 checkpoint 等写在仓库根下 ``logs/``（由 YAML 中 logger 配置决定）；可自行删除。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.cli.conftest import CONFIG, NUM_WORKERS_0, REPO_ROOT, _imagenette_data_ready, _run_cli


@pytest.mark.cli
def test_cli_01_fit_fast_dev_run_exits_ok() -> None:
    """一：fit + fast_dev_run 仅验证进程正常结束（不检查是否写出 .ckpt）。"""
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE（imagenette2-320 与 train.csv）")

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
    """二：fit + max_epochs=1 产生 .ckpt（由 module fixture 执行一次 fit）。"""
    path = Path(imagenette_ckpt)
    assert path.is_file()
    assert path.suffix == ".ckpt"


@pytest.mark.cli
def test_cli_03_fit_resume_with_ckpt_path(imagenette_ckpt: str) -> None:
    """三：fit + --ckpt_path 断点续训（ckpt 来自 5 epoch；``max_epochs=7`` 再训至多 7 轮）。"""
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


PRUNED_OUT = REPO_ROOT / "pretrained_models" / "mobilenet_v3_large_pruned_30_test.pth"


@pytest.mark.cli
def test_cli_09_prune_mobilenet_v3(imagenette_ckpt: str) -> None:
    """九：prune 导出 tp.state_dict 与 meta.json。"""
    pytest.importorskip("torch_pruning")
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE")

    PRUNED_OUT.parent.mkdir(parents=True, exist_ok=True)
    for p in (PRUNED_OUT, PRUNED_OUT.with_suffix(".meta.json")):
        if p.is_file():
            p.unlink()

    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "prune",
            "--config",
            CONFIG,
            "--ckpt_path",
            imagenette_ckpt,
            "--pruning_ratio",
            "0.30",
            "--output_path",
            str(PRUNED_OUT),
            "--ignored_layer_names+",
            "classifier",
            *NUM_WORKERS_0,
        ],
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert PRUNED_OUT.is_file()
    meta_path = PRUNED_OUT.with_suffix(".meta.json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["load_pruned"] is True
    assert meta["pruned_nparams"] < meta["base_nparams"]
    assert re.search(r"剪枝完成:", result.stdout + result.stderr)


@pytest.mark.cli
def test_cli_10_fit_after_prune(imagenette_ckpt: str) -> None:
    """十：剪枝后微调（path_custom 指向 pruned.pth；同目录 .meta.json 触发 tp 加载）。"""
    pytest.importorskip("torch_pruning")
    if not _imagenette_data_ready(REPO_ROOT):
        pytest.skip("缺少 datasets/IMAGENETTE")
    if not PRUNED_OUT.is_file():
        test_cli_09_prune_mobilenet_v3(imagenette_ckpt)

    result = _run_cli(
        [
            "python",
            "scripts/train.py",
            "fit",
            "--config",
            CONFIG,
            "--model.init_args.model.weight.path_custom",
            str(PRUNED_OUT.relative_to(REPO_ROOT)),
            "--trainer.fast_dev_run",
            "true",
            *NUM_WORKERS_0,
        ],
        timeout=300,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
