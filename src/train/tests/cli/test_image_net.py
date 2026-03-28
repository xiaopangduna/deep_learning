import pytest
import subprocess


@pytest.mark.cli
def test_ImageNette_fit():
    result = subprocess.run(
        [
            "python",
            "scripts/train.py",
            "fit",
            "--config",
            "configs/experiments/image_classifiter_IMAGE_NET.yaml",
            "--trainer.fast_dev_run",
            "true",
        ],
        capture_output=True,
        text=True,
        # timeout=60,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    assert result.returncode == 0

