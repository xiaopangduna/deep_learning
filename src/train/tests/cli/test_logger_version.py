"""多个 logger 在实例化前共用同一个 version。"""

from __future__ import annotations

import warnings

import pytest
from jsonargparse import Namespace

from lovely_deep_learning.cli.lightning_cli import LovelyLightningCLI, align_logger_versions

TB = "lightning.pytorch.loggers.TensorBoardLogger"
CSV = "lightning.pytorch.loggers.CSVLogger"


def _pair(save_dir, name="exp", versions=(None, None), class_paths=(TB, CSV)):
    return [
        {
            "class_path": class_paths[0],
            "init_args": {"save_dir": save_dir, "name": name, "version": versions[0]},
        },
        {
            "class_path": class_paths[1],
            "init_args": {"save_dir": save_dir, "name": name, "version": versions[1]},
        },
    ]


def test_null_versions_share_next_number_and_skip_holes(tmp_path):
    root = tmp_path / "exp"
    (root / "version_0").mkdir(parents=True)
    (root / "version_2").mkdir()
    (root / "version_9").write_text("not a directory")
    loggers = _pair(str(tmp_path))

    align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 3
    assert loggers[1]["init_args"]["version"] == 3
    assert not (root / "version_3").exists()


def test_null_on_empty_experiment_starts_at_zero(tmp_path):
    loggers = _pair(str(tmp_path))

    align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 0
    assert loggers[1]["init_args"]["version"] == 0


def test_single_explicit_version_is_copied_to_null(tmp_path):
    (tmp_path / "exp" / "version_8").mkdir(parents=True)
    loggers = _pair(str(tmp_path), versions=(5, None))

    align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 5
    assert loggers[1]["init_args"]["version"] == 5


def test_matching_explicit_versions_stay(tmp_path):
    loggers = _pair(str(tmp_path), versions=(5, 5))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        align_logger_versions(loggers)

    assert recorded == []
    assert loggers[0]["init_args"]["version"] == 5
    assert loggers[1]["init_args"]["version"] == 5


def test_conflicting_explicit_versions_warn_and_stay(tmp_path):
    loggers = _pair(str(tmp_path), versions=(3, 4))

    with pytest.warns(UserWarning, match="不同的 version"):
        align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 3
    assert loggers[1]["init_args"]["version"] == 4


def test_different_experiment_dirs_warn_and_stay_null(tmp_path):
    loggers = _pair(str(tmp_path))
    loggers[1]["init_args"]["name"] = "other"

    with pytest.warns(UserWarning, match="实验目录不一致"):
        align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] is None
    assert loggers[1]["init_args"]["version"] is None


def test_matching_pair_still_aligns_when_another_logger_uses_a_different_dir(tmp_path):
    loggers = _pair(str(tmp_path))
    loggers.append(
        {
            "class_path": "example.OtherLogger",
            "init_args": {"save_dir": str(tmp_path), "name": "other", "version": None},
        }
    )

    with pytest.warns(UserWarning, match="实验目录不一致"):
        align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 0
    assert loggers[1]["init_args"]["version"] == 0
    assert loggers[2]["init_args"]["version"] is None


def test_single_logger_is_left_to_lightning(tmp_path):
    logger = {
        "class_path": TB,
        "init_args": {"save_dir": str(tmp_path), "name": "exp", "version": None},
    }

    align_logger_versions(logger)

    assert logger["init_args"]["version"] is None


def test_disabled_logger_is_ignored():
    align_logger_versions(False)
    align_logger_versions(None)


def test_omitted_name_uses_logger_default(tmp_path):
    (tmp_path / "lightning_logs" / "version_0").mkdir(parents=True)
    loggers = [
        {"class_path": TB, "init_args": {"save_dir": str(tmp_path)}},
        {"class_path": CSV, "init_args": {"save_dir": str(tmp_path)}},
    ]

    align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 1
    assert loggers[1]["init_args"]["version"] == 1


def test_explicit_null_name_scans_save_dir(tmp_path):
    (tmp_path / "version_0").mkdir()
    loggers = _pair(str(tmp_path), name=None)

    align_logger_versions(loggers)

    assert loggers[0]["init_args"]["version"] == 1
    assert loggers[1]["init_args"]["version"] == 1


def test_before_instantiate_classes_writes_the_parsed_config(tmp_path):
    (tmp_path / "exp" / "version_1").mkdir(parents=True)
    cli = LovelyLightningCLI.__new__(LovelyLightningCLI)
    cli.subcommand = "fit"
    cli.config = Namespace(
        fit=Namespace(
            trainer=Namespace(
                logger=[
                    Namespace(
                        class_path=TB,
                        init_args=Namespace(save_dir=str(tmp_path), name="exp", version=None),
                    ),
                    Namespace(
                        class_path=CSV,
                        init_args=Namespace(save_dir=str(tmp_path), name="exp", version=None),
                    ),
                ]
            )
        )
    )

    cli.before_instantiate_classes()

    loggers = cli.config["fit"].trainer.logger
    assert loggers[0].init_args.version == 2
    assert loggers[1].init_args.version == 2
