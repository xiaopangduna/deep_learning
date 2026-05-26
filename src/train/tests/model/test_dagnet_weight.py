import pytest

from lovely_deep_learning.model.weight_loader import DAGNetWeightLoader


def test_normalize_official_opts_in_stage():
    steps = DAGNetWeightLoader.normalize_steps(
        {
            "map_location": "cpu",
            "strict": False,
            "stages": [
                {
                    "format": "official",
                    "path": "pretrained_models/foo.pth",
                    "opts": {
                        "url": "https://example.com/foo.pth",
                        "src_key_prefix": "layers.",
                        "src_key_slice_start": 0,
                    },
                }
            ],
        }
    )
    assert len(steps) == 1
    assert steps[0]["format"] == "official"
    assert steps[0]["url"] == "https://example.com/foo.pth"
    assert steps[0]["src_key_prefix"] == "layers."
    assert steps[0]["map_location"] == "cpu"
    assert steps[0]["strict"] is False


def test_normalize_stages_pipeline():
    steps = DAGNetWeightLoader.normalize_steps(
        {
            "map_location": "cpu",
            "strict": False,
            "stages": [
                {"format": "torch_pruning", "path": "a.pth"},
                {"format": "dense", "path": "b.ckpt"},
            ],
        }
    )
    assert len(steps) == 2
    assert steps[0]["format"] == "torch_pruning"
    assert steps[1]["format"] == "dense"
    assert "url" not in steps[0]


def test_normalize_rejects_top_level_format():
    with pytest.raises(ValueError, match="top-level.*'format'"):
        DAGNetWeightLoader.normalize_steps(
            {
                "format": "official",
                "path": "x.pth",
                "stages": [{"format": "dense", "path": "y.ckpt"}],
            }
        )


def test_normalize_rejects_format_path_without_stages():
    with pytest.raises(ValueError, match="non-empty 'stages'|'format'"):
        DAGNetWeightLoader.normalize_steps(
            {"format": "official", "path": "x.pth", "map_location": "cpu"}
        )


def test_normalize_requires_non_empty_stages():
    with pytest.raises(ValueError, match="non-empty 'stages'"):
        DAGNetWeightLoader.normalize_steps({"map_location": "cpu", "stages": []})


def test_normalize_rejects_top_level_url():
    with pytest.raises(ValueError, match="unknown top-level keys"):
        DAGNetWeightLoader.normalize_steps(
            {
                "map_location": "cpu",
                "url": "https://example.com/foo.pth",
                "stages": [{"format": "official", "path": "foo.pth"}],
            }
        )


def test_normalize_rejects_step_level_url():
    with pytest.raises(ValueError, match="unexpected keys.*url"):
        DAGNetWeightLoader.normalize_steps(
            {
                "stages": [
                    {
                        "format": "official",
                        "path": "foo.pth",
                        "url": "https://example.com/foo.pth",
                    },
                ],
            }
        )


def test_normalize_rejects_unknown_top_level_key():
    with pytest.raises(ValueError, match="unknown top-level keys.*path_custom"):
        DAGNetWeightLoader.normalize_steps({"path_custom": "ckpt.ckpt"})
