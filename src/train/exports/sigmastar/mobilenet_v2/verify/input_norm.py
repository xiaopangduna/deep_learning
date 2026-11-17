"""解析 --mean / --std。单图相似度和多图准确率两条链路共用。

不传时用 RGB ImageNet：123.68,116.28,103.53 和 58.395,57.12,57.375，
与 input_config.ini 一致。两边都写 none 时返回 (None, None)，像素保持原值。
只写其中一个，或一边是数字、另一边是 none，直接退出。
"""
from __future__ import annotations

DEFAULT_MEAN = (123.68, 116.28, 103.53)
DEFAULT_STD = (58.395, 57.12, 57.375)


def _parse_floats(text: str, flag: str) -> list[float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        raise SystemExit(f"{flag} is empty")
    values: list[float] = []
    for item in parts:
        try:
            values.append(float(item))
        except ValueError:
            raise SystemExit(f"{flag} value {item!r} is not a number or none") from None
    return values


def _expand(values: list[float], channels: int, flag: str) -> list[float]:
    if len(values) == 1:
        return values * channels
    if len(values) == channels:
        return values
    raise SystemExit(f"{flag} has {len(values)} values, model has {channels} channels")


def resolve_norm(
    mean_text: str | None,
    std_text: str | None,
    channels: int,
) -> tuple[list[float] | None, list[float] | None]:
    """Return per-channel mean and std, or (None, None) when pixels stay raw."""
    if mean_text is not None:
        mean_text = mean_text.strip()
    if std_text is not None:
        std_text = std_text.strip()
    if mean_text is None and std_text is None:
        mean_vals = list(DEFAULT_MEAN)
        std_vals = list(DEFAULT_STD)
    elif mean_text is None or std_text is None:
        raise SystemExit("--mean and --std must be passed together")
    elif mean_text.lower() == "none" and std_text.lower() == "none":
        return None, None
    elif mean_text.lower() == "none" or std_text.lower() == "none":
        raise SystemExit("--mean and --std must both be none, or both be numbers")
    else:
        mean_vals = _parse_floats(mean_text, "--mean")
        std_vals = _parse_floats(std_text, "--std")

    mean_vals = _expand(mean_vals, channels, "--mean")
    std_vals = _expand(std_vals, channels, "--std")
    if any(value == 0.0 for value in std_vals):
        raise SystemExit("--std contains 0")
    return mean_vals, std_vals
