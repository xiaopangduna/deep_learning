"""比较两个图片输入的 ONNX。同一份输入，逐个输出看是否足够接近。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

ATOL = 1e-4
RTOL = 1e-4
SEED = 0

_FLOAT_DTYPES = {"float16", "float32", "float64"}
_ORT_DTYPE = {
    "tensor(float16)": "float16",
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | None, ...]
    dtype: str
    is_tensor: bool = True


@dataclass(frozen=True)
class OnnxInfo:
    path: str
    nodes: int
    opset: tuple[int, ...]
    input: TensorSpec
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class ImageInput:
    """已经解析成可喂模型的单张图：batch 固定为 1。"""

    name: str
    shape: tuple[int, int, int, int]
    dtype: str


@dataclass(frozen=True)
class Feed:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class OutputStats:
    index: int
    ref_name: str
    other_name: str
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    ok: bool


@dataclass(frozen=True)
class OnnxCompareResult:
    ok: bool
    image: str | None
    ref: OnnxInfo
    other: OnnxInfo
    feed: Feed | None
    outputs: tuple[OutputStats, ...]
    errors: tuple[str, ...]


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _static_dim(dim: object) -> int | None:
    if isinstance(dim, (int, np.integer)) and int(dim) > 0:
        return int(dim)
    if isinstance(dim, str) and dim.isdigit() and int(dim) > 0:
        return int(dim)
    return None


def _tensor_spec(value: ort.NodeArg) -> TensorSpec:
    type_name = value.type
    return TensorSpec(
        name=value.name,
        shape=tuple(_static_dim(dim) for dim in value.shape),
        dtype=_ORT_DTYPE.get(type_name, type_name),
        is_tensor=type_name.startswith("tensor("),
    )


def summarize_onnx(path: Path) -> OnnxInfo:
    try:
        model = onnx.load(str(path))
        session = _session(path)
    except Exception as exc:
        raise ValueError(f"读不了模型: {path}: {exc}") from exc
    inputs = tuple(_tensor_spec(item) for item in session.get_inputs())
    if len(inputs) != 1:
        raise ValueError(f"{path} 须为单个图片输入，当前输入个数 {len(inputs)}")
    outputs = tuple(_tensor_spec(item) for item in session.get_outputs())
    return OnnxInfo(
        path=str(path),
        nodes=len(model.graph.node),
        opset=tuple(item.version for item in model.opset_import),
        input=inputs[0],
        outputs=outputs,
    )


def require_image_input(info: OnnxInfo) -> ImageInput:
    spec = info.input
    if not spec.is_tensor or spec.dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{info.path} 输入 {spec.name} 须为浮点图片，当前 {spec.dtype}")
    if len(spec.shape) != 4:
        raise ValueError(f"{info.path} 输入须为 NCHW，当前 {list(spec.shape)}")
    batch, channels, height, width = spec.shape
    if batch is not None and batch != 1:
        raise ValueError(f"{info.path} batch 须为 1，当前 {batch}")
    if channels not in (1, 3):
        raise ValueError(f"{info.path} 通道须为 1 或 3，当前 {channels}")
    if height is None or width is None:
        raise ValueError(f"{info.path} 高宽须为静态正整数，当前 {list(spec.shape)}")
    return ImageInput(name=spec.name, shape=(1, int(channels), int(height), int(width)), dtype=spec.dtype)


def io_errors(ref: OnnxInfo, other: OnnxInfo, ref_in: ImageInput, other_in: ImageInput) -> list[str]:
    errors: list[str] = []
    if ref_in.dtype != other_in.dtype:
        errors.append(f"输入 dtype {ref_in.dtype} vs {other_in.dtype}")
    if ref_in.shape != other_in.shape:
        errors.append(f"输入形状 {list(ref_in.shape)} vs {list(other_in.shape)}")
    if not ref.outputs and not other.outputs:
        errors.append("没有输出")
        return errors
    if len(ref.outputs) != len(other.outputs):
        errors.append(f"输出个数 {len(ref.outputs)} vs {len(other.outputs)}")
    for index, (left, right) in enumerate(zip(ref.outputs, other.outputs)):
        if not left.is_tensor or not right.is_tensor:
            name = left.name if not left.is_tensor else right.name
            kind = left.dtype if not left.is_tensor else right.dtype
            errors.append(f"out[{index}] {name} 不是张量 ({kind})")
            continue
        if left.dtype != right.dtype:
            errors.append(f"out[{index}] {left.name} dtype {left.dtype} vs {right.dtype}")
        if len(left.shape) != len(right.shape):
            errors.append(f"out[{index}] {left.name} 秩 {len(left.shape)} vs {len(right.shape)}")
            continue
        for dim_ref, dim_other in zip(left.shape, right.shape):
            if dim_ref is not None and dim_other is not None and dim_ref != dim_other:
                errors.append(
                    f"out[{index}] {left.name} 形状 {list(left.shape)} vs {list(right.shape)}"
                )
                break
    return errors


def _load_image(path: Path, channels: int, height: int, width: int) -> np.ndarray:
    try:
        image = Image.open(path)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"读不了图片: {path}") from exc
    mode = "L" if channels == 1 else "RGB"
    array = np.asarray(image.convert(mode).resize((width, height), Image.BILINEAR), dtype=np.float32)
    array = array / 255.0
    if channels == 1:
        return array[None, None]
    return array.transpose(2, 0, 1)[None]


def make_feed(spec: ImageInput, image: Path | None) -> np.ndarray:
    if image is None:
        array = np.random.default_rng(SEED).random(spec.shape, dtype=np.float32)
    else:
        array = _load_image(image, spec.shape[1], spec.shape[2], spec.shape[3])
    dtype = np.dtype(spec.dtype)
    if array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _run(path: str, input_name: str, array: np.ndarray) -> list[np.ndarray]:
    session = _session(Path(path))
    names = [item.name for item in session.get_outputs()]
    return session.run(names, {input_name: array})


def compare_arrays(ref: np.ndarray, other: np.ndarray) -> OutputStats:
    if ref.shape != other.shape:
        raise ValueError(f"输出形状 {list(ref.shape)} vs {list(other.shape)}")
    delta = np.abs(ref.astype(np.float64) - other.astype(np.float64))
    return OutputStats(
        index=0,
        ref_name="",
        other_name="",
        shape=tuple(int(dim) for dim in ref.shape),
        max_abs=float(delta.max()) if delta.size else 0.0,
        mean_abs=float(delta.mean()) if delta.size else 0.0,
        ok=bool(np.allclose(other, ref, atol=ATOL, rtol=RTOL)),
    )


def compare_onnx(ref: Path, other: Path, image: Path | None = None) -> OnnxCompareResult:
    if not ref.is_file():
        raise FileNotFoundError(f"找不到参考模型: {ref}")
    if not other.is_file():
        raise FileNotFoundError(f"找不到待比较模型: {other}")
    if image is not None and not image.is_file():
        raise FileNotFoundError(f"找不到图片: {image}")

    ref_info = summarize_onnx(ref)
    other_info = summarize_onnx(other)
    ref_in = require_image_input(ref_info)
    other_in = require_image_input(other_info)
    errors = io_errors(ref_info, other_info, ref_in, other_in)
    image_text = None if image is None else str(image)
    if errors:
        return OnnxCompareResult(
            ok=False,
            image=image_text,
            ref=ref_info,
            other=other_info,
            feed=None,
            outputs=(),
            errors=tuple(errors),
        )

    array = make_feed(ref_in, image)
    feed = Feed(shape=tuple(int(dim) for dim in array.shape), dtype=str(array.dtype))
    ref_outs = _run(ref_info.path, ref_in.name, array)
    other_outs = _run(other_info.path, other_in.name, array)
    stats: list[OutputStats] = []
    for index, (left, right) in enumerate(zip(ref_outs, other_outs)):
        ref_name = ref_info.outputs[index].name
        other_name = other_info.outputs[index].name
        if left.shape != right.shape:
            errors.append(f"out[{index}] {ref_name} 形状 {list(left.shape)} vs {list(right.shape)}")
            continue
        item = compare_arrays(left, right)
        stats.append(
            OutputStats(
                index=index,
                ref_name=ref_name,
                other_name=other_name,
                shape=item.shape,
                max_abs=item.max_abs,
                mean_abs=item.mean_abs,
                ok=item.ok,
            )
        )
    return OnnxCompareResult(
        ok=not errors and all(item.ok for item in stats) and len(stats) == len(ref_info.outputs),
        image=image_text,
        ref=ref_info,
        other=other_info,
        feed=feed,
        outputs=tuple(stats),
        errors=tuple(errors),
    )
