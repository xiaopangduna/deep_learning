from typing import Dict, Any, Optional, Union
import os
import urllib.request

# weight_loader/base.py
import torch
import torch.nn as nn




class DAGWeightLoader():
    """
    用于加载 TorchVision 预训练权重的加载器。
    专为 DAGNet 设计，假设其键名比标准 TorchVision 模型多了 'layers.' 前缀。
    支持从本地路径加载，或从 URL 下载并缓存到指定路径。
    """

    # 固定的键名前缀
    _KEY_PREFIX = "layers."

    def load_weights(
        self,
        model: nn.Module,
        path: str,
        url: Optional[str] = None,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        strict: bool = False,
    ) -> None:
        """
        加载 TorchVision 预训练权重到指定的 DAGNet 模型。
        权重文件键名（如 'conv1.weight'）会自动加上 'layers.' 前缀（如 'layers.conv1.weight'）
        以匹配 DAGNet 模型的结构。
        如果 path 指向的文件存在，则直接加载。
        如果 path 文件不存在，且提供了 url，则从 url 下载到 path，然后加载。

        Args:
            model (nn.Module): 目标 DAGNet 模型实例。
            path (str): 权重文件 (.pth) 的本地路径。
                        如果文件不存在且提供了 url，将从 url 下载到此路径。
            url (str, optional): 权重文件的网络 URL。如果 path 文件不存在，则从此 URL 下载。
            map_location (str or torch.device, optional): torch.load 的 map_location。
                                                             默认为 'cpu'。
            strict (bool, optional): 是否使用严格模式加载权重。
                                     如果为 True，键名必须完全匹配。
                                     如果为 False (默认)，则忽略不匹配的键。
        Raises:
            ValueError: 如果 path 未提供，或 path 文件不存在且未提供 url。
        """
        if not path:
            raise ValueError("'path' must be provided as the target file location.")

        final_path = path

        # 1. 检查本地路径文件是否存在
        if os.path.isfile(path):
            print(f"TorchVisionWeightLoader: Found local weight file at {path}. Using it.")
        elif url:
            # 2. 如果本地文件不存在，但提供了 URL，则下载
            print(f"TorchVisionWeightLoader: Local file {path} not found. Downloading from {url}...")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                urllib.request.urlretrieve(url, path)
                print(f"TorchVisionWeightLoader: Downloaded weights to {path}.")
            except Exception as e:
                error_msg = f"TorchVisionWeightLoader Error: Failed to download weights from {url} to {path}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            error_msg = (
                f"TorchVisionWeightLoader Error: Weight file not found at {path} and no 'url' provided for download."
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        # 3. 从确定的本地路径加载权重
        try:
            print(f"TorchVisionWeightLoader: Loading weights from {final_path} (map_location={map_location})")
            source_state_dict: Dict[str, Any] = torch.load(final_path, map_location=map_location)
            print(f"TorchVisionWeightLoader: Weights loaded successfully. Total keys: {len(source_state_dict)}")

            # 4. 固定映射键名 (添加 'layers.' 前缀)
            mapped_state_dict: Dict[str, Any] = {}
            model_state_keys = set(model.state_dict().keys())

            print(f"TorchVisionWeightLoader: Mapping keys with fixed prefix '{self._KEY_PREFIX}'...")
            for src_key, value in source_state_dict.items():
                # 将 TorchVision 的键 'conv1.weight' 映射为 DAGNet 的键 'layers.conv1.weight'
                dag_net_key = self._KEY_PREFIX + src_key
                if dag_net_key in model_state_keys:
                    mapped_state_dict[dag_net_key] = value
                # else:
                #     print(f"Info: Mapped key '{dag_net_key}' (from '{src_key}') not found in model. Skipping.")

            print(f"TorchVisionWeightLoader: Mapped {len(mapped_state_dict)} compatible keys.")

            # 5. 加载映射后的权重到模型
            print(
                f"TorchVisionWeightLoader: Loading {len(mapped_state_dict)} mapped keys into model (strict={strict})..."
            )
            missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=strict)

            if strict and (missing_keys or unexpected_keys):
                print(f"Warning (Strict Mode): Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
            elif not strict:
                if missing_keys:
                    print(f"Info (Non-Strict): Missing keys (in model but not in mapped weights): {missing_keys}")

            print("TorchVisionWeightLoader: Weights loading process completed.")

        except Exception as e:
            error_msg = f"TorchVisionWeightLoader Error during loading from {final_path}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
