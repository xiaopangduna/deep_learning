from __future__ import annotations

from lovely_deep_learning.export.base import BaseExporter


class YOLOv8Exporter(BaseExporter):
    """``BaseExporter`` 别名。官方解码请在 YAML 中配置 ``export_head: YOLOv8Decode``。"""
