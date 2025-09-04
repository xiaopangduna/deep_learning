import importlib


def dynamic_class_instantiate_from_string(class_path: str, **kwargs):
    """根据字符串路径动态实例化类"""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**kwargs)
