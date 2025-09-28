from typing import Dict, List, Type
from .base import Transformation


_REGISTRY: Dict[str, Type[Transformation]] = {}


def register(name: str):
    def wrap(cls: Type[Transformation]):
        _REGISTRY[name] = cls
        return cls

    return wrap


from . import ops as _ops  # noqa: F401,E402


def build(name: str, **kwargs) -> Transformation:
    return _REGISTRY[name](**kwargs)


def available_ops() -> List[str]:
    return list(_REGISTRY.keys())
