from pathlib import Path
from argparse import Namespace
from typing import List, Dict
import attrs

from config import Config


@attrs.define(slots=True)
class Type:
    name: str
    labels: Dict[str, int] = attrs.field(factory=dict)


def parse(args: Namespace, cfg: Config) -> List[Type]:
    types: Dict[str, Type] = {type: Type(name=type) for type in cfg.types}

    for root, dirs, files in Path(args.path).walk(on_error=print):
        separator_i: int = root.name.find('_')
        name: str = root.name[:separator_i]
        if name in cfg.types:
            types[name].labels[root.name[separator_i+1:]] = len(files)
    return [type for type in types.values()]
