from typing import List

import attrs


@attrs.define(frozen=True, slots=True)
class Config:
    types: List[str] = attrs.field(default=['Grape', 'Apple'])
    split_sub_dirs: bool = attrs.field(default=False)
