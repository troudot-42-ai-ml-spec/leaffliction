from typing import List

import attrs


@attrs.define(frozen=True, slots=True)
class Config:
    types: List[str] = attrs.field(default=["Grape", "Apple"])
