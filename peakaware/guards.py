from __future__ import annotations

import json
from typing import Any


def static_guard_value(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        encoded = json.dumps(value, sort_keys=True)
    else:
        encoded = repr(value)
    return f"{type(value).__module__}.{type(value).__qualname__}:{encoded}"
