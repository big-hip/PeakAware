from dataclasses import dataclass
from typing import Any

@dataclass
class CustomDType:
    dtype:         str
    ele_size:      float

    def __hash__(self):
        return hash(self.dtype)

    def element_size(self):
        return self.ele_size


@dataclass
class TensorRecord:
    name:          str
    global_shape:  list
    local_shape:   list
    type:          str
    dtype:         Any       # Recommend to use torch.dtype or CustomDType only
    is_dtensor:    bool
    requires_grad: bool
    module_path:   str
    module_id:     int
    device_mesh:   str
    placements:    str
    producer:      list
    consumer:      list