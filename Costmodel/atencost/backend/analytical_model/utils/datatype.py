import re

import torch

from atencost.frontend.utils.tensor_record import CustomDType


def get_dtype_size(dtype) -> float:
    if isinstance(dtype, str):
        if "float8_e4m3fn" in dtype:
            return 1
        if "ele_size" in dtype:
            # dtype="CustomDType(dtype='NvFP4', ele_size=0.5)"
            match = re.search(r'ele_size=([0-9]*\.?[0-9]+)', dtype)
            if match:
                return float(match.group(1))
        return torch.tensor(0, dtype=getattr(torch, dtype.split('.')[-1])).element_size()
    elif isinstance(dtype, CustomDType):
        return dtype.element_size()
    return 0.0
