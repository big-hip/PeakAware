from dataclasses import dataclass
from typing import Any
import re


@dataclass
class OpRecord:
    id:              int
    name:            str
    type:            str
    subtype:         str
    comm_type:       str
    inputs:          list
    outputs:         list
    module_instance: Any
    module_path:     str
    module_id:       int
    fusion_type:     str
    raw_name:        str
    instance:        Any
    op_type:         str  # Used for PDES, vec/cube/h2d/d2h/comm/mix is allowed
    trace_back: str = ''
    graph_name: str = ''

    @property
    def global_rank_list(self):
        return getattr(self.instance, 'global_rank_list', [])


def to_camel(op_name: str) -> str:
    is_inplace = op_name.endswith("_")
    is_out_variant = op_name.endswith(".out")

    clean_name = op_name.replace(".", " ").replace("_", " ")
    parts = clean_name.split()

    camel_parts = [part[0].upper() + part[1:].lower() for part in parts if part]
    camel_name = "".join(camel_parts)

    if is_inplace:
        camel_name += "Inplace"
    elif is_out_variant:
        camel_name += "Out"

    return camel_name


def simplify_op_name(op_str):
    """Extract and simplify operation name"""
    op_name = str(op_str).lower()

    # Remove aten prefix if present
    if op_name.startswith("aten::"):
        op_name = op_name[6:]

    if op_name.startswith("aten."):
        op_name = op_name[5:]

    if op_name.startswith("_c10d_functional."):
        op_name = op_name[17:]

    if op_name.startswith("custom_ops."):
        op_name = op_name[11:]

    if op_name.startswith("c10d."):
        op_name = op_name[5:]

    if op_name.startswith("npu.npu_"):
        op_name = op_name[8:]

    if op_name.startswith("npu._npu_"):
        op_name = op_name[9:]

    if op_name.startswith("mindspeed.npu_"):
        op_name = op_name[14:]

    # Remove suffixes like _Tensor, _default, etc.
    op_name = re.sub(r"\.(Tensor|default|scalar|.*)$", "", op_name)

    #原本是返回，对于通信算子我得重新改名
    op_name=to_camel(op_name)
    pattern = r"^Hccl([a-zA-Z]+)\d*$"#有可能有数字，有可能没数字
    match = re.match(pattern, op_name)
    if match:
        # 提取中间的算子名 (例如 "AllReduce")
        core_op_name = match.group(1)
        # 返回新格式: 核心名 + Kernel
        return f"{core_op_name}Kernel"
    else:
        # 不符合格式，原样返回，不做任何修改
        return op_name
    # # Use mapping if available
    # return to_camel(op_name)


def get_op_type(op_str):
    op_name = str(op_str).lower()
    if "mm" in op_name or "matmul" in op_name or "gmm" in op_name or "ppmatmulaccumatomickernel" in op_name:
        return "cube"
    if "flash" in op_name or "attention" in op_name:
        return "mix"
    if "allreduce" in op_name or "allgather" in op_name or "broadcast" in op_name:
        return "comm"
    if 'py-fake' in op_name:
        return "comm"
    return "vec"
