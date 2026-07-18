from math import prod
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.perf_result import ZhanluPerfResult
import torch

@op_manager.register("NSACompress")
class NSACompress(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def cube_flops_dict(self):

        if not self.inputs_shape:
            return {torch.bfloat16: 0}
        b,s,h,d = self.inputs_shape[0]
        comp_blocksize,kv_group_num,comp_stride = self.inputs_shape[1]

        _, output_len, _, d = self.outputs_shape[0]

        dtype = self.outputs_dtype[0]
        #print(f"[austin debug]: dtype: {dtype}")
        #print(f"[austin debug]: self.outputs_dtype: {self.outputs_dtype}")
        #print(f"[austin debug]: self.inputs_dtype: {self.inputs_dtype}\n\n\n")
        for idx, _ in enumerate((self.inputs_dtype)):
            self.inputs_dtype[idx] = self.outputs_dtype

        flops = prod([output_len,h,d])*comp_blocksize*2*b
        # print(f"\n\n[AUSTIN Register]: {self.inputs_shape}\n\n")
        return {dtype: flops}

    @property
    def memory_size(self):
        if not self.inputs_shape:
            return 0
        b,s,h,d = self.inputs_shape[0]
        output_len, _, _, d = self.outputs_shape[0]
        # print(f"outputlen: {output_len}, h: {h}, d: {d}")
        comp_blocksize,kv_group_num,comp_stride = self.inputs_shape[1]
        dtype = self.outputs_dtype[0]
        # comp_blocksize = 32

        memory_access_weight = prod([comp_blocksize, h, d])*0.5 # for now assume bfloat16, change to dtype later
        memory_access_activation = prod([output_len, comp_blocksize, h, d])*0.5
        memory_access = memory_access_weight + memory_access_activation

        return memory_access