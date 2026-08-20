from math import prod
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.perf_result import OpPerfResult
import torch

@op_manager.register("NSASelectAttn")
class NSASelectAttn(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.head_tail_time = 1

    @property
    def cube_flops_dict(self):
        # print(f"\n\n[AUSTIN Register]: {self.inputs_shape}\n\n")
        # print(f"\n\n[AUSTIN Register]: {type(self.inputs_shape)}\n\n")
        # print(self.inputs_dtype)
        if not self.inputs_shape:
            return {torch.bfloat16: 0}

        # print(f"self.outputs_dtype: {self.outputs_dtype}")
        # print(f"self.inputs_dtype: {self.inputs_dtype}")
        for idx, _ in enumerate((self.inputs_dtype)):
            self.inputs_dtype[idx] = self.outputs_dtype

        dtype = self.outputs_dtype[0]
        sq,b,hq,dqk = self.inputs_shape[0]
        skv,b,hkv,dv = self.inputs_shape[1]
        q_k_flops = 2 * sq * b * hq * skv * dqk
        weight_v_flops = 2 * sq * b * hq * dv * skv
        return {dtype: q_k_flops+weight_v_flops}

    # 根据和恒溪0708的讨论，考虑直接删除memory_size
    @property
    def memory_size(self):
        if not self.inputs_shape:
            return 0
        # 注： NSA Select带有Gather，Gather的内存移动效率应该低于正常的DataCopy？没想好怎么建模这个行为
        for idx, _ in enumerate((self.inputs_dtype)):
            self.inputs_dtype[idx] = self.outputs_dtype

        dtype = self.inputs_dtype[0]
        sq,b,hq,dqk = self.inputs_shape[0]
        skv,b,hkv,dv = self.inputs_shape[1]

        # TODO: 直接取消memory size也可
        qkt = sq*b*hq*dqk + skv*b*hkv*dqk # + sq*b*hq*skv # 融合架构省略qkt的写出和softmax的读入写出

        # 新架构考虑融合架构
        # softmax = 2 * sq * b * hq * skv
        attn_weight_v = skv*b*hkv*dv + sq*b*hq*dv  # sq*b*hq*skv +

        return (qkt + attn_weight_v)*0.5 # FP4

    @property
    def vector_flops_dict(self):
        if not self.inputs_shape:
            return 0

        dtype = self.inputs_dtype[0]
        for idx, _ in enumerate((self.inputs_dtype)):
            self.inputs_dtype[idx] = self.outputs_dtype

        sq,b,hq,dqk = self.inputs_shape[0]
        skv,b,hkv,dv = self.inputs_shape[1]

        softmax_vec = 3 * sq * b * hq * skv

        return {dtype: softmax_vec}


    @property
    def op_time(self):
        overlap_softmax = 0.5
        return max([self.cube_time, self.vec_time + self.memory_time])+self.head_tail_time