import torch
from math import prod

from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_costmodel.matmul import MatmulPrediction
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.analytical_model.utils.helper import (
    create_op_info,
    merge_flops_dicts,
)
from atencost.backend.analytical_model.utils.datatype import get_dtype_size
from atencost.backend.analytical_model.utils.model_utils import (
    calculate_total_bytes,
    get_tasks_time,
    get_compute_type_flops,
    get_dtype_flops,
    TBps2Bpus,
    TFlopS2FlopUs,
)


@op_manager.register("UnpadFlashAttentionMlaNd")
class UnpadFlashAttentionMlaNdPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "mix"
        self.module = op.instance
        self.hardware = hardware
        self.softmax_dtype = self.module.softmax_dtype
        self.init_vector_instruction()

    def init_vector_instruction(self):
        q_shape, k_shape, v_shape = self.inputs_shape
        q_dtype, k_dtype, v_dtype = self.inputs_dtype

        batch, q_seq_len, head_num, head_dim = q_shape
        _, k_seq_len, _, nope_head_dim = v_shape
        bc = 512 if k_seq_len > 1024 else 256

        m = batch * head_num * q_seq_len
        n = k_seq_len / 2
        dtype_name = str(self.softmax_dtype).split(".")[-1]
        self.vec_instruction = [
            ["VADD", dtype_name, m, n],  # mask
            ["VMULS", dtype_name, m, n],
            ["VADD", dtype_name, m, n],
            ["VCGMAX", dtype_name, m, n],
            ["VMAX", dtype_name, m, n / bc],
            ["VSUB", dtype_name, m, n],
            ["VEXP", dtype_name, m, n],
            ["VSUB", dtype_name, m, n / bc],
            ["VEXP", dtype_name, m, n / bc],
            ["VCGADD", dtype_name, m, n],
            ["VADD", dtype_name, m, n / bc],
            ["VMUL", dtype_name, m, n * head_dim / bc],
            ["VADD", dtype_name, m, n * head_dim / bc],
            ["VCONV", dtype_name, m, n],
        ]

    @property
    def cube_flops_dict(self):
        q_shape, k_shape, v_shape = self.inputs_shape
        q_dtype, k_dtype, v_dtype = self.inputs_dtype

        batch, q_seq_len, head_num, head_dim = q_shape
        _, k_seq_len, _, nope_head_dim = v_shape
        score_shape = [batch, head_num, q_seq_len, k_seq_len]

        # qk matmul
        q_shape = [batch, head_num, q_seq_len, head_dim]
        k_shape = [batch, head_num, head_dim, k_seq_len]
        qk_out_shape = MatmulPrediction.infer_shape(q_shape, k_shape)
        qk_op = create_op_info(
            input_list=[q_shape, k_shape],
            input_dtype_list=[q_dtype, k_dtype],
            output_list=[qk_out_shape],
            output_dtype_list=[q_dtype],
        )
        qk_pred = MatmulPrediction(qk_op, self.hardware)

        qk_flops_dict = {k: v // 2 for k, v in qk_pred.cube_flops_dict.items()}

        # score_v matmul
        v_shape = [batch, head_num, k_seq_len, nope_head_dim]
        score_v_out_shape = MatmulPrediction.infer_shape(score_shape, v_shape)
        score_v_op = create_op_info(
            input_list=[score_shape, v_shape],
            input_dtype_list=[v_dtype, v_dtype],
            output_list=[score_v_out_shape],
            output_dtype_list=[v_dtype],
        )

        score_v_pred = MatmulPrediction(score_v_op, self.hardware)

        score_v_flops_dict = {
            k: v // 2 for k, v in score_v_pred.cube_flops_dict.items()
        }

        total_flops_dict = merge_flops_dicts([qk_flops_dict, score_v_flops_dict])

        return total_flops_dict
