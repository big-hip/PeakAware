from numpy import dtype
from zhanlu.frontend.utils.tensor_record import CustomDType
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager

from zhanlu.backend.analytical_model.utils.datatype import get_dtype_size
from zhanlu.backend.analytical_model.utils.helper import broadcast_shapes, create_op_info
from math import prod


@op_manager.register("Matmul", "Mm", "Addmm", "Bmm", "MatMul")
class MatmulPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):
        if self.op.name == "Addmm":
            _, shape1, shape2 = self.inputs_shape
            _, dtype1, dtype2 = self.inputs_dtype

        else:
            shape1, shape2 = self.inputs_shape
            dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        if len(shape1) < 2 or len(shape2) < 2:
            raise ValueError(
                f"Matmul supports only tensors with at least 2 dimensions: {shape1} and {shape2}"
            )

        # if shape1[-1] != shape2[-2]:
        #     shape1, shape2 = shape2, shape1
        if shape1[-1] != shape2[-2]:
            shape2[-2], shape2[-1] = shape2[-1], shape2[-2]

        if shape1[-1] != shape2[-2]:
            shape1[-2], shape1[-1] = shape1[-1], shape1[-2]

        if shape1[-1] != shape2[-2]:
            shape2[-2], shape2[-1] = shape2[-1], shape2[-2]

        if shape1[-1] != shape2[-2]:
            # raise ValueError(
            #     f"Matmul inputs must have compatible dimensions: {shape1} and {shape2}"
            # )
            return {self.compute_type: 0}

        batch_dims1 = shape1[:-2]
        batch_dims2 = shape2[:-2]
        batch_dims1_last = shape1[-2:]
        batch_dims2_last = shape2[-2:]
        batch_dims = broadcast_shapes(batch_dims1, batch_dims2)
        if batch_dims == 0:
            return {self.compute_type: 0}
        flops = 2 * prod(batch_dims) * prod(batch_dims1_last) * batch_dims2_last[-1]
        return {self.compute_type: flops}

    @staticmethod
    def infer_shape(shape1, shape2):
        batch_dims1 = shape1[:-2]
        batch_dims2 = shape2[:-2]
        batch_dims1_last = shape1[-2:]
        batch_dims2_last = shape2[-2:]
        batch_dims = broadcast_shapes(batch_dims1, batch_dims2)

        return batch_dims + [batch_dims1_last[-2], batch_dims2_last[-1]]


@op_manager.register("VarLenQKMatmul")
class VarLenQKMatmulPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.kv_var_lens = op.inputs[0]

    @property
    def cube_flops_dict(self):
        shape1, shape2 = self.inputs_shape
        dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        if shape1[-1] != shape2[-2]:
            raise ValueError(
                f"Matmul inputs must have compatible dimensions: {shape1} and {shape2}"
            )

        if shape1[-3] != shape2[-3]:
            raise ValueError(
                "VarLenMatmul inputs must have -3-dim to be the value of batch size."
            )
        batch_size = shape1[-3]

        # shape1: [decode_len, batch_size, num_heads, hidden_size_per_head]
        # shape2: [batch_size, hidden_size_per_head, max_kv_cache_prefill_decode_len]

        flops = 0
        # print(f"batch_size={batch_size}")
        # print(f"self.kv_var_lens={self.kv_var_lens}")
        for i in range(batch_size):
            # curr_shape1 [decode_len, num_heads, h_per_head], curr_shape2 [h_per_head, kv_len_of_ith_batch]
            flops += 2 * shape1[0] * prod(shape1[-2:]) * self.kv_var_lens[i]

        return {self.compute_type: flops}

    @staticmethod
    def infer_shape(shape1, shape2):
        # shape1: [decode_len, batch_size, num_heads, hidden_size_per_head]
        # shape2: [batch_size, hidden_size_per_head, max_kv_cache_prefill_decode_len]
        batch_dims1 = shape1[:-2]
        batch_dims2 = shape2[:-2]
        batch_dims1_last = shape1[-2:]
        batch_dims2_last = shape2[-2:]
        batch_dims = broadcast_shapes(batch_dims1, batch_dims2)

        # return shape: [decode_len, batch_size, num_heads, max_kv_cache_prefill_decode_len]
        return batch_dims + [batch_dims1_last[-2], batch_dims2_last[-1]]


@op_manager.register("VarLenScoreVMatmul")
class VarLenScoreVMatmulPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.kv_var_lens = op.inputs[0]

    @property
    def cube_flops_dict(self):
        shape1, shape2 = self.inputs_shape
        dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        if shape1[-1] != shape2[-2]:
            raise ValueError(
                f"Matmul inputs must have compatible dimensions: {shape1} and {shape2}"
            )

        if shape1[-3] != shape2[-3]:
            raise ValueError(
                "VarLenMatmul inputs must have -3-dim to be the value of batch size."
            )
        # shape1: [decode_len, batch_size, num_heads, max_kv_cache_prefill_decode_len]
        # shape2: [batch_size, max_prefill_decode_length, kv_lora_rank]
        batch_size = shape1[-3]

        flops = 0
        for i in range(batch_size):
            # curr_shape1 [decode_len, num_heads, kv_len_of_ith_batch], curr_shape2 [kv_len_of_ith_batch, kv_lora_rank]
            flops += 2 * prod(shape1[:2]) * self.kv_var_lens[i] * shape2[-1]

        return {self.compute_type: flops}

    @staticmethod
    def infer_shape(shape1, shape2):
        # shape1: [decode_len, batch_size, num_heads, max_kv_cache_prefill_decode_len]
        # shape2: [batch_size, max_prefill_decode_length, kv_lora_rank]
        batch_dims1 = shape1[:-2]
        batch_dims2 = shape2[:-2]
        batch_dims1_last = shape1[-2:]
        batch_dims2_last = shape2[-2:]
        batch_dims = broadcast_shapes(batch_dims1, batch_dims2)

        # return shape: [decode_len, batch_size, num_heads, kv_lora_rank]
        return batch_dims + [batch_dims1_last[-2], batch_dims2_last[-1]]


@op_manager.register("Ppmatmulaccumatomickernel")
class PpMatmulPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.op = op

    @property
    def cube_flops_dict(self):
        dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        in_shape1, in_shape2 = self.inputs_shape
        out_shape = self.outputs_shape[0]

        M = out_shape[0]
        N = out_shape[-1]
        K_in_1 = in_shape1[0]
        K_in_2 = in_shape2[0]

        assert K_in_1 == K_in_2

        flops = 2 * M * K_in_1 * N

        return {self.compute_type: flops}


@op_manager.register("MatmulBackward")
class MatmulBwdPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.op_type = "cube"
        self.op = op

    @property
    def cube_flops_dict(self):
        dtype1, dtype2, _ = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        grad_out, x, w = self.inputs_shape # [2048, 1, 256]  [2048, 1, 7168]  [7168, 256]
        grad_x, grad_w = self.outputs_shape # [2048, 1, 7168]  [7168, 256]

        # Calculate FLOPs for dX
        grad_x_M = grad_x[0]
        grad_x_N = grad_x[-1]
        grad_out_K = grad_out[-1]
        w_K = w[-1]

        assert grad_out_K == w_K

        x_flops = 2 * grad_x_M * grad_out_K * grad_x_N

        # Calculate FLOPs for dW
        grad_w_M = grad_w[0]
        grad_w_N = grad_w[-1]
        x_K = x[0]
        grad_out_K = grad_out[0]

        assert grad_out_K == x_K

        w_flops = 2 * grad_w_M * x_K * grad_w_N

        flops = x_flops + w_flops

        return {self.compute_type: flops}


@op_manager.register("Convolution")
class ConvPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        input_shape = self.inputs_shape[0]
        weight_shape = self.inputs_shape[1]
        output_shape = self.outputs_shape[0]

        if len(input_shape) == 4:  # Conv2D
            self.conv_nd = "2d"
            self.N, self.C_in, self.H_in, self.W_in = input_shape
            self.C_out, self.C_in_g, self.K_h, self.K_w = weight_shape
            _, _, self.H_out, self.W_out = output_shape
            if self.C_in % self.C_in_g != 0:
                raise ValueError(
                    f"Input channel {self.C_in} is not divisible by weight C_in_per_group {self.C_in_g}")
            self.group = self.C_in//self.C_in_g
        elif len(input_shape) == 3:  # Conv1D
            self.conv_nd = "1d"
            self.N, self.C_in, self.L_in = input_shape
            self.C_out,self.C_in_g, self.K_l = weight_shape
            _, _, self.L_out = output_shape
            if self.C_in % self.C_in_g != 0:
                raise ValueError(
                    f"Input channel {self.C_in} is not divisible by weight C_in_per_group {self.C_in_g}")
            self.group = self.C_in//self.C_in_g
        else:
            raise ValueError("Unsupported Conv dimensionality: input shape length must be 3 or 4.")

        self.add_bias = True if len(self.inputs_shape) >= 3 else False
        self.op_type = "cube"

@op_manager.register("ConvolutionBackward")
class ConvBackPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

        # 反向输入顺序: [dy, x, w]
        dy_shape = self.inputs_shape[0]  # grad_output
        x_shape = self.inputs_shape[1]   # input from forward
        w_shape = self.inputs_shape[2]   # weight

        if len(x_shape) == 4:  # Conv2D
            self.conv_nd = "2d"
            self.N, self.C_in, self.H_in, self.W_in = x_shape
            self.C_out, C_in_per_group, self.K_h, self.K_w = w_shape
            _, _, self.H_out, self.W_out = dy_shape

            # group参数自动推断
            if self.C_in % C_in_per_group != 0:
                raise ValueError(f"Input channel {self.C_in} is not divisible by weight C_in_per_group {C_in_per_group}")
            self.groups = self.C_in // C_in_per_group
            if self.C_out % self.groups != 0:
                raise ValueError(f"Output channel {self.C_out} is not divisible by groups {self.groups}")
        elif len(x_shape) == 3:  # Conv1D
            self.conv_nd = "1d"
            self.N, self.C_in, self.L_in = x_shape
            self.C_out, C_in_per_group, self.K_l = w_shape
            _, _, self.L_out = dy_shape

            # group参数自动推断
            if self.C_in % C_in_per_group != 0:
                raise ValueError(f"Input channel {self.C_in} is not divisible by weight C_in_per_group {C_in_per_group}")
            self.groups = self.C_in // C_in_per_group
            if self.C_out % self.groups != 0:
                raise ValueError(f"Output channel {self.C_out} is not divisible by groups {self.groups}")
        else:
            raise ValueError("Unsupported Conv dimensionality: input shape length must be 3 or 4.")

        self.op_type = "cube"

    def backward_for_x_flops(self):
        # 计算 dX 的 FLOPs
        if self.conv_nd == "2d":
            # 每组分别计算
            group_flops = (
                (self.N * (self.C_in // self.groups) * self.H_in * self.W_in)
                * (self.C_out // self.groups) * self.K_h * self.K_w * 2
            )
            return group_flops * self.groups
        elif self.conv_nd == "1d":
            group_flops = (
                (self.N * (self.C_in // self.groups) * self.L_in)
                * (self.C_out // self.groups) * self.K_l * 2
            )
            return group_flops * self.groups
        else:
            raise ValueError("Unsupported backward mode.")

    def backward_for_w_flops(self):
        # 计算 dW 的 FLOPs
        if self.conv_nd == "2d":
            group_flops = (
                self.N * (self.C_out // self.groups) * (self.C_in // self.groups)
                * self.K_h * self.K_w * self.H_out * self.W_out * 2
            )
            return group_flops * self.groups
        elif self.conv_nd == "1d":
            group_flops = (
                self.N * (self.C_out // self.groups) * (self.C_in // self.groups)
                * self.K_l * self.L_out * 2
            )
            return group_flops * self.groups
        else:
            raise ValueError("Unsupported backward mode.")

    @property
    def cube_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = self.backward_for_w_flops()
        FLOPs += self.backward_for_x_flops()
        return {compute_dtype: FLOPs}



def get_prediction_by_linear(input_shape, input_dtype, linear, hardware, weight_dtype=None):
    weight = linear.weight.T
    if isinstance(weight_dtype, CustomDType):
        weight_dtype = weight_dtype
    else:
        weight_dtype = weight.dtype
    batch_dim = input_shape[:-1]
    output_shape = batch_dim + [linear.out_features]
    op = create_op_info(
			input_list=[input_shape, weight.shape],
			input_dtype_list=[input_dtype, weight_dtype],
			output_list=[output_shape],
			output_dtype_list=[input_dtype]
		)
    prediction = MatmulPrediction(op, hardware)
    return prediction, output_shape, input_dtype