from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager

from atencost.backend.analytical_model.utils.datatype import get_dtype_size
from atencost.backend.analytical_model.utils.helper import broadcast_shapes, create_op_info
from math import prod


@op_manager.register("group_matmul", "GroupMatmul")
class GroupMatmulPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        Args
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]

        output: [..., hidden2]
        """
        super().__init__(op, hardware)
        op = self.op
        self.op_type = "cube"
        if len(self.inputs_shape) != 2:
            raise ValueError(f"GroupMatmul op should have 2 inputs, but got {len(self.inputs_shape)}")

    @property
    def cube_flops_dict(self):

        shape1, shape2 = self.inputs_shape
        dtype1, dtype2 = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        if len(shape1) < 2 or len(shape2) < 2:
            raise ValueError(f"GroupMatmul supports only tensors with at least 2 dimensions: {shape1} and {shape2}")

        if shape1[-1] != shape2[-2]:
            raise ValueError(f"GroupMatmul inputs must have compatible dimensions: {shape1} and {shape2}")

        flops = 2 * prod(shape1) * shape2[-1]
        return {self.compute_type: flops}

@op_manager.register("Groupedmatmul")
class NpuGmmPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        Args
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]

        output: [..., hidden2]
        """
        super().__init__(op, hardware)
        op = self.op
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):

        shape1, shape2, *_ = self.inputs_shape
        dtype1, dtype2, *_ = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        out_shape1 = self.outputs_shape[0]
        M, N = out_shape1
        K = shape1[-1]

        flops = 2 * M * K * N
        return {self.compute_type: flops}

@op_manager.register("Gmmaddkernel")
class GmmaddkernelPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        Args
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]

        output: [..., hidden2]
        """
        super().__init__(op, hardware)
        op = self.op
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):

        shape1, shape2, *_ = self.inputs_shape
        dtype1, dtype2, *_ = self.inputs_dtype
        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        M = shape1[-1]
        N = shape2[-1]
        K_in_1 = shape1[0]
        K_in_2 = shape2[0]

        flops = 2 * M * K_in_1 * N

        return {self.compute_type: flops}

# TODO 这样写两个反向的访存会重复计算
@op_manager.register("GroupMatmulWeightBackward")
class GroupMatmulGradInputPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            output_grad: [..., hidden2]
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]
        output
            input_grad: [..., hidden1]
        """
        super().__init__(op, hardware)
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):
        grad_output_shape, input_shape, weight_shape = self.inputs_shape
        _, dtype1, dtype2 = self.inputs_dtype

        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        group_num, hidden1, hidden2 = weight_shape
        batch_total = prod(grad_output_shape[:-1])

        flops = 2 * batch_total * hidden1 * hidden2

        return {self.compute_type: flops}


@op_manager.register("GroupMatmulActivationBackward")
class GroupMatmulGradWeightPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        input
            output_grad: [..., hidden2]
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]
        output
            weight_grad: [group_num, hidden1, hidden2]
        """
        super().__init__(op, hardware)
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):
        grad_output_shape, input_shape, weight_shape = self.inputs_shape
        _, dtype1, dtype2 = self.inputs_dtype

        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        group_num, hidden1, hidden2 = weight_shape
        batch_total = prod(grad_output_shape[:-1])

        flops = 2 * batch_total * hidden1 * hidden2

        return {self.compute_type: flops}


def get_prediction_by_gmm(input_shape, input_dtype, group_num, out_features, weight_dtype, hardware):
    in_features = input_shape[-1]
    weight_shape = [group_num, in_features, out_features]
    batch_dim = input_shape[:-1]
    output_shape = batch_dim + [out_features]
    op = create_op_info(
        input_list=[input_shape, weight_shape],
        input_dtype_list=[input_dtype, weight_dtype],
        output_list=[output_shape],
        output_dtype_list=[input_dtype],
    )
    prediction = GroupMatmulPrediction(op, hardware)
    return prediction, output_shape, input_dtype


@op_manager.register("gmm", "Gmm")
class GMMPrediction(BaseOp):
    def __init__(self, op, hardware):
        """
        Args
            input: [..., hidden1]
            weight: [group_num, hidden1, hidden2]

        output: [..., hidden2]
        """
        super().__init__(op, hardware)
        self.op_type = "cube"

    @property
    def cube_flops_dict(self):
        if len(self.inputs_shape) == 4:
            _, shape1, shape2, _ = self.inputs_shape
            self.compute_type = self.inputs_dtype[0]

            k, m = shape1
            n = shape2[-1]
            flops = 2 * m * k * n
            return {self.compute_type: flops}
        elif len(self.inputs_shape) == 3:
            # TODO: need to verify correctness
            shape1, shape2, _ = self.inputs_shape
            self.compute_type = self.inputs_dtype[0]

            k, m = shape1
            n = shape2[-1]
            flops = 2 * m * k * n
            return {self.compute_type: flops}
        else:
            shape1, shape2 = self.inputs_shape
            dtype1, dtype2 = self.inputs_dtype

        dtype_val1 = get_dtype_size(dtype1)
        dtype_val2 = get_dtype_size(dtype2)
        self.compute_type = dtype1 if dtype_val1 > dtype_val2 else dtype2

        if len(shape1) < 2 or len(shape2) < 2:
            raise ValueError(f"GroupMatmul supports only tensors with at least 2 dimensions: {shape1} and {shape2}")

        if shape1[-1] != shape2[-2]:
            raise ValueError(f"GroupMatmul inputs must have compatible dimensions: {shape1} and {shape2}")

        flops = 2 * prod(shape1) * shape2[-1]
        return {self.compute_type: flops}
