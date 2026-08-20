from math import prod
from zhanlu.backend.analytical_model.op_costmodel.base_op import BaseOp
from zhanlu.backend.analytical_model.op_manager import op_manager
from zhanlu.backend.perf_result import ZhanluPerfResult


@op_manager.register("SigmoidBackward", "LogSoftmax", "RoPE", "Rope", "NpuRotaryPositionEmbedding")
class Element3Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 3
        return {compute_dtype: FLOPs}


@op_manager.register("Sigmoid", "LogSoftmaxBackwardData", "RMSNorm", "RmsNorm", "FusedCastAddTopkDivKernel",
                     "NativeGroupNorm")
class Element4Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 4
        return {compute_dtype: FLOPs}


@op_manager.register("Swiglu")
class SwigluPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 5 // 2
        return {compute_dtype: FLOPs}


@op_manager.register("SwigluBackward")
class SwigluBwdPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 5
        return {compute_dtype: FLOPs}


@op_manager.register("Silu", "Softmax", "AddRMSNormCast", "AddRmsNorm")
class Element5Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 5
        return {compute_dtype: FLOPs}


@op_manager.register("VarLenSoftmax")
class VarLenSoftmaxPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
        self.var_len_list = op.inputs[0]

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        # attn_scores_shape: [decode_len, batch_size, num_heads, max_kv_cache_prefill_decode_len]
        attn_score_shape = self.inputs_shape[0]
        decode_len, batch_size, num_heads = attn_score_shape[0], attn_score_shape[1], attn_score_shape[2]
        FLOPs = 0
        for i in range(batch_size):
            FLOPs += decode_len * num_heads * self.var_len_list[i] * 5

        return {compute_dtype: FLOPs}


@op_manager.register("NativeLayerNorm")
class Element7Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 7
        return {compute_dtype: FLOPs}


@op_manager.register("Gelu", "RmsNormBackward")
class Element8Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 8
        return {compute_dtype: FLOPs}


@op_manager.register("SiluBackward")
class Element9Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 9
        return {compute_dtype: FLOPs}


@op_manager.register("GeluBackward", "SwigluInputBackward")
class Element12Prediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape]) * 12
        return {compute_dtype: FLOPs}


@op_manager.register(
    "Mul", "Add", "Sub", "AddInplace", "Div", "Sum", "Relu", "ReluBackward", "DivInplace", "MulInplace", "Topk",
    "NllLossForward", "Mean", "Neg", "Exp", "DtypeCast", "FormatCast", "Tanh", "TanhBackward",
    "ThresholdBackward",
)
class ElementPrediction(BaseOp):
    def __init__(self, op, hardware, chip_name="A3", topo_name="A6_2d_fullmesh"):
        super().__init__(op, hardware, chip_name, topo_name)
        self.op_type = "vector"

    @property
    def vec_flops_dict(self):
        compute_dtype = self.compute_power_by_inputs_dtype()[self.op_type][0]
        FLOPs = max([prod(inp) for inp in self.inputs_shape])
        return {compute_dtype: FLOPs}


@op_manager.register("UpsampleNearest2d")
class UpsampleNearest2dPrediction(BaseOp):
    def __init__(self, op, hardware):
        super().__init__(op, hardware)
