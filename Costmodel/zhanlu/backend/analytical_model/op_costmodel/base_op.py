from zhanlu.frontend.utils.tensor_record import TensorRecord
from zhanlu.backend.analytical_model.hardware.hardware_config_loader import (
    load_hardware_config,
)
from zhanlu.backend.analytical_model.utils.utilization_loader import UtilizationLoader
from zhanlu.backend.analytical_model.utils.model_utils import (
    calculate_total_bytes,
    get_tasks_time,
    get_compute_type_flops,
    get_dtype_flops,
    TBps2Bpus,
    TFlopS2FlopUs,
)
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.backend.analytical_model.utils.vector_cycle import InstrPerf


class BaseOp:
    def __init__(self, op, hardware, chip_name="A3", topo_name="A6_2d_fullmesh"):
        self.op = op

        # 提取所有TensorRecord类型的输入元素
        self.tensor_inputs = [
            inp for inp in getattr(op, "inputs", []) if isinstance(inp, TensorRecord)
        ]

        self.parameter_inputs = [
            str(inp) for inp in getattr(op, "inputs", []) if not isinstance(inp, TensorRecord)
        ]
        # 从提取的TensorRecord元素中获取形状和数据类型
        self.inputs_shape = [inp.local_shape for inp in self.tensor_inputs]
        self.inputs_dtype = [str(inp.dtype) for inp in self.tensor_inputs]

        # 对输出做同样的处理
        self.tensor_outputs = [
            out for out in getattr(op, "outputs", []) if isinstance(out, TensorRecord)
        ]

        self.outputs_shape = [out.local_shape for out in self.tensor_outputs]
        self.outputs_dtype = [str(out.dtype) for out in self.tensor_outputs]

        self.result = ZhanluPerfResult(inputs_shape=self.inputs_shape,
                                       inputs_dtype=self.inputs_dtype,
                                       parameters_input=self.parameter_inputs,
                                       outputs_shape=self.outputs_shape,
                                       outputs_dtype=self.outputs_dtype,
                                       module_path=self.op.module_path,
                                       global_rank_list=self.op.instance.global_rank_list if hasattr(self.op.instance, 'global_rank_list') else [],
                                       graph_name=self.op.graph_name)
        # 硬件配置
        self.hardware = hardware
        chip_name, topo_name = hardware.split(',')
        self.hardware_config = load_hardware_config(chip_name, topo_name)
        self.util_loader = UtilizationLoader(chip_name)

        self.op_type = "cube"
        self.memory_bandwidth = self.hardware_config["chip_config"]["memory"]["hbm"][
            "bandwidth"
        ]
        self.head_tail_time = self.hardware_config["chip_config"]["head_tail_time"]
        self.vec_instruction = []
        # self.instruction_perf_model = InstrPerf(chip_name)

    def compute_power_by_inputs_dtype(self):
        if self.op_type in ["cube", "vector"]:
            return {
                self.op_type: get_compute_type_flops(
                    self.op_type, self.inputs_dtype, self.hardware_config
                )
            }
        elif self.op_type in ["mix"]:
            return {
                "cube": get_compute_type_flops(
                    "cube", self.inputs_dtype, self.hardware_config
                ),
                "vector": get_compute_type_flops(
                    "vector", self.inputs_dtype, self.hardware_config
                ),
            }
        else:
            # todo: coc算子的计算类型判断
            return {}

    @property
    def cube_flops_dict(self):
        return {}

    @property
    def total_cube_flops(self):
        total_flops = 0
        for dtype, flops in self.cube_flops_dict.items():
            total_flops += flops # TODO 是否需要统一折算到相同精度 需要再讨论下
        return total_flops

    @property
    def cube_time_dict(self):
        result = dict()
        for data_type, FLOPs in self.cube_flops_dict.items():
            TFLOPS = get_dtype_flops(
                data_type,
                hardware_config=self.hardware_config["chip_config"]["compute"]["cube"],
            )
            cube_utilization = self.util_loader.infer_ratio(FLOPs, "cube", data_type)
            result[data_type] = FLOPs / (TFLOPS * cube_utilization) / TFlopS2FlopUs
        return result

    @property
    def cube_time(self):
        return get_tasks_time(self.cube_time_dict, overlap=0)

    @property
    def cube_ratio(self):
        # cube_ratio = theoretical_time / cube_time
        theoretical_time = 0
        cube_time = self.cube_time
        if cube_time == 0:
            return 0
        for data_type, FLOPs in self.cube_flops_dict.items():
            TFLOPS = get_dtype_flops(
                data_type,
                hardware_config=self.hardware_config["chip_config"]["compute"]["cube"],
            )
            theoretical_time += FLOPs / (TFLOPS) / TFlopS2FlopUs
        return theoretical_time / cube_time


    @property
    def vec_flops_dict(self):
        vector_dtype_flops = dict()
        for instruction_info in self.vec_instruction:
            data_type = instruction_info[1]
            vector_dtype_flops[data_type] = vector_dtype_flops.get(data_type, 0) + self.instruction_perf_model.calculate_execution_ops(*instruction_info)
        return vector_dtype_flops

    @property
    def total_vec_flops(self):
        total_flops = 0
        for dtype, flops in self.vec_flops_dict.items():
            total_flops += flops # TODO 是否需要统一折算到相同精度 需要再讨论下
        return total_flops

    @property
    def vec_time_dict(self):
        result = dict()
        if self.vec_instruction:
            for instruction_info in self.vec_instruction:
                data_type = instruction_info[1]
                result[data_type] = result.get(data_type, 0) + self.instruction_perf_model.calculate_execution_time(*instruction_info)
            return result

        for data_type, FLOPs in self.vec_flops_dict.items():
            TFLOPS = get_dtype_flops(
                data_type,
                hardware_config=self.hardware_config["chip_config"]["compute"][
                    "vector"
                ],
            )
            vector_utilization = self.util_loader.infer_ratio(FLOPs, "vector", data_type)
            # vector_utilization = 0.16
            result[data_type] = FLOPs / (TFLOPS * vector_utilization) / TFlopS2FlopUs
        return result

    @property
    def vec_time(self):
        return get_tasks_time(self.vec_time_dict, overlap=0)

    @property
    def vec_ratio(self):
        # vec_ratio = theoretical_time / vec_time
        theoretical_time = 0
        vec_time = self.vec_time
        if vec_time == 0:
            return 0
        for data_type, FLOPs in self.vec_flops_dict.items():
            TFLOPS = get_dtype_flops(
                data_type,
                hardware_config=self.hardware_config["chip_config"]["compute"][
                    "vector"
                ],
            )
            theoretical_time += FLOPs / (TFLOPS) / TFlopS2FlopUs
        return theoretical_time / vec_time

    @property
    def input_memory_size(self):
        return calculate_total_bytes(self.inputs_shape, self.inputs_dtype)

    @property
    def output_memory_size(self):
        return calculate_total_bytes(self.outputs_shape, self.outputs_dtype)

    @property
    def memory_size(self):
        return self.input_memory_size + self.output_memory_size

    @property
    def memory_time(self):
        memory_utilization = self.util_loader.infer_ratio(self.memory_size, "hbm")
        return (
                self.memory_size / (self.memory_bandwidth * memory_utilization) / TBps2Bpus
        )  # 返回us

    @property
    def memory_ratio(self):
        # memory_ratio = theoretical_time / memory_time
        theoretical_time = self.memory_size / self.memory_bandwidth / TBps2Bpus
        memory_time = self.memory_time
        if memory_time == 0:
            return 0
        return theoretical_time / memory_time

    @property
    def communication_size(self):
        return 0

    @property
    def communication_time(self):
        return 0

    # @property
    # def op_time(self):
    #     return get_tasks_time(
    #         [self.vec_time, self.cube_time, self.memory_time, self.communication_time], overlap=1
    #     ) + self.head_tail_time

    @property
    def op_time(self):
        return max(self.vec_time + self.cube_time, self.memory_time, self.communication_time) + self.head_tail_time

    @property
    def bandwidth_utilization(self):
        if hasattr(self, 'hardware_spec'):
            intra_node_bandwidth, inter_node_bandwidth = 0, 0
            latency = self.hardware_spec.get_latency(self.num_nodes)
            in_node_comm_bytes = self.intra_node_communication_size
            in_node_comm_time = in_node_comm_bytes / self.hardware_spec.get_in_node_x_y_bandwidth(
                world_size=self.num_local_rank)[0] + latency
            if self.num_local_rank <= 1:
                intra_node_bandwidth = 0
            else:
                intra_node_bandwidth = in_node_comm_bytes / (self.num_local_rank - 1) / (1024 ** 3) / (in_node_comm_time * 1e-6)
            if self.num_nodes > 1:
                btw_node_comm_bytes = self.inter_node_communication_size
                btw_node_comm_time = btw_node_comm_bytes / self.hardware_spec.get_among_node_bandwidth() + latency
                inter_node_bandwidth = btw_node_comm_bytes / (self.num_nodes - 1) / (1024 ** 3) / (btw_node_comm_time * 1e-6)
            return intra_node_bandwidth, inter_node_bandwidth
        else:
            return 0, 0

    def __call__(self, *args, **kwargs):
        self.result.cube_flops = self.cube_flops_dict
        self.result.total_cube_flops = self.total_cube_flops
        self.result.cube_time = self.cube_time_dict
        self.result.total_cube_time = self.cube_time
        self.result.cube_ratio = self.cube_ratio
        self.result.vector_flops = self.vec_flops_dict
        self.result.total_vector_flops = self.total_vec_flops
        self.result.vector_time = self.vec_time_dict
        self.result.total_vector_time = self.vec_time
        self.result.vector_ratio = self.vec_ratio
        self.result.memory_access = self.memory_size
        self.result.memory_access_time = self.memory_time
        self.result.memory_ratio = self.memory_ratio
        self.result.communication_access = self.communication_size
        self.result.communication_time =self.communication_time
        self.result.op_time = self.op_time
        self.result.head_tail_time = self.head_tail_time
        self.result.bandwidth_utilization = self.bandwidth_utilization
        return self.result
