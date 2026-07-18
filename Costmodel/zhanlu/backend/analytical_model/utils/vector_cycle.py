import math
from enum import Enum, IntEnum
from typing import Dict, List, Tuple, Union, Callable
import json
from zhanlu.backend.analytical_model.hardware.hardware_config_loader import load_vector_config, load_vector_OPs


class InstrPerf:
    def __init__(self,
                 chip_name: str = "A3"):
        self.param_map = load_vector_config(chip_name)
        self.base_OPs = load_vector_OPs(chip_name)
        self.init_core_num()

    def init_core_num(self):
        baseline_ops = self.get_baseline_OPs("float16")
        frequency = self.get_frequency()
        baseline_throughput = self.get_baseline_throughput("float16")
        self.num_cores = math.ceil(baseline_ops / (frequency * baseline_throughput))

    def get_latency(self, instruction_type: str, data_type: str) -> float:
        """获取指定数据类型的延迟"""
        return self.param_map["instruction"][instruction_type][data_type]["LATENCY"]

    def get_throughput(self, instruction_type: str, data_type: str) -> float:
        """获取指定数据类型的吞吐量"""
        return self.param_map["instruction"][instruction_type][data_type]["PEAK_THROUGHPUT"]

    def get_parallel(self, instruction_type: str, data_type: str) -> int:
        """获取指定数据类型的并行度"""
        return self.param_map["instruction"][instruction_type][data_type]["PARALLEL"]

    def get_baseline_throughput(self, data_type: str):
        return self.param_map["baseline"][data_type]

    def get_baseline_OPs(self, data_type: str):
        vector_OPs = self.base_OPs[data_type]
        return vector_OPs * 1e12

    def get_instruction_ops(self, instruction_type: str, data_type: str):
        instrution_throughput = self.get_throughput(instruction_type, data_type)
        baseline_throughput = self.get_baseline_throughput(data_type)
        baseline_ops = self.get_baseline_OPs(data_type)
        instruction_ops = baseline_ops / baseline_throughput * instrution_throughput
        return instruction_ops

    def get_frequency(self):
        return self.param_map["frequency"] * 1e9

    def calculate_execution_ops(self, instruction_type: str, data_type: str, m: int, n: int) -> float:
        parallel = self.get_parallel(instruction_type, data_type)
        if n < parallel:
            n = parallel

        data_volume = m * n
        return  data_volume

    def calculate_execution_time(self, instruction_type: str, data_type: str, m: int, n: int, cube_k: int = 128) -> int:
        """
        计算执行时间

        参数:
            instruction_type: 指令类型
            data_type: 数据类型
            m: 矩阵行数
            n: 矩阵列数

        返回:
            执行时间(us)
        """
        # 基本计算公式: latency + 数据量/吞吐量
        latency = self.get_latency(instruction_type, data_type)
        ops = self.get_instruction_ops(instruction_type, data_type)  # TOps/s
        parallel = self.get_parallel(instruction_type, data_type)
        frequency = self.get_frequency()

        data_volume = self.calculate_execution_ops(instruction_type, data_type, m, n)

        compute_time = data_volume / ops # s

        latency_time = latency / frequency    # s
        total_latency = compute_time + latency_time
        return total_latency * 1e6     # us


if __name__ == '__main__':
    instruction_perf_model = InstrPerf()
    print(instruction_perf_model.calculate_execution_time("VADD", "float32", 40*16, 512))

    print(instruction_perf_model.calculate_execution_time("VEXP", "float32", 40*16, 512))
