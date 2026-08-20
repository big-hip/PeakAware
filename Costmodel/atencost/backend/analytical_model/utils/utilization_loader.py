from functools import wraps
from math import log
import torch
from .datatype import get_dtype_size
import warnings

ratio_curve = {
    # Ascend 910B2C: same Atlast-910B architecture (cube fp16=376 TF,
    # HBM 1.6 TB/s, 64 GB). Utilization curves below until a
    # dedicated on-chip calibration is collected; infer_*_ratio currently
    # returns fixed priors (0.1 / 0.8 / 0.1) identical across configs.
    'Ascend910B': {
        'cube': {
            'a': 0.15,
            'b': -2.82,
            'max_ratio': 0.75,
            'min_ratio': 0.1532444728230505
        },
        'hbm': {
            'a': 0.07,
            'b': -0.68,
            'max_ratio': 0.6998730030736797,
            'min_ratio': 0.27669700084207
        }
    },
}


def get_ratio_log_linear(a, b, max_ratio, min_ratio, flops):
    if flops <= 0:
        return 1
    ratio_predicted = a * log(flops) + b
    if ratio_predicted > max_ratio:
        return max_ratio
    if ratio_predicted < min_ratio:
        return min_ratio
    return ratio_predicted


def get_cube_ratio_log_linear(flops, hardware='Ascend910B'):
    weight = ratio_curve[hardware]['cube']
    a = weight['a']  # 0.15
    b = weight['b']  # -2.82
    max_ratio = weight['max_ratio']  # 0.9271673085567015
    min_ratio = weight['min_ratio']  # 0.1532444728230505
    return get_ratio_log_linear(a, b, max_ratio, min_ratio, flops)


def get_hbm_ratio_log_linear(flops, hardware='Ascend910B'):
    weight = ratio_curve[hardware]['hbm']
    a = weight['a']  # 0.07
    b = weight['b']  # -0.68
    max_ratio = weight['max_ratio']  # 0.6998730030736797
    min_ratio = weight['min_ratio']  # 0.27669700084207
    # print(f'{get_ratio_log_linear(a, b, max_ratio, min_ratio, flops)=}')
    return get_ratio_log_linear(a, b, max_ratio, min_ratio, flops)


class UtilizationLoader:
    _instances = {}  # 类变量存储不同硬件配置的单例实例
    DEFAULT_HARDWARE = 'Ascend910B'  # 默认硬件配置

    def __new__(cls, hardware='Ascend910B'):
        # 1. 验证硬件配置
        validated_hardware = hardware
        if hardware not in ratio_curve:
            # 2. 生成警告信息
            # warning_msg = (
            #     f"硬件 '{hardware}' 没有利用率建模方案。"
            #     f"使用默认配置 '{cls.DEFAULT_HARDWARE}' 替代。"
            #     f"可用配置: {list(ratio_curve.keys())}"
            # )
            # warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

            # 3. 回退到默认配置
            validated_hardware = cls.DEFAULT_HARDWARE

        # 4. 检查是否已有该硬件的实例
        if validated_hardware not in cls._instances:
            # 5. 创建新实例
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instances[validated_hardware] = instance
        # 6. 返回实例
        return cls._instances[validated_hardware]

    def __init__(self, hardware=DEFAULT_HARDWARE):
        if not self._initialized:
            # 存储实际使用的硬件配置（可能是回退后的）
            self.hardware = hardware
            # 存储验证后的硬件配置
            self.validated_hardware = (
                hardware if hardware in ratio_curve
                else self.DEFAULT_HARDWARE
            )
            self._initialized = True

    def infer_ratio(self, flops, unit, dtype='torch.float8_e4m3fn'):
        dtype_val = get_dtype_size(dtype)
        actual_flops = flops * dtype_val

        if unit == 'cube':
            return self.infer_cube_ratio(actual_flops)
        if unit == 'hbm':
            return self.infer_hbm_ratio(actual_flops)
        if unit == 'vector':
            return self.infer_vector_ratio(actual_flops)

    def infer_cube_ratio(self, flops):
        return 0.1
        # return get_cube_ratio_log_linear(flops, self.validated_hardware)

    def infer_hbm_ratio(self, flops):
        return 0.8
        # return get_hbm_ratio_log_linear(flops, self.validated_hardware)

    def infer_vector_ratio(self, flops):
        return 0.1
        # return 0.6  # Placeholder value, can be adjusted based on actual hardware characteristics


if __name__ == "__main__":
    # Example usage
    util_loader = UtilizationLoader()

    # flops = 10000000 # Example FLOPS value
    flops = 2 * 256 * 7168 * 1536
    dtype = torch.float8_e4m3fn

    cube_ratio = util_loader.infer_ratio(flops, 'cube', dtype)
    hbm_ratio = util_loader.infer_ratio(flops, 'hbm', dtype)
    vector_ratio = util_loader.infer_ratio(flops, 'vector', dtype)

    print(f"Cube Ratio: {cube_ratio}")
    print(f"HBM Ratio: {hbm_ratio}")
    print(f"Vector Ratio: {vector_ratio}")
