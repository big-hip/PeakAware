from math import prod
from atencost.frontend.utils.tensor_record import CustomDType
from atencost.backend.analytical_model.utils.datatype import get_dtype_size

dtype_to_bytes = {
    'torch.int4': 0.5,
    'torch.uint8': 1,
    'torch.int8': 1,
    'torch.int16': 2,
    'torch.short': 2,
    'torch.int32': 4,
    'torch.int': 4,
    'torch.int64': 8,
    'torch.long': 8,
    'torch.float16': 2,
    'torch.half': 2,
    'torch.float32': 4,
    'torch.float': 4,
    'torch.float64': 8,
    'torch.double': 8,
    'torch.complex64': 8,  # 两个float32
    'torch.cfloat': 8,
    'torch.complex128': 16,  # 两个float64
    'torch.cdouble': 16,
    'torch.bool': 1,
    'torch.quint8': 1,
    'torch.qint8': 1,
    'torch.qint32': 4,
}

TBps2Bpus = (1024 ** 4) / (1000 ** 2)
TFlopS2FlopUs = (1000 ** 4) / (1000 ** 2)


def calculate_total_bytes(shapes, dtypes):
    total_bytes = 0
    for dtype, shape in zip(dtypes, shapes):
        total_bytes += prod(shape) * get_dtype_size(dtype)
    return total_bytes


def get_tasks_time(tasks, overlap=0):
    if not tasks:
        return 0

    times = list(tasks.values()) if isinstance(tasks, dict) else tasks
    if len(times) == 1:
        return times[0]

    total_sum = sum(times)
    max_time = max(times)

    # 计算插值：总时间 = 最大值 + (1 - overlap) * (总和 - 最大值)
    total_time = max_time + (1 - overlap) * (total_sum - max_time)
    return total_time


def get_dtype_flops(dtype, hardware_config):
    # TFLOPS
    if isinstance(dtype, CustomDType):
        custom_dtype = dtype.dtype
        # custom_dtype = CustomDType.dtype
        if custom_dtype == 'NvFP4':
            return hardware_config['float4']
        return dtype.ele_size
    if 'NvFP4' in str(dtype):
        return hardware_config['float4']
    if 'float8' in str(dtype):
        return hardware_config['float8']
    dtype_name = str(dtype).split('.')[-1]
    if dtype_name in hardware_config.keys():
        return hardware_config[dtype_name]

    dtype_size = get_dtype_size(dtype)
    return hardware_config["float16"] * 2 / dtype_size


def get_compute_type_flops(unit, dtype_list, hardware_config):
    return min(
        [(dtype, get_dtype_flops(dtype, hardware_config["chip_config"]["compute"][unit])) for dtype in dtype_list],
        key=lambda x: x[1])
