from dataclasses import dataclass, field


@dataclass
class ZhanluPerfResult:
    inputs_shape: list = field(default_factory=list)
    inputs_dtype: list = field(default_factory=list)
    parameters_input: list = field(default_factory=list)
    outputs_shape: list = field(default_factory=list)
    outputs_dtype: list = field(default_factory=list)
    cube_flops: dict = field(default_factory=dict)  # {Dtype: Flops}
    total_cube_flops: int = 0  # Flops
    cube_time: dict = field(default_factory=dict)  # {Dtype: Time(USec)}
    total_cube_time: float = 0  # USec
    cube_ratio: float = 0
    vector_flops: dict = field(default_factory=dict)  # {Dtype: Flops}
    total_vector_flops: int = 0  # Flops
    vector_time: dict = field(default_factory=dict)  # {Dtype: Time(USec)}
    total_vector_time: float = 0  # USec
    vector_ratio: float = 0
    memory_access: int = 0  # KBytes !! 0709 modified for
    memory_access_time: float = 0  # USec
    memory_ratio: float = 0
    communication_access: int = 0  # Bytes
    communication_time: float = 0  # USec
    bandwidth_utilization: tuple = (0, 0) # bandwidth
    head_tail_time: float = 0  # USec
    op_time: float = 0  # USec
    module_path: str = ""
    global_rank_list: list = field(default_factory=list)
    graph_name: str = ""
