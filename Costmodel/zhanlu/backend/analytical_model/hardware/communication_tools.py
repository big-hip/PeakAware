from abc import ABC, abstractmethod
from typing import Type, Optional, Dict
from collections import Counter
import math
from collections import Counter

class HardwareSpec(ABC):
    """Abstract base class for hardware specs."""

    # These will be overridden in subclasses
    NUM_DV_PER_NODE = 1
    MAX_FM_CONN_PER_DV = 1
    NUM_UB_PORTS_BTW_NODE = 1
    SINGLE_FM_BW = 100
    SINGLE_UB_BW = 100
    SINGLE_ROCE_BW = 100
    BW_UTILIZATION_RATE: float = 0.5
    LEVEL_ONE_TOPO_TYPE = ""
    LEVEL_TWO_TOPO_TYPE = ""
    COVER_RATE: 1
    RTT=[[math.inf, 0]]
    LATENCY_HIERARCHICAL_ALLTOALL = 1
    LATENCY_ALLTOALL = 0
    DYNAMIC_RTT = [[8, 1], [64, 5], [math.inf, 8]]
    HBM_BANDWIDTH = 4

    def _get_utilization_rate(self, utilization_rate):
        if utilization_rate:
            if not isinstance(utilization_rate, float) or utilization_rate <= 0 or utilization_rate > 1:
                raise ValueError("utilization_rate must be a float between 0 and 1")
        else:
            utilization_rate = self.__class__.BW_UTILIZATION_RATE
        return utilization_rate

    def get_intra_node_bandwidth(self, num_full_mesh_links, utilization_rate=None):
        if num_full_mesh_links == 0:
            return 0
        utilization_rate = self._get_utilization_rate(utilization_rate)
        if self.__class__.LEVEL_ONE_TOPO_TYPE == "fullmesh":
            port_num = min(self.__class__.MAX_FM_CONN_PER_DV, num_full_mesh_links)
        else:
            port_num = self.__class__.MAX_FM_CONN_PER_DV
        return self.__class__.SINGLE_FM_BW * port_num * utilization_rate

    def get_among_node_bandwidth(self, num_ub_ports=None, utilization_rate=None):
        utilization_rate = self._get_utilization_rate(utilization_rate)
        num_ub_ports = self.__class__.NUM_UB_PORTS_BTW_NODE if not num_ub_ports else num_ub_ports
        if num_ub_ports == 0:
            return 0
        return self.__class__.SINGLE_UB_BW * num_ub_ports * utilization_rate

    def comm_cover(self, x_comm_time, y_comm_time):
        return min(x_comm_time, y_comm_time) * (1 - self.__class__.COVER_RATE) + max(x_comm_time, y_comm_time)

    def get_latency(self, num_node):
        rtt = 0
        sorted_RTT = sorted(self.RTT, key=lambda s: s[0])
        for threshold, value in sorted_RTT:
            if num_node * self.NUM_DV_PER_NODE <= threshold:
                rtt += value
                break

        sorted_dynamic_RTT = sorted(self.DYNAMIC_RTT, key=lambda s: s[0])
        for threshold, value in sorted_dynamic_RTT:
            if num_node * self.NUM_DV_PER_NODE <= threshold:
                rtt += value
                break

        return rtt

    def get_num_node_num_lcoal_rank(self, global_rank_list):
        node_ids = [global_rank // self.NUM_DV_PER_NODE for global_rank in global_rank_list]
        node_rank_count = Counter(node_ids)
        num_node = len(node_rank_count)
        num_lcoal_rank = list(node_rank_count.values())
        # 判断每个node内的local_rank数目是否相等
        assert len(set(num_lcoal_rank)) == 1, f"当前不支持通信域中各node内rank数目不同: {num_lcoal_rank}"
        return num_node, num_lcoal_rank[0]

    def get_hbm_bandwidth(self, hbm_bandwidth=None):
        if hbm_bandwidth:
            if hbm_bandwidth <= 0:
                raise ValueError("hbm_bandwidth must uppper 0")
        else:
            hbm_bandwidth = self.__class__.HBM_BANDWIDTH
        return hbm_bandwidth

    @classmethod
    def summary(cls) -> str:
        return (
            f"{cls.NAME}:\n"
            f"  CUBE_SPEC: {cls.CUBE_SPEC} TFLOPS\n"
            f"  VECTOR_SPEC: {cls.VECTOR_SPEC} TFLOPS\n"
            f"  HBM_BW: {cls.HBM_BW} TB/s"
        )

class Server(HardwareSpec):
    TOPOLOGY_TYPE = "Server"
    NUM_DV_PER_NODE = 8
    MAX_FM_CONN_PER_DV = 7
    NUM_UB_PORTS_BTW_NODE = 4
    COVER_RATE = 1
    RTT = [[8, 0.85], [64, 4.59], [math.inf, 7.19]]

    def get_in_node_x_y_bandwidth(self, world_size, **kwargs):
        # 如果local rank数<= 1，则机内通信带宽非法，当前给1
        if world_size <= 1:
            return 1, 0
        num_full_mesh_links = min(world_size, self.__class__.NUM_DV_PER_NODE) - 1
        return self.get_intra_node_bandwidth(num_full_mesh_links), 0


def _update_platform_variables(hardware_class, platform_dict):
    for key, value in platform_dict.items():
        if hasattr(hardware_class, key):
            setattr(hardware_class, key, value)
        else:
            print(f"Warning: BaseModule has no attribute '{key}'")

class HardwareRegistry:
    _instance = None
    _topology_registry: Dict[str, Type[HardwareSpec]] = {}
    _default: Type[HardwareSpec] = None
    _default_instance: Optional[HardwareSpec] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_registry()
        return cls._instance

    def _init_registry(self):
        self._topology_registry = {
            "Server": Server,
        }
        self._default = Server  # Optional: set a startup default
        self._default_instance = self._default()

    def get(self, name: str) -> Type[HardwareSpec]:
        if name not in self._topology_registry:
            raise ValueError(f"Hardware spec '{name}' not found in registry. Avaialable: {list(self._topology_registry.keys())}")
        return self._topology_registry[name]

    def all_specs(self) -> Dict[str, Type[HardwareSpec]]:
        return dict(self._topology_registry)

    def set(self, platform_name: str, topology_name: str):
        """Set default hardware by name."""
        self._default = self.get(topology_name)
        self._default_instance = self._default()

    def get_default(self) -> Optional[HardwareSpec]:
        """Get the default hardware spec class."""
        return self._default_instance

    def get_by_config(self, name:str = "Server", topo_config:dict = None, chip_config:dict = None) -> HardwareSpec:
        """Get the hardware spec class & modify class variable"""
        c = self._topology_registry[name]
        if topo_config:

            c.BW_UTILIZATION_RATE = topo_config["utilization"]
            c.SINGLE_FM_BW = topo_config["level1"]["bandwidth_per_port"] * (1024 ** 3) * 1e-6
            c.MAX_FM_CONN_PER_DV = topo_config["level1"]["port_num"]
            c.SINGLE_UB_BW = topo_config["level2"]["bandwidth_per_port"] * (1024 ** 3) * 1e-6
            c.NUM_UB_PORTS_BTW_NODE = topo_config["level2"]["port_num"]
            c.RTT = [[topo_config["level1"]["device_num"], topo_config["level1"]["latency"]],
                     [math.inf, topo_config["level2"]["latency"]],]
            c.DYNAMIC_RTT = [[topo_config["level1"]["device_num"], topo_config["level1"]["dynamic_latency"]],
                             [topo_config["level2"]["device_num"], topo_config["level2"]["dynamic_latency"]],
                            [math.inf, topo_config["level3"]["dynamic_latency"]],]
            c.LEVEL_ONE_TOPO_TYPE = topo_config["level1"]["topo"]
            c.LEVEL_TWO_TOPO_TYPE = topo_config["level2"]["topo"]
            c.HBM_BANDWIDTH = chip_config["memory"]["hbm"]["bandwidth"] * (1024 ** 4) * 1e-6
            c.NUM_DV_PER_NODE = topo_config["level1"]["device_num"]
        return c()

registry = HardwareRegistry()

if __name__ == '__main__':
    hardware = registry.get_default()
    print(hardware.get_in_node_x_y_bandwidth(16))