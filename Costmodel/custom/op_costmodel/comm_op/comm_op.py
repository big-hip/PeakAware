from math import prod
from atencost.frontend.utils.tensor_record import CustomDType
from atencost.backend.analytical_model.op_costmodel.base_op import BaseOp
from atencost.backend.analytical_model.op_manager import op_manager
from atencost.backend.perf_result import OpPerfResult
from atencost.backend.analytical_model.utils.model_utils import calculate_total_bytes
from atencost.backend.analytical_model.hardware.communication_tools import registry
from enum import Enum
import math, os


@op_manager.register("AllGatherKernel", "AllgatherInplace", "Py-fake-allgather")
class AllGatherKernel(BaseOp):
    def __init__(self, op, hardware, chip_name="Ascend910B", topo_name="Ascend910B"):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        # self.global_rank_list = [1, 2, 8, 9, 16, 17, 24, 25] # 测试使用
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)
        self.data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
        self.half_data_bytes = self.data_bytes / 2

    @property
    def intra_node_communication_size(self):
        first_data_size = (self.num_local_rank - 1) * self.half_data_bytes # 先x轴
        second_data_size = (self.num_local_rank - 1) * self.num_nodes * self.half_data_bytes # 后x轴
        return first_data_size + second_data_size

    @property
    def inter_node_communication_size(self):
        first_data_size = (self.num_nodes - 1) * self.num_local_rank * self.half_data_bytes # 后y轴
        second_data_size = (self.num_nodes - 1) * self.half_data_bytes # 先y轴
        return first_data_size + second_data_size

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0

        latency = self.hardware_spec.get_latency(self.num_nodes)

        in_node_comm_time = (
            self.intra_node_communication_size
            / self.hardware_spec.get_in_node_x_y_bandwidth(world_size=self.num_local_rank)[0]
        )
        if self.num_nodes > 1:
            btw_node_comm_time = self.inter_node_communication_size / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency
        return in_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time


@op_manager.register("ReduceScatterKernel", "ReduceScatterInplace", "Py-fake-reduceScatter")
class ReduceScatterKernel(BaseOp):
    def __init__(self, op, hardware, chip_name="Ascend910B", topo_name="Ascend910B"):
        """
        :param world_size: communication world size
        :param name: kernel name
        :param device_stride: the stride between two adjacent devices within same communication group in a single node
        """
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        # self.global_rank_list = [1, 2, 8, 9, 16, 17, 24, 25] # 测试使用
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)
        self.data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
        self.half_data_bytes = self.data_bytes / 2

    @property
    def intra_node_communication_size(self):
        first_data_size = (self.num_local_rank - 1) * self.half_data_bytes / self.num_local_rank # 先x轴
        second_data_size = (self.num_local_rank - 1) * self.half_data_bytes / self.num_nodes / self.num_local_rank # 后x轴
        return int(first_data_size + second_data_size)

    @property
    def inter_node_communication_size(self):
        first_data_size = (self.num_nodes - 1) * self.half_data_bytes / self.num_local_rank / self.num_nodes # 后y轴
        second_data_size = (self.num_nodes - 1) * self.half_data_bytes / self.num_nodes # 先y轴
        return int(first_data_size + second_data_size)

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0

        in_node_comm_time = self.intra_node_communication_size / self.hardware_spec.get_in_node_x_y_bandwidth(
            world_size=self.num_local_rank)[0]
        latency = self.hardware_spec.get_latency(self.num_nodes)

        if self.num_nodes > 1:
            btw_node_comm_time = self.inter_node_communication_size / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency

        return in_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time


class AllToAllType(Enum):
    ALL_TO_ALL = 1
    ALL_TO_ALL_TRANSPOSE = 2
    ALL_TO_ALL_DISPATCH = 3
    ALL_TO_ALL_COMBINE = 4


EP_BL = {16: 1.04, 32: 1.05, 64: 1.06, 128: 1.08, 384: 1.1}


# TODO Allredue; All2All; All2All Dispatch; All2All Combine
@op_manager.register("AllToAllKernel", "AlltoallInplace", "Py-fake-alltoall", "Py-fake-alltoallv")
class AllToAllKernel(BaseOp):
    def __init__(self, op, hardware, chip_name="Ascend910B", topo_name="Ascend910B"):
        """
        Pad1D clos all2all通信建模
        :param all_to_all_type: type of all-to-all
        :param top_k: k of top-k in all-to-all in MoE introduced by Expert Parallelism, otherwise default to be 1
        :param name: kernel name
        :param prob_in_node: proportion of communication volume originating from a single device that remains within
        the node
        :param device_stride: the stride between two adjacent devices within same communication group in a single node
        """
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        # self.global_rank_list = [0, 1, 8, 9, 16, 17, 24, 25] # 测试使用
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)

        # 如果没有设置all_to_all_type，使用默认AllToAllType.ALL_TO_ALL
        if hasattr(op.instance, "all_to_all_type"):
            self.all_to_all_type = op.instance.all_to_all_type
        else:
            self.all_to_all_type = AllToAllType.ALL_TO_ALL

        if self.all_to_all_type == AllToAllType.ALL_TO_ALL:
            self.top_k = 1
            self.num_token_max_avg_ratio = 1
            self.alpha_imbalance_ratio = 1
        else:
            self.top_k = op.instance.top_k
            self.num_token_max_avg_ratio = op.instance.num_token_max_avg_ratio
            self.alpha_imbalance_ratio = op.instance.alpha_imbalance_ratio

        self.beta_imbalance_ratio = 1 # y轴不均衡度，当前不考虑，设置为默认1
        # self.all_to_all_type = AllToAllType.ALL_TO_ALL_DISPATCH
        # self.top_k = 8
        # self.num_token_max_avg_ratio = 1.1
        # self.alpha_imbalance_ratio = 1.1
        self.inter_node_member = max(min(self.top_k, self.num_nodes - 1), 1) # 跨节点通信server参与数量 TODO 考虑MoE group限制
        self.input_shape = self.inputs_shape[0]
        self.input_dtype = self.inputs_dtype[0]
        self.hbm_bandwidth = self.hardware_spec.get_hbm_bandwidth()

    def get_data_bytes(self):
        if self.all_to_all_type == AllToAllType.ALL_TO_ALL_COMBINE:
            tokens_per_rank = self.input_shape[0]
            data_bytes = calculate_total_bytes([[int(tokens_per_rank // self.num_token_max_avg_ratio) // self.top_k] + self.input_shape[1:]], [self.input_dtype])
            if 'CustomDType' in self.input_dtype:
                data_bytes *= 2 # TODO 临时规避FP4 combine需要采用FP8
        else:
            data_bytes = calculate_total_bytes([self.input_shape], [self.input_dtype])
        return data_bytes

    @property
    def intra_node_communication_size(self):
        data_bytes = self.get_data_bytes()
        if self.all_to_all_type == AllToAllType.ALL_TO_ALL_DISPATCH or self.all_to_all_type == AllToAllType.ALL_TO_ALL_COMBINE:
            return self.top_k * (self.num_local_rank - 1) * data_bytes / self.num_local_rank * self.alpha_imbalance_ratio
        else:
            return (self.num_local_rank - 1) * data_bytes / self.world_size

    @property
    def inter_node_communication_size(self):
        data_bytes = self.get_data_bytes()
        if self.all_to_all_type == AllToAllType.ALL_TO_ALL_DISPATCH or self.all_to_all_type == AllToAllType.ALL_TO_ALL_COMBINE:
            return data_bytes * self.inter_node_member * self.beta_imbalance_ratio
        else:
            return (self.num_nodes - 1) * self.num_local_rank * data_bytes / self.world_size

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0
        input_bytes = calculate_total_bytes([self.input_shape], [self.input_dtype])
        latency = self.hardware_spec.get_latency(self.num_nodes)
        in_node_comm_bytes = self.intra_node_communication_size
        x_bw, y_bw = self.hardware_spec.get_in_node_x_y_bandwidth(world_size=self.num_local_rank)
        if y_bw != 0:
            in_node_comm_time = self.hardware_spec.comm_cover(in_node_comm_bytes / x_bw, in_node_comm_bytes / y_bw)
        else:
            in_node_comm_time = in_node_comm_bytes / x_bw

        if self.num_nodes > 1:
            btw_node_comm_time = self.inter_node_communication_size / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency
        return in_node_comm_time + latency + input_bytes / self.hbm_bandwidth

    @property
    def op_time(self):
        return self.communication_time


class AllReduceType(Enum):
    ALL_REDUCE_MESH_ONESHOT = 1
    ALL_REDUCE_MESH_TWOSHOT = 2


@op_manager.register("AllReduceKernel", "AllreduceInplace", "Py-fake-allreduce")
class AllReduceKernel(BaseOp):
    def __init__(self, op, hardware, chip_name="Ascend910B", topo_name="Ascend910B"):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)
        self.all_reduce_type = AllReduceType.ALL_REDUCE_MESH_ONESHOT
        self.data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])

    @property
    def intra_node_communication_size(self):
        # 直接一把拉allreduce
        if self.all_reduce_type == AllReduceType.ALL_REDUCE_MESH_ONESHOT:
            communi_size = (self.num_local_rank - 1) * self.data_bytes
        # 拆分成reducescatter+allgather 不管是分层还是不分层都是一样的数据量
        elif self.all_reduce_type == AllReduceType.ALL_REDUCE_MESH_TWOSHOT:
            communi_size = (self.num_local_rank - 1) * math.ceil(self.data_bytes / self.num_local_rank) * 2
        return communi_size

    @property
    def inter_node_communication_size(self):
        # 直接一把拉allreduce
        if self.all_reduce_type == AllReduceType.ALL_REDUCE_MESH_ONESHOT:
            communi_size = (self.num_nodes - 1) * self.data_bytes
        # 拆分成reducescatter+allgather
        elif self.all_reduce_type == AllReduceType.ALL_REDUCE_MESH_TWOSHOT:
            communi_size = (self.num_nodes - 1) * math.ceil(self.data_bytes / self.num_local_rank / self.num_nodes) * 2
        return communi_size

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0

        latency = self.hardware_spec.get_latency(self.num_nodes)
        in_node_comm_bytes = self.intra_node_communication_size
        in_node_comm_time = in_node_comm_bytes / self.hardware_spec.get_in_node_x_y_bandwidth(
            world_size=self.num_local_rank)[0]
        if self.num_nodes > 1:
            btw_node_comm_bytes = self.inter_node_communication_size
            btw_node_comm_time = btw_node_comm_bytes / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency
        return in_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time

@op_manager.register("SendKernel", "Send", "Py-fake-send")
class SendKernel(BaseOp):
    def __init__(self, op, hardware, chip_name='Ascend910B', topo_name='Ascend910B'):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        # self.global_rank_list = [1, 2, 3, 4]
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)

    @property
    def intra_node_communication_size(self):
        if self.num_nodes == 1: # 表明只有节点内的sendrecv
            data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
            communi_size = data_bytes
            return communi_size
        return 0

    @property
    def inter_node_communication_size(self):
        if self.num_nodes > 1: # 表明只有节点间的sendrecv
            data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
            communi_size = data_bytes
            return communi_size
        return 0

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0

        latency = self.hardware_spec.get_latency(self.num_nodes)
        in_node_comm_bytes = self.intra_node_communication_size
        if self.num_local_rank > 1:
            in_node_bandwidth = self.hardware_spec.get_in_node_x_y_bandwidth(world_size=self.num_local_rank)[0]
        else:
            in_node_bandwidth = 1
        in_node_comm_time = in_node_comm_bytes / in_node_bandwidth
        if self.num_nodes > 1:
            btw_node_comm_bytes = self.inter_node_communication_size
            btw_node_comm_time = btw_node_comm_bytes / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency
        return in_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time

@op_manager.register("RecvKernel", "RecvInplace", "Py-fake-recv")
class RecvKernel(BaseOp):
    def __init__(self, op, hardware, chip_name='Ascend910B', topo_name='Ascend910B'):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware_spec = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        if hasattr(op.instance, "global_rank_list"):
            self.global_rank_list = op.instance.global_rank_list
        else:
            self.global_rank_list = list(range(op.instance.world_size))
        self.world_size = len(self.global_rank_list)
        # self.global_rank_list = [1, 8, 16, 24]
        self.num_nodes, self.num_local_rank = self.hardware_spec.get_num_node_num_lcoal_rank(self.global_rank_list)

    @property
    def intra_node_communication_size(self):
        if self.num_nodes == 1: # 表明只有节点内的sendrecv
            data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
            communi_size = data_bytes
            return communi_size
        return 0

    @property
    def inter_node_communication_size(self):
        if self.num_nodes > 1: # 表明只有节点间的sendrecv
            data_bytes = calculate_total_bytes([self.inputs_shape[0]], [self.inputs_dtype[0]])
            communi_size = data_bytes
            return communi_size
        return 0

    @property
    def communication_size(self):
        return self.intra_node_communication_size + self.inter_node_communication_size

    @property
    def communication_time(self):
        if self.world_size <= 1:
            return 0

        latency = self.hardware_spec.get_latency(self.num_nodes)
        in_node_comm_bytes = self.intra_node_communication_size
        if self.num_local_rank > 1:
            in_node_bandwidth = self.hardware_spec.get_in_node_x_y_bandwidth(world_size=self.num_local_rank)[0]
        else:
            in_node_bandwidth = 1
        in_node_comm_time = in_node_comm_bytes / in_node_bandwidth
        if self.num_nodes > 1:
            btw_node_comm_bytes = self.inter_node_communication_size
            btw_node_comm_time = btw_node_comm_bytes / self.hardware_spec.get_among_node_bandwidth()
            return max(in_node_comm_time, btw_node_comm_time) + latency
        return in_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time


@op_manager.register("M2NComKernel")
class M2NComKernel(BaseOp):
    def __init__(self, op, hardware, chip_name='Ascend910B', topo_name='Ascend910B'):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        # self.world_size = op.instance.world_size
        # self.world_size = 16
        # self.in_node_world_size = min(self.hardware.NUM_DV_PER_NODE, self.world_size)
        # if self.world_size > self.hardware.NUM_DV_PER_NODE and self.world_size % self.hardware.NUM_DV_PER_NODE != 0:
        #     raise ValueError("world size must be multiples of node size, otherwise yet to be supported")
        self.num_nodes = math.ceil(op.instance.N / self.hardware.NUM_DV_PER_NODE)
        # if hasattr(op.instance, "global_rank_list"):
        #     self.global_rank_list = op.instance.global_rank_list
        # else:
        #     self.global_rank_list = list(range(op.instance.M))
        # self.num_nodes, self.num_local_rank = self.hardware.get_num_node_num_lcoal_rank(self.global_rank_list)
        # if op.instance.M < 64:
        #     raise ValueError("M must be multiples of 64, otherwise yet to be supported")
        self.received_tokens = self.op.instance.received_tokens

    @property
    def communication_size(self):
        data_bytes = calculate_total_bytes([[self.received_tokens, self.inputs_shape[0][1]]], [self.inputs_dtype[0]])
        return data_bytes

    @property
    def communication_time(self):
        # print(f'received_tokens{self.received_tokens}, self.inputs_shape[0][1]]{self.inputs_shape[0]}')
        data_bytes = calculate_total_bytes([[self.received_tokens, self.inputs_shape[0][1]]], [self.inputs_dtype[0]])

        # print(data_bytes)
        latency = self.hardware.get_latency(self.num_nodes)
        btw_node_comm_time = data_bytes / self.hardware.get_among_node_bandwidth()
        return btw_node_comm_time + latency

    @property
    def op_time(self):
        return self.communication_time


@op_manager.register("N2MComKernel")
class N2MComKernel(BaseOp):
    def __init__(self, op, hardware, chip_name='Ascend910B', topo_name='Ascend910B'):
        super().__init__(op, hardware, chip_name, topo_name)
        self.hardware = registry.get_by_config(topo_config=self.hardware_config["topo_config"], chip_config=self.hardware_config["chip_config"])
        # self.world_size = op.instance.world_size
        # self.world_size = 16
        # self.in_node_world_size = min(self.hardware.NUM_DV_PER_NODE, self.world_size)
        # if self.world_size > self.hardware.NUM_DV_PER_NODE and self.world_size % self.hardware.NUM_DV_PER_NODE != 0:
        #     raise ValueError("world size must be multiples of node size, otherwise yet to be supported")
        self.num_nodes = math.ceil(op.instance.N / self.hardware.NUM_DV_PER_NODE)
        # if hasattr(op.instance, "global_rank_list"):
        #     self.global_rank_list = op.instance.global_rank_list
        # else:
        #     self.global_rank_list = list(range(op.instance.N))
        # self.num_nodes, self.num_local_rank = self.hardware.get_num_node_num_lcoal_rank(self.global_rank_list)
        # if op.instance.M < 64:
        #     raise ValueError("M must be multiples of 64, otherwise yet to be supported")
        self.received_tokens = self.op.instance.received_tokens

    @property
    def communication_size(self):
        data_bytes = calculate_total_bytes([[self.received_tokens, self.inputs_shape[0][1]]], [self.inputs_dtype[0]])
        return data_bytes

    @property
    def communication_time(self):
        # print(f'received_tokens{self.received_tokens}, self.inputs_shape[0][1]]{self.inputs_shape[0]}')
        data_bytes = calculate_total_bytes([[self.received_tokens, self.inputs_shape[0][1]]], [self.inputs_dtype[0]])

        # print(data_bytes)
        latency = self.hardware.get_latency(self.num_nodes)
        btw_node_comm_time = data_bytes / self.hardware.get_among_node_bandwidth()
        return btw_node_comm_time + latency + 0.5

    @property
    def op_time(self):
        return self.communication_time
