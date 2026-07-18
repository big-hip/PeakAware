from dataclasses import asdict
from pprint import pprint

import torch

# noinspection PyUnresolvedReferences
from custom import op_costmodel
from zhanlu.backend.analytical_model import AnalyticalModel
from zhanlu.backend.perf_result import ZhanluPerfResult
from zhanlu.frontend.utils.op_record import OpRecord
from zhanlu.frontend.utils.tensor_record import TensorRecord


def get_dummy_tensor(shape, dtype):
    return TensorRecord(
        name='tensor', global_shape=shape, local_shape=shape,
        type='forward', dtype=dtype, is_dtensor=False, requires_grad=False,
        module_path='', module_id=0, device_mesh='',
        placements='', producer=[], consumer=[]
    )


def get_dummy_op(name, inputs):
    return OpRecord(
        id=0, name=name, type='forward', subtype='', comm_type='',
        inputs=inputs, outputs=[],
        module_instance=None, module_path='', module_id=0,
        fusion_type='', raw_name='', instance=None, op_type=''
    )


class SingleOpPredictor:
    def __init__(self):
        self.hardware = 'A3,A3'

    def predict(self, op: OpRecord):
        self.perf_model = AnalyticalModel(op, self.hardware)
        perf: ZhanluPerfResult = self.perf_model()

        print(f'{perf.op_time}')
        pprint(asdict(perf), sort_dicts=False)
        return perf.op_time


def main():
    sop = SingleOpPredictor()
    print("matmul".center(50, '-'))
    # "4096,4096;27392,4096" FLOAT16;FLOAT16 FORMAT_ND;FORMAT_ND "4096,27392"
    shape1, shape2 = [4096, 4096], [4096, 27392]
    t1 = get_dummy_tensor(shape1, torch.float16)
    t2 = get_dummy_tensor(shape2, torch.float16)
    op = get_dummy_op('MatMul', [t1, t2])
    op_time = sop.predict(op)
    print(f'{op_time=}')

    print("FA".center(50, '-'))
    # [[4096, 16, 192], [4096, 16, 192], [4096, 16, 128], [2048, 2048]]
    shapes = [[4096, 16, 192], [4096, 16, 192], [4096, 16, 128], [2048, 2048]]
    tensors = [get_dummy_tensor(shape, torch.float16) for shape in shapes]
    op = get_dummy_op('npu_fusion_attention', tensors)
    op_time = sop.predict(op)
    print(f'{op_time=}')

    print("FA".center(50, '-'))
    # [[4096, 16, 192], [4096, 16, 192], [4096, 16, 128], [2048, 2048]]
    # shapes = [[4096, 16, 192], [4096, 16, 192], [4096, 16, 128], [2048, 2048]]
    shapes = [[153600]]
    tensors = [get_dummy_tensor(shape, torch.float32) for shape in shapes]
    op = get_dummy_op('AllReduceKernel', tensors)
    class A:
        pass
    op.instance = A()
    op.instance.global_rank_list = [8,9]
    op_time = sop.predict(op)
    print(f'{op_time=}')


if __name__ == '__main__':
    main()
