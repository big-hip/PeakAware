from atencost.frontend.utils.tensor_record import TensorRecord
from atencost.frontend.utils.op_record import OpRecord


def broadcast_shapes(shape1, shape2):
    """
    Broadcast two shapes following the broadcasting rules.

    Args:
        shape1, shape2: Shapes of two tensors to broadcast.

    Returns:
        A tuple of the broadcasted shapes.
    """
    # Ensure shape1 is the larger shape (for easier broadcasting)
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1

    # Prepend dimensions of shape2 with 1s if necessary to match lengths
    shape2 = [1] * (len(shape1) - len(shape2)) + list(shape2)

    for dim1, dim2 in zip(shape1, shape2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            # raise ValueError(f"Dimensions {dim1} and {dim2} do not match or are not broadcastable.")
            return 0

    broadcasted_shape = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2:
            broadcasted_shape.append(dim1)
        elif dim1 == 1:
            broadcasted_shape.append(dim2)
        elif dim2 == 1:
            broadcasted_shape.append(dim1)
        else:
            # raise ValueError(f"Shapes {shape1} and {shape2} cannot be broadcasted.")
            return 0
    return broadcasted_shape


def create_op_info(input_list, input_dtype_list, output_list, output_dtype_list, other_input_list=None, position='behind'):
    inputs = [
        TensorRecord(
            name=None, global_shape=None, local_shape=shape,
            type=None, dtype=dtype, is_dtensor=None, requires_grad=None,
            module_path=None, module_id=None, device_mesh=None,
            placements=None, producer=None, consumer=None
        )
        for shape, dtype in zip(input_list, input_dtype_list)
    ]
    outputs = [
        TensorRecord(
            name=None, global_shape=shape, local_shape=shape,
            type=None, dtype=dtype, is_dtensor=None, requires_grad=None,
            module_path=None, module_id=None, device_mesh=None,
            placements=None, producer=None, consumer=None
        )
        for shape, dtype in zip(output_list, output_dtype_list)
    ]
    if other_input_list:
        if position == 'behind':
            inputs = other_input_list + inputs
        else:
            inputs = inputs + other_input_list
    return OpRecord(
        id=None, name=None, type=None, subtype=None, comm_type=None,
        inputs=inputs, outputs=outputs,
        module_instance=None, module_path=None, module_id=None,
        fusion_type=None, raw_name=None, instance=None, op_type=None
    )

def merge_flops_dicts(dict_list):
    merged_dict = {}

    for d in dict_list:
        for key, value in d.items():
            merged_dict[key] = merged_dict.get(key, 0) + value

    return merged_dict
