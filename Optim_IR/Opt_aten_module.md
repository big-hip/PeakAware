class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[20, 10]", primals_2: "f32[20, 10]", primals_3: "f32[20, 10]", primals_4: "f32[20]", primals_5: "f32[20]", primals_6: "f32[20]", primals_7: "f32[1, 20]", primals_8: "f32[1, 20]", primals_9: "f32[1, 20]", primals_10: "f32[1]", primals_11: "f32[1]", primals_12: "f32[1]"):
        # No stacktrace found for following nodes
        split_default = torch.ops.aten.split.default(primals_1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem = split_default[0];  split_default = getitem = None
        split_default_1 = torch.ops.aten.split.default(primals_2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_1 = split_default_1[0];  split_default_1 = getitem_1 = None
        split_default_2 = torch.ops.aten.split.default(primals_3, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_2 = split_default_2[0];  split_default_2 = getitem_2 = None
        split_default_3 = torch.ops.aten.split.default(primals_4, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_3 = split_default_3[0];  split_default_3 = getitem_3 = None
        split_default_4 = torch.ops.aten.split.default(primals_5, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_4 = split_default_4[0];  split_default_4 = getitem_4 = None
        split_default_5 = torch.ops.aten.split.default(primals_6, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_5 = split_default_5[0];  split_default_5 = getitem_5 = None
        split_default_6 = torch.ops.aten.split.default(primals_7, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_6 = split_default_6[0];  split_default_6 = getitem_6 = None
        split_default_7 = torch.ops.aten.split.default(primals_8, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_7 = split_default_7[0];  split_default_7 = getitem_7 = None
        split_default_8 = torch.ops.aten.split.default(primals_9, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_8 = split_default_8[0];  split_default_8 = getitem_8 = None
        split_default_9 = torch.ops.aten.split.default(primals_10, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_9 = split_default_9[0];  split_default_9 = getitem_9 = None
        split_default_10 = torch.ops.aten.split.default(primals_11, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_10 = split_default_10[0];  split_default_10 = getitem_10 = None
        split_default_11 = torch.ops.aten.split.default(primals_12, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
        getitem_11 = split_default_11[0];  split_default_11 = getitem_11 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:55 in step, code: r = r + torch.square(grads[i])
        pow_1: "f32[20, 10]" = torch.ops.aten.pow.Tensor_Scalar(primals_2, 2)
        add: "f32[20, 10]" = torch.ops.aten.add.Tensor(primals_1, pow_1);  primals_1 = pow_1 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:57 in step, code: update = -1.0 * lr * grads[i] / (torch.sqrt(r) + epsilon)
        mul: "f32[20, 10]" = torch.ops.aten.mul.Tensor(primals_2, -0.01);  primals_2 = None
        sqrt: "f32[20, 10]" = torch.ops.aten.sqrt.default(add);  add = None
        add_1: "f32[20, 10]" = torch.ops.aten.add.Tensor(sqrt, 1e-08);  sqrt = None
        div: "f32[20, 10]" = torch.ops.aten.div.Tensor(mul, add_1);  mul = add_1 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:58 in step, code: params[i] = params[i] + update
        add_2: "f32[20, 10]" = torch.ops.aten.add.Tensor(primals_3, div);  primals_3 = div = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:55 in step, code: r = r + torch.square(grads[i])
        pow_2: "f32[20]" = torch.ops.aten.pow.Tensor_Scalar(primals_5, 2)
        add_3: "f32[20]" = torch.ops.aten.add.Tensor(primals_4, pow_2);  primals_4 = pow_2 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:57 in step, code: update = -1.0 * lr * grads[i] / (torch.sqrt(r) + epsilon)
        mul_1: "f32[20]" = torch.ops.aten.mul.Tensor(primals_5, -0.01);  primals_5 = None
        sqrt_1: "f32[20]" = torch.ops.aten.sqrt.default(add_3);  add_3 = None
        add_4: "f32[20]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-08);  sqrt_1 = None
        div_1: "f32[20]" = torch.ops.aten.div.Tensor(mul_1, add_4);  mul_1 = add_4 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:58 in step, code: params[i] = params[i] + update
        add_5: "f32[20]" = torch.ops.aten.add.Tensor(primals_6, div_1);  primals_6 = div_1 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:55 in step, code: r = r + torch.square(grads[i])
        pow_3: "f32[1, 20]" = torch.ops.aten.pow.Tensor_Scalar(primals_8, 2)
        add_6: "f32[1, 20]" = torch.ops.aten.add.Tensor(primals_7, pow_3);  primals_7 = pow_3 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:57 in step, code: update = -1.0 * lr * grads[i] / (torch.sqrt(r) + epsilon)
        mul_2: "f32[1, 20]" = torch.ops.aten.mul.Tensor(primals_8, -0.01);  primals_8 = None
        sqrt_2: "f32[1, 20]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        add_7: "f32[1, 20]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-08);  sqrt_2 = None
        div_2: "f32[1, 20]" = torch.ops.aten.div.Tensor(mul_2, add_7);  mul_2 = add_7 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:58 in step, code: params[i] = params[i] + update
        add_8: "f32[1, 20]" = torch.ops.aten.add.Tensor(primals_9, div_2);  primals_9 = div_2 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:55 in step, code: r = r + torch.square(grads[i])
        pow_4: "f32[1]" = torch.ops.aten.pow.Tensor_Scalar(primals_11, 2)
        add_9: "f32[1]" = torch.ops.aten.add.Tensor(primals_10, pow_4);  primals_10 = pow_4 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:57 in step, code: update = -1.0 * lr * grads[i] / (torch.sqrt(r) + epsilon)
        mul_3: "f32[1]" = torch.ops.aten.mul.Tensor(primals_11, -0.01);  primals_11 = None
        sqrt_3: "f32[1]" = torch.ops.aten.sqrt.default(add_9);  add_9 = None
        add_10: "f32[1]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-08);  sqrt_3 = None
        div_3: "f32[1]" = torch.ops.aten.div.Tensor(mul_3, add_10);  mul_3 = add_10 = None

         # File: /mnt/sdb/tangchengxiang/zero/Ascend_IR/Dist_IR/Optim_IR/optim_algorithm.py:58 in step, code: params[i] = params[i] + update
        add_11: "f32[1]" = torch.ops.aten.add.Tensor(primals_12, div_3);  primals_12 = div_3 = None
        return (add_2, add_5, add_8, add_11)
