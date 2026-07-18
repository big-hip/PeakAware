import torch
from Dist_IR.Optim_IR import optim_capture




def test_optim_ir(opt_type:str):

    n=4
    params=[torch.randn(2,3) for _ in range(n)]
    #拷贝params到 params_bak
    params_bak=[param.clone() for param in params]
    grads=[torch.randn(2,3) for _ in range(n)]

    #选择优化器类型
    print("opt_type:",opt_type)

    norm_adam=optim_capture(opt_type, params_bak, lr=0.005 ,beta1=0.8, need_compile=False)
    norm_adam(params_bak, grads)
    #优化器计算结果(编译前)
    print("updated params result before compile:",params_bak)

    opt_adam=optim_capture(opt_type, params, lr=0.005, beta1=0.8, need_compile=True)
    opt_adam(params, grads)
    #优化器计算结果(编译后)
    print("updated params result after compile:",params)

if __name__ == "__main__":
    test_optim_ir("adam")
