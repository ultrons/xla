import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
l_in = torch.randn(10, device=xm.xla_device(1))
linear = torch.nn.Linear(10, 20).to(xm.xla_device(2))
linear2 = torch.nn.Linear(20, 20).to(xm.xla_device(3))
l1 = linear(l_in.to(xm.xla_device(2)))
l2 = linear2(l1.to(xm.xla_device(3)))
loss = l2.sum()
loss.backward()
print(linear.weight.grad.norm())
