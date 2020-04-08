import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from torchvision import datasets, transforms

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('xla:1')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('xla:2')

    def forward(self, x):
        x1 = self.relu(self.net1(x.to('xla:1')))
        return self.net2(x1.to('xla:2'))

model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('xla:2')
loss_fn(outputs, labels).backward()
optimizer.step()
