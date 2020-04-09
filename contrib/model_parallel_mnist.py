import numpy as np
import os
import time
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

FLAGS = {}
FLAGS['lr'] = 0.01
FLAGS['datadir'] = "/tmp/mnist"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.01
FLAGS['momentum'] = 0.5
FLAGS['num_epochs'] = 10
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False
FLAGS['tidy'] = False

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = self.bn1(x)
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = self.bn2(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class MNIST_model_parallel(MNIST):
  """ Model Parallel from existing model """
  def __init__(self, gindex=0):
    super(MNIST_model_parallel, self).__init__()
    self.seq1 = nn.Sequential(
        self.conv1,
        nn.MaxPool2d(2),
        nn.ReLU(),
        self.bn1,
        self.conv2,
        nn.MaxPool2d(2),
        nn.ReLU(),
        self.bn2
    ).to(xm.xla_device(gindex))
    self.seq2 = nn.Sequential(
        self.fc1,
        nn.ReLU(),
        self.fc2
    ).to(xm.xla_device(gindex+1))
    self.gindex = gindex

  def forward(self, x):
    x = self.seq1(x)
    x = torch.flatten(x, 1)
    x = self.seq2(x.to(xm.xla_device(self.gindex + 1)))
    return F.log_softmax(x, dim=1)



def train_mnist(gindex):
  if gindex % 2 == 0:
    return 0
  device = xm.xla_device()
  train_loader = xu.SampleGenerator(
      data=(torch.zeros(FLAGS['batch_size'], 1, 28,
                        28), torch.zeros(FLAGS['batch_size'],
                                         dtype=torch.int64)),
      sample_count=60000 // FLAGS['batch_size'] // xm.xrt_world_size())
  # Scale learning rate to num cores
  lr = FLAGS['lr'] * xm.xrt_world_size()
  model = MNIST_model_parallel(gindex)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=FLAGS['momentum'])
  loss_fn = nn.NLLLoss()

  def train_loop_fn(loader,gindex):
    tracker = xm.RateTracker()

    model.train()
    for step, (data, target) in enumerate(loader):
      xm.master_print('Step-1 {} train begin'.format(step))
      optimizer.zero_grad()
      output = model(data)
      target = target.to(xm.xla_device(gindex+1))
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(FLAGS['batch_size'])

  for epoch in range(1, FLAGS['num_epochs'] + 1):
    xm.master_print('Epoch {} train begin'.format(epoch))
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loop_fn(para_loader.per_device_loader(device),gindex)
    xm.master_print('Epoch {} train end '.format(epoch))

    if FLAGS['metrics_debug']:
      xm.master_print(met.metrics_report())

  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(index, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  train_mnist(index+1)
  if FLAGS['tidy'] and os.path.isdir(FLAGS['datadir']):
    shutil.rmtree(FLAGS['datadir'])

if __name__ == '__main__':
  #xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'])
  #exit()
  gindex = 7
  device = xm.xla_device(gindex)
  model = MNIST_model_parallel(gindex)
  loss_fn = nn.NLLLoss()
  
  train_loader = xu.SampleGenerator(
      data=(torch.zeros(FLAGS['batch_size'], 1, 28,
                        28), torch.zeros(FLAGS['batch_size'],
                                         dtype=torch.int64)),
      sample_count=60000 // FLAGS['batch_size'] // xm.xrt_world_size())
  para_loader = pl.ParallelLoader(train_loader, [device])
  loader = para_loader.per_device_loader(device)

  optimizer = optim.SGD(model.parameters(), lr=0.001)
  optimizer.zero_grad()
  tracker = xm.RateTracker()
  model.train()
  for step, (data, target) in enumerate(loader):
    optimizer.zero_grad()
    output = model(data)
    target = target.to(xm.xla_device(gindex+1))
    loss = loss_fn(output, target)
    loss.backward()
    xm.optimizer_step(optimizer)
    tracker.add(FLAGS['batch_size'])
