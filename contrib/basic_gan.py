import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu

# Define Parameters
FLAGS = {}
FLAGS['datadir'] = "/tmp/mnist"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.01
FLAGS['momentum'] = 0.5
FLAGS['num_epochs'] = 10
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False

def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    out_dir = '{}/dataset'.format(FLAGS['datadir'])
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def train_gan():
    torch.manual_seed(1)
    
    if not xm.is_master_ordinal():
        # Barrier: Wait until master is done downloading
        xm.rendezvous('download_only_once')
    # Dataset
    data = mnist_data()
    if xm.is_master_ordinal():
        # Master is done, other workers can proceed now
        xm.rendezvous('download_only_once')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        data,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    

    # Create loader with data, so that we can iterate over it
    #train_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
      data,
      batch_size=FLAGS['batch_size'],
      sampler=train_sampler,
      num_workers=FLAGS['num_workers'],
      drop_last=True)

    # Num batches
    num_batches = len(train_loader)
    
    device=xm.xla_device()
    discriminator = DiscriminatorNet().to(device)
    generator = GeneratorNet().to(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # Loss function
    loss = nn.BCELoss()

    def real_data_target(size):
        '''
        Tensor containing ones, with shape = size
        '''
        data = Variable(torch.ones(size, 1))
        return data.to(device)

    def fake_data_target(size):
        '''
        Tensor containing zeros, with shape = size
        '''
        data = Variable(torch.zeros(size, 1))
        return data.to(device)
    
    # Noise
    def noise(size):
        n = Variable(torch.randn(size, 100)) 
        return n.to(device)
    def images_to_vectors(images):
        return images.view(images.size(0), 784)

    def vectors_to_images(vectors):
        return vectors.view(vectors.size(0), 1, 28, 28)
    
    def train_step_d(optimizer, real_data, fake_data):

        # Reset gradients
        optimizer.zero_grad()
        
        # 1.1 Train on Real Data
        prediction_real = discriminator(real_data)
        # Calculate error and backpropagate
        error_real = loss(prediction_real, real_data_target(real_data.size(0)))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = discriminator(fake_data)
        # Calculate error and backpropagate
        error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
        error_fake.backward()
        
        # 1.3 Update weights with gradients
        xm.optimizer_step(optimizer)
        
        # Return error
        return error_real + error_fake, prediction_real, prediction_fake

    def train_step_g (optimizer, fake_data):
        # 2. Train Generator
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        prediction = discriminator(fake_data)
        # Calculate error and backpropagate
        error = loss(prediction, real_data_target(prediction.size(0)))
        error.backward()
        # Update weights with gradients
        xm.optimizer_step(optimizer, barrier=True)
        # Return error
        return error

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        for n_batch, (real_batch,_) in enumerate(loader):
            # Train Step Descriminator
            real_data = Variable(images_to_vectors(real_batch))
            fake_data = generator(noise(real_data.size(0))).detach()
            d_error, d_pred_real, d_pred_fake = train_step_d(d_optimizer,
                                                                real_data, fake_data)
            #Train Step Generator
            fake_data = generator(noise(real_batch.size(0)))
            g_error = train_step_g(g_optimizer, fake_data)
            print(f'D_ERROR: {d_error}, G_ERROR: {g_error}')
        return d_error, g_error


            # Display Test Images
            # Save Model Checkpoints

    for epoch in range(1, FLAGS['num_epochs'] +1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        d_error, g_error = train_loop_fn (para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}: D_error:{}, G_error".format(epoch, d_error, g_error))
    return 0

# Start training processes
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    train_gan()
    #if rank == 0:
    #  # Retrieve tensors that are on TPU core 0 and plot.
    #  plot_results(data.cpu(), pred.cpu(), target.cpu())

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS['num_cores'],
          start_method='fork')

