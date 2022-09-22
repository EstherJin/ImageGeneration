from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encode1 = nn.Linear(784, 400)
        self.encode2 = nn.Linear(400, latent_size * 2)
        
        self.decode1 = nn.Linear(latent_size, 400)
        self.decode2 = nn.Linear(400, 784)

    def encode(self, x):
        #The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        
        x1 = nn.functional.relu(self.encode1(x))
        return self.encode2(x1)

    def reparameterize(self, means, log_variances):
        #The reparameterization module lies between the encoder and the decoder
        #It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        
        std = torch.sqrt(torch.exp(log_variances))
        n_dist = torch.randn(log_variances.size())
        return means + (std * n_dist)

    def decode(self, z):
        #The decoder will take an input of size latent_size, and will produce an output of size 784
        #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        
        z1 = nn.functional.relu(self.decode1(z))
        return nn.functional.sigmoid(self.decode2(z1))

    def forward(self, x):
        #Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        #Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        
        x = torch.flatten(x, start_dim=1)
        encoded = self.encode(x).view(-1, 2, latent_size)

        means = encoded[:,0]
        log_variances = encoded[:,1]
        
        z = self.reparameterize(means, log_variances)

        decoded = self.decode(z)

        return decoded, means, log_variances

def vae_loss_function(reconstructed_x, x, means, log_variances):
    #Compute the VAE loss
    #The loss is a sum of two terms: reconstruction error and KL divergence
    #Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    #The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    #Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    
    x = torch.flatten(x, start_dim=1)

    CEloss = nn.BCELoss(reduction='sum')
    reconstruction_loss = CEloss(reconstructed_x, x)
    kl = -0.5 * torch.sum(1 + log_variances - torch.square(means) - torch.exp(log_variances))
    loss = reconstruction_loss + kl

    return loss, reconstruction_loss


def train(model, optimizer):
    #Trains the VAE for one epoch on the training dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    
    train_set = len(train_loader.dataset)
    avg_train_loss = 0
    avg_train_reconstruction_loss = 0

    for (x, y) in train_loader:
        model.train()
        optimizer.zero_grad()
        decoded, means, log_variances = model(x)
        loss, reconstruction_loss = vae_loss_function(decoded, x, means, log_variances)
        avg_train_loss += loss
        avg_train_reconstruction_loss += reconstruction_loss
        loss.backward()
        optimizer.step()
    
    avg_train_loss = avg_train_loss / train_set
    avg_train_reconstruction_loss = avg_train_reconstruction_loss / train_set

    return avg_train_loss, avg_train_reconstruction_loss

def test(model):
    #Runs the VAE on the test dataset
    #Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)

    test_set = len(test_loader.dataset)
    avg_test_loss = 0
    avg_test_reconstruction_loss = 0

    for (x, y) in test_loader:
        decoded, means, log_variances = model(x)
        loss, reconstruction_loss = vae_loss_function(decoded, x, means, log_variances)
        avg_test_loss += loss
        avg_test_reconstruction_loss += reconstruction_loss
    
    avg_test_loss = avg_test_loss / test_set
    avg_test_reconstruction_loss = avg_test_reconstruction_loss / test_set

    return avg_test_loss, avg_test_reconstruction_loss

epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

avg_train_reconstruction_losses = [l.item() for l in avg_train_reconstruction_losses]
plt.plot(avg_train_reconstruction_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()

avg_test_reconstruction_losses = [l.item() for l in avg_test_reconstruction_losses]
plt.plot(avg_test_reconstruction_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.show()
