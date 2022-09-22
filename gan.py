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


class Generator(nn.Module):
    #The generator takes an input of size latent_size, and will produce an output of size 784.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
    def __init__(self):
        super(Generator, self).__init__()
        self.gen1 = nn.Linear(latent_size, 400)
        self.gen2 = nn.Linear(400, 784)

    def forward(self, z):
        z1 = nn.functional.relu(self.gen1(z))
        return nn.functional.sigmoid(self.gen2(z1))

class Discriminator(nn.Module):
    #The discriminator takes an input of size 784, and will produce an output of size 1.
    #It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its output
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dist1 = nn.Linear(784, 400)
        self.dist2 = nn.Linear(400, 1)

    def forward(self, x):
        x1 = nn.functional.relu(self.dist1(x))
        return nn.functional.sigmoid(self.dist2(x1))

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    #Trains both the generator and discriminator for one epoch on the training dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)
    
    train_set = len(train_loader.dataset)
    avg_generator_loss = 0
    avg_discriminator_loss = 0

    for (x, y) in train_loader:
        discriminator.train()
        discriminator_optimizer.zero_grad()
        x = torch.flatten(x, start_dim=1)
        disc_outputs_real = discriminator(x)
        disc_labels_real = torch.ones(x.shape[0], 1)

        noise = torch.randn(x.shape[0], latent_size)
        gen_outputs = generator(noise)
        disc_outputs_fake = discriminator(gen_outputs)
        disc_labels_fake = torch.zeros(x.shape[0], 1)

        outputs = torch.cat((disc_outputs_real, disc_outputs_fake), 0)
        labels = torch.cat((disc_labels_real, disc_labels_fake), 0)

        disc_CEloss = nn.BCELoss(reduction='sum')
        disc_loss = disc_CEloss(outputs, labels)
        avg_discriminator_loss += disc_loss
        disc_loss.backward()
        discriminator_optimizer.step()

        generator.train()
        generator_optimizer.zero_grad()

        noise = torch.randn(x.shape[0], latent_size)
        gen_outputs = generator(noise)
        disc_outputs_fake = discriminator(gen_outputs)

        gen_CEloss = nn.BCELoss(reduction='sum')
        gen_loss = disc_CEloss(disc_outputs_fake, disc_labels_real)
        avg_generator_loss += gen_loss
        gen_loss.backward()
        generator_optimizer.step()

    avg_generator_loss = avg_generator_loss / train_set
    avg_discriminator_loss = avg_discriminator_loss / (train_set * 2)

    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    #Runs both the generator and discriminator over the test dataset.
    #Returns the average generator and discriminator loss (scalar values, use the binary cross-entropy appropriately)

    test_set = len(test_loader.dataset)
    avg_generator_loss = 0
    avg_discriminator_loss = 0

    for (x, y) in test_loader:
        x = torch.flatten(x, start_dim=1)
        disc_outputs_real = discriminator(x)
        disc_labels_real = torch.ones(x.shape[0], 1)

        noise = torch.rand(x.shape[0], latent_size)
        gen_outputs = generator(noise)
        disc_outputs_fake = discriminator(gen_outputs)
        disc_labels_fake = torch.zeros(x.shape[0], 1)

        outputs = torch.cat((disc_outputs_real, disc_outputs_fake), 0)
        labels = torch.cat((disc_labels_real, disc_labels_fake), 0)

        disc_CEloss = nn.BCELoss(reduction='sum')
        disc_loss = disc_CEloss(outputs, labels)
        avg_discriminator_loss += disc_loss

        gen_CEloss = nn.BCELoss(reduction='sum')
        gen_loss = disc_CEloss(disc_outputs_fake, disc_labels_real)
        avg_generator_loss += gen_loss
    
    avg_generator_loss = avg_generator_loss / test_set
    avg_discriminator_loss = avg_discriminator_loss / (test_set * 2)

    return avg_generator_loss, avg_discriminator_loss


epochs = 50

discriminator_avg_train_losses = []
discriminator_avg_test_losses = []
generator_avg_train_losses = []
generator_avg_test_losses = []

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
    generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

    discriminator_avg_train_losses.append(discriminator_avg_train_loss)
    generator_avg_train_losses.append(generator_avg_train_loss)
    discriminator_avg_test_losses.append(discriminator_avg_test_loss)
    generator_avg_test_losses.append(generator_avg_test_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = generator(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

discriminator_avg_train_losses = [l.item() for l in discriminator_avg_train_losses]
generator_avg_train_losses = [l.item() for l in generator_avg_train_losses]
plt.plot(discriminator_avg_train_losses)
plt.plot(generator_avg_train_losses)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()

discriminator_avg_test_losses = [l.item() for l in discriminator_avg_test_losses]
generator_avg_test_losses = [l.item() for l in generator_avg_test_losses]
plt.plot(discriminator_avg_test_losses)
plt.plot(generator_avg_test_losses)
plt.title('Test Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Disc','Gen'], loc='upper right')
plt.show()