import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Generator Network
class Generator(nn.Module):
    """
    Generator network that transforms random noise into synthetic images.
    The goal is to generate data that looks real enough to fool the discriminator.
    """
    def __init__(self, latent_dim=100, img_channels=1):
        super(Generator, self).__init__()
        
        # Sequential network to transform latent vectors (noise) into images
        self.main = nn.Sequential(
            # Initial projection and reshape
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Upsampling layers
            # Each layer doubles the image size
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final layer to generate image
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output range: [-1, 1]
        )

    def forward(self, x):
        return self.main(x)

# Discriminator Network
class Discriminator(nn.Module):
    """
    Discriminator network that tries to distinguish between real and fake images.
    Acts as a binary classifier: real (1) or fake (0).
    """
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        
        # Convolutional network that reduces image dimensionality
        self.main = nn.Sequential(
            # Input: img_channels x 64 x 64
            nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output probability in range [0, 1]
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

class GAN:
    """
    GAN wrapper class that handles the training process and interaction between
    Generator and Discriminator networks.
    """
    def __init__(self, latent_dim=100, img_channels=1, device='cuda'):
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize networks
        self.generator = Generator(latent_dim, img_channels).to(device)
        self.discriminator = Discriminator(img_channels).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Binary Cross Entropy loss for adversarial training
        self.criterion = nn.BCELoss()

    def train_step(self, real_images, batch_size):
        """
        Performs one training step for both Generator and Discriminator
        """
        # Create labels for real and fake images
        real_label = torch.ones(batch_size).to(self.device)
        fake_label = torch.zeros(batch_size).to(self.device)

        # ====== Train Discriminator ======
        self.d_optimizer.zero_grad()
        
        # Train on real images
        d_output_real = self.discriminator(real_images)
        d_loss_real = self.criterion(d_output_real, real_label)
        
        # Train on fake images
        noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        fake_images = self.generator(noise)
        d_output_fake = self.discriminator(fake_images.detach())
        d_loss_fake = self.criterion(d_output_fake, fake_label)
        
        # Combined discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()

        # ====== Train Generator ======
        self.g_optimizer.zero_grad()
        
        # Generator tries to fool discriminator
        d_output_fake = self.discriminator(fake_images)
        g_loss = self.criterion(d_output_fake, real_label)
        
        g_loss.backward()
        self.g_optimizer.step()

        return g_loss.item(), d_loss.item()

    def generate_samples(self, num_samples):
        """
        Generate synthetic samples using the trained generator
        """
        with torch.no_grad():
            noise = torch.randn(num_samples, self.latent_dim, 1, 1).to(self.device)
            return self.generator(noise)
