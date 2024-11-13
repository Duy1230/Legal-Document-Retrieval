import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import math


class DiffusionScheduler:
    """
    Manages the noise scheduling process in the diffusion model.
    Implements forward process (adding noise) and reverse process (denoising) scheduling.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        # Define noise schedule parameters
        self.num_timesteps = num_timesteps

        # Create noise schedule (β_t) - linearly increasing from beta_start to beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Calculate alphas (α_t = 1 - β_t)
        self.alphas = 1 - self.betas

        # Calculate cumulative products of alphas (ᾱ_t)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Pre-calculate values for posterior variance
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

        # Add these important pre-computed values
        self.posterior_variance = self.betas * \
            (1. - self.alpha_bars.previous_timestep) / (1. - self.alpha_bars)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2],
                      self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars.previous_timestep) / (1. - self.alpha_bars))
        self.posterior_mean_coef2 = (
            (1. - self.alpha_bars.previous_timestep) * torch.sqrt(self.alphas) / (1. - self.alpha_bars))

    def add_noise(self, x_0, t, noise=None):
        """
        Forward process: q(x_t | x_0)
        Gradually adds noise to the input image according to the schedule
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]

        # Reparameterization trick: x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
        noisy_image = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return noisy_image, noise


class ResidualBlock(nn.Module):
    """
    Basic residual block for U-Net architecture
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class UNet(nn.Module):
    """
    U-Net architecture for the diffusion model.
    Predicts noise given a noisy image and timestep embedding.
    """

    def __init__(self, in_channels=3, time_emb_dim=256):
        super().__init__()

        # Improve time embedding with sinusoidal positional encoding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(
                time_emb_dim),  # New positional embedding
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),  # Replace ReLU with GELU
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Add attention layers
        self.attention = nn.MultiheadAttention(
            256, num_heads=8)  # Add attention mechanism

        # Encoder pathway
        self.enc1 = ResidualBlock(in_channels + time_emb_dim, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)

        # Decoder pathway
        self.dec3 = ResidualBlock(256 + 128, 128)
        self.dec2 = ResidualBlock(128 + 64, 64)
        self.dec1 = ResidualBlock(64 + time_emb_dim, 64)

        # Final layer
        self.final = nn.Conv2d(64, in_channels, 1)

        # Downsampling and upsampling
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        t_emb = t_emb.view(-1, t_emb.shape[1], 1,
                           1).repeat(1, 1, x.shape[2], x.shape[3])

        # Encoder
        x1 = self.enc1(torch.cat([x, t_emb], dim=1))
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))

        # Decoder with skip connections
        x = self.dec3(torch.cat([x3, self.up(x2)], dim=1))
        x = self.dec2(torch.cat([x, self.up(x1)], dim=1))
        x = self.dec1(torch.cat([x, t_emb], dim=1))

        return self.final(x)


class DiffusionModel:
    """
    Main diffusion model class that combines the scheduler and U-Net
    """

    def __init__(self, scheduler, model):
        self.scheduler = scheduler
        self.model = model

    def train_step(self, x_0, optimizer):
        """
        Single training step for the diffusion model
        """
        optimizer.zero_grad()

        # Sample random timesteps
        t = torch.randint(0, self.scheduler.num_timesteps,
                          (x_0.shape[0],), device=x_0.device)

        # Add noise according to schedule
        noisy_images, noise = self.scheduler.add_noise(x_0, t)

        # Predict noise
        predicted_noise = self.model(noisy_images, t)

        # Calculate loss (simple MSE between real and predicted noise)
        loss = F.mse_loss(predicted_noise, noise)

        # Backpropagate and optimize
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size, image_size, device, channels=3):
        """
        Generate new images using the trained model
        """
        # Add DDIM sampling option
        x = torch.randn(batch_size, channels, image_size,
                        image_size).to(device)

        # Add progress bar
        for t in tqdm(reversed(range(self.scheduler.num_timesteps))):
            t_batch = torch.ones(batch_size, device=device) * t

            # Predict noise
            predicted_noise = self.model(x, t_batch)

            # Calculate denoising step
            alpha = self.scheduler.alphas[t]
            alpha_bar = self.scheduler.alpha_bars[t]
            beta = self.scheduler.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 -
                                                                         alpha_bar)) * predicted_noise) + torch.sqrt(beta) * noise

            # # Add classifier-free guidance (optional)
            # if self.guidance_scale > 1:
            #     uncond_predicted_noise = self.model(x, t_batch, None)
            #     cond_predicted_noise = self.model(x, t_batch, context)
            #     predicted_noise = uncond_predicted_noise + self.guidance_scale * (cond_predicted_noise - uncond_predicted_noise)

        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
