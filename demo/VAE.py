import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Variational Autoencoder (VAE) implementation
        
        Args:
            input_dim: dimension of input data
            hidden_dim: dimension of hidden layers
            latent_dim: dimension of latent space
        """
        super(VAE, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Latent space parameters (μ and log(σ²))
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # For images in range [0,1]
        )
        
    def encode(self, x):
        """Encode input into latent space parameters μ and log(σ²)"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) while maintaining
        differentiability
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        """Decode latent vector into reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass through the VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    """
    VAE loss function = Reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed input
        x: original input
        mu: mean of the latent distribution
        log_var: log variance of the latent distribution
    """
    # Reconstruction loss (binary cross entropy for images)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence between N(mu, var) and N(0, 1)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

# Example usage
def train_vae(model, optimizer, data_loader, epochs):
    """Training loop for VAE"""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.view(data.size(0), -1)  # Flatten input
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}')
