import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_channel, latent_dim=2):
        super(VAE, self).__init__()
        self.input_channel = input_channel
        self.latent_dim = latent_dim

        # --- Encoder ---
        # 1x28x28 -> 16x28x28
        self.encoder1 = nn.Conv2d(input_channel, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        # 16x14x14 -> 32x7x7
        self.encoder2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 32x7x7 -> latent_dim
        self.fc_mu = nn.Linear(32 * 7 * 7, self.latent_dim)
        self.fc_logvar = nn.Linear(32 * 7 * 7, self.latent_dim)

        # --- Decoder ---
        # latent_dim -> 32x7x7
        self.fc_decode = nn.Linear(self.latent_dim, 32 * 7 * 7)
        # 32x7x7 -> 16x14x14
        self.decoder1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        # 16x14x14 -> 1x28x28
        self.decoder2 = nn.ConvTranspose2d(16, input_channel, kernel_size=2, stride=2)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization Trick:
        Allows backpropagation through random sampling.
        z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * log_var) # Convert log_var to standard deviation
            eps = torch.randn_like(std)    # Sample random noise from standard normal distribution
            return mu + eps * std          # Add noise
        else:
            return mu # During inference, just use the mean
        
    def encode(self, x):
        x = torch.relu(self.encoder1(x))
        x = self.pool1(x)
        x = torch.relu(self.encoder2(x))
        x = self.pool2(x)
        x = x.view(-1, 32 * 7 * 7) # Flatten
        return self.fc_mu(x), self.fc_logvar(x)
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 32, 7, 7) # Reshape back to feature map
        x = torch.relu(self.decoder1(x))
        x = torch.sigmoid(self.decoder2(x)) # Sigmoid puts pixels in [0, 1] range
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)        
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var