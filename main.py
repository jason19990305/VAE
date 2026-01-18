import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from VAE import VAE
import matplotlib.pyplot as plt


def loss_function(reconstructed, original, mu, log_var, beta=4.0):
    
    # Reconstruction loss (Mean Squared Error)
    recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
    # BCE(binary cross entropy) can also be used:
    # recon_loss = nn.functional.binary_cross_entropy(reconstructed, original, reduction='sum')
    
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + beta * kl_div

def main():
    
    # Hyperparameters
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    LATENT_DIM = 10
    MAX_BETA = 4.0 # Maximum weight for KL Divergence
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize model, optimizer, and loss function
    model = VAE(input_channel=1, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # shuffle=True for training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        BETA = min(MAX_BETA, MAX_BETA * epoch / (EPOCHS // 2)) # Linear increase of beta
        # unpack data : (data, labels)
        # batch_idx : 0 ~ (batch_size // total_size)
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            
            reconstructed, mu, log_var = model(data)
            
            # Compute reconstruction loss and KL divergence
            loss = loss_function(reconstructed, data, mu, log_var, beta=BETA)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss / len(train_loader.dataset):.4f}")
        
    # Visualization / Inference
    model.eval() # Switch to evaluation mode
    with torch.no_grad():
        # Sample random noise from latent space (z ~ Normal(0, 1))
        z_sample = torch.randn(16, LATENT_DIM).to(DEVICE)
        print("z_sample.shape:", z_sample.shape)
        generated = model.decode(z_sample).cpu()

        # Plot generated images
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(generated[i].squeeze(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
