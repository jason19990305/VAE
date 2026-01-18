# VAE (Variational Autoencoder) Example


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)](https://pytorch.org/)

HackMD Article : https://hackmd.io/@bGCXESmGSgeAArScMaBxLA/H1e3KZqrbx

This is a simple **PyTorch** implementation of a **Variational Autoencoder (VAE)** trained on MNIST. The project demonstrates a complete pipeline for training, evaluating, and visualizing a VAE including reconstruction, sampling from the latent space, and KL-divergence visualization.

## 📂 Project Structure

```text
.
├── main.py             # Training and evaluation script
├── VAE.py              # VAE model: encoder, decoder, reparameterization
├── KL_visualize.py     # Utilities to visualize KL divergence and latent space
├── KL_visualize.png    # (optional) generated visualization
├── data/               # MNIST dataset (raw files stored under data/MNIST/raw/)
│   └── MNIST/
│       └── raw/
├── __pycache__/
└── README.md           # Project documentation
```

## 🚀 Installation

### 1. Prerequisites
Use a Python virtual environment (recommended).

### 2. Install Dependencies
```bash
# Install core dependencies
pip install torch torchvision numpy matplotlib tqdm
```

## 🖥️ Usage

### Train the VAE
The `main.py` script trains the VAE on MNIST and saves checkpoints / outputs (if enabled).

```bash
python main.py
```

Typical behavior:
- **Training**: runs training loop computing reconstruction + KL loss.
- **Checkpointing**: saves model weights (if enabled in the script).

### Visualize KL / Latent Space
Run the KL visualization utility to inspect KL divergence across latent dims and to plot 2D latent scatter / reconstructions.

```bash
python KL_visualize.py
```

Outputs: plots showing KL per-dimension, reconstructions, and optionally a grid of decoded samples from the latent prior.

## 💡 Technical Highlights

- **Encoder / Decoder**: simple convolutional or MLP backbone (see `VAE.py`).
- **Reparameterization trick**: samples z ~ N(mu, sigma^2) using mu and logvar from encoder.
- **Loss**: reconstruction loss (e.g., BCE or MSE) + KL divergence regularizer.
- **Sampling & Generation**: sample from prior N(0,I) and decode to generate new digit images.


## 📈 Example Commands

Train quickly (example):
```bash
python main.py
```

Visualize KL and reconstructions:
```bash
python KL_visualize.py
```




