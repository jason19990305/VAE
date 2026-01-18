import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_kl_divergence_2d():
    # 1. Set up the grid range
    x, y = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x, y))

    # 2. Define two distributions: P (Target/True) and Q (Predicted)
    # P: Standard Normal Distribution (centered at 0,0)
    mu_p = np.array([0.0, 0.0])
    cov_p = np.array([[1.0, 0.0], [0.0, 1.0]]) # Circular shape
    
    # Q: Shifted and distorted Normal Distribution (centered at 1.5, 1.0)
    mu_q = np.array([1.5, 1.0])
    cov_q = np.array([[0.8, 0.3], [0.3, 0.6]]) # Elliptical and rotated
    
    # 3. Calculate Probability Density Function (PDF)
    rv_p = multivariate_normal(mu_p, cov_p)
    rv_q = multivariate_normal(mu_q, cov_q)
    
    P = rv_p.pdf(pos)
    Q = rv_q.pdf(pos)

    # 4. Calculate the "density" (Integrand) of KL Divergence
    # Add a tiny epsilon to avoid division by zero or log(0)
    epsilon = 1e-10
    kl_density = P * np.log((P + epsilon) / (Q + epsilon))
    
    # Calculate total KL value (Approximate integral via summation)
    total_kl = np.sum(kl_density) * 0.01 * 0.01 # Multiply by dx * dy

    # === Start Plotting ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: P (True Distribution)
    ax1 = axes[0]
    c1 = ax1.contourf(x, y, P, cmap='Blues', levels=20)
    ax1.set_title(r'$P(x)$ - Target Distribution (Standard Normal)', fontsize=12)
    ax1.scatter(mu_p[0], mu_p[1], c='red', marker='x', label='Center P')
    plt.colorbar(c1, ax=ax1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q (Predicted Distribution)
    ax2 = axes[1]
    c2 = ax2.contourf(x, y, Q, cmap='Oranges', levels=20)
    ax2.set_title(r'$Q(x)$ - Predicted Distribution (Shifted)', fontsize=12)
    ax2.scatter(mu_q[0], mu_q[1], c='red', marker='x', label='Center Q')
    plt.colorbar(c2, ax=ax2)
    ax2.grid(True, alpha=0.3)

    # Plot 3: KL Divergence Density Map (Key Visualization!)
    ax3 = axes[2]
    # Using 'Reds' colormap: Darker red indicates high penalty (loss contribution)
    # This shows where P is high but Q is low.
    limit = np.max(np.abs(kl_density))
    c3 = ax3.contourf(x, y, kl_density, cmap='Reds', levels=30)
    ax3.set_title(f'KL Divergence Density\nTotal KL Value $\\approx$ {total_kl:.4f}', fontsize=12, fontweight='bold')
    
    # Overlay contours of P and Q for easier comparison
    ax3.contour(x, y, P, colors='blue', linewidths=1, alpha=0.5, linestyles='--')
    ax3.contour(x, y, Q, colors='orange', linewidths=1, alpha=0.5, linestyles='--')
    
    plt.colorbar(c3, ax=ax3, label='Contribution to Loss')
    ax3.grid(True, alpha=0.3)
    ax3.text(-3.5, 3.5, "Blue Dashed: P(x)\nOrange Dashed: Q(x)", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_kl_divergence_2d()