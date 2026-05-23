import numpy as np
import matplotlib.pyplot as plt
import os


out_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'Fluid_Simulation', 'figures')
os.makedirs(out_dir, exist_ok=True)
q = np.linspace(0.0, 1.0, 1000)
pi = np.pi

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'lines.linewidth': 2.0
})

def plot_bw_kernel(q, W, dW, name, filename):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    ax1.set_xlabel(r'$q = r/h$')
    ax1.set_ylabel(r'$W(q)$')
    line1, = ax1.plot(q, W, color='black', linestyle='-', label=f'Jądro {name}')
    ax1.set_ylim(0, max(W) * 1.1)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Градієнт (пунктирна сіра лінія)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\partial W / \partial q$')
    line2, = ax2.plot(q, dW, color='gray', linestyle='--', label='Gradient')
    ax2.set_ylim(min(dW) * 1.1, 0.5) # Трохи місця зверху для нуля
    ax2.axhline(0, color='black', linewidth=0.8) # Лінія нуля

    # Легенда
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', framealpha=1.0)

    plt.title(f'Charakterystyka jądra {name}')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()

k_c = 8.0 / pi
W_cubic = np.where(q <= 0.5, k_c * (6*q**3 - 6*q**2 + 1), 
                   np.where(q <= 1.0, k_c * 2*(1 - q)**3, 0.0))
dW_cubic = np.where(q <= 0.5, k_c * (18*q**2 - 12*q),
                    np.where(q <= 1.0, k_c * (-6*(1 - q)**2), 0.0))

k_w = 21.0 / (2.0 * pi)
W_wendland = np.where(q <= 1.0, k_w * (1 - q)**4 * (4*q + 1), 0.0)
dW_wendland = np.where(q <= 1.0, k_w * (-20 * q * (1 - q)**3), 0.0)

# Рендеринг
plot_bw_kernel(q, W_cubic, dW_cubic, 'Cubic Spline', 'kernel_cubic_bw.png')
plot_bw_kernel(q, W_wendland, dW_wendland, 'Wendland C2', 'kernel_wendland_bw.png')