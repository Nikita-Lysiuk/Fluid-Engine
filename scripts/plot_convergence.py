"""Generate the DFSPH solver convergence plot from CSV produced by the Rust benchmark.

Pipeline:
    1. Run the GPU benchmark (writes scripts/convergence.csv):
           cargo test --release -p fluid_engine -- --ignored convergence --nocapture
    2. Run this script (writes docs/Fluid_Simulation/figures/convergence.png):
           python scripts/plot_convergence.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'convergence.csv')
out_dir = os.path.join(script_dir, '..', 'docs', 'Fluid_Simulation', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'convergence.png')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'lines.linewidth': 2.0,
})

df = pd.read_csv(csv_path).sort_values('iters')

fig, (ax_density, ax_divergence) = plt.subplots(1, 2, figsize=(12, 5))


ax_density.plot(
    df['iters'], df['density_error'],
    color='black', marker='o', linestyle='-', markersize=7,
)
ax_density.set_title('(a)', loc='left', fontweight='bold') 
ax_density.set_xlabel(r'Liczba iteracji solvera $N_\rho$')
ax_density.set_ylabel(r'Średni błąd gęstości [kg/m$^3$]')
ax_density.set_yscale('log')
ax_density.grid(True, which='both', linestyle=':', alpha=0.7)

ax_divergence.plot(
    df['iters'], df['divergence_error'],
    color='gray', marker='s', linestyle='--', markersize=7,
)
ax_divergence.set_title('(b)', loc='left', fontweight='bold') 
ax_divergence.set_xlabel(r'Liczba iteracji solvera $N_{\nabla}$')
ax_divergence.set_ylabel(r'Średni błąd dywergencji [kg/(m$^3\cdot$s)]')
ax_divergence.set_yscale('log')
ax_divergence.grid(True, which='both', linestyle=':', alpha=0.7)

fig.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f'wrote: {out_path}')