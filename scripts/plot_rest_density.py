"""Plot rest-density histogram to check SPH kernel + particle-mass calibration.

If the average of the histogram is far from `target_density` (1000), then the
solver's source term `(rho_0 - rho_i) / dt` is non-zero even when the fluid
is at rest, and the pressure solver has no equilibrium to settle into. This
shows up as the convergence benchmark plateauing at a large non-zero error
regardless of iteration count.

Pipeline:
    1. cargo test --release -p fluid_engine -- --ignored rest_density --nocapture
    2. python scripts/plot_rest_density.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'rest_density.csv')
out_dir = os.path.join(script_dir, '..', 'docs', 'Fluid_Simulation', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'rest_density.png')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
})

df = pd.read_csv(csv_path)
densities = df['density'].values
target = 1000.0

avg = densities.mean()
std = densities.std()
mn = densities.min()
mx = densities.max()
rel_err = 100.0 * (avg - target) / target

print(f"particles : {len(densities)}")
print(f"target    : {target:.3f}")
print(f"avg       : {avg:.3f}")
print(f"min       : {mn:.3f}")
print(f"max       : {mx:.3f}")
print(f"std       : {std:.3f}")
print(f"avg-target: {avg - target:+.3f} ({rel_err:+.2f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(densities, bins=60, color='lightgray', edgecolor='black', alpha=0.85)
ax.axvline(target, color='red', linestyle='--', linewidth=2,
           label=fr'$\rho_0$ = {target:.0f}')
ax.axvline(avg, color='blue', linestyle='-', linewidth=2,
           label=fr'$\overline{{\rho_i}}$ = {avg:.1f}')

ax.set_xlabel(r'Gęstość $\rho_i$ [kg/m$^3$]')
ax.set_ylabel('Liczba cząstek')
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.7)

fig.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f'wrote: {out_path}')
