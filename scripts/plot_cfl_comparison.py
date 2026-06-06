"""Generuje wykres porównujący strategię statycznego Δt z adaptacyjnym CFL.

Potok:
    1. cargo test --release -p fluid_engine -- --ignored cfl_comparison --nocapture
    2. python scripts/plot_cfl_comparison.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'cfl_comparison.csv')
out_dir = os.path.join(script_dir, '..', 'docs', 'Fluid_Simulation', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'cfl_comparison.png')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'lines.linewidth': 1.8,
})

df = pd.read_csv(csv_path)
static   = df[df['mode'] == 'static'].reset_index(drop=True)
adaptive = df[df['mode'] == 'adaptive'].reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# ── Panel 1: krok czasowy ─────────────────────────────────────────────────────
ax = axes[0]
ax.plot(static['frame'],   static['dt']   * 1000, color='black',
        linestyle='-',  label='Statyczny')
ax.plot(adaptive['frame'], adaptive['dt'] * 1000, color='gray',
        linestyle='--', label='Adaptacyjny CFL')
ax.set_ylabel(r'$\Delta t$ [ms]')
ax.set_title('(a) Krok czasowy')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)

# ── Panel 2: liczba iteracji Jacobiego ────────────────────────────────────────
ax = axes[1]
ax.plot(static['frame'],   static['iters'],   color='black',
        linestyle='-',  label='Statyczny')
ax.plot(adaptive['frame'], adaptive['iters'], color='gray',
        linestyle='--', label='Adaptacyjny CFL')
ax.set_ylabel('Iteracje Jacobiego')
ax.set_title('(b) Liczba iteracji solvera gęstości')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)

# ── Panel 3: maksymalna prędkość cząstki ─────────────────────────────────────
ax = axes[2]
ax.plot(static['frame'],   static['max_speed'],   color='black',
        linestyle='-',  label='Statyczny')
ax.plot(adaptive['frame'], adaptive['max_speed'], color='gray',
        linestyle='--', label='Adaptacyjny CFL')
ax.set_xlabel('Klatka symulacji')
ax.set_ylabel(r'$v_{\max}$ [m/s]')
ax.set_title('(c) Maksymalna prędkość cząstki')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)

fig.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f'wrote: {out_path}')
