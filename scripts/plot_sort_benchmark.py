"""Generate the sort benchmark plot from CSV produced by the Rust benchmark.

Pipeline:
    1. Run the GPU benchmark (writes scripts/sort_benchmark.csv):
           cargo test --release -p fluid_engine -- --ignored sort_benchmark --nocapture
    2. Run this script (writes docs/Fluid_Simulation/figures/sort_benchmark.png):
           python scripts/plot_sort_benchmark.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, 'sort_benchmark.csv')
out_dir = os.path.join(script_dir, '..', 'docs', 'Fluid_Simulation', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sort_benchmark.png')

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'lines.linewidth': 2.0,
})

df = pd.read_csv(csv_path)

styles = {
    'Bitonic': {'color': 'black',  'marker': 'o', 'linestyle': '-',  'label': 'Bitonic sort'},
    'Radix':   {'color': 'gray',   'marker': 's', 'linestyle': '--', 'label': 'Radix sort'},
}

fig, ax = plt.subplots(figsize=(7, 5))

for alg, style in styles.items():
    sub = df[df['algorithm'] == alg].sort_values('n')
    if sub.empty:
        continue
    ax.plot(sub['n'], sub['time_ms'],
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            label=style['label'],
            markersize=7)

ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xlabel(r'Liczba elementów $N$')
ax.set_ylabel(r'Czas sortowania [ms]')
ax.set_title('Porównanie wydajności sortowania bitonicznego i radix na GPU')
ax.grid(True, which='both', linestyle=':', alpha=0.7)
ax.legend(loc='upper left', framealpha=1.0)

fig.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()
print(f'wrote: {out_path}')
