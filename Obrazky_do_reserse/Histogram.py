import numpy as np
import matplotlib.pyplot as plt

# 1. Generování syntetických dat
np.random.seed(42)
peak1 = np.random.normal(loc=50, scale=8, size=6000)
peak2 = np.random.normal(loc=105, scale=10, size=35000)  # Vyšší pík
peak3 = np.random.normal(loc=190, scale=7, size=5000)
data = np.concatenate([peak1, peak2, peak3])
data = data[(data >= 0) & (data <= 255)]

# 2. Nastavení grafu
fig, (ax_hist, ax_gradient) = plt.subplots(
    nrows=2,
    gridspec_kw={"height_ratios": [15, 1]},
    figsize=(8, 5)
)

# --- NOVÁ KLÍČOVÁ ČÁST: Pruhovaný histogram ---
counts, bins = np.histogram(data, bins=256, range=(0, 256))
bin_width = bins[1] - bins[0]
blue_color = '#1f77b4'

for i in range(len(counts)):
    bin_left = bins[i]
    count_value = counts[i]

    if i % 2 == 0:
        ax_hist.bar(bin_left, count_value, width=bin_width, align='edge',
                    color=blue_color, edgecolor='none')
    else:
        ax_hist.bar(bin_left, count_value, width=bin_width, align='edge',
                    color='none', edgecolor=blue_color, linewidth=0.5)

# --- 3. Nastavení os a vzhledu ---
ax_hist.set_xlim(0, 255)
ax_hist.set_ylim(bottom=0)

# PŘIDÁNO: Popis osy Y pro histogram
ax_hist.set_ylabel('Počet pixelů / Frekvence', fontsize=10)

ax_hist.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
ax_hist.tick_params(direction='in', right=True, top=True)

# --- 4. Vykreslení šedého přechodu ---
gradient = np.linspace(0, 255, 256).reshape(1, -1)
ax_gradient.imshow(gradient, aspect='auto', cmap='gray', extent=[0, 255, 0, 1])

ax_gradient.set_yticks([])
ax_gradient.set_xticks([0, 50, 100, 150, 200, 250])
ax_gradient.tick_params(direction='in', top=True)

# PŘIDÁNO: Popis osy X pro spodní přechod (který slouží jako osa X celého grafu)
ax_gradient.set_xlabel('Hodnota jasu (0 - 255)', fontsize=10)

# 5. Finální úpravy a uložení
plt.subplots_adjust(hspace=0)

plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
plt.show()