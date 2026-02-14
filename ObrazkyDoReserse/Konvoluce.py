import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 1. NASTAVENÍ DAT
input_grid = np.array([
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50]
])

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

row, col = 2, 2

# 2. VÝPOČET
output_grid = np.zeros_like(input_grid)
sub_matrix = input_grid[row - 1:row + 2, col - 1:col + 2]
calculation_result = np.sum(sub_matrix * kernel)
output_grid[row, col] = calculation_result

# 3. FUNKCE VYKRESLOVÁNÍ
def draw_grid(ax, data, title, highlight_coords=None, highlight_color='red', cell_format="{:.0f}", cmap='Blues',
              is_kernel=False, hide_zeros=False):

    ax.imshow(data, cmap=cmap, vmin=0, vmax=np.max(data) if np.max(data) > 0 else 1)

    r, c = data.shape
    ax.set_xticks(np.arange(c) - 0.5)
    ax.set_yticks(np.arange(r) - 0.5)
    ax.grid(which="major", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)

    for i in range(r):
        for j in range(c):
            val = data[i, j]
            text_str = cell_format.format(val)

            # --- ZMĚNA ZDE ---
            # Pokud máme skrývat nuly, vykreslíme text JEN pokud hodnota není 0.
            if hide_zeros and val == 0:
                text_str = ""
            # -----------------

            # Barva textu
            if is_kernel:
                text_color = "black" if val <= 0 else "white"
            else:
                text_color = "white" if np.abs(val) > np.max(np.abs(data)) / 2 else "black"
                # Pokud by se náhodou nula vykreslila (např. v kernelu), bude šedá
                if val == 0: text_color = "gray"

            ax.text(j, i, text_str, ha="center", va="center",
                    color=text_color, fontsize=14, fontweight='bold')

    if highlight_coords:
        hr, hc, hh, hw = highlight_coords
        rect = Rectangle((hc - 0.5, hr - 0.5), hw, hh, fill=False, edgecolor=highlight_color, linewidth=4)
        ax.add_patch(rect)


# 4. PLOTOVÁNÍ
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# Panel 1: Vstup
draw_grid(axs[0], input_grid, "Vstup",
          highlight_coords=(row - 1, col - 1, 3, 3))

# Panel 2: Kernel
draw_grid(axs[1], kernel, "Kernel (Filtrační maska)", highlight_color='red', cmap='Oranges', is_kernel=True)

# Panel 3: Výsledek (zapnuto hide_zeros)
draw_grid(axs[2], output_grid, "Výsledek", hide_zeros=True)

# 5. TEXT
col1_res = np.sum(sub_matrix[:, 0] * kernel[:, 0])
col2_res = np.sum(sub_matrix[:, 1] * kernel[:, 1])
col3_res = np.sum(sub_matrix[:, 2] * kernel[:, 2])

col1_calc = f"({sub_matrix[0, 0]}×{kernel[0, 0]}) + ({sub_matrix[1, 0]}×{kernel[1, 0]}) + ({sub_matrix[2, 0]}×{kernel[2, 0]})"
col2_calc = f"({sub_matrix[0, 1]}×{kernel[0, 1]}) + ({sub_matrix[1, 1]}×{kernel[1, 1]}) + ({sub_matrix[2, 1]}×{kernel[2, 1]})"
col3_calc = f"({sub_matrix[0, 2]}×{kernel[0, 2]}) + ({sub_matrix[1, 2]}×{kernel[1, 2]}) + ({sub_matrix[2, 2]}×{kernel[2, 2]})"

calculation_text = (
    f"Detailní výpočet pro zvýrazněný pixel (konvoluce):\n"
    f"1. Sloupec: {col1_calc} = {col1_res}\n"
    f"2. Sloupec: {col2_calc} =   {col2_res}\n"
    f"3. Sloupec: {col3_calc} =  {col3_res}\n"
    f"----------------------------------------------------------------------\n"
    f"Celkový součet: {col1_res} + {col2_res} + {col3_res} = {calculation_result}"
)

fig.text(0.5, 0.05, calculation_text, ha='center', va='bottom', fontsize=16,
         family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=1, pad=0.8))

plt.subplots_adjust(bottom=0.35)
plt.savefig("konvoluce.png", dpi=300)
plt.show()