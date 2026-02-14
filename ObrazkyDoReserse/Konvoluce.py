import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 1. NASTAVENÍ DAT
# Vstupní matice (Input) - Svislá hrana mezi 3. a 4. sloupcem
input_grid = np.array([
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],  # <-- Tady probíhá naše ukázka
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50]
])

# Kernel (Jádro) - Detekce svislé hrany (Sobel)
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Pozice "okna" pro vizualizaci (střed)
row, col = 2, 2

# 2. VÝPOČET CELÉ KONVOLUCE (Feature Map)
# Abychom ukázali, že to funguje všude, vypočítáme to pro celou matici
output_grid = np.zeros_like(input_grid)
pad = 1  # Kernel je 3x3, takže okraj je 1 pixel
rows, cols = input_grid.shape

for i in range(pad, rows - pad):
    for j in range(pad, cols - pad):
        # Výřez 3x3
        region = input_grid[i - pad:i + pad + 1, j - pad:j + pad + 1]
        # Součet součinů
        output_grid[i, j] = np.sum(region * kernel)

# Pro vizualizaci konkrétního výpočtu si uložíme data z našeho zvoleného bodu
sub_matrix = input_grid[row - 1:row + 2, col - 1:col + 2]
calculation_result = output_grid[row, col]


# 3. VYKRESLOVACÍ FUNKCE
def draw_grid(ax, data, title, highlight_coords=None, highlight_color='red', cell_format="{:.0f}", cmap='Blues'):
    # Zobrazíme matici
    ax.imshow(data, cmap=cmap, vmin=0, vmax=np.max(data) if np.max(data) > 0 else 1)

    # Mřížka
    r, c = data.shape
    ax.set_xticks(np.arange(c) - 0.5)
    ax.set_yticks(np.arange(r) - 0.5)
    ax.grid(which="major", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15)

    # Čísla v buňkách
    for i in range(r):
        for j in range(c):
            val = data[i, j]
            # Barva textu podle pozadí (aby byla čitelná)
            text_color = "white" if np.abs(val) > np.max(np.abs(data)) / 2 else "black"
            # Pokud je hodnota 0 a není to kernel, zobrazíme ji šedě (méně nápadně)
            if val == 0 and title != "2. Kernel (Filtrační maska)":
                text_color = "gray"

            ax.text(j, i, cell_format.format(val), ha="center", va="center",
                    color=text_color, fontsize=14, fontweight='bold')

    # Červený rámeček (zvýraznění)
    if highlight_coords:
        hr, hc, hh, hw = highlight_coords
        rect = Rectangle((hc - 0.5, hr - 0.5), hw, hh, fill=False, edgecolor=highlight_color, linewidth=4)
        ax.add_patch(rect)


# 4. PLOTOVÁNÍ
fig, axs = plt.subplots(1, 3, figsize=(20, 8))  # Širší a vyšší plátno

# Panel 1: Input
draw_grid(axs[0], input_grid, "1. Vstupní obraz (Input)",
          highlight_coords=(row - 1, col - 1, 3, 3))

# Panel 2: Kernel
draw_grid(axs[1], kernel, "2. Kernel (Filtrační maska)",
          highlight_coords=(0, 0, 3, 3), highlight_color='red', cmap='Oranges')

# Panel 3: Output
# Zvýrazníme jeden pixel, ale vidět budou i ostatní vypočítané hodnoty!
draw_grid(axs[2], output_grid, "3. Výsledek (Feature Map)",
          highlight_coords=(row, col, 1, 1))

# 5. DETAILNÍ VÝPOČET (Textové pole dole)
# Rozepíšeme to po sloupcích, aby to bylo jasné
col1_calc = f"({sub_matrix[0, 0]}×{kernel[0, 0]}) + ({sub_matrix[1, 0]}×{kernel[1, 0]}) + ({sub_matrix[2, 0]}×{kernel[2, 0]})"
col2_calc = f"({sub_matrix[0, 1]}×{kernel[0, 1]}) + ({sub_matrix[1, 1]}×{kernel[1, 1]}) + ({sub_matrix[2, 1]}×{kernel[2, 1]})"
col3_calc = f"({sub_matrix[0, 2]}×{kernel[0, 2]}) + ({sub_matrix[1, 2]}×{kernel[1, 2]}) + ({sub_matrix[2, 2]}×{kernel[2, 2]})"

col1_res = np.sum(sub_matrix[:, 0] * kernel[:, 0])  # -40
col2_res = np.sum(sub_matrix[:, 1] * kernel[:, 1])  # 0
col3_res = np.sum(sub_matrix[:, 2] * kernel[:, 2])  # 200

# Sestavení textu
calculation_text = (
    f"Detailní výpočet pro zvýrazněný pixel (konvoluce):\n"
    f"1. Sloupec: {col1_calc} = {col1_res}\n"
    f"2. Sloupec: {col2_calc} =   {col2_res}\n"
    f"3. Sloupec: {col3_calc} =  {col3_res}\n"
    f"----------------------------------------------------------------------\n"
    f"Celkový součet: {col1_res} + {col2_res} + {col3_res} = {calculation_result}"
)

# Vložení textu do rámečku
fig.text(0.5, 0.02, calculation_text, ha='center', va='bottom', fontsize=16,
         family='monospace',  # Monospace písmo zarovná čísla pod sebe
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=1, pad=0.8))

plt.subplots_adjust(bottom=0.3)  # Uděláme místo dole pro text
plt.savefig("schema_konvoluce_vylepsene.png", dpi=300)
plt.show()