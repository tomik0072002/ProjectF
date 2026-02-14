import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 1. Nastavení dat
# Vytvoříme jednoduchou vstupní matici (např. hrana: vlevo 10, vpravo 50)
input_grid = np.array([
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],  # <-- Tady proběhne výpočet
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50],
    [10, 10, 10, 50, 50, 50]
])

# Kernel (Jádro) - Detekce svislé hrany (Sobel-ish)
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Pozice "okna", které chceme vizualizovat (střed okna)
row, col = 2, 2  # Indexy ve vstupní matici (0-based), takže 3. řádek, 3. sloupec

# 2. Výpočet jednoho pixelu (konvoluce)
# Vyřízneme oblast 3x3 z Inputu
sub_matrix = input_grid[row - 1:row + 2, col - 1:col + 2]
# Element-wise násobení a suma
calculation = np.sum(sub_matrix * kernel)

# Vytvoříme prázdnou výstupní matici (jen pro vizualizaci)
output_grid = np.zeros_like(input_grid)
output_grid[row, col] = calculation


# 3. Funkce pro vykreslení mřížky s čísly
def draw_grid(ax, data, title, highlight_coords=None, highlight_color='red', cell_format="{:.0f}"):
    ax.imshow(data, cmap='Blues', vmin=0, vmax=np.max(data) if np.max(data) > 0 else 1)

    # Mřížka
    rows, cols = data.shape
    ax.set_xticks(np.arange(cols) - 0.5)
    ax.set_yticks(np.arange(rows) - 0.5)
    ax.grid(which="major", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="major", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Vepsání čísel do buněk
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            color = "white" if val > np.max(data) / 2 else "black"
            ax.text(j, i, cell_format.format(val), ha="center", va="center", color=color, fontsize=14,
                    fontweight='bold')

    # Zvýraznění (červený rámeček)
    if highlight_coords:
        r, c, h, w = highlight_coords
        rect = Rectangle((c - 0.5, r - 0.5), w, h, fill=False, edgecolor=highlight_color, linewidth=4)
        ax.add_patch(rect)


# 4. Vykreslení celého schématu
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Vstupní matice (Input)
draw_grid(axs[0], input_grid, "1. Vstupní obraz (Input)",
          highlight_coords=(row - 1, col - 1, 3, 3))  # Zvýrazníme 3x3 okolí

# Panel 2: Kernel
draw_grid(axs[1], kernel, "2. Kernel (Filtrační maska)",
          highlight_coords=(0, 0, 3, 3), highlight_color='red')  # Celý kernel je aktivní

# Panel 3: Výstup (Output)
# Zobrazíme jen nuly, ale s tím jedním vypočítaným číslem
# Zvýrazníme jen ten jeden pixel
draw_grid(axs[2], output_grid, "3. Výsledek (Feature Map)",
          highlight_coords=(row, col, 1, 1))

# 5. Přidání vysvětlujícího textu (Rovnice) pod obrázek
calculation_text = (
    f"Princip výpočtu pro zvýrazněný pixel:\n"
    f"Součet součinů (Input * Kernel) = \n"
    f"({sub_matrix[0, 0]}*{kernel[0, 0]}) + ({sub_matrix[0, 1]}*{kernel[0, 1]}) + ... + ({sub_matrix[2, 2]}*{kernel[2, 2]}) = {calculation}"
)
fig.text(0.5, 0.05, calculation_text, ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout(rect=[0, 0.1, 1, 1])  # Necháme místo dole pro text
plt.savefig("schema_konvoluce.png", dpi=300)
plt.show()