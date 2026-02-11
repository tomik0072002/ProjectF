import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

filename = "deformovane_tri_schody.png"

# Detekce profilu
def detekce_profilu(image_path, threshold=50):
    img = cv2.imread(image_path)
    if img is None: return None, None, None

    r_channel = img[:, :, 2]
    r_channel = cv2.GaussianBlur(r_channel, (5, 5), 0)

    h, w = r_channel.shape
    profile_x = []
    profile_y = []

    for x in range(w):
        col = r_channel[:, x].astype(float)
        mask = col > threshold
        if np.sum(mask) > 0:
            y_indices = np.arange(h)
            intensity_sum = np.sum(col[mask])
            weighted_sum = np.sum(col[mask] * y_indices[mask])
            center_y = weighted_sum / intensity_sum
            profile_x.append(x)
            profile_y.append(center_y)

    return img, np.array(profile_x), np.array(profile_y)


# Analýza hladin
def evaluate_multiple_levels(ys, min_distance=20):

    # Histogram
    hist_range = int(np.max(ys) - np.min(ys)) + 10
    hist, bin_edges = np.histogram(ys, bins=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # B) Detekce vrcholů >> find_peaks najde lokální maxima v datech (eliminuje rozkmit jedné hladiny)
    peaks, properties = find_peaks(hist, distance=min_distance, height=5)

    # Získání Y-pozice těchto vrcholů
    found_levels_y = bin_centers[peaks]

    # Seřazení nalezené úrovně podle Y souřadnice (sestupně)
    sorted_indices = np.argsort(found_levels_y)[::-1]
    sorted_levels = found_levels_y[sorted_indices]

    return sorted_levels, hist, bin_centers


# Získání profilu
original_img, xs, ys = detekce_profilu(filename, threshold=80)

if xs is not None:
    # Analýza hladin - Hledáme více úrovní s odstupem minimálně min_distance=x)
    levels, hist, bins = evaluate_multiple_levels(ys, min_distance=30)

    print(f"--- Výsledek měření ---")
    print(f"Nalezeno úrovní: {len(levels)}")

    if len(levels) > 0:
        # První hladina je vždy brána jako referenční základna
        base_level = levels[0]
        print(f"Základna (Podlaha): {base_level:.2f} px")

        # Výpočet výšky všech ostatních hladin vůči základně
        measured_heights = []
        for i in range(1, len(levels)):
            h_diff = base_level - levels[i]
            measured_heights.append(h_diff)
            print(f" -> Objekt {i} (úroveň {levels[i]:.2f} px): VÝŠKA {h_diff:.2f} px")
    else:
        print("Nenalezeny žádné úrovně.")
        base_level = 0

    # Vizualizace
    plt.figure(figsize=(10, 8))

    # Graf 1: Profil
    plt.subplot(2, 1, 1)
    if original_img is not None:
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.plot(xs, ys, 'c-', linewidth=1, alpha=0.6, label='Profil')

    # Vykreslení všech nalezených hladin
    colors = ['r', 'g', 'orange', 'm']
    for i, lvl in enumerate(levels):
        c = colors[i % len(colors)]
        label = "Základna" if i == 0 else f"Objekt {i} ({base_level - lvl:.1f}px)"
        plt.axhline(lvl, color=c, linestyle='--', linewidth=2, label=label)

    plt.title(f"Detekce profilu - Nalezeno {len(levels)} úrovní")
    plt.legend()

    # Graf 2: Histogram
    plt.subplot(2, 1, 2)
    plt.bar(bins, hist, width=1.0, color='gray', alpha=0.7, label='Histogram')

    for i, lvl in enumerate(levels):
        c = colors[i % len(colors)]
        plt.axvline(lvl, color=c, linestyle='--', linewidth=2)

    plt.xlabel("Y souřadnice (pixel)")
    plt.ylabel("Počet bodů")
    plt.title("Analýza histogramu (lokální maxima)")

    cv2.imshow("Originalni obrazek", original_img)

    plt.tight_layout()
    plt.show()

else:
    print("Chyba: Obrázek nenalezen nebo detekce selhala.")