import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Nastavení

# Cesta ke složce s obázky
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, 'orig')
INPUT_MASK = os.path.join(DATA_FOLDER, '*.png')

LASER_AXIS = 0  # 0 = Laser je vertikálně, 1 = laser je horizontálně
THRESHOLD_VALUE = 50  # Minimální jas laseru
PEAK_WINDOW = 10  # Okno pro těžiště


# Funkce, která hledá pixel s nejvyšším jasem ve 3 krocích >> zvýší se přesnst
    #  1. nadje body s nejvyšším jasem, musí mít vyšší jas než THRESHOLD_VALUE
    #  2. použije PEAK_WINDOW, aby se podíval 10 pixelů do okolí bodu
    #  3. výpočet těžiště
def get_laser_center_subpixel(img_slice):
    img_slice = img_slice.astype(float)
    max_idx = int(np.argmax(img_slice))
    max_val = img_slice[max_idx]

    if max_val < THRESHOLD_VALUE:
        return np.nan

    start = max(0, max_idx - PEAK_WINDOW)
    end = min(len(img_slice), max_idx + PEAK_WINDOW + 1)

    window_vals = img_slice[start:end]
    window_idxs = np.arange(start, end)

    total = np.sum(window_vals)
    if total == 0: return np.nan

    return np.sum(window_idxs * window_vals) / total

# HLavní smyčka
def main():
    print(f"Načítám obrázky z: {INPUT_MASK}")
    files = sorted(glob.glob(INPUT_MASK))

    if not files:
        print("Obrázky nenalezeny.")
        print(f"Hledaná složka: {DATA_FOLDER}")
        return

    print(f"Zpracovávám {len(files)} snímků...")

    depth_map = []

    # Smyčka s OpenCV
    for i, fpath in enumerate(files):
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None: continue

        # Červený kanál minus Zelený
        laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1])
        h, w = laser_signal.shape
        profile = []

        if LASER_AXIS == 0:
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)

        if i % 50 == 0:
            print(f"Hotovo {i}/{len(files)}")

    # Převedeme na matici a otočíme (Transpozice), aby osy seděly
    scan_result = np.array(depth_map).T
    print("Výpočet hotov. Vykresluji a ukládám graf...")

    # Vizualizace

    # Nastavení velikosti okna
    plt.figure(figsize=(16, 5))

    viz_data = np.nan_to_num(scan_result)

    # Vykreslení
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    # Titulky a popisky os
    plt.title("Laserová linie", fontsize=16)
    plt.ylabel("Šířka [px]", fontsize=12)
    plt.xlabel("Délka [px]", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(img_plot)
    cbar.set_label('Výška [px]', rotation=90, labelpad=15)

    plt.tight_layout()

    # Uložení výsledku
    output_filename = "vysledny_sken.png"
    # dpi=300 zajistí vysokou kvalitu, bbox_inches='tight' ořízne zbytečné bílé okraje
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--> Obrázek uložen jako: {output_filename}")

    # Zobrazení na obrazovku
    plt.show()


if __name__ == "__main__":
    main()
