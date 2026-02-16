import numpy as np
import matplotlib.pyplot as plt
import os
import cv2  # Stále importujeme pro případné operace, i když data načítáme z numpy

# --- Nastavení cest a souborů (Podle prvního skriptu) ---
outAdr = './OUT/'
N = '251'
input_file = os.path.join(outAdr, f'kamera_{N}.npy')

# --- Nastavení parametrů detekce ---
LASER_AXIS = 0  # 0 = Laser je vertikálně, 1 = laser je horizontálně
THRESHOLD_VALUE = 20  # Minimální jas laseru (možná bude potřeba upravit podle citlivosti kamery)
PEAK_WINDOW = 10  # Okno pro těžiště


# Funkce pro subpixelovou detekci (zůstává stejná)
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

    # Posuneme indexy relativně k oknu a přičteme start
    return np.sum(window_idxs * window_vals) / total


# Hlavní funkce
def main():
    print(f"Načítám data ze souboru: {input_file}")

    if not os.path.exists(input_file):
        print("Soubor .npy neexistuje! Zkontroluj cestu.")
        return

    # Načtení .npy souboru (stejně jako ve skriptu A)
    snimky_3d_loaded = np.load(input_file)

    # Převedení na list (pokud je potřeba, ale numpy array je lepší pro výkon)
    # Zde pracujeme přímo s numpy polem, je to rychlejší
    n_snimku = len(snimky_3d_loaded)
    print(f"Načteno {n_snimku} snímků. Zahajuji výpočet...")

    depth_map = []

    for i in range(n_snimku):
        img = snimky_3d_loaded[i]

        # --- Detekce signálu ---
        # Původní skript dělal: Red - Green.
        # Musíme zjistit, jestli je vstup barevný (3 dimenze) nebo černobílý (2 dimenze)

        if img.ndim == 3:
            # Je to barevné (H, W, Channels) -> Použijeme logiku odečítání kanálů
            # Pozor: OpenCV načítá BGR, ale Matplotlib/Numpy často RGB.
            # Zde předpokládáme pořadí jako v cv2.subtract(R, G)
            # Pokud by to nefungovalo, zkus jen převod na šedou: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Ošetření přetečení (numpy uint8 při odečítání může blbnout, cv2.subtract je bezpečné)
            laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1])
        else:
            # Je to černobílé (H, W) -> Použijeme přímo jasové hodnoty
            laser_signal = img

        h, w = laser_signal.shape
        profile = []

        if LASER_AXIS == 0:
            # Procházíme řádky
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            # Procházíme sloupce
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)

        if i % 50 == 0:
            print(f"Zpracováno {i}/{n_snimku}")

    # Převedeme na matici a otočíme
    # Tady záleží, jak chceš data vidět.
    # scan_result bude mít rozměry: [Počet pixelů na řádek] x [Počet snímků]
    scan_result = np.array(depth_map).T

    print("Výpočet hotov. Vykresluji...")

    # --- Vizualizace ---
    plt.figure(figsize=(16, 6))

    # Nahradíme NaN nuly pro zobrazení
    viz_data = np.nan_to_num(scan_result)

    # Vykreslení
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"Výsledný sken ze souboru {N}", fontsize=16)
    plt.ylabel("Pozice na senzoru [px]", fontsize=12)
    plt.xlabel("Číslo snímku (čas) [-]", fontsize=12)

    cbar = plt.colorbar(img_plot)
    cbar.set_label('Intenzita/Výška', rotation=90, labelpad=15)

    plt.tight_layout()

    # Uložení
    output_filename = f"vysledny_sken_{N}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--> Obrázek uložen jako: {output_filename}")

    plt.show()


if __name__ == "__main__":
    main()