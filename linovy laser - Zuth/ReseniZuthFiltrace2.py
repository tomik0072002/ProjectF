# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# --- NASTAVENÍ ---
outAdr = './OUT/'
N = '251'  # Změň na 'b251' dle potřeby
input_file = os.path.join(outAdr, f'kamera_{N}.npy')

# --- DETEKCE ---
LASER_AXIS = 0
THRESHOLD_VALUE = 25
PEAK_WINDOW = 10

# --- PŘED-ZPRACOVÁNÍ (Vstupní signál) ---
ENABLE_GAUSSIAN_BLUR = True
GAUSSIAN_KSIZE = (5, 5)

# --- POKROČILÁ FILTRACE (Výsledný obraz) ---
# 1. Ořezání pozadí (Podle osy Z - viz legenda vpravo na grafu)
# Nastav tyto hodnoty podle toho, co ukazuje tvůj graf.
# Vše pod MIN a nad MAX bude smazáno (černá).
ENABLE_Z_CROP = True
MIN_Z_THRESHOLD = 50  # Odstraní podlahu/šum s nízkou hodnotou (např. fialové oblasti)
MAX_Z_THRESHOLD = 400  # Odstraní odlesky/strop s vysokou hodnotou

# 2. Morfologické uzavření (Zacelí díry v paletě)
ENABLE_MORPHOLOGY = True
MORPH_KERNEL_SIZE = 5  # Velikost "záplaty" na díry (čím větší, tím větší díry zalepí)

# 3. Bilaterální filtr (Hladké plochy, ostré hrany)
# Nahrazuje prostý Medián, je výpočetně náročnější, ale hezčí.
ENABLE_BILATERAL = True
BILATERAL_SIGMA_COLOR = 75  # Jak moc se liší barvy, aby se ještě průměrovaly
BILATERAL_SIGMA_SPACE = 75  # Jak daleko od sebe pixely ovlivňují výsledek


def get_laser_center_subpixel(img_slice):
    # (Stejná funkce jako minule)
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


def main():
    print(f"--- START: {N} ---")

    if not os.path.exists(input_file):
        print(f"Soubor neexistuje: {input_file}")
        return

    snimky_3d_loaded = np.load(input_file)
    n_snimku = len(snimky_3d_loaded)
    print(f"Načteno {n_snimku} snímků.")

    depth_map = []

    # 1. Výpočet profilů
    for i in range(n_snimku):
        img = snimky_3d_loaded[i]

        if img.ndim == 3:
            laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1])
        else:
            laser_signal = img

        if ENABLE_GAUSSIAN_BLUR:
            laser_signal = cv2.GaussianBlur(laser_signal, GAUSSIAN_KSIZE, 0)

        h, w = laser_signal.shape
        profile = []

        if LASER_AXIS == 0:
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)

        if i % 100 == 0: print(f"Zpracováno {i}/{n_snimku}...")

    scan_result = np.array(depth_map).T
    print("Základní výpočet hotov. Aplikuji pokročilé filtry...")

    # --- POST-PROCESSING ---

    # Práce s daty (převedeme na float32 pro filtry OpenCV)
    viz_data = np.nan_to_num(scan_result).astype(np.float32)

    # KROK A: Ořezání pozadí (Thresholding)
    if ENABLE_Z_CROP:
        # Vytvoříme masku: kde je hodnota mimo limity, nastavíme 0
        maska_pozadi = (viz_data < MIN_Z_THRESHOLD) | (viz_data > MAX_Z_THRESHOLD)
        viz_data[maska_pozadi] = 0
        print(f"Ořezáno Z mimo rozsah {MIN_Z_THRESHOLD}-{MAX_Z_THRESHOLD}")

    # KROK B: Morfologické uzavření (vyplnění děr)
    if ENABLE_MORPHOLOGY:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        # Closing = Dilatace následovaná Erozí
        viz_data = cv2.morphologyEx(viz_data, cv2.MORPH_CLOSE, kernel)
        print("Morfologické uzavření aplikováno.")

    # KROK C: Bilaterální filtr (nebo Medián)
    if ENABLE_BILATERAL:
        # Bilaterální filtr vyžaduje 32bit float
        viz_data = cv2.bilateralFilter(viz_data, 9, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        print("Bilaterální filtr aplikován.")
    else:
        # Fallback na Medián, pokud by Bilateral nefungoval dobře
        viz_data = cv2.medianBlur(viz_data, 5)

    # --- VIZUALIZACE ---
    fig_width = 16

    if N == 'b351':
        fig_height = 3
    elif N == 'b251':
        fig_height = 6
    else:
        fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))

    # Pokud jsme ořezali data na 0, chceme, aby 0 byla černá (nebo průhledná)
    # cmap='magma' má černou dole, což je fajn.
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"High-Quality Sken: {N}", fontsize=16)
    plt.ylabel("Pozice na senzoru [px]", fontsize=12)
    plt.xlabel("Číslo snímku", fontsize=12)

    cbar = plt.colorbar(img_plot)
    cbar.set_label('Z-Hloubka', rotation=90, labelpad=15)

    plt.tight_layout()

    filename = f"sken_HQ_{N}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {filename}")
    plt.show()


if __name__ == "__main__":
    main()