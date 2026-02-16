# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# --- NASTAVENÍ SOUBORU ---
outAdr = './OUT/'
N = '251'  # Aplikujeme na pohled shora
input_file = os.path.join(outAdr, f'kamera_{N}.npy')

# --- 1. DETEKCE ---
LASER_AXIS = 0
THRESHOLD_VALUE = 25
PEAK_WINDOW = 10

# --- 2. PŘED-ZPRACOVÁNÍ ---
ENABLE_GAUSSIAN_BLUR = True
GAUSSIAN_KSIZE = (5, 5)

# --- 3. AGRESIVNÍ FILTRACE (ZDE JSOU ZMĚNY) ---

# A) OŘEZ PODLE ČASU (Osa X)
# Podle tvého grafu bordel končí na snímku 90.
CROP_START_INDEX = 90  # <--- TOTO SMAŽE TEN PRUH VLEVO
CROP_END_INDEX = 10  # Pro jistotu ořízneme i kousek konce

# B) OŘEZ PODLE VÝŠKY (Osa Z / Barva)
# Podle legendy je paleta žlutá (>200), podlaha fialová (<100).
ENABLE_Z_CROP = True
MIN_Z_THRESHOLD = 120  # <--- ZVÝŠENO Z 50 NA 120 (Odstraní podlahu mezi prkny)
MAX_Z_THRESHOLD = 400

# C) Morfologie a vyhlazení
ENABLE_MORPHOLOGY = True
MORPH_KERNEL_SIZE = 5

ENABLE_BILATERAL = True
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# D) Blob Filter (teď už bude fungovat, protože jsme přerušili spojení)
ENABLE_BLOB_FILTER = True


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


def keep_largest_blob(img_data):
    """
    Ponechá pouze největší souvislý objekt v obraze.
    """
    mask = (img_data > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels < 2:
        return img_data

    # stats: [x, y, width, height, area]
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    final_mask = (labels == largest_label)
    img_data[~final_mask] = 0

    print(f"  -> Blob Filter: Ponechán objekt {largest_label} (plocha {stats[largest_label, cv2.CC_STAT_AREA]} px)")
    return img_data


def main():
    print(f"--- START: {N} ---")

    if not os.path.exists(input_file):
        print(f"Soubor neexistuje: {input_file}")
        return

    snimky_3d_loaded = np.load(input_file)
    n_snimku = len(snimky_3d_loaded)
    print(f"Načteno {n_snimku} snímků.")

    depth_map = []

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
    print("Výpočet hotov. Aplikuji finální filtry...")

    # --- POST-PROCESSING ---
    viz_data = np.nan_to_num(scan_result).astype(np.float32)

    # 1. ČASOVÝ OŘEZ (Nejspolehlivější metoda na ten pruh vlevo)
    if CROP_START_INDEX > 0:
        viz_data[:, :CROP_START_INDEX] = 0
        print(f"  -> Oříznuto prvních {CROP_START_INDEX} snímků.")

    if CROP_END_INDEX > 0:
        viz_data[:, -CROP_END_INDEX:] = 0

    # 2. VÝŠKOVÝ OŘEZ (Odstraní podlahu)
    if ENABLE_Z_CROP:
        maska_pozadi = (viz_data < MIN_Z_THRESHOLD) | (viz_data > MAX_Z_THRESHOLD)
        viz_data[maska_pozadi] = 0
        print(f"  -> Oříznuta výška Z pod {MIN_Z_THRESHOLD} a nad {MAX_Z_THRESHOLD}")

    # 3. Morfologie (Zacelení)
    if ENABLE_MORPHOLOGY:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        viz_data = cv2.morphologyEx(viz_data, cv2.MORPH_CLOSE, kernel)

    # 4. Blob Filter (Teď už vyčistí jen zbytky)
    if ENABLE_BLOB_FILTER:
        print("  -> Hledám největší objekt...")
        viz_data = keep_largest_blob(viz_data)

    # 5. Vyhlazení
    if ENABLE_BILATERAL:
        viz_data = cv2.bilateralFilter(viz_data, 9, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    # --- VIZUALIZACE ---
    fig_width = 16

    if N == 'b351':
        fig_height = 3
    elif N == 'b251':
        fig_height = 6
    else:
        fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))

    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"Final Clean Sken: {N}", fontsize=16)
    plt.ylabel("Pozice [px]", fontsize=12)
    plt.xlabel("Číslo snímku", fontsize=12)

    cbar = plt.colorbar(img_plot)
    cbar.set_label('Z-Hloubka', rotation=90, labelpad=15)

    plt.tight_layout()

    filename = f"sken_FINAL_{N}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Uloženo: {filename}")
    plt.show()


if __name__ == "__main__":
    main()