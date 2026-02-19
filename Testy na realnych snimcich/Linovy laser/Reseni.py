import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from typing import List

# Konfigurace
OUT_ADR = './OUT/'
FILE_NAME = '251'
INPUT_FILE = os.path.join(OUT_ADR, f'kamera_{FILE_NAME}.npy')


def get_laser_center_subpixel(img_slice: np.ndarray, threshold: int = 25, peak_window: int = 10) -> float:
    img_slice = img_slice.astype(float)
    max_idx = int(np.argmax(img_slice))
    max_val = img_slice[max_idx]

    if max_val < threshold:
        return np.nan

    start = max(0, max_idx - peak_window)
    end = min(len(img_slice), max_idx + peak_window + 1)

    window_vals = img_slice[start:end]
    window_idxs = np.arange(start, end)
    total = np.sum(window_vals)

    return np.sum(window_idxs * window_vals) / total if total != 0 else np.nan


def process_laser_scan(snimky_3d: np.ndarray, laser_axis: int = 0, enable_median: bool = True) -> np.ndarray:
    n_snimku = len(snimky_3d)
    depth_map = []

    for i in range(n_snimku):
        img = snimky_3d[i]
        laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1]) if img.ndim == 3 else img

        # Aplikace mediánu
        if enable_median:
            laser_signal = cv2.medianBlur(laser_signal, 3)  # Okno 3x3

        h, w = laser_signal.shape
        profile = []

        if laser_axis == 0:
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)
        if i % 50 == 0: print(f"Zpracováno {i} / {n_snimku}")

    return np.nan_to_num(np.array(depth_map).T)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"CHYBA: Soubor {INPUT_FILE} neexistuje!")
        return

    snimky_3d_loaded = np.load(INPUT_FILE)
    print(f"Načteno {len(snimky_3d_loaded)} snímků. Zahajuji zpracování...")

    # Vizualizace
    viz_data = process_laser_scan(snimky_3d_loaded)

    fig_height = 3 if FILE_NAME == 'b351' else 6
    plt.figure(figsize=(16, fig_height))
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"Filtrovaný sken: {FILE_NAME}", fontsize=16)
    plt.ylabel("Pozice na senzoru [px]", fontsize=12)
    plt.xlabel("Číslo snímku", fontsize=12)
    plt.colorbar(img_plot).set_label('Intenzita/Výška', rotation=90, labelpad=15)
    plt.tight_layout()

    output_filename = f"vysledny_sken_{FILE_NAME}_filtrovany.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--> Uloženo: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()