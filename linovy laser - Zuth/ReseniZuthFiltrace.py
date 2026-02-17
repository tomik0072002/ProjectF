# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# --- NASTAVENÍ ---
outAdr = './OUT/'
N = 'b351'  # ZDE MĚNÍŠ NÁZEV SOUBORU (např. 'b351' nebo 'b251')
input_file = os.path.join(outAdr, f'kamera_{N}.npy')

# --- Parametry detekce a filtrace ---
LASER_AXIS = 0  # 0 = Laser je vertikálně, 1 = Laser je horizontálně
THRESHOLD_VALUE = 25  # Lehce jsem zvýšil práh (původně 20), pomůže odfiltrovat slabý šum
PEAK_WINDOW = 10  # Poloměr okna pro výpočet těžiště

# NOVÉ: Parametry filtrace
ENABLE_GAUSSIAN_BLUR = True  # Zapnout rozostření před detekcí (vyhladí vstup)
GAUSSIAN_KSIZE = (5, 5)  # Velikost jádra pro rozostření (musí být lichá čísla, např. 3x3, 5x5, 7x7)

ENABLE_MEDIAN_FILTER = True  # Zapnout mediánový filtr na výsledku (odstraní tečky)
MEDIAN_KSIZE = 5  # Velikost okna mediánu (musí být liché číslo > 1)


def get_laser_center_subpixel(img_slice):
    """
    Vypočítá přesné centrum laseru (subpixelová přesnost) v jednom řádku/sloupci.
    """
    img_slice = img_slice.astype(float)
    max_idx = int(np.argmax(img_slice))
    max_val = img_slice[max_idx]

    # Pokud je signál příliš slabý, vrátíme NaN (nedefinováno)
    if max_val < THRESHOLD_VALUE:
        return np.nan

    # Určení okna okolo maxima
    start = max(0, max_idx - PEAK_WINDOW)
    end = min(len(img_slice), max_idx + PEAK_WINDOW + 1)

    window_vals = img_slice[start:end]
    window_idxs = np.arange(start, end)

    total = np.sum(window_vals)
    if total == 0:
        return np.nan

    # Vážený průměr (těžiště)
    return np.sum(window_idxs * window_vals) / total


def main():
    print(f"--- START ---")
    print(f"Hledám soubor: {input_file}")

    # 1. Kontrola a načtení souboru
    if not os.path.exists(input_file):
        print(f"CHYBA: Soubor {input_file} neexistuje!")
        return

    snimky_3d_loaded = np.load(input_file)
    n_snimku = len(snimky_3d_loaded)
    print(f"Načteno {n_snimku} snímků. Zahajuji zpracování...")

    depth_map = []

    # 2. Zpracování snímků
    for i in range(n_snimku):
        img = snimky_3d_loaded[i]

        # a) Získání signálu (rozdíl barev nebo jas)
        if img.ndim == 3:
            laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1])
        else:
            laser_signal = img

        # b) NOVÉ: Předzpracování - Gaussovské rozostření
        # Vyhladí šum senzoru předtím, než začneme hledat maximum.
        if ENABLE_GAUSSIAN_BLUR:
            laser_signal = cv2.GaussianBlur(laser_signal, GAUSSIAN_KSIZE, 0)

        h, w = laser_signal.shape
        profile = []

        # c) Detekce profilu
        if LASER_AXIS == 0:
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)

        # Výpis průběhu
        if i % 50 == 0:
            print(f"Zpracováno {i} / {n_snimku}")

    # Převedení na numpy array a transpozice
    scan_result = np.array(depth_map).T
    print("Výpočet profilů hotov.")

    # 3. NOVÉ: Následné zpracování (Post-processing)
    viz_data = np.nan_to_num(scan_result)  # Převedeme NaN na nuly pro filtraci

    if ENABLE_MEDIAN_FILTER:
        print("Aplikuji mediánový filtr pro odstranění šumu...")
        # Mediánový filtr funguje nejlépe na datech typu uint8 nebo float32.
        # Pro jistotu převedeme data na float32.
        viz_data_float = viz_data.astype(np.float32)
        viz_data = cv2.medianBlur(viz_data_float, MEDIAN_KSIZE)
        print("Filtrace hotova.")

    # 4. Nastavení vizualizace
    print("Připravuji graf...")

    fig_width = 16
    fig_height = 6

    if N == 'b351':
        print(f"Režim zobrazení: Širokoúhlý (pro b351)")
        fig_height = 3
    elif N == 'b251':
        print(f"Režim zobrazení: Standardní (pro b251)")
        fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))

    # Vykreslení
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"Filtrovaný sken: {N}", fontsize=16)
    plt.ylabel("Pozice na senzoru [px]", fontsize=12)
    plt.xlabel("Číslo snímku (čas)", fontsize=12)

    # Přidání colorbaru
    cbar = plt.colorbar(img_plot)
    cbar.set_label('Intenzita/Výška', rotation=90, labelpad=15)

    plt.tight_layout()

    # Uložení do souboru
    output_filename = f"vysledny_sken_{N}_filtrovany.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--> Obrázek uložen jako: {output_filename}")

    # Zobrazení
    plt.show()


if __name__ == "__main__":
    main()