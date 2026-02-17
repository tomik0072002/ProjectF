# Vycházím ze skriptu pro zpracování laseru ze simulace
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Tahání dat
outAdr = './OUT/'
N = 'b351'  # Název souboru ("251", "b351")
input_file = os.path.join(outAdr, f'kamera_{N}.npy')

# Parametry detekce
LASER_AXIS = 0  # 0 = Laser je vertikálně, 1 = Laser je horizontálně
THRESHOLD_VALUE = 20  # Minimální jas pro detekci (odfiltrování šumu)
PEAK_WINDOW = 10  # Poloměr okna pro výpočet těžiště


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
    if total == 0:
        return np.nan

    # Vážený průměr (těžiště)
    return np.sum(window_idxs * window_vals) / total


def main():
    print(f"START")
    print(f"Hledám soubor: {input_file}")

    # Kontrola a načtení souboru
    if not os.path.exists(input_file):
        print(f"CHYBA: Soubor {input_file} neexistuje!")
        return

    snimky_3d_loaded = np.load(input_file)
    n_snimku = len(snimky_3d_loaded)
    print(f"Načteno {n_snimku} snímků. Zahajuji zpracování...")

    depth_map = []

    # Zpracování snímků
    for i in range(n_snimku):
        img = snimky_3d_loaded[i]

        if img.ndim == 3:

            laser_signal = cv2.subtract(img[:, :, 2], img[:, :, 1])
        else:
            # Obraz už je jasová mapa !
            laser_signal = img

        h, w = laser_signal.shape
        profile = []

        # Procházení řezů obrazem
        if LASER_AXIS == 0:
            for y in range(h):
                profile.append(get_laser_center_subpixel(laser_signal[y, :]))
        else:
            for x in range(w):
                profile.append(get_laser_center_subpixel(laser_signal[:, x]))

        depth_map.append(profile)

        # Výpis průběhu zpracování snímků
        if i % 50 == 0:
            print(f"Zpracováno {i} / {n_snimku}")

    scan_result = np.array(depth_map).T
    print("Výpočet hotov. Připravuji graf...")

    # Vizualizace

    # Výchozí rozměry okna pro vizualizaci
    fig_width = 16
    fig_height = 6

    if N == 'b351':
        fig_height = 3

    elif N == 'b251':
        fig_height = 6

    plt.figure(figsize=(fig_width, fig_height))

    viz_data = np.nan_to_num(scan_result)

    # Vykreslení
    img_plot = plt.imshow(viz_data, cmap='magma', interpolation='nearest', aspect='auto', origin='lower')

    plt.title(f"Výsledný sken: {N}", fontsize=16)
    plt.ylabel("Pozice na senzoru [px]", fontsize=12)
    plt.xlabel("Číslo snímku", fontsize=12)

    # Přidání colorbaru
    cbar = plt.colorbar(img_plot)
    cbar.set_label('Výška', rotation=90, labelpad=15)

    plt.tight_layout()

    # Uložení do souboru
    output_filename = f"vysledny_sken_{N}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"--> Obrázek uložen jako: {output_filename}")

    # Zobrazení
    plt.show()


if __name__ == "__main__":
    main()