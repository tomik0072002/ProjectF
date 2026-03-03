import numpy as np
import matplotlib.pyplot as plt

# --- 1. Definice velikosti textu (v bodech) ---
# Tady si můžeš velikosti snadno upravit
LABEL_FONTSIZE = 16    # Popisky os (Čas, Frekvence, Intenzita)
TICK_FONTSIZE = 12     # Čísla na osách
ANNOTATE_FONTSIZE = 14 # Popis poruchy
# ---------------------------------------------

# 1. Generování syntetického signálu (vibrace)
# Signál se mění v čase (např. rozběh motoru)
fs = 1000  # Vzorkovací frekvence
t = np.linspace(0, 2, 2 * fs)
# Frekvence roste lineárně z 50 Hz na 150 Hz (chirp)
signal = np.sin(2 * np.pi * 50 * t + 20 * t**2)
# Přidáme "poruchu" - krátký ráz na vysoké frekvenci v čase 1.0s
signal[1000:1100] += 0.5 * np.sin(2 * np.pi * 300 * t[1000:1100])

# 2. Výpočet a vykreslení spektrogramu
plt.figure(figsize=(10, 5))
Pxx, freqs, bins, im = plt.specgram(signal, NFFT=256, Fs=fs, noverlap=128, cmap='inferno')

# --- Úpravy pro DP ---

# Odstraníme název grafu (zakomentováním)
# plt.title("Časově-frekvenční spektrogram (Analýza signálu)")

# Zvětšíme popisky os
plt.ylabel('Frekvence [Hz]', fontsize=LABEL_FONTSIZE)
plt.xlabel('Čas [s]', fontsize=LABEL_FONTSIZE)

# Zvětšíme čísla na osách
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)

# Zvětšíme popisek barevné škály
cbar = plt.colorbar()
cbar.set_label('Intenzita (dB)', fontsize=LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE) # I značky škály zvětšíme

# Zvětšíme šipku na poruchu a posuneme text, aby se vešel
plt.annotate('Simulovaná porucha', xy=(1.05, 300), xytext=(1.3, 370),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=6),
             color='white', fontsize=ANNOTATE_FONTSIZE)

plt.tight_layout()
# Uložíme s novým názvem
plt.savefig("spektrogram_zvetseny_text.png", dpi=300)
plt.show()