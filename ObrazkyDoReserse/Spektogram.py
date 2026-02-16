import numpy as np
import matplotlib.pyplot as plt

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
plt.title("Časově-frekvenční spektrogram (Analýza signálu)") #
plt.ylabel('Frekvence [Hz]')
plt.xlabel('Čas [s]')
plt.colorbar(label='Intenzita (dB)')

# Šipka na poruchu
plt.annotate('Simulovaná porucha', xy=(1.05, 300), xytext=(1.3, 350),
             arrowprops=dict(facecolor='white', shrink=0.05), color='white')

plt.tight_layout()
plt.savefig("signal_to_image_spektrogram.png", dpi=300)
plt.show()