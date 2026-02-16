# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:23:40 2025

@author: zuth
"""

import numpy as np
import matplotlib.pyplot as plt

#import cv2
#from time import time


outAdr='./OUT/'
N='251'
do=1000
od=0
snimky_3d_loaded = np.load(outAdr+f'kamera_{N}.npy')
snimky_loaded = [arr for arr in snimky_3d_loaded]
nSnimku=len(snimky_loaded)

maxJas = 0
plt.figure(1)
jasy = []

# Inicializace pole pro jasy
for i in range(nSnimku):
    jasy.append(None)

# --- ZMĚNA 1: Zapnout interaktivní režim ---
plt.ion()

for idx in range(nSnimku):

    if idx < od:
        continue

    if idx == do:
        break

    img_array = snimky_loaded[idx]

    jas = np.max(img_array)
    jasy[idx] = jas
    if jas > maxJas:
        maxJas = jas

    jasX = np.max(img_array, axis=1)
    jasXmean = np.mean(img_array, axis=1)
    jasY = np.linspace(0, len(jasX), len(jasX))

    plt.clf()  # Vymazat starý graf

    # 1. Podgraf: Snímek
    plt.subplot(1, 3, 1)
    plt.title(f'{idx} / {nSnimku}')
    plt.imshow(img_array, cmap='jet', origin='lower')

    # 2. Podgraf: Jasový průběh
    plt.subplot(1, 3, 2)
    plt.title('Jasovy prubeh')
    plt.plot(jasX, jasY, label='max')
    plt.plot(jasXmean, jasY, label='mean')
    plt.legend()
    plt.ylim([0, len(jasX)])
    plt.xlim([0, maxJas if maxJas > 0 else 255])  # Ošetření pro začátek
    plt.grid()

    # 3. Podgraf: Historie max jasu
    plt.subplot(1, 3, 3)
    plt.title(f' maxJas: {jas} / maxCelkove {maxJas}')
    plt.plot(jasy)
    plt.xlim([0, nSnimku])
    plt.grid()

    plt.tight_layout()

    # --- ZMĚNA 2: Pouze pause, show není uvnitř nutné ---
    plt.pause(0.001)

# Na konci vypnout interaktivní režim a zobrazit výsledek (aby se okno nezavřelo)
plt.ioff()
plt.show()