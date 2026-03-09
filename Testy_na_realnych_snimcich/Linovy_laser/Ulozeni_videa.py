# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:23:40 2025

@author: zuth
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

outAdr = './OUT/'
N = '251'
od = 125   # první snímek (včetně)
do = 320   # poslední snímek (včetně)

snimky_3d_loaded = np.load(outAdr + f'kamera_{N}.npy')
snimky_loaded = [arr for arr in snimky_3d_loaded]
nSnimku = len(snimky_loaded)

# Předvýpočet maxJas pro celý rozsah (aby osa X byla stabilní)
maxJas = 0
jasy = [None] * nSnimku
for i in range(od, do + 1):
    if i >= nSnimku:
        break
    jas = np.max(snimky_loaded[i])
    jasy[i] = jas
    if jas > maxJas:
        maxJas = jas

fig = plt.figure(figsize=(12, 4))
frames = list(range(od, min(do + 1, nSnimku)))

def make_frame(idx):
    plt.clf()

    img_array = snimky_loaded[idx]
    jas = np.max(img_array)

    jasX = np.max(img_array, axis=1)
    jasXmean = np.mean(img_array, axis=1)
    jasY = np.linspace(0, len(jasX), len(jasX))

    plt.subplot(1, 2, 1)
    plt.title(f'{idx} / {nSnimku}')
    plt.imshow(img_array, cmap='jet', origin='lower')

    plt.subplot(1, 2, 2)
    plt.title('Jasovy prubeh')
    plt.plot(jasX, jasY, label='max')
    plt.plot(jasXmean, jasY, label='mean')
    plt.legend(loc='upper right')
    plt.ylim([0, len(jasX)])
    plt.xlim([0, maxJas if maxJas > 0 else 255])
    plt.grid()

    plt.tight_layout()

print(f"Renderuji snímky {od}–{do} ({len(frames)} snímků)...")

ani = animation.FuncAnimation(fig, make_frame, frames=frames, repeat=False)

output_path = outAdr + f'video_{N}_snimky_{od}_{do}.mp4'
writer = animation.FFMpegWriter(fps=25, bitrate=1800)
ani.save(output_path, writer=writer)

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
print(f"Video uloženo: {output_path}")
plt.close()