import cv2
import numpy as np
import matplotlib.pyplot as plt

# Zvýšíme rozlišení plátna (z 400 na 1024) pro hladší hrany
img_size = 1024
img = np.ones((img_size, img_size), dtype=np.uint8) * 50

# Parametry šestiúhelníku (zvětšené pro nové rozlišení)
center_coordinates = (img_size//2, img_size//2)
radius = 350
color = 200

# Výpočet vrcholů
theta = np.linspace(0, 2*np.pi, 7)
x_hex = center_coordinates[0] + radius * np.cos(theta)
y_hex = center_coordinates[1] + radius * np.sin(theta)
hexagon_pts = np.array([list(zip(x_hex, y_hex))], np.int32)

# Vykreslení
cv2.fillPoly(img, hexagon_pts, color)
cv2.circle(img, center_coordinates, 180, 50, -1)

# Šum a rozostření
noise = np.random.normal(0, 10, img.shape).astype(np.int16)
img_noisy = cv2.add(img.astype(np.int16), noise)
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

# Rozostření musí být větší kvůli většímu rozlišení
img_blurred = cv2.GaussianBlur(img_noisy, (9, 9), 0)

# Sobel
sobelx = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobelx**2 + sobely**2)
sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

# Canny
canny = cv2.Canny(img_blurred, 100, 200)

# Vykreslení (ořízneme okraje grafu, aby byl obrázek co největší)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Vstup (High-Res)")
plt.imshow(img_noisy, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobelův operátor")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Cannyho detektor")
plt.imshow(canny, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("detekce_hran_high_res.png", dpi=300)
plt.show()