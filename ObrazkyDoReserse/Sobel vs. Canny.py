import cv2
import numpy as np
import matplotlib.pyplot as plt

img_size = 1024
img = np.ones((img_size, img_size), dtype=np.uint8) * 50

center_coordinates = (img_size//2, img_size//2)
radius = 350
color = 200

theta = np.linspace(0, 2*np.pi, 7)
x_hex = center_coordinates[0] + radius * np.cos(theta)
y_hex = center_coordinates[1] + radius * np.sin(theta)
hexagon_pts = np.array([list(zip(x_hex, y_hex))], np.int32)

cv2.fillPoly(img, hexagon_pts, color)
cv2.circle(img, center_coordinates, 180, 50, -1)

noise = np.random.normal(0, 10, img.shape).astype(np.int16)
img_noisy = cv2.add(img.astype(np.int16), noise)
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

img_blurred = cv2.GaussianBlur(img_noisy, (9, 9), 0)

sobelx = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = np.sqrt(sobelx**2 + sobely**2)
sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

canny = cv2.Canny(img_blurred, 100, 200)



plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Vstup", fontsize=24)
plt.imshow(img_noisy, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobelův operátor", fontsize=24)
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Cannyho detektor", fontsize=24)
plt.imshow(canny, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("detekce_hran_high_res.png", dpi=300)
plt.show()