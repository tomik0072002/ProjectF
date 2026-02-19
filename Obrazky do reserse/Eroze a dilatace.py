import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

height, width = 600, 1200
img = np.zeros((height, width), dtype=np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, 'J', (480, 500), font, 15, 255, 45)

def add_noise(image):
    noisy_img = image.copy()
    h, w = image.shape

    for _ in range(150):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        if image[y, x] == 0:
            cv2.circle(noisy_img, (x, y), random.randint(2, 4), 255, -1)


    for _ in range(30):
        x = random.randint(400, 800)
        y = random.randint(100, 500)
        if image[y, x] == 255:

            pt1 = (x, y)
            pt2 = (x + random.randint(-20, 20), y + random.randint(-20, 20))
            cv2.line(noisy_img, pt1, pt2, 0, 6)

    return noisy_img


img = add_noise(img)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

erosion = cv2.erode(img, kernel, iterations=1)

dilation = cv2.dilate(img, kernel, iterations=1)


plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Vstup", fontsize=24)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Eroze", fontsize=24)
plt.imshow(erosion, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Dilatace", fontsize=24)
plt.imshow(dilation, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("Eroze_vs_Dilatace.png", dpi=300)
plt.show()