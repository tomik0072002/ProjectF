import cv2
import numpy as np
import matplotlib.pyplot as plt

background = np.random.normal(50, 15, (300, 300))
object_ = np.random.normal(200, 15, (300, 300))
mask = np.zeros((300, 300))
cv2.circle(mask, (150, 150), 100, 1, -1)


img = np.where(mask == 1, object_, background).astype(np.uint8)


ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(img, cmap='gray')
axs[0].set_title("Vstup", fontsize=24)
axs[0].axis('off')

axs[1].hist(img.ravel(), 256)
axs[1].axvline(ret, color='r', linestyle='dashed', linewidth=2)
axs[1].set_title(f"Histogram a Otsu práh ({int(ret)})", fontsize=24)
axs[1].set_xlabel("Intenzita pixelu")
axs[1].set_ylabel("Počet pixelů")

axs[2].imshow(thresh, cmap='gray')
axs[2].set_title("Výsledek segmentace", fontsize=24)
axs[2].axis('off')

plt.tight_layout()
plt.savefig("histogram_otsu.png", dpi=300)
plt.show()