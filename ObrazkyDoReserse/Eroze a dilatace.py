import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Nastavení plátna
height, width = 600, 1200
img = np.zeros((height, width), dtype=np.uint8)

# 2. Vykreslení písmene "J" - použijeme zaoblený font a tloušťku tak akorát
font = cv2.FONT_HERSHEY_SIMPLEX
# Tloušťka 45 je ideální kompromis
cv2.putText(img, 'J', (480, 500), font, 15, 255, 45)


# 3. Funkce pro generování "chytrého" šumu
def add_noise(image):
    noisy_img = image.copy()
    h, w = image.shape

    # A) JEMNÝ PRACH v pozadí (pro Erozi)
    # Generujeme hodně malých teček (poloměr 2-4 px)
    # Jsou malé, takže eroze je snadno vymaže.
    for _ in range(150):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        # Kreslíme jen do černého pozadí
        if image[y, x] == 0:
            cv2.circle(noisy_img, (x, y), random.randint(2, 4), 255, -1)

    # B) PRASKLINY uvnitř objektu (pro Dilataci)
    # Generujeme černé čáry/praskliny uvnitř bílého písmena
    for _ in range(30):
        x = random.randint(400, 800)  # Oblast písmena
        y = random.randint(100, 500)
        if image[y, x] == 255:
            # Náhodná délka a úhel praskliny
            pt1 = (x, y)
            pt2 = (x + random.randint(-20, 20), y + random.randint(-20, 20))
            cv2.line(noisy_img, pt1, pt2, 0, 6)  # Tloušťka praskliny 6px

    return noisy_img


img = add_noise(img)

# 4. Morfologické operace s KULATÝM KERNELEM
# To je ten trik! Místo čtverce použijeme elipsu (kruh).
# Velikost 21x21 zajistí viditelný efekt, ale zachová tvar.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

# Eroze: Vymaže malé tečky v pozadí, ztenčí písmeno
erosion = cv2.erode(img, kernel, iterations=1)

# Dilatace: Zacelí praskliny, písmeno se zakulatí a ztloustne
dilation = cv2.dilate(img, kernel, iterations=1)

# 5. Vykreslení
plt.figure(figsize=(18, 6))

# Panel 1: Originál
plt.subplot(1, 3, 1)
plt.title("Vstup", fontsize=24)
plt.imshow(img, cmap='gray')
plt.axis('off')

# Panel 2: Eroze
plt.subplot(1, 3, 2)
plt.title("Eroze", fontsize=24)
plt.imshow(erosion, cmap='gray')
plt.axis('off')

# Panel 3: Dilatace
plt.subplot(1, 3, 3)
plt.title("Dilatace", fontsize=24)
plt.imshow(dilation, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig("Eroze_vs_Dilatace.png", dpi=300)
plt.show()