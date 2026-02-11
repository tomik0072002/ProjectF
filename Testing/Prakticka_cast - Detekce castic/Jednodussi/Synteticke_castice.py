import numpy as np
import cv2
import random
import math


# Parametry
WIDTH, HEIGHT = 2000, 2000
BACKGROUND = 255

# Počty částic
COUNT_X = 1000
COUNT_Y = 500
COUNT_Z = 100

# Velikosti (poloměry elips)
SIZE_X = (4, 7)
SIZE_Y = (10, 20)
SIZE_Z = (25, 45)


# Aby se vytvořené částice nepřekrývali
def overlaps(existing, x, y, r):
    for (ex, ey, er) in existing:
        dist = math.hypot(x - ex, y - ey)
        if dist < (r + er + 2):  # 2 px tolerance
            return True
    return False


# Generátor částic
def generator_castic(count, size_range, img, particle_list):
    placed = 0
    attempts = 0
    max_attempts = count * 20

    while placed < count and attempts < max_attempts:
        attempts += 1

        # Generování tvaru (morfologie) částice
        rx = random.randint(*size_range)
        ry = random.randint(*size_range)
        r = max(rx, ry)

        # Pozice částice v obraze
        x = random.randint(r + 5, WIDTH - r - 5)
        y = random.randint(r + 5, HEIGHT - r - 5)

        # Kontrola, zda místo není obsazeno jinou částicí
        if overlaps(particle_list, x, y, r):
            continue

        # Vykreslení částice
        angle = random.randint(0, 180)          # Orientace částice
        color = (0, 0, 0)                             # Barva čásice

        cv2.ellipse(img, (x, y), (rx, ry), angle, 0, 360, color, -1)

        particle_list.append((x, y, r))
        placed += 1

    print(f"Vytvořeno {placed}/{count} částic.")
    return img

# Vytvoření prázdného obrazu
img = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
particles = []

# Volání generátoru částic pro 3 velikosti částic
img = generator_castic(COUNT_X, SIZE_X, img, particles)
img = generator_castic(COUNT_Y, SIZE_Y, img, particles)
img = generator_castic(COUNT_Z, SIZE_Z, img, particles)

# Rozmazání
img = cv2.GaussianBlur(img, (5, 5), 0)

# Uložrní výsledku
cv2.imwrite("synteticke_castice.png", img)
print("Vytvořen: synteticke castice.png")
