import cv2
import numpy as np
import random
import math

# Generování 2 čtverců, 2 obdélníků a 2 kruhů v náhodném pořadí o náhodných velikostí se šumem

def nahodne_tvary(filename, width=1000, height=700):

    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Barvy
    COLOR_WHITE = (255, 255, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 100, 0)

    # Mřížka
    cols = 3
    rows = 2
    cell_w = width // cols
    cell_h = height // rows

    safe_diameter = min(cell_w, cell_h) * 0.90
    min_dimension = 30

    object_types = ['square', 'square', 'rect', 'rect', 'circle', 'circle']
    random.shuffle(object_types)

    print(f"Rozměr buňky: {cell_w}x{cell_h}, Bezpečný průměr: {safe_diameter:.1f}")

    idx = 0
    for r in range(rows):
        for c in range(cols):
            cx = int(c * cell_w + cell_w / 2)
            cy = int(r * cell_h + cell_h / 2)
            center = (cx, cy)

            obj_type = object_types[idx]
            idx += 1

            angle = random.uniform(0, 360)

            # Náhodné obrazce
            if obj_type == 'square':

                max_side = safe_diameter / math.sqrt(2)

                # Náhodná strana mezi minimem a vypočteným maximem
                side = random.uniform(min_dimension, max_side)

                rect_struct = (center, (side, side), angle)
                box = cv2.boxPoints(rect_struct)
                cv2.drawContours(image, [np.intp(box)], 0, COLOR_WHITE, -1)
                print(f"Buňka [{r},{c}]: Čtverec, strana={side:.1f}, úhel={angle:.1f}")

            elif obj_type == 'rect':

                max_dim = safe_diameter / math.sqrt(2)

                # Generujeme náhodnou šířku a výšku
                w = random.uniform(min_dimension, max_dim)
                # Aby obdélník nebyl příliš úzká (aspoň polovina šířky)
                h = random.uniform(min_dimension, max_dim)

                rect_struct = (center, (w, h), angle)
                box = cv2.boxPoints(rect_struct)
                cv2.drawContours(image, [np.intp(box)], 0, COLOR_GREEN, -1)
                print(f"Buňka [{r},{c}]: Obdélník, w={w:.1f}, h={h:.1f}, úhel={angle:.1f}")

            elif obj_type == 'circle':

                max_radius = safe_diameter / 2
                radius = random.uniform(min_dimension / 2, max_radius)

                cv2.circle(image, center, int(radius), COLOR_BLUE, -1)
                print(f"Buňka [{r},{c}]: Kruh, R={radius:.1f}")

    # Šum
    noise = np.random.normal(0, 1.0, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    # Uložení
    cv2.imwrite(filename, noisy_image)
    print(f"\nVygenerován obrázek: {filename}")


nahodne_tvary('syntetika_random_sizes.png')