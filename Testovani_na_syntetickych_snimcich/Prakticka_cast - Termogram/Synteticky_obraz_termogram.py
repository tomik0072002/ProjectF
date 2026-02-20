import cv2
import numpy as np

# Funkde na vytvoření syntetického termogramu "solárního panelu"
def synteticky_termogram(nazev_souboru="solar_termogram.jpg"):
    # Nastavení rozměrů termogramu
    width, height = 640, 480

    # Vytvoření pozadí - simulace normální teploty panelu
    image = np.full((height, width), 100, dtype=np.uint8)

    # Šum
    noise = np.random.normal(0, 5, (height, width)).astype(np.uint8)
    image = cv2.add(image, noise)

    # Vykreslení mřížky článků solárního panelu
    cell_w, cell_h = 80, 120
    for x in range(0, width, cell_w):
        cv2.line(image, (x, 0), (x, height), (50), 2)  # Tmavší čáry
    for y in range(0, height, cell_h):
        cv2.line(image, (0, y), (width, y), (50), 2)


    # Vytvoříme "poruch"
    mask = np.zeros_like(image)

    pocet_hotspotu = 2
    coords_log = []  # Pro výpis do konzole

    print(f"Generuji {pocet_hotspotu} hotspoty...")

    for i in range(pocet_hotspotu):
        # Náhodná pozice
        hx = np.random.randint(80, width - 80)
        hy = np.random.randint(80, height - 80)
        coords_log.append((hx, hy))

        # Změna velikost poruchy
        radius = np.random.randint(12, 18)

        cv2.circle(mask, (hx, hy), radius, (255), -1)
        print(f" -> Hotspot {i + 1}: pozice [{hx}, {hy}], poloměr {radius}")

    # Gaussovské rozostření poruchy
    mask = cv2.GaussianBlur(mask, (45, 45), 0)


    final_gray = cv2.add(image, mask)


    #  Barevné palety
    thermal_image = cv2.applyColorMap(final_gray, cv2.COLORMAP_INFERNO)

    # Uložení
    cv2.imwrite(nazev_souboru, thermal_image)

    # Zobrazení
    cv2.imshow("Synteticky Termogram - 2 Hotspoty", thermal_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    synteticky_termogram()