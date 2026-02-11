import cv2
import numpy as np
import random

# Definování funkce pro generování obrazu
def synteticky_obrazek(vystup="synteticke_castice.jpg", sirka=800, vyska=600, pocet_castic=60):

    img = np.ones((vyska, sirka, 3), dtype=np.uint8) * 255

    # Generování částic (elipsy)
    for _ in range(pocet_castic):
        # Náhodný střed
        center_x = random.randint(20, sirka - 20)
        center_y = random.randint(20, vyska - 20)

        # Osy elips
        axis_major = random.randint(10, 35)
        axis_minor = random.randint(5, axis_major - 2)

        # Rotace
        angle = random.randint(0, 360)

        # Náhodná barva (odstíny šedé)
        intensity = random.randint(0, 50)
        color = (intensity, intensity, intensity)


        cv2.ellipse(img, (center_x, center_y), (axis_major, axis_minor), angle, 0, 360, color, -1)

    # Šum
    tecky_sumu = 3000
    for _ in range(tecky_sumu):
        x = random.randint(0, sirka - 1)
        y = random.randint(0, vyska - 1)

        # Náhodná velikost šumu
        radius = random.randint(1, 2)

        # Barva - šedá
        noise_intensity = random.randint(50, 150)
        noise_color = (noise_intensity, noise_intensity, noise_intensity)

        cv2.circle(img, (x, y), radius, noise_color, -1)

    # Rozmazání
    img_blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Uložení a zobrazení
    cv2.imwrite(vystup, img_blurred)
    print(f"Obrázek úspěšně uložen jako: {vystup}")

    cv2.imshow("Vygenerovany synteticky obrazek", img_blurred)
    print("Stiskněte libovolnou klávesu pro zavření okna.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Spuštění funkce
synteticky_obrazek()