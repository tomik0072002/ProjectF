import cv2
import numpy as np


def hledani_poruchy(cesta_k_obrazku):
    # Načtení obrázku
    img = cv2.imread(cesta_k_obrazku)
    if img is None:
        print("Obrázek nenalezen.")
        return

    output = img.copy()

    # Převod na grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Rozmazání pro odstranění šumu
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hledání nejteplejších bodů >> body s nejvyšším jasem (Thresholding)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_blurred)
    print(f"Maximální nalezená teplota (hodnota pixelu): {max_val}")

    if max_val < 180:
        print("Žádná výrazná porucha nenalezena.")
        return

    # Aplikace prahu - vše nad 190 se stane bílou, zbytek černou
    _, thresh = cv2.threshold(gray_blurred, 190, 255, cv2.THRESH_BINARY)

    # Nalezení kontur poruch
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_defects = 0
    for contour in contours:
        # Filtrace šumuu
        area = cv2.contourArea(contour)
        if area > 50:
            found_defects += 1


            x, y, w, h = cv2.boundingRect(contour)

            # Vykreslení detekce poruch
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Přidání textu
            cv2.putText(output, "Porucha", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Zobrazení výsledků
    print(f"🔍 Detekováno potenciálních poruch: {found_defects}")

    # Co vidí počítač vs výsledek
    cv2.imshow("Maska (Prahovani)", thresh)
    cv2.imshow("Vysledek detekce", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Spustíme detekci na obrázku vytvořeném prvním skriptem
    hledani_poruchy("solar_termogram.jpg")