import cv2
import numpy as np

# Je tady změna v hledání středů částic, místo globálního přístupy se hledají lokální maxima
# Rozpoznává částice lépe než předchozí metoda ale v pár případech nadje v jedné částici víc než jeden střed
# a to vede k její "rozpůlení"

def vyhodnoceni(cesta_k_obrazku):

    img = cv2.imread(cesta_k_obrazku)

    if img is None:
        print("Chyba: Obrázek nenalezen.")
        return

    vysledek = img.copy()

    # Předzpracování
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Odstranění šumu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Hledání středů částic

    # A) Normalizace
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # B) Nalezení lokálních maxim
    local_max = cv2.dilate(dist_norm, np.ones((20, 20), np.uint8))

    seeds_mask = (dist_norm == local_max) & (dist_norm > 40)

    # Převedení na uint8 pro hledání komponent
    seeds_uint8 = np.uint8(seeds_mask) * 255

    cv2.imshow('Nalezene stredy', seeds_uint8)

    # Markery
    ret, markers = cv2.connectedComponents(seeds_uint8)

    markers = markers + 1

    # Pro přesný watershed definujeme neznámou oblast:
    sure_fg = np.uint8(seeds_mask) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)
    markers[unknown == 255] = 0

    # Spuštění Watershed algoritmu
    markers = cv2.watershed(img, markers)

    #  Vykreslení kontur
    labels = np.unique(markers)
    pocet_objektu = 0

    for label in labels:
        if label <= 1: continue  # 0 je hranice, 1 je pozadí

        # Maska pro aktuální objekt
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255

        # Najdeme kontury pro tento jeden objekt
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            # Filtrace malých smítek (volitelné)
            area = cv2.contourArea(c)
            if area < 50: continue

            pocet_objektu += 1

            # Klasifikace podle velikosti (jako ve vašem příkladu)
            color = (0, 255, 0)  # Malé - Zelená
            if area > 500: color = (255, 0, 0)  # Střední - Modrá
            if area > 1500: color = (0, 0, 255)  # Velké - Červená

            cv2.drawContours(vysledek, [c], -1, color, 2)

    # Výpis výsledku
    text = f"Nalezeno: {pocet_objektu}"
    cv2.putText(vysledek, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    print(f"Analýza dokončena. {text}")

    cv2.imshow('Vysledek detekce', vysledek)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



vyhodnoceni('synteticke_castice.jpg')