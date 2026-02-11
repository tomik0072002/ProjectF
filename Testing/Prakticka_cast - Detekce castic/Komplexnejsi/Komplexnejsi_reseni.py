import cv2
import numpy as np

# Hodně podobný jednoduššímu řešení s malými změny
# Hlavní algoritmus je Watershed pro rozdělení dotýkajících se částic (teoreticky)

def vyhodnoceni(cesta_k_obrazku):
    print(f"Načítám a analyzuji: {cesta_k_obrazku}...")
    img = cv2.imread("synteticke_castice.jpg")
    if img is None:
        print("Chyba: Obrázek nenalezen.")
        return

    vizualizace_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Prahování
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Odstranění šumu - otevření
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Příprava Watershed
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    faktor_separace = 0.2
    ret, sure_fg = cv2.threshold(dist_transform, faktor_separace * dist_transform.max(), 255, 0)
    cv2.imshow("DEBUG: Nalezene stredy (seeds)", sure_fg)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Značky (markery
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed algoritmus
    markers = cv2.watershed(img, markers)

    # Analýza dat
    plochy = []
    finalni_kontury = []
    unique_markers = np.unique(markers)

    for marker_id in unique_markers:
        if marker_id <= 1: continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker_id] = 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = cnts[0]
            area = cv2.contourArea(c)
            # Filtrace šumu
            if area > 10:
                plochy.append(area)
                finalni_kontury.append(c)

    pocet_castic = len(plochy)
    print(f"Nalezeno {pocet_castic} objektů.")

    # Klasifikace a vykreslení
    if pocet_castic > 0:
        hranice_mala = np.percentile(plochy, 33)
        hranice_stredni = np.percentile(plochy, 66)

        skupina_mala = 0
        skupina_stredni = 0
        skupina_velka = 0

        for cnt, area in zip(finalni_kontury, plochy):
            if area <= hranice_mala:
                color = (0, 255, 0)
                skupina_mala += 1
            elif area <= hranice_stredni:
                color = (255, 0, 0)
                skupina_stredni += 1
            else:
                color = (0, 0, 255)
                skupina_velka += 1

            cv2.drawContours(vizualizace_img, [cnt], -1, color, 2)

        # Prezentace výsledků
        info_img = np.ones((400, 500, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(info_img, "Vysledky analyzy", (30, 50), font, 0.8, (0, 0, 0), 2)
        cv2.putText(info_img, f"Nalezeno: {pocet_castic}", (30, 100), font, 0.8, (0, 0, 0), 2)

        cv2.circle(info_img, (30, 150), 10, (0, 255, 0), -1)
        cv2.putText(info_img, f"Male: {skupina_mala}", (50, 155), font, 0.6, (0, 0, 0), 1)

        cv2.circle(info_img, (30, 190), 10, (255, 0, 0), -1)
        cv2.putText(info_img, f"Stredni: {skupina_stredni}", (50, 195), font, 0.6, (0, 0, 0), 1)

        cv2.circle(info_img, (30, 230), 10, (0, 0, 255), -1)
        cv2.putText(info_img, f"Velke: {skupina_velka}", (50, 235), font, 0.6, (0, 0, 0), 1)

        cv2.imshow("1 - Detekce", vizualizace_img)
        cv2.imshow("2 - Statistika", info_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Spuštění funkce
vyhodnoceni('synteticke_castice.jpg')