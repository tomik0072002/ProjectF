import cv2
import numpy as np

# Vytvoření funkce, která: načte obraz, provede segmentaci pomocí algoritmu Watershed,
# klasifikuje částice podle velikosti (percentily) a vygeneruje statistiku

def zpracovani(vstupni_cesta):
    print(f"--- Načítám originál: {vstupni_cesta} ---")

    # Načtení vstupního obrazu
    img_original = cv2.imread(vstupni_cesta)

    if img_original is None:
        print("Chyba: Vstupní obrázek nebyl nalezen.")
        return

    # Kopie pro vykreslování výsledných kontur (abych nepokreslil originál v paměti)
    img_pro_vykresleni = img_original.copy()

    # Převod na grayscale (odstíny šedi)
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    # Prahování (použito inverzní prahování, protože je tmavá čístice na světlém pozadí
    # + pro nalezení optimálního práhu Otsu prahování)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Odstranění šumu (morfologická oprace > otevření)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # Nalezení pozadí
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Nalezení popředí
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Neznámá oblast
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Značky pro Watershed
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # Algoritmus Watershed
    markers = cv2.watershed(img_pro_vykresleni, markers)

    # Analýza dat
    plochy = []
    finalni_kontury = []
    # Projde nalezené objekty
    unique_markers = np.unique(markers)
    for marker_id in unique_markers:
        if marker_id <= 1: continue
        # Vytvoří masku pro aktuální částici
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker_id] = 255
        # Najde její konturu
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = cnts[0]
            area = cv2.contourArea(c)
            if area > 5:
                plochy.append(area)
                finalni_kontury.append(c)

    # Klasifikace částic
    if len(plochy) > 0:
        # Pomocí percentil se vytvoří hranice
        h_mala = np.percentile(plochy, 33)
        h_stredni = np.percentile(plochy, 66)

        pocty = [0, 0, 0]  # Malé, střední, velké

        for cnt, area in zip(finalni_kontury, plochy):
            if area <= h_mala:
                color = (0, 255, 0)
                pocty[0] += 1
            elif area <= h_stredni:
                color = (255, 0, 0)
                pocty[1] += 1
            else:
                color = (0, 0, 255)
                pocty[2] += 1

            # Barevná kontura do obrazu
            cv2.drawContours(img_pro_vykresleni, [cnt], -1, color, 2)
    else:
        pocty = [0, 0, 0]
        h_mala, h_stredni = 0, 0

    # Okno s výsledky
    vyska_info, sirka_info = 400, 600
    img_statistika = np.ones((vyska_info, sirka_info, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_statistika, "Vysledek analyzy", (50, 50), font, 1, (0, 0, 0), 2)
    cv2.putText(img_statistika, f"Pocet castic: {len(plochy)}", (50, 100), font, 0.8, (0, 0, 0), 2)

    # Legenda
    cv2.circle(img_statistika, (40, 150), 10, (0, 255, 0), -1)
    cv2.putText(img_statistika, f"Male: {pocty[0]} (< nez {h_mala:.1f} pixelu)", (70, 155), font, 0.6, (0, 0, 0), 1)

    cv2.circle(img_statistika, (40, 200), 10, (255, 0, 0), -1)
    cv2.putText(img_statistika, f"Stredni: {pocty[1]}", (70, 205), font, 0.6, (0, 0, 0), 1)

    cv2.circle(img_statistika, (40, 250), 10, (0, 0, 255), -1)
    cv2.putText(img_statistika, f"Velke: {pocty[2]} (> nez {h_stredni:.1f} pixelu)", (70, 255), font, 0.6, (0, 0, 0), 1)

    # Uložení vyvořených souborů
    nazev_vystup_1 = "vysledek_1_obrazek.jpg"
    nazev_vystup_2 = "vysledek_2_statistika.jpg"

    cv2.imwrite(nazev_vystup_1, img_pro_vykresleni)
    cv2.imwrite(nazev_vystup_2, img_statistika)

    print(f"1. {nazev_vystup_1} (Obrázek s konturami)")
    print(f"2. {nazev_vystup_2} (Tabulka s textem)")

    # Zobrazení oken
    cv2.imshow("Vystup 1", img_pro_vykresleni)
    cv2.imshow("Vystup 2", img_statistika)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Spuštění funkce
zpracovani("synteticke_castice.png")