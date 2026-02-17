import cv2
import numpy as np
import os


def vytvor_masku_panelu(img_gray):
    """
    Tato funkce se snaží najít solární panely na obrázku a vytvořit černobílou masku,
    kde bílá = panel, černá = pozadí.
    """
    # 1. Rozmazání pro odstranění detailů (chceme jen hrubé obrysy panelů)
    blurred = cv2.GaussianBlur(img_gray, (11, 11), 0)

    # 2. Detekce hran (Canny)
    # Tresholdy 30/100 jsou nastaveny volněji, aby chytily obrysy panelů
    edges = cv2.Canny(blurred, 30, 100)

    # 3. Spojení přerušených hran (Dilatace)
    # Tím zajistíme, že obrys panelu bude uzavřený
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 4. Nalezení kontur
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vytvoření prázdné černé masky
    mask = np.zeros_like(img_gray)

    # Seřadíme kontury podle velikosti (od největší)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtr: Pokud je oblast příliš malá, není to panel (ignorujeme šum)
        # Hodnota 2000 je experimentální pro tvé rozlišení, možná bude třeba ladit
        if area < 2000:
            continue

        # Zjednodušení tvaru kontury (aproximace polygonu)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Pokud má tvar 4 rohy (nebo je prostě velký konvexní útvar), bereme ho jako panel
        # Použijeme convexHull, abychom vyhladili výřezy
        hull = cv2.convexHull(contour)

        # Vykreslíme nalezený panel bílou barvou do masky
        cv2.drawContours(mask, [hull], -1, 255, -1)

    return mask


def hledani_poruchy(cesta_k_obrazku):
    if not os.path.exists(cesta_k_obrazku):
        print(f"Soubor nenalezen: {cesta_k_obrazku}")
        return

    img = cv2.imread(cesta_k_obrazku)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- KROK 1: Vytvoření masky panelů (NOVÉ) ---
    panel_mask = vytvor_masku_panelu(gray)

    # --- KROK 2: Detekce hotspotů (Stejné jako minule) ---

    # Bilateral filter pro vyhlazení šumu uvnitř panelů
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Top-Hat transformace (zvýraznění světlých bodů oproti okolí)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray_filtered, cv2.MORPH_TOPHAT, kernel_tophat)

    # Prahování hotspotů
    _, defect_thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

    # Čištění drobných nečistot
    defect_thresh = cv2.erode(defect_thresh, None, iterations=1)
    defect_thresh = cv2.dilate(defect_thresh, None, iterations=2)

    # --- KROK 3: Aplikace masky (NOVÉ) ---
    # Logický součin (AND): Pixel musí být DETEKOVÁN JAKO VADA (defect_thresh)
    # A ZÁROVEŇ musí ležet UVNITŘ PANELU (panel_mask)
    final_defects_mask = cv2.bitwise_and(defect_thresh, defect_thresh, mask=panel_mask)

    # --- KROK 4: Vykreslení výsledků ---
    contours, _ = cv2.findContours(final_defects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_defects = 0
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtrace podle velikosti a tvaru hotspotu
        if area < 10 or area > 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        # Hotspoty jsou spíše kulaté/čtvercové, ne dlouhé čáry
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            continue

        found_defects += 1

        # Změření intenzity (teploty) v bodě
        roi = gray[y:y + h, x:x + w]
        min_val, max_val, _, _ = cv2.minMaxLoc(roi)

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, f"{int(max_val)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"Obrázek: {cesta_k_obrazku} | Nalezeno hotspotů: {found_defects}")

    # Zobrazení pro ladění
    # Zmenšíme okna, aby se vešla na obrazovku
    scale = 0.6
    h, w = img.shape[:2]
    new_dim = (int(w * scale), int(h * scale))

    cv2.imshow("1. Maska panelu (Kde hledame)", cv2.resize(panel_mask, new_dim))
    cv2.imshow("2. Top-Hat (Vsechny svetle body)", cv2.resize(tophat, new_dim))
    cv2.imshow("3. Vysledek", cv2.resize(output, new_dim))

    cv2.waitKey(0)


cv2.destroyAllWindows()

if __name__ == "__main__":
    # Zde doplň názvy svých souborů
    files = [
        "solar_color122_jpg.rf.f18eaa0a96e33fbab0902ce3a0e44931.jpg",
        "solar_color206_jpg.rf.d72f14ca5f13d75bed222349659f2cea.jpg",
        "solar_color622_jpg.rf.2b12f20d90aa701bfd9b2dbbe2a08025.jpg"
    ]

    for f in files:
        hledani_poruchy(f)