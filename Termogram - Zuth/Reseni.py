import cv2
import numpy as np
import os


def zpracuj_termalni_snimek(cesta_k_obrazku):
    # Kontrola existence souboru
    if not os.path.exists(cesta_k_obrazku):
        print(f"Soubor nenalezen: {cesta_k_obrazku}")
        return

    # Načtení obrázku
    img = cv2.imread(cesta_k_obrazku)
    output = img.copy()

    # 1. Převod na odstíny šedi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Předzpracování - odstranění šumu
    # Bilateral filter je lepší než Gaussian, protože zachovává hrany (okraje hotspotu)
    gray_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Morphological Top-Hat transformace
    # Tato operace zvýrazní malé světlé objekty na tmavším pozadí (ideální pro hotspoty)
    # Velikost kernelu (15,15) určuje maximální velikost objektu, který hledáme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray_filtered, cv2.MORPH_TOPHAT, kernel)

    # 4. Prahování na výsledku Top-Hat
    # Nyní nehledáme absolutní jas 190, ale jas "vystupující" nad okolí
    # Použijeme Otsuho metodu pro automatické nalezení prahu, nebo fixní práh na tophat
    # Zde volím fixní práh na tophat vrstvu - experimentálně cca 30-50 rozdíl jasu
    _, thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)

    # 5. Čištění masky (Eroze + Dilatace pro odstranění malých teček šumu)
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 6. Nalezení kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_defects = 0
    print(f"--- Zpracování: {cesta_k_obrazku} ---")

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filtr 1: Velikost (ignorujeme příliš malé smítka a obrovské plochy)
        if area < 10 or area > 1000:
            continue

        # Získání ohraničujícího obdélníku
        x, y, w, h = cv2.boundingRect(contour)

        # Filtr 2: Tvar (Aspect Ratio)
        # Hotspoty jsou obvykle kulaté nebo čtvercové.
        # Pokud je poměr stran příliš velký (např. > 4), je to spíše čára/odlesk.
        aspect_ratio = float(w) / h
        if aspect_ratio > 3.0 or aspect_ratio < 0.33:
            continue

        # Pokud prošlo filtry, považujeme za poruchu
        found_defects += 1

        # Vykreslení: Červený obdélník kolem hotspotu
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Volitelně: Vypíšeme max teplotu (jas) uvnitř oblasti
        roi = gray[y:y + h, x:x + w]
        min_val, max_val, _, _ = cv2.minMaxLoc(roi)
        label = f"Hotspot ({max_val})"
        cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    print(f"🔍 Nalezeno hotspotů: {found_defects}")

    # Zobrazení výsledků
    # Ukážeme Top-Hat (co vidí algoritmus jako 'výstupky') a výsledek
    cv2.imshow("Top-Hat (Zvyrazneni)", tophat)
    cv2.imshow("Maska po prahovani", thresh)
    cv2.imshow("Vysledek detekce", output)

    print("Stiskni libovolnou klávesu pro další obrázek...")
    cv2.waitKey(0)


cv2.destroyAllWindows()

if __name__ == "__main__":
    seznam_obrazku = [
        "Obr1.jpg",
        "Obr2.jpg",
        "Obr3.jpg",
        "Obr4.jpg"
    ]

    for img_path in seznam_obrazku:
        zpracuj_termalni_snimek(img_path)