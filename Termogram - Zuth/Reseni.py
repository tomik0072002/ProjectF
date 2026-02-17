import cv2
import numpy as np
import os
import glob


def zpracuj_vsechny_obrazky():
    # 1. Příprava složky pro výsledky
    vystupni_slozka = "vysledky"
    if not os.path.exists(vystupni_slozka):
        os.makedirs(vystupni_slozka)
        print(f"📁 Vytvořena složka pro ukládání: {vystupni_slozka}")

    # 2. Hledání všech obrázků
    pripory = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
    seznam_obrazku = []
    for p in pripory:
        seznam_obrazku.extend(glob.glob(p))

    if not seznam_obrazku:
        print("❌ Žádné obrázky nenalezeny.")
        return

    print(f"🔎 Nalezeno {len(seznam_obrazku)} obrázků. Začínám zpracování...")

    # 3. Hlavní smyčka
    for cesta_k_obrazku in seznam_obrazku:
        print(f"➡️ Zpracovávám: {cesta_k_obrazku}")

        # Načtení
        img = cv2.imread(cesta_k_obrazku)
        if img is None:
            continue

        output = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # --- PŮVODNÍ LOGIKA DETEKCE ---

        # Identifikace panelů
        thresh_value = 110
        _, panels_binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        panels_binary = cv2.morphologyEx(panels_binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(panels_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maska_panelu = np.zeros_like(gray)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000: continue
            hull = cv2.convexHull(cnt)
            cv2.drawContours(maska_panelu, [hull], -1, 255, -1)

        maska_panelu = cv2.erode(maska_panelu, kernel, iterations=3)

        # Hledání hotspotů
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_tophat)

        _, defect_thresh = cv2.threshold(tophat, 50, 255, cv2.THRESH_BINARY)
        final_defects = cv2.bitwise_and(defect_thresh, defect_thresh, mask=maska_panelu)

        _, absolute_hot_thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
        absolute_hot_masked = cv2.bitwise_and(absolute_hot_thresh, absolute_hot_thresh, mask=maska_panelu)

        final_combined = cv2.bitwise_or(final_defects, absolute_hot_masked)

        # Vykreslení (Černé čtverečky)
        contours_defects, _ = cv2.findContours(final_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for d_cnt in contours_defects:
            area = cv2.contourArea(d_cnt)
            if area < 15: continue

            x, y, w, h = cv2.boundingRect(d_cnt)
            aspect = w / float(h)
            if aspect > 4 or aspect < 0.25: continue

            count += 1
            # Černý obdélník
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # --- ZOBRAZENÍ VEDLE SEBE (NOVÉ) ---

        # Spojíme originál (img) a výsledek (output) vedle sebe
        # np.hstack vyžaduje, aby měly oba obrázky stejnou výšku (což mají)
        porovnani = np.hstack((img, output))

        # Volitelné: Přidání popisků přímo do obrazu pro přehlednost
        cv2.putText(porovnani, "ORIGINAL", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Musíme vypočítat pozici pro druhý nápis (šířka jednoho obrázku + odsazení)
        sirka_obr = img.shape[1]
        cv2.putText(porovnani, "DETEKCE", (sirka_obr + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Uložení (Ukládáme jen výsledek s rámečky, ne to dvojité porovnání)
        nazev_souboru = os.path.basename(cesta_k_obrazku)
        cesta_ulozeni = os.path.join(vystupni_slozka, "res_" + nazev_souboru)
        cv2.imwrite(cesta_ulozeni, output)
        print(f"   ✅ Uloženo: {cesta_ulozeni} (Hotspotů: {count})")

        # Zobrazení
        nazev_okna = "Porovnani (Dalsi = MEZERNIK, Konec = Q)"
        cv2.namedWindow(nazev_okna, cv2.WINDOW_NORMAL)
        # Nastavíme širší okno, aby se tam vešly oba obrázky vedle sebe
        cv2.resizeWindow(nazev_okna, 1600, 700)

        cv2.imshow(nazev_okna, porovnani)

        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            print("Ukončuji skript...")
            break

    cv2.destroyAllWindows()
    print("HOTOVO.")


if __name__ == "__main__":
    zpracuj_vsechny_obrazky()