import cv2
import numpy as np

# Funkce pro rozpoznání tvaru
def rozpoznani_tvaru(contour, width_mm, height_mm):

    # Zjednodušení tvaru
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    num_vertices = len(approx)

    # Rozhodovací logika
    if num_vertices == 4:

        # Rozlišíme je podle poměru stran
        short_side = min(width_mm, height_mm)
        long_side = max(width_mm, height_mm)

        if long_side == 0: return "Chyba"

        aspect_ratio = short_side / long_side

        # Pokud je poměr stran větší než 0.90, považujeme to za čtverec
        if aspect_ratio > 0.90:
            return "Ctverec"
        else:
            return "Obdelnik"

    elif num_vertices > 6:
        # Mnoho vrcholů v aproximaci značí oblý tvar
        return "Kruh"

    else:
        # Tvary, které neuvažujeme
        return "Jiny tvar"


# Funkce pro určení a vypsání rozměrů
def rozmery_tvaru(image_path):
    # poměr pixelů ku milimetru (vymyšlené)
    PIXELS_PER_MM = 10.0
    # Korekce
    CORRECTION = 0.5

    print(f"--- ANALÝZA SOUBORU: {image_path} ---")

    # Načtení snímku z předchozího kroku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba: Obrázek '{image_path}' nenalezen. Spusť nejprve generátor.")
        return

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Filtrace a prahování
    blurred = cv2.medianBlur(gray, 9)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # Morfologie pro začištění
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Detekce kontur
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Detekováno objektů: {len(cnts)}")
    print("-" * 60)
    print(f"{'KLASIFIKACE':<12} | {'ROZMĚRY (mm)':<15} | {'POMĚR STRAN':<12}")
    print("-" * 60)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500: continue

        # Získání rozměrů
        rect = cv2.minAreaRect(c)
        (x_c, y_c), (w_px, h_px), angle = rect

        # Přepočet na mm a korekce
        dimA = (max(w_px, h_px) / PIXELS_PER_MM) - CORRECTION
        dimB = (min(w_px, h_px) / PIXELS_PER_MM) - CORRECTION

        # Ošetření záporných čísel
        dimA = max(0.1, dimA)
        dimB = max(0.1, dimB)

        # Identifikace tvaru
        # Voláme naši novou funkci
        shape_name = rozpoznani_tvaru(c, dimA, dimB)

        # Výpis
        ar = min(dimA, dimB) / max(dimA, dimB)
        print(f"{shape_name:<12} | {dimA:.2f} x {dimB:.2f}   | {ar:.2f}")

        # Vykreslení
        box = np.intp(cv2.boxPoints(rect))

        # Barvy
        if shape_name == "Ctverec":
            color = (0, 0, 255)
        elif shape_name == "Obdelnik":
            color = (0, 0, 255)
        elif shape_name == "Kruh":
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        if shape_name == "Kruh":
            radius_px = int((w_px + h_px) / 4)
            cv2.circle(output_image, (int(x_c), int(y_c)), radius_px, color, 3)
            diameter = (dimA + dimB) / 2
            label_dim = f"D = {diameter:.1f} mm"
        else:
            cv2.drawContours(output_image, [box], 0, color, 3)
            label_dim = f"{dimA:.1f}x{dimB:.1f}mm"

        # Název
        cv2.putText(output_image, shape_name, (int(x_c) - 40, int(y_c) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4)
        cv2.putText(output_image, shape_name, (int(x_c) - 40, int(y_c) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        #  Rozměry
        cv2.putText(output_image, label_dim, (int(x_c) - 40, int(y_c) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 4)
        cv2.putText(output_image, label_dim, (int(x_c) - 40, int(y_c) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow("Rozpoznane tvary", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


rozmery_tvaru('syntetika_random_sizes.png')