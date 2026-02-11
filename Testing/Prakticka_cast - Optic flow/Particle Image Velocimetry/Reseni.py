import cv2
import numpy as np
import glob
import os

# Zdroj: https://www.pivchallenge.org/pub/index.html#e
# Particle Image Velocimetry (PIV)

def OpticFlow_PIV(image_path1, image_path2, output_filename):
    # Načtení vstupních obrázků
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"CHYBA: Nelze načíst {image_path1}")
        return

    # Výpočet Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Vizualizace výsledku
    res_img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    img1_disp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_disp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Parametry vykreslování
    step = 16  # Krok mřížky
    scale = 5.0  # Zvětšení šipek
    h, w = img1.shape
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    fx = flow[y, x][:, 0]
    fy = flow[y, x][:, 1]

    # Kreslení šipek
    for i in range(len(x)):
        px, py = x[i], y[i]
        dx, dy = fx[i], fy[i]
        length = np.sqrt(dx ** 2 + dy ** 2)

        if 0.2 < length < 30:
            angle = np.arctan2(dy, dx) * 180 / np.pi
            hue = int((angle + 180) / 2)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

            end_x = int(px + dx * scale)
            end_y = int(py + dy * scale)
            cv2.arrowedLine(res_img, (px, py), (end_x, end_y), color, 1, tipLength=0.3)

    # Popisky
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_font = 1.0  # Velikost písma
    color_font = (0, 255, 255)
    thick = 2

    cv2.putText(img1_disp, "Obrazek 1 (Zacatek)", (30, 50), font, scale_font, color_font, thick)
    cv2.putText(img2_disp, "Obrazek 2 (Konec)", (30, 50), font, scale_font, color_font, thick)
    cv2.putText(res_img, "Vysledek: Optic Flow", (30, 50), font, scale_font, color_font, thick)

    # Spojení obrázků
    combo_image = np.hstack((img1_disp, img2_disp, res_img))

    # Uložení
    cv2.imwrite(output_filename, combo_image)
    return combo_image


def vytvoreni_vysledk(folder_path):
    search_pattern = os.path.join(folder_path, "*_1.tif")
    first_frames = sorted(glob.glob(search_pattern))

    if not first_frames:
        print("Nenalezeny žádné soubory '_1.tif'.")
        return

    # Složka pro kombinované výsledky
    output_dir = os.path.join(folder_path, "Vysledky")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Nalezeno {len(first_frames)}. Vytváří se výsledky...")

    for i, frame1_path in enumerate(first_frames):
        frame2_path = frame1_path.replace("_1.tif", "_2.tif")

        if not os.path.exists(frame2_path):
            continue

        base_name = os.path.basename(frame1_path)
        result_name = "Vysledek_" + base_name.replace("_1.tif", ".png")
        output_path = os.path.join(output_dir, result_name)

        print(f"[{i + 1}/{len(first_frames)}] Spojuji: {result_name}")

        # Zpracování
        combo = OpticFlow_PIV(frame1_path, frame2_path, output_path)

        if combo is not None:

            scale = 0.4
            width = int(combo.shape[1] * scale)
            height = int(combo.shape[0] * scale)
            dim = (width, height)

            resized_preview = cv2.resize(combo, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow('Ukazka OpticFlow', resized_preview)
            cv2.waitKey(200)

    print(f"\nHotovo! Výsledky vytvořeny ve:\n{output_dir}")
    cv2.destroyAllWindows()


if __name__ == "__main__":

    folder_path = r"C:\Users\dejme\PycharmProjects\DP_project\Testing\Prakticka_cast - Optic flow\Particle Image Velocimetry"

    if os.path.exists(folder_path):
        vytvoreni_vysledk(folder_path)
    else:
        print("Cesta neexistuje.")