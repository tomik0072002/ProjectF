import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

INPUT_FOLDER = '.'
OUTPUT_FOLDER = 'vysledky'


def get_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detekci hran Canny s nízkým prahem
    canny = cv2.Canny(blurred, 10, 100)

    # Dilatace (roztažení)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    canny_connected = cv2.dilate(canny, kernel_connect, iterations=2)

    # Vyplnění děr uvnitř uzavřených tvarů
    contours_c, _ = cv2.findContours(canny_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_metal = np.zeros_like(gray)
    cv2.drawContours(mask_metal, contours_c, -1, 255, thickness=cv2.FILLED)

    # Adaptivní threshold (Otsu)
    _, mask_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sloučení masek
    combined = cv2.bitwise_or(mask_metal, mask_dark)

    # Finální začištění
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    return final_mask


def analyze_exact_shape(contour):

    # Určení plochy
    exact_area = cv2.contourArea(contour)

    if exact_area < 50: return None

    # Rozměry částic
    rect = cv2.minAreaRect(contour)
    (center, (w, h), angle) = rect

    length = max(w, h)
    width = min(w, h)
    aspect_ratio = length / width if width > 0 else 0

    return {
        "area": exact_area,
        "length": length,
        "width": width,
        "ar": aspect_ratio,
        "rect": rect,
        "center": center
    }


def process_batch_exact_fix(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = glob.glob(os.path.join(input_folder, "*.png")) + glob.glob(os.path.join(input_folder, "*.jpg"))
    print(f"Zpracovávám {len(files)} souborů...")

    for file_path in files:
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path)
        if img is None: continue

        mask = get_mask(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated_img = img.copy()
        data_list = []

        # Seřazení kontur
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for i, cnt in enumerate(contours):
            stats = analyze_exact_shape(cnt)
            if stats is None: continue

            data_list.append({
                "ID": i + 1,
                "Plocha (px)": int(stats["area"]),
                "Délka (px)": round(stats["length"], 1),
                "Šířka (px)": round(stats["width"], 1),
                "Poměr": round(stats["ar"], 2)
            })

            # Vykreslení
            # Tvar částice >> plocha částice
            cv2.drawContours(annotated_img, [cnt], -1, (0, 0, 255), 1)

            # Opsaný obdélník >> rozměry částice
            box = cv2.boxPoints(stats["rect"])
            box = np.int64(box)

            cv2.drawContours(annotated_img, [box], 0, (0, 255, 0), 2)

            cx, cy = stats["center"]
            cv2.putText(annotated_img, str(i + 1), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Export texťáku
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        ax[0].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Soubor: {filename}")
        ax[0].axis('off')

        ax[1].axis('off')
        ax[1].set_title("Plocha)")

        if data_list:
            df = pd.DataFrame(data_list)
            vis_df = df.head(15)
            table = ax[1].table(cellText=vis_df.values, colLabels=vis_df.columns, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            for (r, c), cell in table.get_celld().items():
                if r == 0: cell.set_facecolor('#dddddd')
        else:
            ax[1].text(0.5, 0.5, "Žádné částice", ha='center')

        out_file = os.path.join(output_folder, f"FIXED_{filename}")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.close(fig)
        print(f"Uloženo: {out_file}")


# Spuštění
if __name__ == "__main__":
    process_batch_exact_fix(INPUT_FOLDER, OUTPUT_FOLDER)