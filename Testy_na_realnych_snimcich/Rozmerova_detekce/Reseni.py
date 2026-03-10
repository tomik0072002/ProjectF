import cv2
import numpy as np
import sys
import os


# Konfigurace

CHECKERBOARD_IMAGE = "Obrazky/sachovnice1a.jpg"   # kalibrační šachovnice
OBJECT_IMAGE       = "Obrazky/lego2c.jpg"     # měřený předmět
OUTPUT_FOLDER      = "Vysledky"                # složka pro ukládání výsledků

# Parametry šachovnice

CHECKERBOARD_ROWS        = 23      # počet vnitřních rohů na výšku
CHECKERBOARD_COLS        = 32     # počet vnitřních rohů na šířku

CHECKERBOARD_SQUARE_MM   = 5.0   # velikost jednoho čtverce [mm]

# Detekce objektů
MIN_OBJECT_AREA_PX  = 500    # minimální plocha kontury v pixelech
MAX_OBJECTS         = 1      # max. počet výsledných objektů (0 = neomezeno)
MERGE_NEARBY        = True   # sloučit blízké kontury do jednoho objektu
MERGE_DISTANCE_PX   = 40     # vzdálenost slučování kontur v pixelech
CANNY_LOW           = 50     # spodní práh hranového detektoru Canny
CANNY_HIGH          = 150    # horní práh hranového detektoru Canny

# Zobrazení oken
WINDOW_MAX_WIDTH  = 1280    # maximální šířka okna
WINDOW_MAX_HEIGHT = 800     # maximální výška okna




def load_image(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(path, flags)
    if img is None:
        print(f"[CHYBA] Nelze načíst obrázek: {path}")
        sys.exit(1)
    return img


def show_image(title: str, img: np.ndarray):
    h, w = img.shape[:2]
    scale = min(WINDOW_MAX_WIDTH / w, WINDOW_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        preview = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        preview = img
    cv2.imshow(title, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def next_output_path(folder: str, base: str = "vysledek", ext: str = ".jpg") -> str:
    os.makedirs(folder, exist_ok=True)
    idx = 1
    while True:
        path = os.path.join(folder, f"{base}{idx}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


def _try_find_corners(gray: np.ndarray, pattern_size: tuple):
    flags_variants = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        0,
    ]

    def preprocessed_variants(g):
        yield g
        yield cv2.equalizeHist(g)
        yield cv2.GaussianBlur(g, (5, 5), 0)
        yield cv2.adaptiveThreshold(g, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yield clahe.apply(g)

    scales = [1.0, 0.75, 0.5, 0.25]

    for scale in scales:
        h, w = gray.shape
        if scale != 1.0:
            g_scaled = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            g_scaled = gray

        for variant in preprocessed_variants(g_scaled):
            for flags in flags_variants:
                ret, corners = cv2.findChessboardCorners(variant, pattern_size, flags)
                if ret:
                    return ret, corners, g_scaled, scale

    return False, None, gray, 1.0


def calibrate_from_checkerboard(image_path: str, rows: int, cols: int,
                                 square_size_mm: float):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pattern_size = (cols, rows)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    print("  Hledám rohy šachovnice ...")
    ret, corners, gray_scaled, found_scale = _try_find_corners(gray, pattern_size)

    if not ret:
        print("[CHYBA] Rohy šachovnice nebyly nalezeny.")
        sys.exit(1)


    corners_refined_scaled = cv2.cornerSubPix(
        gray_scaled, corners.astype(np.float32), (11, 11), (-1, -1), criteria
    )

    # přepočet rohů pro kalibraci kamery
    if found_scale != 1.0:
        corners_refined = corners_refined_scaled / found_scale
    else:
        corners_refined = corners_refined_scaled

    print(f"  Detekce uspesna (meritko: {found_scale:.2f})")

    # jednoobrázková kalibrace
    h, w = gray.shape
    ret_val, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners_refined], (w, h), None, None
    )

    # výpočet px/mm z projekce rohů
    proj, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], cam_mtx, dist)
    proj = proj.reshape(-1, 2)


    dists_x, dists_y = [], []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            dists_x.append(np.linalg.norm(proj[i + 1] - proj[i]))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            dists_y.append(np.linalg.norm(proj[i + cols] - proj[i]))

    px_per_mm_x = np.mean(dists_x) / square_size_mm
    px_per_mm_y = np.mean(dists_y) / square_size_mm
    px_per_mm = (px_per_mm_x + px_per_mm_y) / 2.0

    print(f"Kalibrace probehla uspesne.")
    print(f"     Rozliseni: {px_per_mm:.4f} px/mm  "
          f"(x: {px_per_mm_x:.4f}, y: {px_per_mm_y:.4f})")

    # vizualizace rohů šachovnice
    debug = img.copy()
    n_corners = corners_refined.reshape(-1, 2)
    for i, (cx, cy) in enumerate(n_corners):
        color = (
            int(255 * (i / len(n_corners))),
            int(255 * (1 - i / len(n_corners))),
            180
        )
        cv2.circle(debug, (int(cx), int(cy)), 18, color, -1)
        cv2.circle(debug, (int(cx), int(cy)), 18, (255, 255, 255), 2)
    show_image("Kalibrace - detekovane rohy (stiskni klavesu)", debug)

    return cam_mtx, dist, px_per_mm


def find_and_measure_objects(image_path: str, px_per_mm: float,
                              cam_mtx=None, dist=None):
    """
    Najde objekty v obraze a změří jejich rozměry.
    """
    img = load_image(image_path)


    if cam_mtx is not None and dist is not None:
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, cam_mtx, dist, None, new_mtx)
        x, y, rw, rh = roi
        if all(v > 0 for v in (x, y, rw, rh)):
            img = img[y:y + rh, x:x + rw]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Předzpracování
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)


    merge_k = max(5, MERGE_DISTANCE_PX) if MERGE_NEARBY else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_k, merge_k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    if MERGE_NEARBY:
        closed = cv2.dilate(closed, kernel, iterations=2)
        closed = cv2.erode(closed, kernel, iterations=2)

    # Hledání kontur
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[INFO] Zadne kontury nenalezeny.")
        return


    contours = [c for c in contours if cv2.contourArea(c) >= MIN_OBJECT_AREA_PX]

    if not contours:
        print("[INFO] Zadne kontury nad minimalni plochou. Zkuste snizit MIN_OBJECT_AREA_PX.")
        return

    # Omezení počtu nalezených ploch (od nějvětších)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if MAX_OBJECTS > 0:
        contours = contours[:MAX_OBJECTS]

    result = img.copy()
    print("\n── Nalezene objekty ────")
    for idx, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)

        # Ohraničující obdélník
        x, y, bw, bh = cv2.boundingRect(cnt)
        bw_mm = bw / px_per_mm
        bh_mm = bh / px_per_mm
        area_mm2 = area_px / (px_per_mm ** 2)

        # Rotovaný obdélník
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        rw_mm = rw / px_per_mm
        rh_mm = rh / px_per_mm

        # Obvod
        perimeter_px = cv2.arcLength(cnt, True)
        perimeter_mm = perimeter_px / px_per_mm

        print(f"\nObjekt #{idx + 1}")
        print(f"  Ohranicujici obdelnik (zarovnany):  {bw_mm:.1f} x {bh_mm:.1f} mm")
        print(f"  Rotovany obdelnik (min. plocha):    {max(rw_mm, rh_mm):.1f} x "
              f"{min(rw_mm, rh_mm):.1f} mm  (uhel: {angle:.1f} deg)")
        print(f"  Plocha kontury:                {area_mm2:.1f} mm2")
        print(f"  Obvod:                         {perimeter_mm:.1f} mm")


        box_pts = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(result, [box_pts], 0, (0, 255, 0), 3)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (255, 100, 0), 1)

        # Popisek
        lines = [
            f"Sirka:  {max(rw_mm, rh_mm):.1f} mm",
            f"Vyska:  {min(rw_mm, rh_mm):.1f} mm",
            f"Plocha: {area_mm2:.1f} mm2",
        ]
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness  = 2
        padding    = 10
        line_gap   = 8


        (_, line_h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
        row_step = line_h + line_gap
        block_h  = len(lines) * row_step + padding * 2
        block_w  = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0]
                       for l in lines) + padding * 2


        bx = max(x, 0)
        by = max(y - block_h - 6, 0)


        overlay = result.copy()
        cv2.rectangle(overlay, (bx, by), (bx + block_w, by + block_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # text
        for i, line in enumerate(lines):
            ty = by + padding + line_h + i * row_step
            cv2.putText(result, line, (bx + padding, ty),
                        font, font_scale, (0, 255, 180), thickness, cv2.LINE_AA)

    print("\n────────")

    out_path = next_output_path(OUTPUT_FOLDER)
    cv2.imwrite(out_path, result)
    print(f"[OK] Vysledny obrazek ulozen: {out_path}")

    show_image("Mereni objektu (stiskni klavesu)", result)


# Hlavní program

def main():
    # ověření souborů
    for path in (CHECKERBOARD_IMAGE, OBJECT_IMAGE):
        if not os.path.isfile(path):
            print(f"[CHYBA] Soubor nenalezen: {path}")
            print("  Zkontrolujte cesty v sekci KONFIGURACE na začátku skriptu.")
            sys.exit(1)

    print("=" * 60)
    print(" Měření rozměrů předmětu – OpenCV")
    print("=" * 60)
    print(f"\nKalibrace ze souboru:  {CHECKERBOARD_IMAGE}")
    print(f"Šachovnice:            {CHECKERBOARD_COLS} x {CHECKERBOARD_ROWS} vnitřních rohů")
    print(f"Velikost čtverce:      {CHECKERBOARD_SQUARE_MM} mm\n")

    cam_mtx, dist, px_per_mm = calibrate_from_checkerboard(
        CHECKERBOARD_IMAGE,
        CHECKERBOARD_ROWS,
        CHECKERBOARD_COLS,
        CHECKERBOARD_SQUARE_MM,
    )

    print(f"\nMěření objektu ze souboru: {OBJECT_IMAGE}\n")
    find_and_measure_objects(
        OBJECT_IMAGE, px_per_mm,
        cam_mtx=cam_mtx, dist=dist,
    )


if __name__ == "__main__":
    main()