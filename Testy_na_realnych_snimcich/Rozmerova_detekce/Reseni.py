"""
Měření rozměrů předmětu pomocí kalibrační šachovnice
=====================================================
Stačí upravit sekci KONFIGURACE níže a spustit skript:
    python measure_object.py
"""

import cv2
import numpy as np
import sys
import os


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        KONFIGURACE                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Upravte hodnoty níže podle svého nastavení.                    ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Cesty k obrázkům ────────────────────────────────────────────────
CHECKERBOARD_IMAGE = "Obrazky/sachovnice1a.jpg"   # fotografie kalibrační šachovnice
OBJECT_IMAGE       = "Obrazky/lego2c.jpg"     # fotografie měřeného předmětu
OUTPUT_FOLDER      = "Vysledky"                # složka pro ukládání výsledků

# ── Parametry šachovnice ────────────────────────────────────────────
#   Počítejte VNITŘNÍ rohy (= počet políček - 1 v každém směru).
#   Příklad: šachovnice 8×11 políček  →  ROWS = 7, COLS = 10
CHECKERBOARD_ROWS        = 23      # počet vnitřních rohů na výšku
CHECKERBOARD_COLS        = 32     # počet vnitřních rohů na šířku
CHECKERBOARD_SQUARE_MM   = 5.0   # fyzická velikost jednoho čtverce [mm]

# ── Detekce objektů ─────────────────────────────────────────────────
#   Zvyšte MIN_OBJECT_AREA_PX, pokud se detekují nežádoucí malé objekty.
#   Snižte prahy Canny (CANNY_LOW / CANNY_HIGH) pro slabší hrany.
#   MAX_OBJECTS = 1 zobrazí pouze největší objekt, 2 dva největší atd. (0 = neomezeno)
#   MERGE_NEARBY = True sloučí blízké kontury do jednoho objektu (vhodné pro lego apod.)
#   MERGE_DISTANCE_PX – jak blízké kontury se sloučí dohromady (v pixelech)
MIN_OBJECT_AREA_PX  = 500    # minimální plocha kontury v pixelech
MAX_OBJECTS         = 1      # max. počet výsledných objektů (0 = neomezeno)
MERGE_NEARBY        = True   # sloučit blízké kontury do jednoho objektu
MERGE_DISTANCE_PX   = 40     # vzdálenost slučování kontur v pixelech
CANNY_LOW           = 50     # spodní práh hranového detektoru Canny
CANNY_HIGH          = 150    # horní práh hranového detektoru Canny

# ── Zobrazení oken ───────────────────────────────────────────────────
#   Maximální rozměry náhledového okna v pixelech (přizpůsobte monitoru).
WINDOW_MAX_WIDTH  = 1280    # maximální šířka okna
WINDOW_MAX_HEIGHT = 800     # maximální výška okna

# ════════════════════════════════════════════════════════════════════


# ── pomocné funkce ──────────────────────────────────────────────────────────

def load_image(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(path, flags)
    if img is None:
        print(f"[CHYBA] Nelze načíst obrázek: {path}")
        sys.exit(1)
    return img


def show_image(title: str, img: np.ndarray):
    """Zobrazí obrázek v okně přizpůsobeném velikosti monitoru."""
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
    """Vrátí cestu vysledek1.jpg, vysledek2.jpg, ... ve složce folder."""
    os.makedirs(folder, exist_ok=True)
    idx = 1
    while True:
        path = os.path.join(folder, f"{base}{idx}{ext}")
        if not os.path.exists(path):
            return path
        idx += 1


def _try_find_corners(gray: np.ndarray, pattern_size: tuple):
    """Zkouší detekci rohů šachovnice s různými předzpracováními.
    Vrací (ret, corners, gray_at_scale, scale) – corners i gray jsou ve stejném měřítku."""

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
                    # vracíme corners a gray VE STEJNÉM měřítku (cornerSubPix musí souhlasit)
                    return ret, corners, g_scaled, scale

    return False, None, gray, 1.0


def calibrate_from_checkerboard(image_path: str, rows: int, cols: int,
                                 square_size_mm: float):
    """
    Detekuje šachovnici a vrátí:
        camera_matrix, dist_coeffs, pixels_per_mm (průměr po ose X i Y)
    """
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pattern_size = (cols, rows)

    # subpixelové kritérium
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3-D souřadnice rohů šachovnice (Z = 0, rovina)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    print("  Hledám rohy šachovnice (zkouším různá předzpracování)...")
    ret, corners, gray_scaled, found_scale = _try_find_corners(gray, pattern_size)

    if not ret:
        print("[CHYBA] Rohy šachovnice nebyly nalezeny ani po více pokusech.")
        print("  Tipy:")
        print("   • Zkontrolujte CHECKERBOARD_ROWS a CHECKERBOARD_COLS")
        print("     (pocet VNITRNICH rohu = policka - 1 v kazdem smeru)")
        print("   • Ujistete se, ze sachovnice zabira velkou cast snimku")
        print("   • Zkuste fotku s lepsim osvetlenim bez odlesku")
        sys.exit(1)

    # subpixelové zpřesnění na škálovaném gray (corners a gray_scaled jsou stejné měřítko)
    corners_refined_scaled = cv2.cornerSubPix(
        gray_scaled, corners.astype(np.float32), (11, 11), (-1, -1), criteria
    )

    # přepočet rohů zpět do původního rozlišení pro kalibraci kamery
    if found_scale != 1.0:
        corners_refined = corners_refined_scaled / found_scale
    else:
        corners_refined = corners_refined_scaled

    print(f"  Detekce uspesna (meritko: {found_scale:.2f})")

    # --- jednoobrázková kalibrace (dostatečná pro odhad px/mm) ---
    h, w = gray.shape
    ret_val, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners_refined], (w, h), None, None
    )

    # --- výpočet px/mm z projekce rohů zpět do obrazu ---
    proj, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], cam_mtx, dist)
    proj = proj.reshape(-1, 2)

    # vzdálenosti sousedních rohů podél osy X a Y
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

    print(f"[OK] Kalibrace probehla uspesne.")
    print(f"     Rozliseni: {px_per_mm:.4f} px/mm  "
          f"(x: {px_per_mm_x:.4f}, y: {px_per_mm_y:.4f})")

    # vizualizace – kreslíme rohy ručně jako barevné kroužky
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

    # --- korekce zkreslení (pokud máme parametry) ---
    if cam_mtx is not None and dist is not None:
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, cam_mtx, dist, None, new_mtx)
        x, y, rw, rh = roi
        if all(v > 0 for v in (x, y, rw, rh)):
            img = img[y:y + rh, x:x + rw]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- předzpracování ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # zvětšíme kernel při slučování, aby se blízké kontury spojily
    merge_k = max(5, MERGE_DISTANCE_PX) if MERGE_NEARBY else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge_k, merge_k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    if MERGE_NEARBY:
        closed = cv2.dilate(closed, kernel, iterations=2)
        closed = cv2.erode(closed, kernel, iterations=2)

    # --- hledání kontur ---
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[INFO] Zadne kontury nenalezeny.")
        return

    # --- filtrování: pouze kontury nad minimální plochou ---
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_OBJECT_AREA_PX]

    if not contours:
        print("[INFO] Zadne kontury nad minimalni plochou. Zkuste snizit MIN_OBJECT_AREA_PX.")
        return

    # --- seřazení podle plochy (největší první) + omezení počtu ---
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if MAX_OBJECTS > 0:
        contours = contours[:MAX_OBJECTS]

    result = img.copy()
    print("\n── Nalezene objekty ──────────────────────────────────────────")
    for idx, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)

        # ohraničující obdélník (zarovnaný na osy)
        x, y, bw, bh = cv2.boundingRect(cnt)
        bw_mm = bw / px_per_mm
        bh_mm = bh / px_per_mm
        area_mm2 = area_px / (px_per_mm ** 2)

        # rotovaný obdélník (minimální plocha)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        rw_mm = rw / px_per_mm
        rh_mm = rh / px_per_mm

        # obvod
        perimeter_px = cv2.arcLength(cnt, True)
        perimeter_mm = perimeter_px / px_per_mm

        print(f"\nObjekt #{idx + 1}")
        print(f"  Ohranicujici box (zarovnany):  {bw_mm:.1f} x {bh_mm:.1f} mm")
        print(f"  Rotovany box (min. plocha):    {max(rw_mm, rh_mm):.1f} x "
              f"{min(rw_mm, rh_mm):.1f} mm  (uhel: {angle:.1f} deg)")
        print(f"  Plocha kontury:                {area_mm2:.1f} mm2")
        print(f"  Obvod:                         {perimeter_mm:.1f} mm")

        # --- kreslení ---
        box_pts = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(result, [box_pts], 0, (0, 255, 0), 3)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (255, 100, 0), 1)

        # popisek – dvě řádky s tmavým podkladem
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

        # změř výšku jednoho řádku
        (_, line_h), baseline = cv2.getTextSize("A", font, font_scale, thickness)
        row_step = line_h + line_gap
        block_h  = len(lines) * row_step + padding * 2
        block_w  = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0]
                       for l in lines) + padding * 2

        # umísti blok nad konturou, ale ne mimo obraz
        bx = max(x, 0)
        by = max(y - block_h - 6, 0)

        # tmavý poloprůhledný podklad
        overlay = result.copy()
        cv2.rectangle(overlay, (bx, by), (bx + block_w, by + block_h),
                      (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # text
        for i, line in enumerate(lines):
            ty = by + padding + line_h + i * row_step
            cv2.putText(result, line, (bx + padding, ty),
                        font, font_scale, (0, 255, 180), thickness, cv2.LINE_AA)

    print("\n─────────────────────────────────────────────────────────────")

    out_path = next_output_path(OUTPUT_FOLDER)
    cv2.imwrite(out_path, result)
    print(f"[OK] Vysledny obrazek ulozen: {out_path}")

    show_image("Mereni objektu (stiskni klavesu)", result)


# ── hlavní program ───────────────────────────────────────────────────────────

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