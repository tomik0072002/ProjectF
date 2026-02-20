"""
╔══════════════════════════════════════════════════════════════╗
║     DETEKCE HOTSPOTŮ – Termogramy FV panelů                  ║
║  Nastavení: upravte proměnné v sekci KONFIGURACE             ║
║  Spuštění:  python hotspoty.py                               ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import os
import glob
import logging
import csv
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE
# ─────────────────────────────────────────────────────────────

INPUT_FOLDER   = '.'
OUTPUT_FOLDER  = 'vysledky'

# ── Detekce panelů ───────────────────────────────────────────
PANEL_DETECTION    = True    # False = analyzuj celý snímek bez masky
PANEL_MIN_AREA     = 1000    # minimální plocha panelu [px²]
PANEL_ERODE_ITER   = 2       # eroze okrajů (odstraní tepelný vliv rámu)
PANEL_MIN_SOLIDITY = 0.35    # minimální plnost kontury (0–1) – nižší = povolí složitější tvary
PANEL_MAX_COVERAGE = 0.70    # max. podíl snímku který může maska pokrýt
                             # > 60 % → segmentace selhala → strategie se zahodí

# ── Top-Hat filtr ────────────────────────────────────────────
TOPHAT_KERNEL_SIZE = 51      # [px] větší = ignoruje texturu buněk
TOPHAT_PERCENTILE  = 97      # práh top-hat signálu (95–99)

# ── Absolutní jas ────────────────────────────────────────────
ABS_PERCENTILE     = 97      # absolutní práh jasu (95–99)

# ── Kombinace metod ──────────────────────────────────────────
REQUIRE_BOTH       = True    # True=AND (méně FP), False=OR (více detekcí)

# ── Filtrování hotspotů ──────────────────────────────────────
HOTSPOT_MIN_AREA   = 40      # minimální plocha [px²]
HOTSPOT_MAX_ASPECT = 4.0     # max poměr stran
HOTSPOT_PADDING    = 4       # rozšíření bbox [px]

# ── Slučování blízkých hotspotů ──────────────────────────────
MERGE_ENABLED      = True    # sloučit blízké hotspoty
MERGE_DISTANCE     = 30      # vzdálenost pro sloučení [px]

# ── Filtr okrajů panelů ──────────────────────────────────────
EDGE_FILTER_ENABLED = True   # odstraní hotspoty blízko okraje panelu
EDGE_MARGIN_PX      = 15     # minimální vzdálenost od okraje [px]

# ── Skóre spolehlivosti ──────────────────────────────────────
CONFIDENCE_WEIGHTS = dict(
    both_methods = 40,   # za detekci oběma metodami
    intensity    = 35,   # za relativní intenzitu
    not_on_edge  = 25,   # za vzdálenost od okraje
)

# ── Vizualizace ──────────────────────────────────────────────
SHOW_WINDOW        = True
SHOW_MATPLOTLIB    = True
SAVE_PNG           = True
SAVE_CSV           = True
COLOR_TEXT         = (255, 255, 255)
COLOR_PANEL        = (0, 200, 0)

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  DATOVÉ TŘÍDY
# ─────────────────────────────────────────────────────────────

@dataclass
class Hotspot:
    id:             int
    x: int; y: int; w: int; h: int
    area:           float
    aspect_ratio:   float
    mean_intensity: float
    max_intensity:  float
    typ:            str
    confidence:     int = 0

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


@dataclass
class VysledekSnimku:
    soubor:        str
    n_panelu:      int
    n_hotspotu:    int
    hotspoty:      List[Hotspot] = field(default_factory=list)
    tophat_thresh: int = 0
    abs_thresh:    int = 0
    panel_thresh:  int = 0


# ─────────────────────────────────────────────────────────────
#  DETEKCE PANELŮ – kombinace gradient + Otsu
# ─────────────────────────────────────────────────────────────

def _zkus_segmentaci(gray: np.ndarray, thresh_val: int,
                      inverse: bool) -> np.ndarray:
    """Prahování + morfologie → kandidátní maska panelů."""
    flags = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, b = cv2.threshold(gray, thresh_val, 255, flags)
    k5   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    b    = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k5, iterations=1)
    b    = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k5, iterations=3)
    return b


def _filtruj_kontury(bin_mask: np.ndarray, gray: np.ndarray,
                      min_area: int, min_solidity: float,
                      max_pokryti: float) -> Tuple[np.ndarray, int]:
    """
    Z binární masky vybere validní kontury:
      - plocha >= min_area
      - solidity (plnost) >= min_solidity  → odfiltruje vegetaci, budovy
      - výsledná maska nesmí pokrýt více než max_pokryti snímku
        (pokud ano, segmentace selhala a vrátíme prázdnou masku)
    """
    total_px = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    maska = np.zeros_like(gray)
    n = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity  = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue
        cv2.drawContours(maska, [hull], -1, 255, -1)
        n += 1

    pokryti = maska.sum() / 255 / total_px
    if pokryti > max_pokryti:
        log.debug(f"    Segmentace zamítnuta – pokrytí {pokryti:.0%} > {max_pokryti:.0%}")
        return np.zeros_like(gray), 0

    return maska, n


def najdi_panely(gray: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Detekce FV panelů – tři strategie, vybere nejlepší.

    Každá strategie se hodnotí dvěma kritérii:
      1. Počet validních kontur (splňují plochu + solidity)
      2. Pokrytí masky < PANEL_MAX_COVERAGE – pokud segmentace
         pokryje celý snímek (>60 %), výsledek je neplatný
         a tato strategie se zahodí.

    Strategie:
      A) Otsu INV  – panely jsou chladnější/tmavší než pozadí
      B) Otsu NORM – panely jsou teplejší/světlejší (přehřáté pole,
                     inverzní pseudobarvy)
      C) Gradientní – panely mají homogenní vnitřek (nízký gradient).
                      Funguje i při podobném jasu panelů a pozadí.

    Pokud všechny strategie selžou (pokrytí příliš velké nebo žádné
    kontury), vrátí masku celého snímku jako fallback – analýza
    pak proběhne bez prostorového omezení.
    """
    otsu_val, _ = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = int(otsu_val)

    max_cov = PANEL_MAX_COVERAGE
    sol     = PANEL_MIN_SOLIDITY

    # ── A: Otsu INV ──────────────────────────────────────────
    b_inv = _zkus_segmentaci(gray, otsu_val, inverse=True)
    mA, nA = _filtruj_kontury(b_inv, gray, PANEL_MIN_AREA, sol, max_cov)

    # ── B: Otsu NORM ─────────────────────────────────────────
    b_nrm = _zkus_segmentaci(gray, otsu_val, inverse=False)
    mB, nB = _filtruj_kontury(b_nrm, gray, PANEL_MIN_AREA, sol, max_cov)

    # ── C: Gradientní (homogenní oblasti) ────────────────────
    blur    = cv2.GaussianBlur(gray, (15, 15), 0)
    lapl    = cv2.Laplacian(blur, cv2.CV_64F)
    lapl_u8 = np.uint8(np.clip(np.abs(lapl), 0, 255))
    _, gm   = cv2.threshold(lapl_u8, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    gm  = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, k25)
    mC, nC = _filtruj_kontury(gm, gray, PANEL_MIN_AREA, sol, max_cov)

    # ── Výběr nejlepší strategie ──────────────────────────────
    kandidati = [(nA, mA, "INV"), (nB, mB, "NORM"), (nC, mC, "GRAD")]
    best_n, best_mask, best_name = max(kandidati, key=lambda x: x[0])

    log.debug(f"  Panely – INV:{nA}  NORM:{nB}  GRAD:{nC}  → {best_name}")

    if best_n == 0:
        log.warning("  Panely nenalezeny – analýza bez masky (celý snímek)")
        best_mask = np.ones_like(gray) * 255

    k_e   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best_mask = cv2.erode(best_mask, k_e, iterations=PANEL_ERODE_ITER)

    return best_mask, best_n, otsu_val


# ─────────────────────────────────────────────────────────────
#  DETEKCE HOTSPOTŮ
# ─────────────────────────────────────────────────────────────

def detekuj_hotspoty(gray, maska_panelu):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
    tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    px_panel = tophat[maska_panelu > 0]
    tophat_t = int(np.percentile(px_panel, TOPHAT_PERCENTILE)) if px_panel.size else 50
    _, tb    = cv2.threshold(tophat, tophat_t, 255, cv2.THRESH_BINARY)
    tophat_m = cv2.bitwise_and(tb, tb, mask=maska_panelu)

    px_gray = gray[maska_panelu > 0]
    abs_t   = int(np.percentile(px_gray, ABS_PERCENTILE)) if px_gray.size else 190
    _, ab   = cv2.threshold(gray, abs_t, 255, cv2.THRESH_BINARY)
    abs_m   = cv2.bitwise_and(ab, ab, mask=maska_panelu)

    combined = cv2.bitwise_and(tophat_m, abs_m) if REQUIRE_BOTH else cv2.bitwise_or(tophat_m, abs_m)
    return tophat_m, abs_m, combined, tophat_t, abs_t


# ─────────────────────────────────────────────────────────────
#  SLUČOVÁNÍ BLÍZKÝCH HOTSPOTŮ
# ─────────────────────────────────────────────────────────────

def sluc_blizke(combined: np.ndarray, distance: int) -> np.ndarray:
    """
    Dilatace → sloučení blízkých oblastí → eroze zpět.
    Zachová jen oblasti kde byl původně signál.
    """
    if distance <= 0:
        return combined
    ks = distance // 2 * 2 + 1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    d1 = cv2.dilate(combined, k)
    d2 = cv2.erode(d1, k)
    return cv2.bitwise_and(d2, cv2.dilate(combined, k))


# ─────────────────────────────────────────────────────────────
#  SKÓRE SPOLEHLIVOSTI
# ─────────────────────────────────────────────────────────────

def vypocti_confidence(hs: Hotspot, dist_map: np.ndarray, global_max: float) -> int:
    """
    Skóre 0–100 ze tří složek:
      - Obě metody detekovaly hotspot       → max 40 bodů
      - Relativní intenzita vůči snímku     → max 35 bodů
      - Vzdálenost od okraje panelu         → max 25 bodů
    """
    w = CONFIDENCE_WEIGHTS
    score = 0

    if hs.typ == 'combined':
        score += w['both_methods']

    if global_max > 0:
        score += int(min(1.0, hs.max_intensity / global_max) * w['intensity'])

    h_map, w_map = dist_map.shape
    if 0 <= hs.cy < h_map and 0 <= hs.cx < w_map:
        norm = min(1.0, dist_map[hs.cy, hs.cx] / 50.0)
        score += int(norm * w['not_on_edge'])

    return min(100, score)


# ─────────────────────────────────────────────────────────────
#  FILTROVÁNÍ A KLASIFIKACE
# ─────────────────────────────────────────────────────────────

def filtruj_hotspoty(combined, tophat_mask, abs_mask, gray, maska_panelu) -> List[Hotspot]:
    if MERGE_ENABLED:
        combined = sluc_blizke(combined, MERGE_DISTANCE)

    dist_map   = cv2.distanceTransform(maska_panelu, cv2.DIST_L2, 5) if maska_panelu.any() \
                 else np.zeros_like(gray, float)
    global_max = float(gray[maska_panelu > 0].max()) if maska_panelu.any() else float(gray.max())

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspoty: List[Hotspot] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < HOTSPOT_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if aspect > HOTSPOT_MAX_ASPECT or aspect < 1.0 / HOTSPOT_MAX_ASPECT:
            continue

        cx, cy = x + w // 2, y + h // 2
        if EDGE_FILTER_ENABLED and maska_panelu.any():
            hm, wm = dist_map.shape
            if 0 <= cy < hm and 0 <= cx < wm:
                if dist_map[cy, cx] < EDGE_MARGIN_PX:
                    continue

        roi_g  = gray[y:y+h, x:x+w]
        cm     = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(cm, [cnt - [x, y]], -1, 255, -1)
        px     = roi_g[cm > 0]
        mean_i = float(px.mean()) if px.size else 0.0
        max_i  = float(px.max())  if px.size else 0.0

        in_t   = bool(tophat_mask[y:y+h, x:x+w][cm > 0].any())
        in_a   = bool(abs_mask[y:y+h, x:x+w][cm > 0].any())
        typ    = 'combined' if (in_t and in_a) else ('tophat' if in_t else 'absolute')

        hs = Hotspot(id=len(hotspoty)+1, x=x, y=y, w=w, h=h,
                     area=area, aspect_ratio=round(aspect, 2),
                     mean_intensity=round(mean_i, 1),
                     max_intensity=round(max_i, 1), typ=typ)
        hs.confidence = vypocti_confidence(hs, dist_map, global_max)
        hotspoty.append(hs)

    hotspoty.sort(key=lambda h: h.confidence, reverse=True)
    for i, hs in enumerate(hotspoty):
        hs.id = i + 1
    return hotspoty


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

def conf_barva(conf: int):
    """Plynulý přechod barvy: červená(100) → oranžová(50) → žlutá(0)."""
    t = conf / 100.0
    return (0, int((1 - t) * 255), 255)


def anotuj_vysledek(img, maska_panelu, hotspoty, vysledek) -> np.ndarray:
    out = img.copy()
    conts, _ = cv2.findContours(maska_panelu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, conts, -1, COLOR_PANEL, 1)

    p = HOTSPOT_PADDING
    for hs in hotspoty:
        barva     = conf_barva(hs.confidence)
        thickness = 3 if hs.confidence >= 70 else 2 if hs.confidence >= 40 else 1
        x1 = max(0, hs.x - p);  y1 = max(0, hs.y - p)
        x2 = min(img.shape[1]-1, hs.x+hs.w+p)
        y2 = min(img.shape[0]-1, hs.y+hs.h+p)
        cv2.rectangle(out, (x1, y1), (x2, y2), barva, thickness)
        lbl = f"#{hs.id} {hs.confidence}%"
        cv2.putText(out, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, barva, 1, cv2.LINE_AA)

    for i, line in enumerate([f"Panely: {vysledek.n_panelu}",
                               f"Hotspoty: {vysledek.n_hotspotu}",
                               f"Tophat T: {vysledek.tophat_thresh}",
                               f"Abs T: {vysledek.abs_thresh}"]):
        yp = 22 + i*20
        cv2.putText(out, line, (8,yp), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8,yp), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB
# ─────────────────────────────────────────────────────────────

def zobraz_matplotlib(img, output, tophat_mask, abs_mask, hotspoty, soubor, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"Analýza hotspotů – {os.path.basename(soubor)}",
                 fontsize=13, fontweight='bold')
    for ax, (data, title, cmap) in zip(axes, [
        (cv2.cvtColor(img,    cv2.COLOR_BGR2RGB), "Originální snímek", None),
        (tophat_mask,                              "Top-Hat maska",     'hot'),
        (abs_mask,                                 "Absolutní maska",   'hot'),
        (cv2.cvtColor(output, cv2.COLOR_BGR2RGB),  "Výsledek detekce",  None),
    ]):
        ax.imshow(data, cmap=cmap); ax.set_title(title, fontsize=10); ax.axis('off')

    fig.legend(handles=[
        mpatches.Patch(color='red',    label=f'Vysoká ≥70%  ({sum(1 for h in hotspoty if h.confidence>=70)})'),
        mpatches.Patch(color='orange', label=f'Střední 40–69% ({sum(1 for h in hotspoty if 40<=h.confidence<70)})'),
        mpatches.Patch(color='yellow', label=f'Nízká <40%   ({sum(1 for h in hotspoty if h.confidence<40)})'),
    ], loc='lower center', ncol=3, fontsize=9, framealpha=0.8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        log.info(f"  Graf: {save_path}")
    plt.show(); plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────

def uloz_csv(vysledky, output_folder):
    path = os.path.join(output_folder, '_hotspoty_souhrn.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Soubor','ID','Typ','Confidence [%]','X','Y',
                    'Šířka','Výška','Plocha [px²]','Poměr stran',
                    'Střední jas','Max jas','Tophat práh','Abs práh','Počet panelů'])
        for v in vysledky:
            for hs in v.hotspoty:
                w.writerow([v.soubor, hs.id, hs.typ, hs.confidence,
                             hs.x, hs.y, hs.w, hs.h, round(hs.area,1),
                             hs.aspect_ratio, hs.mean_intensity, hs.max_intensity,
                             v.tophat_thresh, v.abs_thresh, v.n_panelu])
    log.info(f"CSV: {path}")


# ─────────────────────────────────────────────────────────────
#  ZPRACOVÁNÍ JEDNOHO SNÍMKU
# ─────────────────────────────────────────────────────────────

def zpracuj_snimek(cesta, output_folder) -> Optional[VysledekSnimku]:
    img = cv2.imread(cesta)
    if img is None:
        log.warning(f"Nelze načíst: {cesta}"); return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if PANEL_DETECTION:
        maska_panelu, n_panelu, panel_t = najdi_panely(gray)
        if not maska_panelu.any():
            log.warning("  Žádné panely – analyzuji celý snímek")
            maska_panelu = np.full_like(gray, 255); n_panelu = 0; panel_t = 0
    else:
        maska_panelu = np.full_like(gray, 255); n_panelu = 0; panel_t = 0

    tophat_m, abs_m, combined, tophat_t, abs_t = detekuj_hotspoty(gray, maska_panelu)
    hotspoty = filtruj_hotspoty(combined, tophat_m, abs_m, gray, maska_panelu)

    v = VysledekSnimku(soubor=cesta, n_panelu=n_panelu, n_hotspotu=len(hotspoty),
                       hotspoty=hotspoty, tophat_thresh=tophat_t,
                       abs_thresh=abs_t, panel_thresh=panel_t)

    log.info(f"  Panely: {n_panelu} | Hotspoty: {len(hotspoty)} | "
             f"Tophat T={tophat_t} Abs T={abs_t}")
    if hotspoty:
        log.info("  Top 3: " + ", ".join(
            f"#{h.id} conf={h.confidence}% max={h.max_intensity:.0f}"
            for h in hotspoty[:3]))

    output_img = anotuj_vysledek(img, maska_panelu, hotspoty, v)
    nazev = os.path.splitext(os.path.basename(cesta))[0]

    if SAVE_PNG:
        p = os.path.join(output_folder, f"vysledek_{nazev}.png")
        cv2.imwrite(p, output_img); log.info(f"  PNG: {p}")

    mpl_path = os.path.join(output_folder, f"graf_{nazev}.png") if SAVE_PNG else None
    if SHOW_MATPLOTLIB:
        zobraz_matplotlib(img, output_img, tophat_m, abs_m, hotspoty, cesta, mpl_path)

    if SHOW_WINDOW:
        por = np.hstack((img, output_img))
        sw  = img.shape[1]
        for txt, x in [("Originál", 15), ("Detekce", sw+15)]:
            cv2.putText(por, txt, (x,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(por, txt, (x,30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 1, cv2.LINE_AA)
        win = f"Hotspoty – {nazev}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1600, 700)
        cv2.imshow(win, por); cv2.destroyWindow(win)

    return v


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ FUNKCE
# ─────────────────────────────────────────────────────────────

def zpracuj_vsechny_obrazky():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    seznam = sorted(sum([glob.glob(os.path.join(INPUT_FOLDER, p))
                         for p in ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']], []))
    if not seznam:
        log.warning(f"Žádné snímky v: {INPUT_FOLDER}"); return

    log.info(f"Nalezeno {len(seznam)} snímků")
    vysledky: List[VysledekSnimku] = []

    for i, cesta in enumerate(seznam):
        log.info(f"[{i+1}/{len(seznam)}] {os.path.basename(cesta)}")
        v = zpracuj_snimek(cesta, OUTPUT_FOLDER)
        if v: vysledky.append(v)
        if SHOW_WINDOW:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                log.info("Ukončuji..."); break

    cv2.destroyAllWindows()
    if SAVE_CSV and vysledky: uloz_csv(vysledky, OUTPUT_FOLDER)

    celkem = sum(v.n_hotspotu for v in vysledky)
    log.info(f"Zpracováno: {len(vysledky)} snímků | "
             f"Celkem hotspotů: {celkem} | "
             f"Průměr: {celkem/len(vysledky):.1f}" if vysledky else "")
    log.info("HOTOVO.")


if __name__ == "__main__":
    zpracuj_vsechny_obrazky()