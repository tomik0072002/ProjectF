"""
╔══════════════════════════════════════════════════════════════╗
║     DETEKCE HOTSPOTŮ – Termogramy FV panelů  v4             ║
║  Nastavení: upravte proměnné v sekci KONFIGURACE             ║
╚══════════════════════════════════════════════════════════════╝

Přístup: výhradně lokální Z-skóre.
  1. Gaussian blur pro potlačení šumu
  2. Lokální Z-skóre = (pixel − lokální průměr) / lokální std
  3. Maska: Z >= ZSCORE_THRESH
  4. Morfologické čištění masky (eroze drobných artefaktů)
  5. Extrakce kontur + filtrování geometrií
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
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE
# ─────────────────────────────────────────────────────────────

INPUT_FOLDER  = '.'
OUTPUT_FOLDER = 'vysledky'

# Předzpracování
PRE_BLUR      = 5         # Gaussian blur před Z-skóre [px, liché]
                          # Potlačí šum a JPEG artefakty. 0 = vypnuto.

# Z-skóre
Z_WINDOW      = 51        # Velikost okna pro lokální průměr/std [px, liché]
                          # Příliš malé → zachytí texturu buněk
                          # Příliš velké → nezachytí hotspoty blízko u sebe
Z_THRESH      = 2.5       # Minimální Z pro hotspot (typicky 2.0–4.0)
                          # Snižte pokud hotspoty chybí, zvyšte při false positives

# Čištění masky
MORPH_OPEN_K  = 3         # Kernel eroze/otevření masky [px] – odstraní drobné body
                          # 0 = vypnuto, 3–5 doporučeno

# Filtrování kontur
HS_MIN_AREA   = 10        # Min. plocha hotspotu [px²]
HS_MAX_AREA   = 5000      # Max. plocha hotspotu [px²] – odfiltruje velké oblasti
                          # které nejsou hotspot ale teplotní gradient
HS_MAX_ASPECT = 5.0       # Max. poměr stran (protáhlé = kabel/artefakt)
HS_MERGE_DIST = 15        # Sloučení fragmentů [px], 0 = vypnuto

# Skóre spolehlivosti
# C = f(max_Z, plocha, kruhovitost)
CONF_MIN      = 0         # Minimální skóre pro zobrazení

# Výstup
SHOW_WINDOW   = True
SHOW_PLOT     = True
SAVE_PNG      = True
SAVE_CSV      = True
DPI           = 200

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
    circularity:    float
    max_z:          float
    mean_z:         float
    max_intensity:  float
    confidence:     int   # 0–5

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


@dataclass
class Vysledek:
    soubor:     str
    n_hotspotu: int
    hotspoty:   List[Hotspot] = field(default_factory=list)
    z_thresh:   float = 0.0
    z_window:   int   = 0


# ─────────────────────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────────────────────

def priprav(img: np.ndarray, blur_k: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_k >= 3:
        k = blur_k | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def zscore_mapa(gray: np.ndarray, window: int) -> np.ndarray:
    """
    Lokální Z-skóre pro každý pixel.

    Místo globálního prahu porovnáváme každý pixel se svým okolím
    definovaným klouzavým oknem. Výsledek je invariantní vůči:
      - absolutní teplotní úrovni snímku
      - teplotním gradientům přes celé pole (jeden konec chladnější)
      - různému nastavení pseudobarev termokamery

    Hotspot = lokálně výrazně teplejší bod → vysoké Z.
    """
    k     = window | 1
    mu    = cv2.boxFilter(gray, -1, (k, k))
    sq_mu = cv2.boxFilter(gray ** 2, -1, (k, k))
    std   = np.sqrt(np.clip(sq_mu - mu ** 2, 0, None)) + 1e-6
    return (gray - mu) / std


def sestavmasku(zmap: np.ndarray, thresh: float, morph_k: int) -> np.ndarray:
    """
    Z-skóre → binární maska → morfologické čištění.

    Čištění (morfologické otevření) odstraní izolované pixely a drobné
    artefakty šumu/komprese, ale zachová kompaktnější oblasti hotspotů.
    """
    maska = (zmap >= thresh).astype(np.uint8) * 255

    if morph_k >= 3:
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
        maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, k)

    return maska


def sluc(maska: np.ndarray, dist: int) -> np.ndarray:
    if dist <= 0:
        return maska
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist | 1, dist | 1))
    return cv2.erode(cv2.dilate(maska, k), k)


def confidence(max_z: float, area: float, circ: float, thresh: float) -> int:
    """
    Skóre 0–5:
      +1  Z ≥ thresh × 1.5   (výrazný hotspot)
      +1  Z ≥ thresh × 2.5   (velmi silný hotspot)
      +1  Z ≥ thresh × 4.0   (extrémní teplotní anomálie)
      +1  plocha v rozumném rozsahu (10–500 px²)
      +1  kompaktní tvar (kruhovitost > 0.4)
    """
    s = 0
    if max_z >= thresh * 1.5: s += 1
    if max_z >= thresh * 2.5: s += 1
    if max_z >= thresh * 4.0: s += 1
    if 10 <= area <= 500:     s += 1
    if circ > 0.4:            s += 1
    return s


def detekuj(img: np.ndarray,
            pre_blur:    int   = PRE_BLUR,
            z_window:    int   = Z_WINDOW,
            z_thresh:    float = Z_THRESH,
            morph_open:  int   = MORPH_OPEN_K,
            min_area:    int   = HS_MIN_AREA,
            max_area:    int   = HS_MAX_AREA,
            max_aspect:  float = HS_MAX_ASPECT,
            merge_dist:  int   = HS_MERGE_DIST,
            conf_min:    int   = CONF_MIN,
            ) -> Tuple[np.ndarray, np.ndarray, List[Hotspot]]:
    """
    Kompletní detekční pipeline.
    Vrací (zmap, maska, hotspoty).
    """
    gray  = priprav(img, pre_blur)
    zmap  = zscore_mapa(gray, z_window)
    maska = sestavmasku(zmap, z_thresh, morph_open)
    merged = sluc(maska, merge_dist)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspoty = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        asp = (w / h) if h > 0 else 0
        if asp > max_aspect or asp < 1.0 / max_aspect:
            continue

        # Metriky v oblasti kontury
        cm     = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px_z   = zmap[cm > 0]
        px_g   = gray[cm > 0]
        max_z  = float(px_z.max())  if px_z.size else 0.0
        mean_z = float(px_z.mean()) if px_z.size else 0.0
        max_i  = float(px_g.max())  if px_g.size else 0.0

        perim  = cv2.arcLength(cnt, True)
        circ   = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0

        conf = confidence(max_z, area, circ, z_thresh)
        if conf < conf_min:
            continue

        hotspoty.append(Hotspot(
            id=len(hotspoty) + 1,
            x=x, y=y, w=w, h=h,
            area=round(area, 1),
            aspect_ratio=round(asp, 2),
            circularity=round(circ, 3),
            max_z=round(max_z, 2),
            mean_z=round(mean_z, 2),
            max_intensity=round(max_i, 1),
            confidence=conf,
        ))

    hotspoty.sort(key=lambda h: h.max_z, reverse=True)
    for i, h in enumerate(hotspoty):
        h.id = i + 1

    return zmap, maska, hotspoty


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

CONF_BGR = {
    0: (160, 160, 160),
    1: (0,   210, 210),
    2: (0,   165, 255),
    3: (0,    80, 255),
    4: (0,    20, 220),
    5: (0,     0, 180),
}


def anotuj(img: np.ndarray, hotspoty: List[Hotspot],
           z_thresh: float, z_window: int) -> np.ndarray:
    out = img.copy()
    pad = 5
    for hs in hotspoty:
        b  = CONF_BGR.get(min(hs.confidence, 5), (0, 0, 255))
        x1 = max(0, hs.x - pad);  y1 = max(0, hs.y - pad)
        x2 = min(img.shape[1]-1, hs.x + hs.w + pad)
        y2 = min(img.shape[0]-1, hs.y + hs.h + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), b, 2)
        lbl = f"#{hs.id} Z{hs.max_z:.1f}"
        cv2.putText(out, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, b, 1, cv2.LINE_AA)

    info = [f"Hotspoty: {len(hotspoty)}",
            f"Z-thresh: {z_thresh}",
            f"Z-window: {z_window}px"]
    for i, line in enumerate(info):
        y = 22 + i * 20
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB VÝSTUP
# ─────────────────────────────────────────────────────────────

def vykresli(img, annotated, zmap, maska, hotspoty, soubor, save_path=None):
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(f"Detekce hotspotů – {os.path.basename(soubor)}",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

    panels = [
        (cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originální snímek",  None),
        (np.clip(zmap, 0, None),                      "Z-skóre mapa",       'RdYlBu_r'),
        (maska,                                        "Detekční maska",     'gray'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  "Anotovaný výsledek", None),
    ]

    for i, (ax, (data, title, cmap)) in enumerate(
            zip([fig.add_subplot(gs[j]) for j in range(4)], panels)):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis('off')
        if cmap == 'RdYlBu_r':
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        log.info(f"  Graf: {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────

def uloz_csv(vysledky: List[Vysledek], folder: str):
    path = os.path.join(folder, '_hotspoty_souhrn.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Soubor', 'ID', 'Spolehlivost', 'Max Z', 'Mean Z',
                    'X', 'Y', 'Š', 'V', 'Plocha', 'Poměr stran',
                    'Kruhovitost', 'Max jas', 'Z práh', 'Z okno'])
        for v in vysledky:
            for h in v.hotspoty:
                w.writerow([v.soubor, h.id, h.confidence, h.max_z, h.mean_z,
                             h.x, h.y, h.w, h.h, h.area, h.aspect_ratio,
                             h.circularity, h.max_intensity,
                             v.z_thresh, v.z_window])
    log.info(f"CSV: {path}")


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ SMYČKA
# ─────────────────────────────────────────────────────────────

def zpracuj(cesta: str, output_folder: str) -> Optional[Vysledek]:
    img = cv2.imread(cesta)
    if img is None:
        log.warning(f"Nelze načíst: {cesta}")
        return None

    zmap, maska, hotspoty = detekuj(img)

    log.info(f"  → Hotspoty: {len(hotspoty)}  "
             f"(Z>={Z_THRESH}, okno {Z_WINDOW}px)")

    v = Vysledek(soubor=cesta, n_hotspotu=len(hotspoty),
                 hotspoty=hotspoty, z_thresh=Z_THRESH, z_window=Z_WINDOW)

    annotated = anotuj(img, hotspoty, Z_THRESH, Z_WINDOW)

    nazev = os.path.splitext(os.path.basename(cesta))[0]

    if SAVE_PNG:
        cv2.imwrite(os.path.join(output_folder, f"vysledek_{nazev}.png"), annotated)

    mpl_p = os.path.join(output_folder, f"graf_{nazev}.png") if SAVE_PNG else None
    if SHOW_PLOT:
        vykresli(img, annotated, zmap, maska, hotspoty, cesta, mpl_p)

    if SHOW_WINDOW:
        porovnani = np.hstack((img, annotated))
        win = f"{nazev}  (MEZERNIK = další,  Q = konec)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 700)
        cv2.imshow(win, porovnani)
        cv2.destroyWindow(win)

    return v


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    seznam = sorted(sum([
        glob.glob(os.path.join(INPUT_FOLDER, p))
        for p in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    ], []))

    if not seznam:
        log.warning(f"Žádné snímky v: {INPUT_FOLDER}")
        return

    log.info(f"Nalezeno {len(seznam)} snímků")
    vysledky = []

    for i, cesta in enumerate(seznam):
        log.info(f"[{i+1}/{len(seznam)}] {os.path.basename(cesta)}")
        v = zpracuj(cesta, OUTPUT_FOLDER)
        if v:
            vysledky.append(v)
        if SHOW_WINDOW:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    cv2.destroyAllWindows()
    if SAVE_CSV and vysledky:
        uloz_csv(vysledky, OUTPUT_FOLDER)

    celkem = sum(v.n_hotspotu for v in vysledky)
    log.info(f"HOTOVO – {len(vysledky)} snímků, {celkem} hotspotů celkem")


if __name__ == "__main__":
    main()