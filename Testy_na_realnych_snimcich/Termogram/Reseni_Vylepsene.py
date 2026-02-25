"""
╔══════════════════════════════════════════════════════════════╗
║     DETEKCE HOTSPOTŮ – Termogramy FV panelů  v5             ║
║  Nastavení: upravte proměnné v sekci KONFIGURACE             ║
╚══════════════════════════════════════════════════════════════╝

Přístup: výhradně lokální Z-skóre.
  1. Gaussian blur pro potlačení šumu
  2. Lokální Z-skóre = (pixel − lokální průměr) / lokální std
  3. Maska: Z >= ZSCORE_THRESH
  4. Morfologické čištění masky (eroze drobných artefaktů)
  5. Slučování fragmentů
  6. Extrakce kontur + filtrování geometrií
  7. Skóre závažnosti (Slabý / Střední / Kritický)
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
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE
# ─────────────────────────────────────────────────────────────

INPUT_FOLDER  = '.'
OUTPUT_FOLDER = 'Výsledky_vylepšeno'

# Předzpracování
PRE_BLUR      = 7         # Gaussian blur před Z-skóre [px, liché]
                          # Potlačí šum a JPEG artefakty. 0 = vypnuto.

# Z-skóre
Z_WINDOW      = 91        # Velikost okna pro lokální průměr/std [px, liché]
                          # Příliš malé → zachytí texturu buněk
                          # Příliš velké → nezachytí hotspoty blízko u sebe
Z_THRESH      = 3.2       # Minimální Z pro hotspot (typicky 2.0–4.0)
                          # Snižte pokud hotspoty chybí, zvyšte při false positives

# Čištění masky
MORPH_OPEN_K  = 5         # Kernel eroze/otevření masky [px] – odstraní drobné body
                          # 0 = vypnuto, 3–5 doporučeno

# Sloučení fragmentů
MERGE_DIST    = 15        # Sloučení fragmentů [px], 0 = vypnuto

# Filtrování kontur
HS_MIN_AREA   = 15        # Min. plocha hotspotu [px²]
HS_MAX_AREA   = 800       # Max. plocha hotspotu [px²]
HS_MAX_ASPECT = 3.5       # Max. poměr stran (protáhlé = kabel/artefakt)

# Filtr závažnosti
# "vse"      → zobrazit vše včetně slabých (conf_min = 0)
# "stredni"  → střední a kritické         (conf_min = 1)
# "kriticke" → pouze kritické             (conf_min = 3)
FILTER_ZAVAZNOST = "stredni"

# Rychlé předvolby – přepište FILTER_PRESET na jméno předvolby,
# nebo nechte "" pro ruční nastavení výše.
# Dostupné předvolby: "standard", "sensitive", "strict"
FILTER_PRESET = ""

# Výstup
SHOW_WINDOW   = False     # Zobrazit CV2 okno s porovnáním
SHOW_PLOT     = False     # Zobrazit matplotlib graf
SAVE_PNG      = True      # Uložit anotovaný PNG
SAVE_CSV      = True      # Uložit CSV souhrn
SAVE_DEBUG    = True      # Uložit debug graf (4 panely)
DPI           = 200

# ─────────────────────────────────────────────────────────────
#  PŘEDVOLBY (přepisují konfiguraci výše)
# ─────────────────────────────────────────────────────────────

PRESETS = {
    "standard":  dict(blur_k=7,  z_window=91,  z_thresh=3.2, morph_k=5,
                      merge_dist=15, min_area=15, max_area=800,
                      max_aspect=3.5, filter_zavaznost="stredni"),
    "sensitive": dict(blur_k=3,  z_window=51,  z_thresh=2.5, morph_k=3,
                      merge_dist=15, min_area=5,  max_area=3000,
                      max_aspect=5.0, filter_zavaznost="vse"),
    "strict":    dict(blur_k=7,  z_window=91,  z_thresh=3.5, morph_k=5,
                      merge_dist=10, min_area=10, max_area=500,
                      max_aspect=3.5, filter_zavaznost="stredni"),
}

FILTER_ZAVAZNOST_MAP = {
    "vse":       0,
    "stredni":   1,
    "kriticke":  3,
}

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
    confidence:     int   # 0–5, interní skóre

    @property
    def cx(self): return self.x + self.w // 2

    @property
    def cy(self): return self.y + self.h // 2

    @property
    def zavaznost(self) -> str:
        """Lidsky čitelná závažnost odvozená z confidence skóre."""
        if self.confidence >= 3: return "Kritický"
        if self.confidence >= 1: return "Střední"
        return "Slabý"

    @property
    def zavaznost_color_hex(self) -> str:
        if self.confidence >= 3: return "#ef4444"
        if self.confidence >= 1: return "#f97316"
        return "#eab308"

    @property
    def zavaznost_bgr(self) -> tuple:
        """BGR barva pro OpenCV anotaci."""
        if self.confidence >= 3: return (0,   50, 239)   # červená
        if self.confidence >= 1: return (0,  115, 249)   # oranžová
        return (0, 163, 234)                              # žlutá


@dataclass
class Vysledek:
    soubor:     str
    n_hotspotu: int
    hotspoty:   List[Hotspot] = field(default_factory=list)
    z_thresh:   float = 0.0
    z_window:   int   = 0
    n_krit:     int   = 0
    n_str:      int   = 0
    n_slab:     int   = 0
    mask_pct:   float = 0.0


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
      - teplotním gradientům přes celé pole
      - různému nastavení pseudobarev termokamery
    """
    k     = window | 1
    mu    = cv2.boxFilter(gray, -1, (k, k))
    sq_mu = cv2.boxFilter(gray ** 2, -1, (k, k))
    std   = np.sqrt(np.clip(sq_mu - mu ** 2, 0, None)) + 1e-6
    return (gray - mu) / std


def sestav_masku(zmap: np.ndarray, thresh: float, morph_k: int) -> np.ndarray:
    """Z-skóre → binární maska → morfologické čištění."""
    maska = (zmap >= thresh).astype(np.uint8) * 255
    if morph_k >= 3:
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (morph_k | 1, morph_k | 1))
        maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, k)
    return maska


def sluc(maska: np.ndarray, dist: int) -> np.ndarray:
    """Sloučí blízké fragmenty dilatací + erozí."""
    if dist <= 0:
        return maska
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist | 1, dist | 1))
    return cv2.erode(cv2.dilate(maska, k), k)


def spocti_confidence(max_z: float, area: float,
                      circ: float, thresh: float) -> int:
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
            merge_dist:  int   = MERGE_DIST,
            conf_min:    int   = 0,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Hotspot]]:
    """
    Kompletní detekční pipeline.
    Vrací (zmap, maska, merged_maska, hotspoty).
    """
    gray   = priprav(img, pre_blur)
    zmap   = zscore_mapa(gray, z_window)
    maska  = sestav_masku(zmap, z_thresh, morph_open)
    merged = sluc(maska, merge_dist)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    hotspoty = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        asp = (w / h) if h > 0 else 0
        if asp > max_aspect or asp < 1.0 / max_aspect:
            continue

        cm = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px_z   = zmap[cm > 0]
        px_g   = gray[cm > 0]
        max_z  = float(px_z.max())  if px_z.size else 0.0
        mean_z = float(px_z.mean()) if px_z.size else 0.0
        max_i  = float(px_g.max())  if px_g.size else 0.0

        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0

        conf = spocti_confidence(max_z, area, circ, z_thresh)
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

    return zmap, maska, merged, hotspoty


# ─────────────────────────────────────────────────────────────
#  METRIKY
# ─────────────────────────────────────────────────────────────

def mask_coverage_pct(maska: np.ndarray) -> float:
    """Procento aktivních pixelů v detekční masce."""
    return 100.0 * int((maska > 0).sum()) / maska.size


def mask_status(pct: float) -> str:
    """Slovní hodnocení pokrytí masky (jako v aplikaci)."""
    if pct == 0:      return "Prázdná – snižte Z práh"
    if pct < 1:       return "OK"
    if pct < 5:       return "Mírně vysoké pokrytí"
    if pct < 10:      return "Vysoké – zvyšte Z práh"
    return "Příliš mnoho pixelů"


def z_status(max_z: float, z_thresh: float) -> str:
    """Slovní hodnocení maxima Z-skóre."""
    if max_z < z_thresh:       return "Pod prahem"
    if max_z < z_thresh * 2:   return "Hotspot (slabý)"
    if max_z < z_thresh * 4:   return "Hotspot (střední)"
    return "Hotspot (kritický)"


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

def anotuj(img: np.ndarray, hotspoty: List[Hotspot],
           z_thresh: float, z_window: int) -> np.ndarray:
    """
    Kreslí barevné obdélníky podle závažnosti:
      Žlutá    = Slabý    (confidence 0)
      Oranžová = Střední  (confidence 1–2)
      Červená  = Kritický (confidence 3+)

    Popisek u každého hotspotu: pouze číslo (#1, #2, …)
    Vlevo nahoře: pouze celkový počet hotspotů.
    """
    out = img.copy()
    pad = 5
    for hs in hotspoty:
        b  = hs.zavaznost_bgr
        x1 = max(0, hs.x - pad);  y1 = max(0, hs.y - pad)
        x2 = min(img.shape[1] - 1, hs.x + hs.w + pad)
        y2 = min(img.shape[0] - 1, hs.y + hs.h + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), b, 2)
        lbl = f"#{hs.id}"
        cv2.putText(out, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, b, 1, cv2.LINE_AA)

    # Vlevo nahoře – pouze počet hotspotů
    line = f"Hotspoty: {len(hotspoty)}"
    cv2.putText(out, line, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, line, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB VÝSTUP  (4-panelový debug graf)
# ─────────────────────────────────────────────────────────────

def _legenda_zavaznosti(ax):
    """Přidá legendu závažnosti do osy."""
    patches = [
        mpatches.Patch(color='#ef4444', label='Kritický (C≥3)'),
        mpatches.Patch(color='#f97316', label='Střední  (C 1–2)'),
        mpatches.Patch(color='#eab308', label='Slabý    (C 0)'),
    ]
    ax.legend(handles=patches, loc='upper right',
              fontsize=7, framealpha=0.7)


def vykresli(img: np.ndarray, annotated: np.ndarray,
             zmap: np.ndarray, maska: np.ndarray,
             hotspoty: List[Hotspot],
             soubor: str,
             z_thresh: float,
             mask_pct: float,
             save_path: Optional[str] = None):
    """
    4-panelový debug graf:
      1. Originální snímek
      2. Z-skóre mapa s colorbar
      3. Detekční maska + info o pokrytí
      4. Anotovaný výsledek s legendou závažnosti
    """
    n_krit = sum(1 for h in hotspoty if h.confidence >= 3)
    n_str  = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
    n_slab = sum(1 for h in hotspoty if h.confidence == 0)
    max_z  = max((h.max_z for h in hotspoty), default=0.0)

    fig = plt.figure(figsize=(22, 6))
    fig.patch.set_facecolor('#080b10')
    fig.suptitle(
        f"Detekce hotspotů – {os.path.basename(soubor)}\n"
        f"Celkem: {len(hotspoty)}  │  Kritické: {n_krit}  │  "
        f"Střední: {n_str}  │  Slabé: {n_slab}  │  "
        f"Max Z: {max_z:.2f}  │  Maska: {mask_pct:.1f}% ({mask_status(mask_pct)})",
        fontsize=10, color='#e5e7eb', fontweight='bold'
    )

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06)
    axes = [fig.add_subplot(gs[j]) for j in range(4)]

    panels = [
        (cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originální snímek",  None),
        (np.clip(zmap, -2, None),                    "Z-skóre mapa",       'RdYlBu_r'),
        (maska,                                       "Detekční maska",     'gray'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Anotovaný výsledek", None),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='#9ca3af', fontsize=9, pad=5)
        ax.axis('off')
        ax.set_facecolor('#080b10')
        if cmap == 'RdYlBu_r':
            cbar = plt.colorbar(im, ax=ax, fraction=0.046,
                                pad=0.04, shrink=0.85)
            cbar.ax.tick_params(colors='#6b7280', labelsize=7)
            # Čára prahu
            cbar.ax.axhline(y=z_thresh, color='white',
                            linewidth=1.2, linestyle='--', alpha=0.8)
        if title == "Anotovaný výsledek":
            _legenda_zavaznosti(ax)
        if title == "Detekční maska":
            ax.set_title(
                f"Detekční maska  ({mask_pct:.1f}%  –  {mask_status(mask_pct)})",
                color='#9ca3af', fontsize=8, pad=5
            )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        log.info(f"  Debug graf: {save_path}")

    if SHOW_PLOT:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV
# ─────────────────────────────────────────────────────────────

def uloz_csv(vysledky: List[Vysledek], folder: str):
    path = os.path.join(folder, '_hotspoty_souhrn.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow([
            'Soubor', 'ID', 'Závažnost', 'Confidence',
            'Max Z', 'Mean Z', 'Max jas',
            'X', 'Y', 'Šířka', 'Výška',
            'Plocha [px²]', 'Poměr stran', 'Kruhovitost',
            'Z práh', 'Z okno', 'Maska pokrytí [%]',
        ])
        for v in vysledky:
            for h in v.hotspoty:
                w.writerow([
                    v.soubor, h.id, h.zavaznost, h.confidence,
                    h.max_z, h.mean_z, h.max_intensity,
                    h.x, h.y, h.w, h.h,
                    h.area, h.aspect_ratio, h.circularity,
                    v.z_thresh, v.z_window, round(v.mask_pct, 2),
                ])
    log.info(f"CSV: {path}")


# ─────────────────────────────────────────────────────────────
#  ZPRACOVÁNÍ JEDNOHO SNÍMKU
# ─────────────────────────────────────────────────────────────

def zpracuj(cesta: str, output_folder: str,
            pre_blur: int, z_window: int, z_thresh: float,
            morph_open: int, merge_dist: int,
            min_area: int, max_area: int, max_aspect: float,
            conf_min: int) -> Optional[Vysledek]:

    img = cv2.imread(cesta)
    if img is None:
        log.warning(f"Nelze načíst: {cesta}")
        return None

    zmap, maska, merged, hotspoty = detekuj(
        img,
        pre_blur   = pre_blur,
        z_window   = z_window,
        z_thresh   = z_thresh,
        morph_open = morph_open,
        min_area   = min_area,
        max_area   = max_area,
        max_aspect = max_aspect,
        merge_dist = merge_dist,
        conf_min   = conf_min,
    )

    pct    = mask_coverage_pct(maska)
    n_krit = sum(1 for h in hotspoty if h.confidence >= 3)
    n_str  = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
    n_slab = sum(1 for h in hotspoty if h.confidence == 0)
    max_z  = max((h.max_z for h in hotspoty), default=0.0)

    log.info(
        f"  Hotspoty: {len(hotspoty)} "
        f"(krit={n_krit}, stř={n_str}, slab={n_slab})  "
        f"Max Z={max_z:.2f}  Maska={pct:.1f}% [{mask_status(pct)}]  "
        f"({z_status(max_z, z_thresh)})"
    )

    v = Vysledek(
        soubor     = cesta,
        n_hotspotu = len(hotspoty),
        hotspoty   = hotspoty,
        z_thresh   = z_thresh,
        z_window   = z_window,
        n_krit     = n_krit,
        n_str      = n_str,
        n_slab     = n_slab,
        mask_pct   = pct,
    )

    annotated = anotuj(img, hotspoty, z_thresh, z_window)
    nazev     = os.path.splitext(os.path.basename(cesta))[0]

    if SAVE_PNG:
        out_png = os.path.join(output_folder, f"vysledek_{nazev}.png")
        cv2.imwrite(out_png, annotated)
        log.info(f"  PNG: {out_png}")

    debug_path = (os.path.join(output_folder, f"debug_{nazev}.png")
                  if SAVE_DEBUG else None)
    if SHOW_PLOT or SAVE_DEBUG:
        vykresli(img, annotated, zmap, maska, hotspoty,
                 cesta, z_thresh, pct,
                 save_path=debug_path if SAVE_DEBUG else None)

    if SHOW_WINDOW:
        porovnani = np.hstack((img, annotated))
        win = f"{nazev}  (MEZERNIK = další,  Q = konec)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 700)
        cv2.imshow(win, porovnani)

    return v


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ SMYČKA
# ─────────────────────────────────────────────────────────────

def main():
    # ── Načtení předvolby (pokud je nastavena) ──────────────
    global PRE_BLUR, Z_WINDOW, Z_THRESH, MORPH_OPEN_K, MERGE_DIST
    global HS_MIN_AREA, HS_MAX_AREA, HS_MAX_ASPECT, FILTER_ZAVAZNOST

    if FILTER_PRESET and FILTER_PRESET in PRESETS:
        p = PRESETS[FILTER_PRESET]
        PRE_BLUR         = p["blur_k"]
        Z_WINDOW         = p["z_window"]
        Z_THRESH         = p["z_thresh"]
        MORPH_OPEN_K     = p["morph_k"]
        MERGE_DIST       = p["merge_dist"]
        HS_MIN_AREA      = p["min_area"]
        HS_MAX_AREA      = p["max_area"]
        HS_MAX_ASPECT    = p["max_aspect"]
        FILTER_ZAVAZNOST = p["filter_zavaznost"]
        log.info(f"Použita předvolba: {FILTER_PRESET}")

    conf_min = FILTER_ZAVAZNOST_MAP.get(FILTER_ZAVAZNOST, 1)

    log.info(
        f"Parametry: blur={PRE_BLUR}  z_window={Z_WINDOW}  "
        f"z_thresh={Z_THRESH}  morph={MORPH_OPEN_K}  "
        f"merge={MERGE_DIST}  area={HS_MIN_AREA}–{HS_MAX_AREA}  "
        f"aspect≤{HS_MAX_ASPECT}  filter={FILTER_ZAVAZNOST}(conf≥{conf_min})"
    )

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
        v = zpracuj(
            cesta, OUTPUT_FOLDER,
            pre_blur   = PRE_BLUR,
            z_window   = Z_WINDOW,
            z_thresh   = Z_THRESH,
            morph_open = MORPH_OPEN_K,
            merge_dist = MERGE_DIST,
            min_area   = HS_MIN_AREA,
            max_area   = HS_MAX_AREA,
            max_aspect = HS_MAX_ASPECT,
            conf_min   = conf_min,
        )
        if v:
            vysledky.append(v)

        if SHOW_WINDOW:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                log.info("Uživatel ukončil zpracování.")
                break

    cv2.destroyAllWindows()

    if SAVE_CSV and vysledky:
        uloz_csv(vysledky, OUTPUT_FOLDER)

    # ── Souhrnný report ──────────────────────────────────────
    celkem    = sum(v.n_hotspotu for v in vysledky)
    krit_tot  = sum(v.n_krit    for v in vysledky)
    str_tot   = sum(v.n_str     for v in vysledky)
    slab_tot  = sum(v.n_slab    for v in vysledky)

    log.info("─" * 60)
    log.info(f"HOTOVO – {len(vysledky)} snímků  │  "
             f"Celkem hotspotů: {celkem}  "
             f"(krit={krit_tot}, stř={str_tot}, slab={slab_tot})")
    log.info("─" * 60)


if __name__ == "__main__":
    main()