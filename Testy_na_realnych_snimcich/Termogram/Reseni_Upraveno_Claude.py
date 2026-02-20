"""
╔══════════════════════════════════════════════════════════════╗
║     DETEKCE HOTSPOTŮ – Termogramy FV panelů  v3             ║
║  Nastavení: upravte proměnné v sekci KONFIGURACE             ║
╚══════════════════════════════════════════════════════════════╝

Přístup k detekci:
  Celý snímek se analyzuje bez segmentace panelů (ta byla nespolehlivá).
  Hotspot = místo které je zároveň:
    1. Lokálně teplejší než okolí   → Top-Hat filtr (relativní anomálie)
    2. Globálně mezi nejteplejšími  → Absolutní percentil (globální anomálie)
    3. Lokálně výrazně odlišné      → Z-skóre mapa (statistická odchylka)
  Výsledek je AND/OR těchto masek → výrazné snížení false positives.
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

INPUT_FOLDER  = '.'
OUTPUT_FOLDER = 'vysledky'

# Předzpracování
PRE_BLUR_KERNEL = 3       # gaussovský blur před detekcí pro potlačení šumu
                          # 0 = vypnuto, 3 nebo 5 doporučeno

# Metoda 1 – Top-Hat (lokální relativní anomálie)
TOPHAT_KERNEL   = 51      # kernel top-hat [px] – větší = ignoruje texturu buněk
                          # doporučeno: 3–5× velikost jedné solární buňky
TOPHAT_PCT      = 97      # percentil pro práh top-hat signálu (95–99)

# Metoda 2 – Absolutní percentil (globální anomálie)
ABS_PCT         = 97      # percentil pro absolutní práh jasu (95–99)

# Metoda 3 – Lokální Z-skóre (statistická odchylka od lokálního průměru)
ZSCORE_WINDOW   = 61      # velikost okna pro výpočet lokálního průměru a std [px]
ZSCORE_THRESH   = 2.5     # minimální Z-skóre pro hotspot (typicky 2.0–3.5)
                          # vyšší = přísnější = méně detekcí

# Kombinace metod – váhovaný součet
# Z-skóre má váhu 2 (samo o sobě stačí k detekci), Top-Hat a Abs mají váhu 1.
# Hotspot = pixel jehož váhovaný součet >= SCORE_THRESHOLD.
# Efektivní chování:
#   threshold=2 → Z samo stačí NEBO Top-Hat+Abs dohromady
#   threshold=3 → Z+jedna další NEBO všechny tři
#   threshold=4 → Z musí být + obě ostatní (velmi přísné)
ZSCORE_WEIGHT   = 2       # váha Z-skóre metody (doporučeno 2)
TOPHAT_WEIGHT   = 1       # váha Top-Hat metody
ABS_WEIGHT      = 1       # váha absolutní metody
SCORE_THRESHOLD = 2       # minimální váhovaný součet pro detekci hotspotu

# Filtrování kontur
HS_MIN_AREA    = 40       # minimální plocha hotspotu [px²]
HS_MAX_ASPECT  = 4.0      # maximální poměr stran (příliš protáhlé = artefakt)
HS_MERGE_DIST  = 20       # vzdálenost pro sloučení blízkých fragmentů [px]
                          # 0 = vypnuto

# Skóre spolehlivosti (váhované, odpovídá logice kombinace)
#   +2 pokud byl detekován Z-skóre metodou
#   +1 pokud byl detekován Top-Hat metodou
#   +1 pokud byl detekován absolutní metodou
#   +1 pokud max jas > abs_práh + 20
#   +1 pokud max jas > abs_práh + 40
#   +1 pokud kruhovitost > 0.5
#   +1 pokud max Z > 1.5 × ZSCORE_THRESH
#   +1 pokud max Z > 2.5 × ZSCORE_THRESH  (velmi silný hotspot)
# Minimální dosažitelné skóre při detekci samotným Z = 2
CONF_MIN       = 2        # minimální skóre pro zobrazení (doporučeno 2)

# Vizualizace
SHOW_WINDOW    = True
SHOW_MATLPLOT  = True
SAVE_PNG       = True
SAVE_CSV       = True
DPI            = 200

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
    circularity:    float
    max_zscore:     float
    metody:         List[str]     # které metody ho potvrdily
    confidence:     int

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


@dataclass
class Vysledek:
    soubor:      str
    n_hotspotu:  int
    hotspoty:    List[Hotspot] = field(default_factory=list)
    tophat_t:    int = 0
    abs_t:       int = 0
    zscore_t:    float = 0.0


# ─────────────────────────────────────────────────────────────
#  PŘEDZPRACOVÁNÍ
# ─────────────────────────────────────────────────────────────

def priprav_gray(img: np.ndarray) -> np.ndarray:
    """Převod na šedotón + volitelný blur pro potlačení šumu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if PRE_BLUR_KERNEL >= 3:
        k = PRE_BLUR_KERNEL | 1  # zajistíme liché číslo
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


# ─────────────────────────────────────────────────────────────
#  METODA 1 – TOP-HAT
# ─────────────────────────────────────────────────────────────

def tophat_maska(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Top-Hat morfologický filtr detekuje lokálně světlé objekty
    menší než je kernel. Výsledkem je obraz kde jsou zachovány
    pouze lokální teplotní vrcholy (hotspoty) a okolní pozadí = 0.

    Práh = percentil top-hat signálu → adaptivní na každý snímek.
    Vrací (maska, tophat_signal, použitý_práh).
    """
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    t      = int(np.percentile(tophat, TOPHAT_PCT))
    _, m   = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8), tophat, t


# ─────────────────────────────────────────────────────────────
#  METODA 2 – ABSOLUTNÍ PERCENTIL
# ─────────────────────────────────────────────────────────────

def abs_maska(gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Detekuje globálně nejsvětlejší pixely v celém snímku.
    Práh = percentil jasu → zachytí jen nejvyšší X % hodnot.
    """
    t    = int(np.percentile(gray, ABS_PCT))
    _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8), t


# ─────────────────────────────────────────────────────────────
#  METODA 3 – LOKÁLNÍ Z-SKÓRE
# ─────────────────────────────────────────────────────────────

def zscore_maska(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lokální Z-skóre = (pixel - lokální_průměr) / lokální_std

    Každý pixel je porovnán se svým okolím definovaným oknem
    ZSCORE_WINDOW × ZSCORE_WINDOW. Hotspot = pixel jehož teplota
    je výrazně nad lokálním průměrem (Z > ZSCORE_THRESH).

    Výhoda oproti top-hat: není citlivý na absolutní hodnoty jasu,
    funguje i na snímcích s nerovnoměrným osvětlením nebo gradientem
    teploty přes celé pole panelů.
    """
    g_f  = gray.astype(np.float32)
    k    = ZSCORE_WINDOW | 1   # liché číslo

    # Lokální průměr a E[X²] přes klouzavé okno
    mean_loc = cv2.boxFilter(g_f, -1, (k, k))
    sq_mean  = cv2.boxFilter(g_f ** 2, -1, (k, k))
    var_loc  = np.clip(sq_mean - mean_loc ** 2, 0, None)
    std_loc  = np.sqrt(var_loc) + 1e-6    # +epsilon aby nedošlo k dělení nulou

    zscore   = (g_f - mean_loc) / std_loc
    mask_z   = (zscore >= ZSCORE_THRESH).astype(np.uint8) * 255

    return mask_z, zscore


# ─────────────────────────────────────────────────────────────
#  KOMBINACE MASEK
# ─────────────────────────────────────────────────────────────

def kombinuj_masky(m_tophat: np.ndarray,
                   m_abs:    np.ndarray,
                   m_zscore: np.ndarray) -> np.ndarray:
    """
    Váhovaný součet masek – Z-skóre má váhu 2, ostatní váhu 1.

    Logika prahu SCORE_THRESHOLD=2:
      Z aktivní                   → 2 body → detekováno ✓
      Top-Hat + Abs aktivní       → 2 body → detekováno ✓
      Pouze Top-Hat nebo Abs      → 1 bod  → ignorováno ✗
      Z + cokoliv dalšího         → 3+ bodů → detekováno ✓

    Z-skóre tak může samo detekovat hotspot i bez potvrzení ostatními
    metodami, ale Top-Hat nebo Abs samotné nestačí – musí se shodovat.
    """
    vazeny = (m_zscore > 0).astype(np.uint8) * ZSCORE_WEIGHT + \
             (m_tophat > 0).astype(np.uint8) * TOPHAT_WEIGHT + \
             (m_abs    > 0).astype(np.uint8) * ABS_WEIGHT
    return ((vazeny >= SCORE_THRESHOLD) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  SLOUČENÍ BLÍZKÝCH FRAGMENTŮ
# ─────────────────────────────────────────────────────────────

def sluc_fragmenty(maska: np.ndarray, dist: int) -> np.ndarray:
    """Dilatace + eroze → sloučí fragmenty blíže než dist px."""
    if dist <= 0:
        return maska
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist | 1, dist | 1))
    return cv2.erode(cv2.dilate(maska, k), k)


# ─────────────────────────────────────────────────────────────
#  SKÓRE SPOLEHLIVOSTI
# ─────────────────────────────────────────────────────────────

def skore(metody: List[str], max_int: float, abs_t: int,
          circularity: float, max_z: float) -> int:
    """
    Skóre spolehlivosti reflektuje váhování kombinace:
      - Z-skóre přítomnost = 2 body (vyšší váha)
      - Top-Hat nebo Abs   = 1 bod každá
      - Bonus za výrazný jas, kompaktní tvar, vysoké Z
    """
    s = 0
    if "zscore"   in metody: s += ZSCORE_WEIGHT
    if "tophat"   in metody: s += TOPHAT_WEIGHT
    if "absolute" in metody: s += ABS_WEIGHT
    # Bonusy za kvalitu hotspotu
    if max_int > abs_t + 20:        s += 1
    if max_int > abs_t + 40:        s += 1
    if circularity > 0.5:           s += 1
    if max_z > ZSCORE_THRESH * 1.5: s += 1   # výrazné Z = silný hotspot
    if max_z > ZSCORE_THRESH * 2.5: s += 1   # velmi silné Z
    return s


# ─────────────────────────────────────────────────────────────
#  EXTRAKCE A FILTROVÁNÍ HOTSPOTŮ
# ─────────────────────────────────────────────────────────────

def extrahuj_hotspoty(combined:  np.ndarray,
                       m_tophat: np.ndarray,
                       m_abs:    np.ndarray,
                       m_zscore: np.ndarray,
                       gray:     np.ndarray,
                       zscore:   np.ndarray,
                       abs_t:    int) -> List[Hotspot]:

    merged   = sluc_fragmenty(combined, HS_MERGE_DIST)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspoty = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < HS_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if aspect > HS_MAX_ASPECT or aspect < 1.0 / HS_MAX_ASPECT:
            continue

        # Maska kontury
        cm = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)

        # Metriky jasu
        px       = gray[cm > 0]
        mean_int = float(px.mean()) if px.size else 0.0
        max_int  = float(px.max())  if px.size else 0.0

        # Max Z-skóre v oblasti
        px_z    = zscore[cm > 0]
        max_z   = float(px_z.max()) if px_z.size else 0.0

        # Kruhovitost
        perim       = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0

        # Které metody hotspot potvrdily
        metody = []
        if m_tophat[cm > 0].any(): metody.append("tophat")
        if m_abs[cm > 0].any():    metody.append("absolute")
        if m_zscore[cm > 0].any(): metody.append("zscore")

        conf = skore(metody, max_int, abs_t, circularity, max_z)
        if conf < CONF_MIN:
            continue

        hotspoty.append(Hotspot(
            id=len(hotspoty) + 1,
            x=x, y=y, w=w, h=h,
            area=round(area, 1),
            aspect_ratio=round(aspect, 2),
            mean_intensity=round(mean_int, 1),
            max_intensity=round(max_int, 1),
            circularity=round(circularity, 3),
            max_zscore=round(max_z, 2),
            metody=metody,
            confidence=conf,
        ))

    hotspoty.sort(key=lambda h: h.confidence, reverse=True)
    for i, h in enumerate(hotspoty): h.id = i + 1
    return hotspoty


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

CONF_BGR = {
    1: (0,   220, 220),   # žlutá
    2: (0,   165, 255),   # oranžová
    3: (0,    60, 255),   # červeno-oranžová
    4: (0,     0, 255),   # červená
    5: (50,    0, 220),   # tmavě červená
    6: (100,   0, 180),
    7: (120,   0, 140),
}

def anotuj(img: np.ndarray, hotspoty: List[Hotspot],
            tophat_t: int, abs_t: int, n: int) -> np.ndarray:
    out = img.copy()
    pad = 4
    for hs in hotspoty:
        barva = CONF_BGR.get(hs.confidence, (0, 0, 255))
        x1 = max(0, hs.x - pad);  y1 = max(0, hs.y - pad)
        x2 = min(img.shape[1]-1, hs.x+hs.w+pad)
        y2 = min(img.shape[0]-1, hs.y+hs.h+pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), barva, 2)
        label = f"#{hs.id} C{hs.confidence}"
        cv2.putText(out, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, barva, 1, cv2.LINE_AA)

    for i, line in enumerate([f"Hotspoty: {n}",
                               f"TopHat T: {tophat_t}",
                               f"Abs T:    {abs_t}",
                               f"Z-thresh: {ZSCORE_THRESH}"]):
        yp = 22 + i * 20
        cv2.putText(out, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0,0,0),         3, cv2.LINE_AA)
        cv2.putText(out, line, (8, yp), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255,255,255),   1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB VÝSTUP
# ─────────────────────────────────────────────────────────────

def vykresli(img, annotated, m_tophat, m_abs, m_zscore, combined,
             zscore_map, hotspoty, soubor, save_path=None):

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Analýza hotspotů – {os.path.basename(soubor)}",
                 fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25)

    data = [
        (fig.add_subplot(gs[0, 0]), cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originál",         None),
        (fig.add_subplot(gs[0, 1]), m_tophat,                                    "Top-Hat maska",    'hot'),
        (fig.add_subplot(gs[0, 2]), m_abs,                                       "Absolutní maska",  'hot'),
        (fig.add_subplot(gs[0, 3]), m_zscore,                                    "Z-skóre maska",    'hot'),
        (fig.add_subplot(gs[1, 0]), zscore_map,                                  "Z-skóre mapa",     'RdYlBu_r'),
        (fig.add_subplot(gs[1, 1]), combined,                                    "Kombinovaná maska",'hot'),
        (fig.add_subplot(gs[1, 2:]), cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Výsledek detekce", None),
    ]

    for ax, im, title, cmap in data:
        kwargs = dict(cmap=cmap) if cmap and im.ndim == 2 else {}
        if cmap == 'RdYlBu_r' and im.ndim != 2:
            kwargs = {}
        ax.imshow(im, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    patches = [
        mpatches.Patch(color='#ff4444', label=f'Vysoká C≥4 ({sum(1 for h in hotspoty if h.confidence>=4)})'),
        mpatches.Patch(color='#ff8800', label=f'Střední C=2–3 ({sum(1 for h in hotspoty if 2<=h.confidence<4)})'),
        mpatches.Patch(color='#dddd00', label=f'Nízká C=1 ({sum(1 for h in hotspoty if h.confidence==1)})'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
        log.info(f"  Graf uložen: {save_path}")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV EXPORT
# ─────────────────────────────────────────────────────────────

def uloz_csv(vysledky: List[Vysledek], folder: str):
    path = os.path.join(folder, '_hotspoty_souhrn.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Soubor','ID','Metody','Spolehlivost',
                    'X','Y','Š','V','Plocha','Poměr stran',
                    'Střední jas','Max jas','Kruhovitost','Max Z-skóre',
                    'TopHat práh','Abs práh','Z práh'])
        for v in vysledky:
            for h in v.hotspoty:
                w.writerow([
                    v.soubor, h.id, '+'.join(h.metody), h.confidence,
                    h.x, h.y, h.w, h.h, h.area, h.aspect_ratio,
                    h.mean_intensity, h.max_intensity,
                    h.circularity, h.max_zscore,
                    v.tophat_t, v.abs_t, ZSCORE_THRESH
                ])
    log.info(f"CSV uloženo: {path}")


# ─────────────────────────────────────────────────────────────
#  ZPRACOVÁNÍ JEDNOHO SNÍMKU
# ─────────────────────────────────────────────────────────────

def zpracuj(cesta: str, output_folder: str) -> Optional[Vysledek]:
    img = cv2.imread(cesta)
    if img is None:
        log.warning(f"  Nelze načíst: {cesta}")
        return None

    gray = priprav_gray(img)

    m_th,  tophat_sig, tophat_t = tophat_maska(gray)
    m_abs, abs_t                = abs_maska(gray)
    m_z,   zscore_map           = zscore_maska(gray)

    log.info(f"  Prahy → TopHat: {tophat_t}  Abs: {abs_t}  Z: {ZSCORE_THRESH}")
    log.info(f"  Aktivní px → TopHat: {(m_th>0).sum()}  "
             f"Abs: {(m_abs>0).sum()}  Z: {(m_z>0).sum()}")

    combined = kombinuj_masky(m_th, m_abs, m_z)
    hotspoty = extrahuj_hotspoty(combined, m_th, m_abs, m_z,
                                  gray, zscore_map, abs_t)

    log.info(f"  Nalezeno hotspotů: {len(hotspoty)}")

    vysledek = Vysledek(soubor=cesta, n_hotspotu=len(hotspoty),
                         hotspoty=hotspoty, tophat_t=tophat_t,
                         abs_t=abs_t, zscore_t=ZSCORE_THRESH)

    annotated = anotuj(img, hotspoty, tophat_t, abs_t, len(hotspoty))

    nazev = os.path.splitext(os.path.basename(cesta))[0]
    if SAVE_PNG:
        p = os.path.join(output_folder, f"vysledek_{nazev}.png")
        cv2.imwrite(p, annotated)
        log.info(f"  PNG: {p}")

    mpl_path = os.path.join(output_folder, f"graf_{nazev}.png") if SAVE_PNG else None
    if SHOW_MATLPLOT:
        vykresli(img, annotated, m_th, m_abs, m_z, combined,
                 zscore_map, hotspoty, cesta, mpl_path)

    if SHOW_WINDOW:
        porovnani = np.hstack((img, annotated))
        for text, x in [("Originál", 15), ("Detekce", img.shape[1]+15)]:
            cv2.putText(porovnani, text, (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),       3, cv2.LINE_AA)
            cv2.putText(porovnani, text, (x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        win = f"{nazev}  (MEZERNIK=další  Q=konec)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1600, 700)
        cv2.imshow(win, porovnani)
        cv2.destroyWindow(win)

    return vysledek


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ SMYČKA
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    seznam = sorted(sum([
        glob.glob(os.path.join(INPUT_FOLDER, p))
        for p in ['*.jpg','*.jpeg','*.png','*.bmp','*.tif','*.tiff']
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
                log.info("Ukončuji...")
                break

    cv2.destroyAllWindows()

    if SAVE_CSV and vysledky:
        uloz_csv(vysledky, OUTPUT_FOLDER)

    celkem = sum(v.n_hotspotu for v in vysledky)
    log.info("─────────────────────────────────────────")
    log.info(f"Zpracováno:  {len(vysledky)} snímků")
    log.info(f"Hotspoty:    {celkem}  "
             f"(průměr {celkem/len(vysledky):.1f}/snímek)" if vysledky else "")
    log.info("HOTOVO.")


if __name__ == "__main__":
    main()