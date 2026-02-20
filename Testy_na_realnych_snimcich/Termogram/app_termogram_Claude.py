"""
╔══════════════════════════════════════════════════════════════╗
║   HOTSPOTY FV PANELŮ – Streamlit aplikace                    ║
║   Spuštění: streamlit run hotspoty_app.py                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import io
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE STRÁNKY
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FV Hotspot Detektor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Pozadí */
.stApp { background: #0a0c10; }
section[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #1e2530; }

/* Metriky */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-card .val {
    font-size: 1.9rem;
    font-weight: 600;
    color: #f97316;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Sekce v sidebaru */
.param-section {
    background: #111827;
    border-left: 3px solid #f97316;
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 8px 0 4px 0;
    font-size: 0.78rem;
    font-weight: 600;
    color: #f97316;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

/* Confidence badge */
.badge-high   { color: #ef4444; font-weight: 700; }
.badge-mid    { color: #f97316; font-weight: 600; }
.badge-low    { color: #eab308; font-weight: 500; }

/* Tlačítka */
.stDownloadButton > button {
    background: #1e3a5f !important;
    color: #93c5fd !important;
    border: 1px solid #2563eb !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: #2563eb !important;
    color: white !important;
}

/* Headings */
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e5e7eb !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATOVÉ TŘÍDY
# ─────────────────────────────────────────────────────────────

@dataclass
class Hotspot:
    id: int
    x: int; y: int; w: int; h: int
    area: float
    aspect_ratio: float
    mean_intensity: float
    max_intensity: float
    circularity: float
    typ: str
    confidence: int
    edge_distance: float

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


# ─────────────────────────────────────────────────────────────
#  DETEKČNÍ PIPELINE (stejná logika jako hotspoty.py)
# ─────────────────────────────────────────────────────────────

def _zkus_segmentaci(gray, thresh_val, inverse):
    flags = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    _, b = cv2.threshold(gray, thresh_val, 255, flags)
    k5   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    b    = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k5, iterations=1)
    b    = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k5, iterations=3)
    return b


def _filtruj_kontury(bin_mask, gray, min_area, min_solidity, max_coverage):
    total_px = gray.shape[0] * gray.shape[1]
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    if pokryti > max_coverage:
        return np.zeros_like(gray), 0, pokryti
    return maska, n, pokryti


def najdi_panely(gray, p):
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = int(otsu_val)

    b_inv = _zkus_segmentaci(gray, otsu_val, True)
    mA, nA, covA = _filtruj_kontury(b_inv, gray, p["panel_min_area"], p["panel_solidity"], p["panel_max_cov"])

    b_nrm = _zkus_segmentaci(gray, otsu_val, False)
    mB, nB, covB = _filtruj_kontury(b_nrm, gray, p["panel_min_area"], p["panel_solidity"], p["panel_max_cov"])

    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    lapl = cv2.Laplacian(blur, cv2.CV_64F)
    lapl_u8 = np.uint8(np.clip(np.abs(lapl), 0, 255))
    _, gm = cv2.threshold(lapl_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    gm  = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, k25)
    mC, nC, covC = _filtruj_kontury(gm, gray, p["panel_min_area"], p["panel_solidity"], p["panel_max_cov"])

    kandidati = [(nA, mA, "Otsu INV",  covA),
                 (nB, mB, "Otsu NORM", covB),
                 (nC, mC, "Gradient",  covC)]
    best_n, best_mask, best_name, best_cov = max(kandidati, key=lambda x: x[0])

    if best_n == 0:
        best_mask = np.ones_like(gray) * 255
        best_name = "Fallback (celý snímek)"

    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best_mask = cv2.erode(best_mask, k_e, iterations=p["panel_erode"])

    debug = {
        "strategie": best_name,
        "pokryti":   best_cov,
        "INV":  (nA, f"{covA:.0%}"),
        "NORM": (nB, f"{covB:.0%}"),
        "GRAD": (nC, f"{covC:.0%}"),
    }
    return best_mask, best_n, otsu_val, debug


def detekuj_hotspoty(gray, maska, p):
    ks = p["tophat_kernel"] | 1  # zajistíme liché číslo
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)

    px_th = tophat[maska > 0]
    px_gr = gray[maska > 0]
    tophat_t = int(np.percentile(px_th, p["tophat_pct"])) if px_th.size else 50
    abs_t    = int(np.percentile(px_gr, p["abs_pct"]))    if px_gr.size else 190

    _, tb = cv2.threshold(tophat, tophat_t, 255, cv2.THRESH_BINARY)
    th_m  = cv2.bitwise_and(tb, tb, mask=maska)

    _, ab = cv2.threshold(gray, abs_t, 255, cv2.THRESH_BINARY)
    ab_m  = cv2.bitwise_and(ab, ab, mask=maska)

    combined = (cv2.bitwise_and(th_m, ab_m)
                if p["require_both"] else
                cv2.bitwise_or(th_m, ab_m))

    return th_m, ab_m, combined, tophat_t, abs_t


def sluc_blizke(combined, distance):
    if distance <= 0:
        return combined
    ks = distance | 1
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    return cv2.erode(cv2.dilate(combined, k), k)


def vypocti_confidence(typ, max_int, abs_t, edge_dist, circularity, edge_margin):
    score = 0
    if typ == 'combined':      score += 2
    elif typ in ('tophat', 'absolute'): score += 1
    if max_int > abs_t + 20:   score += 1
    if max_int > abs_t + 40:   score += 1
    if edge_dist > edge_margin * 3: score += 1
    if circularity > 0.5:      score += 1
    return score


def filtruj_hotspoty(combined, th_m, ab_m, gray, maska, abs_t, p):
    merged   = sluc_blizke(combined, p["merge_dist"])
    dist_map = cv2.distanceTransform(maska, cv2.DIST_L2, 5)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspoty = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < p["hs_min_area"]:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if aspect > p["hs_max_aspect"] or aspect < 1.0 / p["hs_max_aspect"]:
            continue

        cx_hs = x + w // 2; cy_hs = y + h // 2
        edge_dist = float(dist_map[
            max(0, min(cy_hs, dist_map.shape[0]-1)),
            max(0, min(cx_hs, dist_map.shape[1]-1))
        ])
        if edge_dist < p["edge_margin"]:
            continue

        cnt_mask = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        pixely   = gray[cnt_mask > 0]
        mean_int = float(pixely.mean()) if pixely.size else 0.0
        max_int  = float(pixely.max())  if pixely.size else 0.0

        perim       = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / perim**2) if perim > 0 else 0.0

        roi_m = cnt_mask[y:y+h, x:x+w]
        in_th = th_m[y:y+h, x:x+w][roi_m > 0].any()
        in_ab = ab_m[y:y+h, x:x+w][roi_m > 0].any()
        if in_th and in_ab: typ = 'combined'
        elif in_th:          typ = 'tophat'
        else:                typ = 'absolute'

        conf = vypocti_confidence(typ, max_int, abs_t, edge_dist,
                                   circularity, p["edge_margin"])
        if conf < p["conf_min"]:
            continue

        hotspoty.append(Hotspot(
            id=len(hotspoty)+1, x=x, y=y, w=w, h=h,
            area=round(area, 1), aspect_ratio=round(aspect, 2),
            mean_intensity=round(mean_int, 1), max_intensity=round(max_int, 1),
            circularity=round(circularity, 3), typ=typ,
            confidence=conf, edge_distance=round(edge_dist, 1),
        ))

    hotspoty.sort(key=lambda h: h.confidence, reverse=True)
    for i, h in enumerate(hotspoty): h.id = i + 1
    return hotspoty


def run_pipeline(img_bytes, params):
    arr  = np.frombuffer(img_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p    = json.loads(params)

    maska, n_pan, otsu_t, debug = najdi_panely(gray, p)
    th_m, ab_m, combined, tophat_t, abs_t = detekuj_hotspoty(gray, maska, p)
    hotspoty = filtruj_hotspoty(combined, th_m, ab_m, gray, maska, abs_t, p)

    return img, gray, maska, th_m, ab_m, combined, hotspoty, n_pan, tophat_t, abs_t, debug


# ─────────────────────────────────────────────────────────────
#  ANOTACE VÝSLEDKU
# ─────────────────────────────────────────────────────────────

CONF_COLORS_BGR = [
    (100,100,100),(0,255,255),(0,165,255),(0,80,255),(0,0,255),(0,0,200),(0,0,150)
]

def anotuj(img, maska, hotspoty, pad=4):
    out = img.copy()
    pc, _ = cv2.findContours(maska, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, pc, -1, (0, 200, 80), 1)
    for hs in hotspoty:
        barva = CONF_COLORS_BGR[min(hs.confidence, 6)]
        x1 = max(0, hs.x-pad); y1 = max(0, hs.y-pad)
        x2 = min(img.shape[1]-1, hs.x+hs.w+pad)
        y2 = min(img.shape[0]-1, hs.y+hs.h+pad)
        cv2.rectangle(out, (x1,y1), (x2,y2), barva, 2)
        label = f"#{hs.id} C{hs.confidence}"
        cv2.putText(out, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, barva, 1, cv2.LINE_AA)
    return out


def img_to_png(img_bgr):
    _, enc = cv2.imencode(".png", img_bgr)
    return enc.tobytes()


# ─────────────────────────────────────────────────────────────
#  MATPLOTLIB DETAIL PANEL
# ─────────────────────────────────────────────────────────────

def detail_figure(img, maska, th_m, ab_m, combined, annotated, hotspoty):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0a0c10")
    titles = ["Originál", "Maska panelů", "Top-Hat", "Absolutní", "Kombinovaná", "Výsledek"]
    imgs   = [
        cv2.cvtColor(img,       cv2.COLOR_BGR2RGB),
        maska,
        th_m,
        ab_m,
        combined,
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
    ]
    cmaps = [None, 'gray', 'hot', 'hot', 'hot', None]
    for ax, im_data, title, cmap in zip(axes.flat, imgs, titles, cmaps):
        ax.imshow(im_data, cmap=cmap)
        ax.set_title(title, color='#9ca3af', fontsize=10, pad=6)
        ax.axis('off')
        ax.set_facecolor("#0a0c10")

    patches = [
        mpatches.Patch(color='#ef4444', label=f'C≥4 ({sum(1 for h in hotspoty if h.confidence>=4)})'),
        mpatches.Patch(color='#f97316', label=f'C=2–3 ({sum(1 for h in hotspoty if 2<=h.confidence<4)})'),
        mpatches.Patch(color='#eab308', label=f'C=1 ({sum(1 for h in hotspoty if h.confidence==1)})'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               facecolor='#111827', labelcolor='white', fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


# ─────────────────────────────────────────────────────────────
#  SIDEBAR – parametry
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔥 FV Hotspot Detektor")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Nahrát termogram (JPG / PNG)",
                                 type=["jpg","jpeg","png","bmp","tif"])

    # ── PANELY ───────────────────────────────────────────────
    st.markdown('<div class="param-section">🟩 Detekce panelů</div>', unsafe_allow_html=True)

    panel_min_area = st.slider("Min. plocha panelu [px²]", 200, 10000, 1000, 100,
        help="Kontury menší než tato hodnota jsou ignorovány jako šum")
    panel_solidity = st.slider("Min. plnost tvaru (solidity)", 0.1, 0.9, 0.45, 0.05,
        help="Poměr plochy ke konvexnímu obalu. Nižší = povolí složitější tvary (vegetace, šikmé panely)")
    panel_max_cov  = st.slider("Max. pokrytí snímku [%]", 30, 95, 60, 5,
        help="Pokud maska pokryje více než X % snímku, strategie se zamítne jako neplatná") / 100
    panel_erode    = st.slider("Eroze okrajů panelu [iterace]", 0, 6, 2,
        help="Více iterací = menší maska, odstraní více okrajových artefaktů rámu")

    st.markdown('<div class="param-section">🌡️ Detekce hotspotů</div>', unsafe_allow_html=True)

    tophat_kernel = st.slider("Top-Hat kernel [px]", 11, 101, 51, 2,
        help="Větší kernel = detekuje jen velké anomálie, ignoruje texturu buněk")
    tophat_pct    = st.slider("Top-Hat percentil [%]", 85, 99, 97,
        help="Práh citlivosti. Vyšší = přísnější = méně detekcí")
    abs_pct       = st.slider("Absolutní práh percentil [%]", 85, 99, 97,
        help="Minimální absolutní jas hotspotu (percentil z oblasti panelů)")
    require_both  = st.toggle("Vyžadovat obě metody (AND)", value=True,
        help="True = hotspot musí být potvrzen oběma metodami → méně false positives")

    st.markdown('<div class="param-section">🔍 Filtrování</div>', unsafe_allow_html=True)

    hs_min_area  = st.slider("Min. plocha hotspotu [px²]", 5, 300, 40, 5,
        help="Velmi malé oblasti jsou šum nebo kompresní artefakty")
    hs_max_aspect= st.slider("Max. poměr stran (w/h)", 1.5, 8.0, 4.0, 0.5,
        help="Příliš protáhlé obdélníky jsou artefakty okrajů nebo kabelů")
    edge_margin  = st.slider("Min. vzdálenost od okraje panelu [px]", 0, 30, 8,
        help="Hotspoty příliš blízko okraje = tepelný vliv rámu → zahodit")
    merge_dist   = st.slider("Sloučení blízkých fragmentů [px]", 0, 60, 20,
        help="Fragmenty blíže než X px se sloučí do jednoho hotspotu")
    conf_min     = st.slider("Min. skóre spolehlivosti (0–6)", 0, 5, 1,
        help="0 = zobraz vše, vyšší = jen spolehlivé hotspoty")

    st.markdown("---")
    st.markdown('<div class="param-section">🎨 Zobrazení</div>', unsafe_allow_html=True)
    show_debug = st.toggle("Zobrazit debug panel (6 masek)", value=False)


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ OBSAH
# ─────────────────────────────────────────────────────────────

st.title("🔥 Detekce hotspotů FV panelů")

if uploaded is None:
    st.info("👈  Nahrajte termogram v levém panelu.")

    # Vysvětlení parametrů
    with st.expander("📖  Jak parametry ovlivňují výsledek", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**🟩 Detekce panelů**
- **Min. plocha** – odfiltruje malé objekty (stromy, budovy)
- **Plnost (solidity)** – snižte pro šikmé / oříznuté panely
- **Max. pokrytí** – klíčový parametr: pokud maska = celý snímek → segmentace selhala
- **Eroze okrajů** – více = striktnější maska, méně okrajových artefaktů

**🌡️ Hotspoty – Top-Hat**
Top-Hat filtr detekuje lokálně světlé oblasti relativně k okolí.
- **Kernel** – menší = citlivější na malé anomálie, ale zachytí i texturu buněk
- **Percentil** – zvyšte pokud je maska přeplněná (95→99)
""")
        with col2:
            st.markdown("""
**🌡️ Hotspoty – Absolutní práh**
Detekuje globálně nejhořejší oblasti v celé scéně.
- **Percentil** – 97 = nejhořejší 3 % pixelů v oblasti panelů

**AND vs OR kombinace**
- **AND (oba)** – hotspot musí být potvrzen oběma metodami → spolehlivé, méně detekcí
- **OR (jeden)** – stačí jedna metoda → více detekcí, více false positives

**🔍 Filtrování**
- **Skóre spolehlivosti** – kombinace více kritérií (0–6), červená = nejvyšší
- **Sloučení** – rozpadlé fragmenty jednoho hotspotu se sloučí

**Skóre spolehlivosti (C)**
`C6` 🟥 obě metody + vysoký jas + dobrý tvar
`C3` 🟧 průměrná detekce
`C1` 🟨 pouze jedna metoda
""")
    st.stop()

# ── Sestavení params dict ────────────────────────────────────
params = json.dumps({
    "panel_min_area": panel_min_area,
    "panel_solidity": panel_solidity,
    "panel_max_cov":  panel_max_cov,
    "panel_erode":    panel_erode,
    "tophat_kernel":  tophat_kernel,
    "tophat_pct":     tophat_pct,
    "abs_pct":        abs_pct,
    "require_both":   require_both,
    "hs_min_area":    hs_min_area,
    "hs_max_aspect":  hs_max_aspect,
    "edge_margin":    edge_margin,
    "merge_dist":     merge_dist,
    "conf_min":       conf_min,
}, sort_keys=True)

img_bytes = uploaded.getvalue()

@st.cache_data(show_spinner=False)
def cached_pipeline(img_bytes, params):
    return run_pipeline(img_bytes, params)

with st.spinner("⚙️ Zpracovávám..."):
    img, gray, maska, th_m, ab_m, combined, hotspoty, n_pan, tophat_t, abs_t, debug = \
        cached_pipeline(img_bytes, params)

annotated = anotuj(img, maska, hotspoty)

# ── Metriky ──────────────────────────────────────────────────
st.markdown("### 📊 Výsledky")
m1,m2,m3,m4,m5,m6 = st.columns(6)

def metric(col, val, lbl):
    col.markdown(f'<div class="metric-card"><div class="val">{val}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

n_high = sum(1 for h in hotspoty if h.confidence >= 4)
n_mid  = sum(1 for h in hotspoty if 2 <= h.confidence < 4)
n_low  = sum(1 for h in hotspoty if h.confidence == 1)

metric(m1, n_pan,        "Panely")
metric(m2, len(hotspoty),"Hotspoty celkem")
metric(m3, n_high,       "Vysoká C (≥4)")
metric(m4, n_mid,        "Střední C (2–3)")
metric(m5, tophat_t,     "TopHat práh")
metric(m6, abs_t,        "Abs. práh")

st.markdown("<br>", unsafe_allow_html=True)

# ── Debug info o strategii panelů ───────────────────────────
with st.expander(f"🟩 Detekce panelů – strategie: **{debug['strategie']}**  "
                 f"(pokrytí {debug['pokryti']:.0%})", expanded=False):
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Otsu INV",  f"{debug['INV'][0]} panelů",  debug['INV'][1])
    dc2.metric("Otsu NORM", f"{debug['NORM'][0]} panelů", debug['NORM'][1])
    dc3.metric("Gradient",  f"{debug['GRAD'][0]} panelů", debug['GRAD'][1])

# ── Snímky ───────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Originální termogram**")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Maska panelů**")
    st.image(maska, use_container_width=True)
with c3:
    st.markdown("**Výsledek detekce**")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown("---")

# ── Debug panel (6 masek) ────────────────────────────────────
if show_debug:
    st.markdown("### 🔬 Debug – všechny mezikroky")
    fig_bytes = detail_figure(img, maska, th_m, ab_m, combined, annotated, hotspoty)
    st.image(fig_bytes, use_container_width=True)
    st.markdown("---")

# ── Tabulka hotspotů ─────────────────────────────────────────
st.markdown(f"### 🔥 Nalezené hotspoty  `{len(hotspoty)}`")
if hotspoty:
    import pandas as pd
    rows = []
    for h in hotspoty:
        badge = ("🔴" if h.confidence >= 4 else
                 "🟠" if h.confidence >= 2 else "🟡")
        rows.append({
            "ID":          h.id,
            "Skóre":       f"{badge} C{h.confidence}",
            "Typ":         h.typ,
            "Max jas":     h.max_intensity,
            "Střední jas": h.mean_intensity,
            "Plocha [px²]":h.area,
            "Poměr stran": h.aspect_ratio,
            "Kruhovitost": h.circularity,
            "Okraj [px]":  h.edge_distance,
            "X,Y":         f"{h.x},{h.y}",
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=min(400, 40+len(df)*35))
else:
    st.warning("⚠️ Žádné hotspoty nebyly detekovány. "
               "Zkuste snížit percentily nebo minimální skóre spolehlivosti.")

st.markdown("---")

# ── Export ───────────────────────────────────────────────────
st.markdown("### 💾 Export")
e1, e2, e3 = st.columns(3)
fname = uploaded.name.rsplit(".", 1)[0]

with e1:
    st.download_button("⬇️ Anotovaný snímek (PNG)",
        data=img_to_png(annotated),
        file_name=f"{fname}_hotspoty.png",
        mime="image/png", use_container_width=True)
with e2:
    if hotspoty:
        import pandas as pd
        st.download_button("⬇️ CSV – hotspoty",
            data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{fname}_hotspoty.csv",
            mime="text/csv", use_container_width=True)
    else:
        st.button("⬇️ CSV", disabled=True, use_container_width=True)
with e3:
    fig_bytes_dl = detail_figure(img, maska, th_m, ab_m, combined, annotated, hotspoty)
    st.download_button("⬇️ Debug graf (PNG)",
        data=fig_bytes_dl,
        file_name=f"{fname}_debug.png",
        mime="image/png", use_container_width=True)

st.markdown("---")
st.caption("FV Hotspot Detektor  |  OpenCV + Streamlit  |  Confidence: C0–C6")