"""
╔══════════════════════════════════════════════════════════════╗
║   HOTSPOTY FV PANELŮ – Streamlit aplikace  v3               ║
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
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────────
#  STRÁNKA
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="FV Hotspot Detektor", page_icon="🔥",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #080b10; }
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1a2332;
}

.metric-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 10px 10px;
    text-align: center;
}
.metric-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f97316;
    line-height: 1;
}
.metric-box .lbl {
    font-size: 0.68rem;
    color: #6b7280;
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.sec-hdr {
    background: #111827;
    border-left: 3px solid #f97316;
    border-radius: 0 6px 6px 0;
    padding: 6px 12px;
    margin: 10px 0 4px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #f97316;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

.method-pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 1px;
}
.pill-th  { background: #1e3a5f; color: #60a5fa; }
.pill-abs { background: #3b1a08; color: #fb923c; }
.pill-z   { background: #1a2e1a; color: #4ade80; }

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e5e7eb !important; }

.stDownloadButton > button {
    background: #111827 !important;
    color: #93c5fd !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: #1e3a5f !important; color: white !important;
}
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
    max_zscore: float
    metody: List[str]
    confidence: int

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


# ─────────────────────────────────────────────────────────────
#  DETEKČNÍ PIPELINE
# ─────────────────────────────────────────────────────────────

def priprav_gray(img: np.ndarray, blur_k: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_k >= 3:
        k = blur_k | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def tophat_maska(gray: np.ndarray, kernel: int, pct: int):
    k      = kernel | 1
    se     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    t      = int(np.percentile(tophat, pct))
    _, m   = cv2.threshold(tophat, t, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8), tophat, t


def abs_maska(gray: np.ndarray, pct: int):
    t    = int(np.percentile(gray, pct))
    _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    return m.astype(np.uint8), t


def zscore_maska(gray: np.ndarray, window: int, thresh: float):
    g_f     = gray.astype(np.float32)
    k       = window | 1
    mu      = cv2.boxFilter(g_f, -1, (k, k))
    sq_mu   = cv2.boxFilter(g_f**2, -1, (k, k))
    std     = np.sqrt(np.clip(sq_mu - mu**2, 0, None)) + 1e-6
    zmap    = (g_f - mu) / std
    mask_z  = (zmap >= thresh).astype(np.uint8) * 255
    return mask_z.astype(np.uint8), zmap


def kombinuj(m_th, m_ab, m_z, score_thresh, z_w, th_w, ab_w):
    """Váhovaný součet – Z-skóre má vyšší váhu, samo stačí k detekci."""
    vazeny = (m_z  > 0).astype(np.uint8) * z_w + \
             (m_th > 0).astype(np.uint8) * th_w + \
             (m_ab > 0).astype(np.uint8) * ab_w
    return ((vazeny >= score_thresh) * 255).astype(np.uint8)


def sluc(mask, dist):
    if dist <= 0: return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist|1, dist|1))
    return cv2.erode(cv2.dilate(mask, k), k)


def vypocti_skore(metody, max_int, abs_t, circ, max_z, z_thresh, z_w, th_w, ab_w):
    """Skóre kopíruje váhování kombinace + bonusy za kvalitu hotspotu."""
    s = 0
    if "zscore"   in metody: s += z_w
    if "tophat"   in metody: s += th_w
    if "absolute" in metody: s += ab_w
    if max_int > abs_t + 20:        s += 1
    if max_int > abs_t + 40:        s += 1
    if circ > 0.5:                  s += 1
    if max_z > z_thresh * 1.5:      s += 1
    if max_z > z_thresh * 2.5:      s += 1
    return s


def extrahuj(combined, m_th, m_ab, m_z, gray, zmap, abs_t,
             min_area, max_aspect, merge_dist, conf_min, z_thresh,
             z_w=2, th_w=1, ab_w=1):
    merged   = sluc(combined, merge_dist)
    cnts, _  = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result   = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < min_area: continue
        x, y, w, h = cv2.boundingRect(cnt)
        asp = w / float(h) if h > 0 else 0
        if asp > max_aspect or asp < 1.0 / max_aspect: continue

        cm = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px = gray[cm > 0]
        if px.size == 0: continue

        mean_i = float(px.mean())
        max_i  = float(px.max())
        pz     = zmap[cm > 0]
        max_z  = float(pz.max()) if pz.size else 0.0
        perim  = cv2.arcLength(cnt, True)
        circ   = (4 * np.pi * area / perim**2) if perim > 0 else 0.0

        metody = []
        if m_th[cm > 0].any():  metody.append("tophat")
        if m_ab[cm > 0].any():  metody.append("absolute")
        if m_z[cm > 0].any():   metody.append("zscore")

        conf = vypocti_skore(metody, max_i, abs_t, circ, max_z, z_thresh, z_w, th_w, ab_w)
        if conf < conf_min: continue

        result.append(Hotspot(
            id=len(result)+1, x=x, y=y, w=w, h=h,
            area=round(area,1), aspect_ratio=round(asp,2),
            mean_intensity=round(mean_i,1), max_intensity=round(max_i,1),
            circularity=round(circ,3), max_zscore=round(max_z,2),
            metody=metody, confidence=conf,
        ))

    result.sort(key=lambda h: h.confidence, reverse=True)
    for i, h in enumerate(result): h.id = i + 1
    return result


def run_pipeline(img_bytes: bytes, params: str):
    p   = json.loads(params)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray             = priprav_gray(img, p["blur_k"])
    m_th, th_sig, tt = tophat_maska(gray, p["th_kernel"], p["th_pct"])
    m_ab, abs_t      = abs_maska(gray, p["abs_pct"])
    m_z, zmap        = zscore_maska(gray, p["z_window"], p["z_thresh"])
    combined         = kombinuj(m_th, m_ab, m_z, p["score_thresh"], p["z_w"], p["th_w"], p["ab_w"])
    hotspoty         = extrahuj(combined, m_th, m_ab, m_z, gray, zmap, abs_t,
                                p["min_area"], p["max_aspect"],
                                p["merge_dist"], p["conf_min"], p["z_thresh"],
                                p["z_w"], p["th_w"], p["ab_w"])
    return img, gray, m_th, m_ab, m_z, zmap, combined, hotspoty, tt, abs_t


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

CONF_BGR = {
    1:(0,220,220), 2:(0,165,255), 3:(0,80,255),
    4:(0,0,255),   5:(30,0,220),  6:(60,0,180), 7:(80,0,140)
}

def anotuj(img, hotspoty, tt, abs_t, z_thresh):
    out = img.copy()
    for hs in hotspoty:
        b = CONF_BGR.get(min(hs.confidence, 7), (0,0,255))
        x1,y1 = max(0,hs.x-4), max(0,hs.y-4)
        x2,y2 = min(img.shape[1]-1,hs.x+hs.w+4), min(img.shape[0]-1,hs.y+hs.h+4)
        cv2.rectangle(out,(x1,y1),(x2,y2),b,2)
        lbl = f"#{hs.id} C{hs.confidence}"
        cv2.putText(out,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(out,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,b,1,cv2.LINE_AA)
    for i,line in enumerate([f"Hotspoty:{len(hotspoty)}",
                              f"TopHat T:{tt}",f"Abs T:{abs_t}",f"Z:{z_thresh}"]):
        y = 22+i*20
        cv2.putText(out,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    return out


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def debug_figure(img, annotated, m_th, m_ab, m_z, zmap, combined, hotspoty):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.patch.set_facecolor('#080b10')
    panels = [
        (cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originál",          None),
        (m_th,                                        "Top-Hat maska",     'hot'),
        (m_ab,                                        "Absolutní maska",   'hot'),
        (m_z,                                         "Z-skóre maska",     'hot'),
        (np.clip(zmap, 0, None),                      "Z-skóre mapa",      'RdYlBu_r'),
        (combined,                                     "Kombinovaná maska", 'hot'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  "Výsledek",          None),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap if data.ndim == 2 else None)
        ax.set_title(title, color='#9ca3af', fontsize=10, pad=5)
        ax.axis('off')
        ax.set_facecolor('#080b10')
    axes.flat[-1].set_visible(False)  # 8. panel prázdný

    patches = [
        mpatches.Patch(color='#ef4444', label=f'C≥4  ({sum(1 for h in hotspoty if h.confidence>=4)})'),
        mpatches.Patch(color='#f97316', label=f'C=2–3 ({sum(1 for h in hotspoty if 2<=h.confidence<4)})'),
        mpatches.Patch(color='#eab308', label=f'C=1  ({sum(1 for h in hotspoty if h.confidence==1)})'),
    ]
    fig.legend(handles=patches, loc='lower right', fontsize=9,
               facecolor='#111827', labelcolor='white', framealpha=0.9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔥 FV Hotspot Detektor")
    st.caption("v3 – bez segmentace panelů")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Nahraj termogram",
                                 type=["jpg","jpeg","png","bmp","tif"])
    st.markdown("---")

    # ── Předzpracování ───────────────────────────────────────
    st.markdown('<div class="sec-hdr">⚙️ Předzpracování</div>', unsafe_allow_html=True)
    blur_k = st.slider("Blur kernel [px]", 0, 9, 3, 2,
        help="Gaussovský blur před detekcí – potlačí šum a kompresní artefakty. 0 = vypnuto.")

    # ── Top-Hat ──────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔵 Top-Hat (lokální anomálie)</div>', unsafe_allow_html=True)
    th_kernel = st.slider("Kernel [px]", 11, 121, 51, 2,
        help="Větší = detekuje jen velké hotspoty, ignoruje texturu solárních buněk.\n"
             "Menší = citlivější, ale zachytí mřížku buněk jako false positives.")
    th_pct = st.slider("Percentil prahu [%]", 85, 99, 97,
        help="97 = nejsvětlejší 3 % top-hat signálu. Zvyšte pokud je maska přeplněná.")

    # ── Absolutní práh ───────────────────────────────────────
    st.markdown('<div class="sec-hdr">🟠 Absolutní práh (globální anomálie)</div>', unsafe_allow_html=True)
    abs_pct = st.slider("Percentil prahu [%]", 85, 99, 97,
        help="97 = nejsvětlejší 3 % pixelů v celém snímku.")

    # ── Z-skóre ──────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🟢 Z-skóre (statistická odchylka)</div>', unsafe_allow_html=True)
    z_window = st.slider("Okno Z-skóre [px]", 21, 121, 61, 4,
        help="Velikost okolí pro výpočet lokálního průměru a std.\n"
             "Větší okno = ignoruje pomalé teplotní gradienty (přechody přes pole).")
    z_thresh = st.slider("Z-skóre práh", 1.0, 5.0, 2.5, 0.1,
        help="Kolik směrodatných odchylek nad lokálním průměrem = hotspot.\n"
             "2.5 = standardní statistická anomálie. Zvyšte pro přísnější detekci.")

    # ── Kombinace ────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔗 Kombinace metod</div>', unsafe_allow_html=True)
    st.caption("Z-skóre má vyšší váhu – samo stačí k detekci hotspotu")
    z_w = st.slider("Váha Z-skóre", 1, 4, 2,
        help="Váha 2 = Z-skóre samo překročí práh 2 a hotspot je detekován.\n"
             "Zvyšte na 3–4 pokud chcete Z-skóre jako dominantní metodu.")
    th_w = st.slider("Váha Top-Hat", 0, 2, 1,
        help="0 = Top-Hat se ignoruje úplně. 1 = standard. 2 = zvýšená váha.")
    ab_w = st.slider("Váha Absolutní", 0, 2, 1,
        help="0 = absolutní metoda se ignoruje. 1 = standard.")
    score_thresh = st.slider("Práh váhovaného součtu", 1, 6, 2,
        help="Hotspot = pixely kde váhovaný součet >= práh.\n"
             "Příklady při Z=2, TH=1, ABS=1:\n"
             "  Práh 2: Z samo stačí, nebo TH+ABS dohromady\n"
             "  Práh 3: Z + jedna další metoda\n"
             "  Práh 4: Z + obě ostatní (velmi přísné)")

    # ── Filtrování ───────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔍 Filtrování kontur</div>', unsafe_allow_html=True)
    min_area   = st.slider("Min. plocha [px²]", 5, 500, 40, 5,
        help="Odstraní šumové body a kompresní artefakty.")
    max_aspect = st.slider("Max. poměr stran", 1.5, 8.0, 4.0, 0.5,
        help="Příliš protáhlé tvary = artefakty linií nebo kabelů.")
    merge_dist = st.slider("Sloučení fragmentů [px]", 0, 80, 20,
        help="Fragmenty blíže než X px se sloučí do jednoho hotspotu.\n0 = vypnuto.")
    conf_min   = st.slider("Min. skóre spolehlivosti", 0, 8, 2,
        help="0 = zobraz vše  |  vyšší = jen spolehlivé hotspoty\n"
             "Skóre závisí na počtu metod, jasu, tvaru a Z-skóre.")

    st.markdown("---")
    show_debug = st.toggle("🔬 Zobrazit debug (7 masek)", value=False)


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ OBSAH
# ─────────────────────────────────────────────────────────────

st.title("🔥 Detekce hotspotů FV panelů")

if uploaded is None:
    st.info("👈  Nahrajte termogram v levém panelu.")

    with st.expander("📖  Jak fungují tři detekční metody", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
**🔵 Top-Hat filtr**
Detekuje oblasti které jsou *lokálně* teplejší než bezprostřední okolí.

Morfologická operace: snímek mínus jeho otevření velkým kernelem.
Výsledek = pouze lokální vrcholy.

**Kdy selže:** pokud je hotspot velký (přes celý panel) a kernel je menší.
**Ladění:** zvětšit kernel pokud zachytí mřížku buněk.
""")
        with c2:
            st.markdown("""
**🟠 Absolutní percentil**
Detekuje oblasti které jsou *globálně* mezi nejteplejšími X % pixelů.

Práh = percentil jasu celého snímku.

**Kdy selže:** na snímcích s velkým teplotním gradientem přes pole
(jedna část pole vždy teplejší) → celá teplá oblast = "hotspot".
**Ladění:** zvýšit percentil (97→99).
""")
        with c3:
            st.markdown("""
**🟢 Lokální Z-skóre**
Detekuje oblasti které jsou statisticky výrazně teplejší než lokální okolí.

Z = (pixel − lokální průměr) / lokální std

**Kdy selže:** pokud je std v okolí velmi malé (homogenní panel)
→ i malý šum překročí práh.
**Ladění:** zvětšit okno nebo zvýšit Z-práh.

**Skóre spolehlivosti (C):** součet bodů za počet metod,
výšku jasu, tvar a velikost Z-skóre. Vyšší = spolehlivější.
""")
    st.stop()

# ── Sestavení cache klíče ────────────────────────────────────
params = json.dumps({
    "blur_k":    blur_k,
    "th_kernel": th_kernel, "th_pct": th_pct,
    "abs_pct":   abs_pct,
    "z_window":  z_window,  "z_thresh": z_thresh,
    "score_thresh": score_thresh, "z_w": z_w, "th_w": th_w, "ab_w": ab_w,
    "min_area":  min_area,  "max_aspect": max_aspect,
    "merge_dist":merge_dist,"conf_min": conf_min,
}, sort_keys=True)

if uploaded is None:
    st.stop()

img_bytes = uploaded.getvalue()

@st.cache_data(show_spinner=False)
def cached(img_bytes, params):
    return run_pipeline(img_bytes, params)

with st.spinner("⚙️  Zpracovávám..."):
    img, gray, m_th, m_ab, m_z, zmap, combined, hotspoty, tt, abs_t = \
        cached(img_bytes, params)

annotated = anotuj(img, hotspoty, tt, abs_t, z_thresh)

# ── Metriky ──────────────────────────────────────────────────
n_h = sum(1 for h in hotspoty if h.confidence >= 4)
n_m = sum(1 for h in hotspoty if 2 <= h.confidence < 4)
n_l = sum(1 for h in hotspoty if h.confidence == 1)

cols = st.columns(6)
for col, val, lbl in zip(cols, [
    len(hotspoty), n_h, n_m, n_l, tt, abs_t
], [
    "Hotspoty celkem","Vysoká C≥4","Střední C2–3","Nízká C1","TopHat práh","Abs. práh"
]):
    col.markdown(f'<div class="metric-box"><div class="val">{val}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Snímky ───────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Originál**")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Výsledek detekce**")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
with c3:
    st.markdown("**Z-skóre mapa**")
    fig_z, ax_z = plt.subplots(figsize=(5, 4))
    fig_z.patch.set_facecolor('#080b10')
    im_z = ax_z.imshow(np.clip(zmap, 0, None), cmap='RdYlBu_r')
    plt.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04)
    ax_z.axis('off')
    st.pyplot(fig_z, use_container_width=True)
    plt.close(fig_z)

# ── Debug ────────────────────────────────────────────────────
if show_debug:
    st.markdown("---")
    st.markdown("### 🔬 Debug – mezikroky")
    fig_d = debug_figure(img, annotated, m_th, m_ab, m_z, zmap, combined, hotspoty)
    st.image(fig_to_bytes(fig_d), use_container_width=True)

st.markdown("---")

# ── Aktivní px per metoda ────────────────────────────────────
px_total = gray.size
with st.expander("📊  Aktivní pixely per metoda (před kombinací)", expanded=False):
    d1, d2, d3 = st.columns(3)
    d1.metric("🔵 Top-Hat",    f"{(m_th>0).sum():,} px",
              f"{100*(m_th>0).sum()/px_total:.1f} % snímku")
    d2.metric("🟠 Absolutní",  f"{(m_ab>0).sum():,} px",
              f"{100*(m_ab>0).sum()/px_total:.1f} % snímku")
    d3.metric("🟢 Z-skóre",   f"{(m_z>0).sum():,} px",
              f"{100*(m_z>0).sum()/px_total:.1f} % snímku")

# ── Tabulka hotspotů ─────────────────────────────────────────
st.markdown(f"### 🔥 Nalezené hotspoty &nbsp; `{len(hotspoty)}`")
if hotspoty:
    rows = []
    for h in hotspoty:
        badge = "🔴" if h.confidence >= 4 else ("🟠" if h.confidence >= 2 else "🟡")
        pills = " ".join([
            f'<span class="method-pill pill-th">TOP</span>'  if "tophat"   in h.metody else "",
            f'<span class="method-pill pill-abs">ABS</span>' if "absolute" in h.metody else "",
            f'<span class="method-pill pill-z">Z</span>'     if "zscore"   in h.metody else "",
        ])
        rows.append({
            "ID":           h.id,
            "Skóre":        f"{badge} C{h.confidence}",
            "Metody":        "+".join(h.metody),
            "Max jas":       h.max_intensity,
            "Střední jas":   h.mean_intensity,
            "Max Z":         h.max_zscore,
            "Plocha [px²]":  h.area,
            "Poměr stran":   h.aspect_ratio,
            "Kruhovitost":   h.circularity,
            "X,Y":           f"{h.x},{h.y}",
            "W×H":           f"{h.w}×{h.h}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 height=min(420, 45 + len(df)*38))
else:
    st.warning("⚠️  Žádné hotspoty. Zkuste snížit percentily, Z-práh nebo min. skóre.")

st.markdown("---")

# ── Export ───────────────────────────────────────────────────
st.markdown("### 💾 Export")
fname = uploaded.name.rsplit(".", 1)[0]
e1, e2, e3, e4 = st.columns(4)

_, png_enc = cv2.imencode(".png", annotated)
with e1:
    st.download_button("⬇️ Anotovaný PNG",
        data=png_enc.tobytes(), file_name=f"{fname}_hotspoty.png",
        mime="image/png", use_container_width=True)

if hotspoty:
    with e2:
        st.download_button("⬇️ CSV – hotspoty",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{fname}_hotspoty.csv",
            mime="text/csv", use_container_width=True)

fig_exp = debug_figure(img, annotated, m_th, m_ab, m_z, zmap, combined, hotspoty)
with e3:
    st.download_button("⬇️ Debug graf PNG",
        data=fig_to_bytes(fig_exp),
        file_name=f"{fname}_debug.png",
        mime="image/png", use_container_width=True)

# Z-skóre mapa jako PNG
fig_zexp, ax_zexp = plt.subplots(figsize=(6,5))
ax_zexp.imshow(np.clip(zmap, 0, None), cmap='RdYlBu_r')
ax_zexp.axis('off')
with e4:
    st.download_button("⬇️ Z-skóre mapa PNG",
        data=fig_to_bytes(fig_zexp),
        file_name=f"{fname}_zscore.png",
        mime="image/png", use_container_width=True)

st.caption("FV Hotspot Detektor v3  |  Top-Hat + Absolutní percentil + Z-skóre  |  OpenCV + Streamlit")