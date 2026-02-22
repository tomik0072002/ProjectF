"""
╔══════════════════════════════════════════════════════════════╗
║   HOTSPOTY FV PANELŮ – Streamlit aplikace  v4               ║
║   Spuštění: streamlit run hotspoty_app.py                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import io
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    margin-bottom: 4px;
}
.metric-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.9rem;
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
.info-box {
    background: #0f1e35;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #93c5fd;
    margin-bottom: 8px;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #e5e7eb !important; }
.stDownloadButton > button {
    background: #111827 !important; color: #93c5fd !important;
    border: 1px solid #1e3a5f !important; border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important; width: 100%;
}
.stDownloadButton > button:hover { background: #1e3a5f !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DATOVÁ TŘÍDA
# ─────────────────────────────────────────────────────────────

@dataclass
class Hotspot:
    id: int
    x: int; y: int; w: int; h: int
    area: float
    aspect_ratio: float
    circularity: float
    max_z: float
    mean_z: float
    max_intensity: float
    confidence: int

    @property
    def cx(self): return self.x + self.w // 2
    @property
    def cy(self): return self.y + self.h // 2


# ─────────────────────────────────────────────────────────────
#  DETEKCE
# ─────────────────────────────────────────────────────────────

def priprav(img: np.ndarray, blur_k: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_k >= 3:
        k = blur_k | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def zscore_mapa(gray: np.ndarray, window: int) -> np.ndarray:
    k     = window | 1
    mu    = cv2.boxFilter(gray, -1, (k, k))
    sq_mu = cv2.boxFilter(gray ** 2, -1, (k, k))
    std   = np.sqrt(np.clip(sq_mu - mu ** 2, 0, None)) + 1e-6
    return (gray - mu) / std


def sestav_masku(zmap: np.ndarray, thresh: float, morph_k: int) -> np.ndarray:
    maska = (zmap >= thresh).astype(np.uint8) * 255
    if morph_k >= 3:
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k | 1, morph_k | 1))
        maska = cv2.morphologyEx(maska, cv2.MORPH_OPEN, k)
    return maska


def sluc(maska: np.ndarray, dist: int) -> np.ndarray:
    if dist <= 0: return maska
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dist | 1, dist | 1))
    return cv2.erode(cv2.dilate(maska, k), k)


def spocti_confidence(max_z: float, area: float, circ: float, thresh: float) -> int:
    s = 0
    if max_z >= thresh * 1.5: s += 1
    if max_z >= thresh * 2.5: s += 1
    if max_z >= thresh * 4.0: s += 1
    if 10 <= area <= 500:     s += 1
    if circ > 0.4:            s += 1
    return s


def detekuj(img_bytes: bytes, params: str):
    p   = json.loads(params)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray   = priprav(img, p["blur_k"])
    zmap   = zscore_mapa(gray, p["z_window"])
    maska  = sestav_masku(zmap, p["z_thresh"], p["morph_k"])
    merged = sluc(maska, p["merge_dist"])

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspoty = []

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < p["min_area"] or area > p["max_area"]:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        asp = (w / h) if h > 0 else 0
        if asp > p["max_aspect"] or asp < 1.0 / p["max_aspect"]:
            continue

        cm    = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px_z  = zmap[cm > 0]
        px_g  = gray[cm > 0]
        max_z  = float(px_z.max())  if px_z.size else 0.0
        mean_z = float(px_z.mean()) if px_z.size else 0.0
        max_i  = float(px_g.max())  if px_g.size else 0.0
        perim  = cv2.arcLength(cnt, True)
        circ   = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
        conf   = spocti_confidence(max_z, area, circ, p["z_thresh"])

        if conf < p["conf_min"]:
            continue

        hotspoty.append(Hotspot(
            id=len(hotspoty)+1, x=x, y=y, w=w, h=h,
            area=round(area,1), aspect_ratio=round(asp,2),
            circularity=round(circ,3), max_z=round(max_z,2),
            mean_z=round(mean_z,2), max_intensity=round(max_i,1),
            confidence=conf,
        ))

    hotspoty.sort(key=lambda h: h.max_z, reverse=True)
    for i, h in enumerate(hotspoty): h.id = i + 1

    return img, gray, zmap, maska, merged, hotspoty


# ─────────────────────────────────────────────────────────────
#  ANOTACE
# ─────────────────────────────────────────────────────────────

CONF_BGR = {
    0:(160,160,160), 1:(0,210,210), 2:(0,165,255),
    3:(0,80,255),    4:(0,20,220),  5:(0,0,180),
}

def anotuj(img, hotspoty, z_thresh, z_window):
    out = img.copy()
    for hs in hotspoty:
        b  = CONF_BGR.get(min(hs.confidence, 5), (0,0,255))
        x1 = max(0, hs.x-5); y1 = max(0, hs.y-5)
        x2 = min(img.shape[1]-1, hs.x+hs.w+5)
        y2 = min(img.shape[0]-1, hs.y+hs.h+5)
        cv2.rectangle(out, (x1,y1), (x2,y2), b, 2)
        lbl = f"#{hs.id} Z{hs.max_z:.1f}"
        cv2.putText(out,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(out,lbl,(x1,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.4,b,1,cv2.LINE_AA)
    for i, line in enumerate([f"Hotspoty:{len(hotspoty)}",
                               f"Z≥{z_thresh} okno:{z_window}px"]):
        y = 22 + i * 20
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


def render_panels(img, annotated, zmap, maska, merged):
    """Čtyřpanelový debug výstup."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor('#080b10')
    panels = [
        (cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originální snímek",  None),
        (np.clip(zmap, -2, None),                    "Z-skóre mapa",       'RdYlBu_r'),
        (maska,                                       "Detekční maska",     'gray'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  "Anotovaný výsledek", None),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='#9ca3af', fontsize=10, pad=5)
        ax.axis('off')
        ax.set_facecolor('#080b10')
        if cmap == 'RdYlBu_r':
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔥 FV Hotspot Detektor")
    st.caption("v4 – čistě Z-skóre přístup")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Nahraj termogram",
                                 type=["jpg","jpeg","png","bmp","tif"])
    st.markdown("---")

    # ── Předzpracování ───────────────────────────────────────
    st.markdown('<div class="sec-hdr">⚙️ Předzpracování</div>', unsafe_allow_html=True)
    blur_k = st.slider("Gaussian blur [px]", 0, 15, 5, 2,
        help="Potlačí JPEG šum a texturu buněk před výpočtem Z-skóre.\n"
             "Doporučeno: 5–9. Příliš velký rozmaže skutečné hotspoty.")

    # ── Z-skóre ──────────────────────────────────────────────
    st.markdown('<div class="sec-hdr">📊 Z-skóre parametry</div>', unsafe_allow_html=True)
    z_window = st.slider("Okno Z-skóre [px]", 21, 201, 71, 10,
        help="Velikost okolí pro výpočet lokálního průměru a std.\n\n"
             "⚠️ Toto je nejdůležitější parametr:\n"
             "• Příliš malé (< 51px) → zachytí mřížku buněk a rámy jako anomálie\n"
             "• Příliš velké (> 150px) → sousední hotspoty splývají s pozadím\n"
             "• Doporučeno: 2–3× větší než průměrná solární buňka v px")
    z_thresh = st.slider("Z-skóre práh", 1.0, 8.0, 3.0, 0.1,
        help="Hotspot = pixel s Z ≥ práh.\n"
             "• Nízký (< 2.5) → více detekcí, více false positives\n"
             "• Vysoký (> 4.0) → jen velmi výrazné hotspoty\n"
             "• Start: 3.0, dolaďte podle výsledků")

    # ── Čištění masky ────────────────────────────────────────
    st.markdown('<div class="sec-hdr">🧹 Čištění masky</div>', unsafe_allow_html=True)
    morph_k = st.slider("Morfologické otevření [px]", 0, 11, 3, 2,
        help="Odstraní izolované pixely a drobný šum z detekční masky.\n"
             "0 = vypnuto. 3–5 doporučeno.")
    merge_dist = st.slider("Sloučení fragmentů [px]", 0, 40, 10, 5,
        help="Blízké fragmenty jednoho hotspotu se sloučí do jednoho.\n"
             "0 = vypnuto.")

    # ── Filtrování kontur ─────────────────────────────────────
    st.markdown('<div class="sec-hdr">🔍 Filtrování kontur</div>', unsafe_allow_html=True)
    min_area   = st.slider("Min. plocha [px²]", 5, 200, 10, 5,
        help="Odstraní šumové body. Hotspot typicky > 10 px².")
    max_area   = st.slider("Max. plocha [px²]", 100, 10000, 2000, 100,
        help="⚠️ Klíčový filtr: velké bílé oblasti v masce nejsou hotspoty\n"
             "ale teplotní gradienty nebo rámy. Nastavte na max. očekávanou\n"
             "plochu skutečného hotspotu.")
    max_aspect = st.slider("Max. poměr stran", 1.5, 10.0, 4.0, 0.5,
        help="Příliš protáhlé tvary (kabely, rámy) se odfiltrují.")
    conf_min   = st.slider("Min. spolehlivost (0–5)", 0, 4, 0,
        help="0 = zobraz vše\n"
             "Skóre: +1 za Z≥1.5×práh, +1 za Z≥2.5×práh, +1 za Z≥4×práh,\n"
             "+1 za plochu 10–500px², +1 za kulatý tvar")

    st.markdown("---")
    show_debug = st.toggle("🔬 Zobrazit debug panel", value=True)

    # ── Rychlé předvolby ─────────────────────────────────────
    st.markdown('<div class="sec-hdr">⚡ Rychlé předvolby</div>', unsafe_allow_html=True)
    st.caption("Zkopíruj hodnoty do sliderů výše")
    with st.expander("Přísnější (méně false positives)"):
        st.code("blur=7, okno=91, práh=3.5\nmorph=5, max_plocha=500")
    with st.expander("Citlivější (více detekcí)"):
        st.code("blur=3, okno=51, práh=2.5\nmorph=3, max_plocha=3000")
    with st.expander("Velké snímky / velké panely"):
        st.code("blur=9, okno=121, práh=3.0\nmorph=5, max_plocha=2000")


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ OBSAH
# ─────────────────────────────────────────────────────────────

st.title("🔥 Detekce hotspotů FV panelů")

if uploaded is None:
    st.info("👈  Nahrajte termogram v levém panelu.")
    with st.expander("📖  Jak ladit parametry – průvodce", expanded=True):
        st.markdown("""
### Příznaky a řešení

| Problém | Příčina | Řešení |
|---|---|---|
| Maska zachycuje celé rámy a přechody | Z-skóre okno příliš malé | Zvýšit **Okno** na 71–101px |
| Velké bílé bloky v masce | Max. plocha příliš vysoká | Snížit **Max. plochu** na 500–1000px² |
| Protáhlé oblasti (kabely) | Max. aspect ratio příliš vysoký | Snížit na 3.0 |
| Hotspot v originále není detekován | Práh příliš vysoký | Snížit **Z práh** na 2.0–2.5 |
| Příliš mnoho false positives | Práh příliš nízký | Zvýšit **Z práh** na 3.5–4.0 |
| Fragmentovaný hotspot | Sloučení příliš malé | Zvýšit **Sloučení** na 15–25px |

### Postup ladění
1. Start: **blur=5, okno=71, práh=3.0**
2. Podívej se na Z-skóre mapu – hotspot by měl být červený bod
3. Pokud je maska přeplněná → zvyšte okno nebo práh
4. Pokud hotspot chybí → snižte práh nebo zmenšete okno
5. Pokud jsou tam velké bloky → snižte Max. plochu
""")
    st.stop()

# ── Cache + výpočet ──────────────────────────────────────────
params = json.dumps({
    "blur_k": blur_k, "z_window": z_window, "z_thresh": z_thresh,
    "morph_k": morph_k, "merge_dist": merge_dist,
    "min_area": min_area, "max_area": max_area,
    "max_aspect": max_aspect, "conf_min": conf_min,
}, sort_keys=True)

img_bytes = uploaded.getvalue()

@st.cache_data(show_spinner=False)
def cached(img_bytes, params):
    return detekuj(img_bytes, params)

with st.spinner("⚙️  Počítám Z-skóre mapu..."):
    img, gray, zmap, maska, merged, hotspoty = cached(img_bytes, params)

annotated = anotuj(img, hotspoty, z_thresh, z_window)

# ── Metriky ──────────────────────────────────────────────────
n_h = sum(1 for h in hotspoty if h.confidence >= 3)
n_m = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
n_l = sum(1 for h in hotspoty if h.confidence == 0)
avg_z = round(float(np.mean([h.max_z for h in hotspoty])), 2) if hotspoty else 0
max_z_val = round(float(max([h.max_z for h in hotspoty])), 2) if hotspoty else 0

cols = st.columns(6)
for col, val, lbl in zip(cols,
    [len(hotspoty), n_h, n_m, n_l, max_z_val, avg_z],
    ["Hotspoty celkem", "Silné (C≥3)", "Střední (C1–2)", "Slabé (C0)",
     "Max Z-skóre", "Prům. max Z"]):
    col.markdown(f'<div class="metric-box"><div class="val">{val}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Hlavní panely ────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Originální snímek**")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Detekční maska**")
    st.image(maska, use_container_width=True)
with c3:
    st.markdown("**Anotovaný výsledek**")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

# ── Z-skóre mapa + debug ─────────────────────────────────────
if show_debug:
    st.markdown("---")
    st.markdown("### 🔬 Debug")
    fig = render_panels(img, annotated, zmap, maska, merged)
    st.image(fig_to_bytes(fig), use_container_width=True)

    # Diagnostika masky
    px_total  = maska.size
    px_active = int((maska > 0).sum())
    pct       = 100 * px_active / px_total
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Aktivní px v masce", f"{px_active:,}",
                  f"{pct:.1f} % snímku")
    col_d2.metric("Max Z-skóre v snímku",
                  f"{float(zmap.max()):.2f}")
    col_d3.metric("Z-skóre práh",
                  f"{z_thresh}")

    if pct > 20:
        st.warning(f"⚠️  Maska pokrývá {pct:.0f} % snímku – pravděpodobně příliš citlivá. "
                   f"Zvyšte **Z práh** nebo **Okno Z-skóre**.")
    elif px_active == 0:
        st.error("❌  Maska je prázdná – práh je příliš vysoký. Snižte **Z práh**.")

st.markdown("---")

# ── Tabulka hotspotů ─────────────────────────────────────────
st.markdown(f"### 🔥 Nalezené hotspoty &nbsp; `{len(hotspoty)}`")
if hotspoty:
    rows = []
    for h in hotspoty:
        badge = "🔴" if h.confidence >= 3 else ("🟠" if h.confidence >= 1 else "⚪")
        rows.append({
            "ID":           h.id,
            "Skóre":        f"{badge} C{h.confidence}",
            "Max Z":        h.max_z,
            "Mean Z":       h.mean_z,
            "Max jas":      h.max_intensity,
            "Plocha [px²]": h.area,
            "Poměr stran":  h.aspect_ratio,
            "Kruhovitost":  h.circularity,
            "X,Y":          f"{h.x},{h.y}",
            "W×H":          f"{h.w}×{h.h}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 height=min(450, 48 + len(df) * 38))
else:
    st.warning("⚠️  Žádné hotspoty. Zkuste snížit Z práh nebo Max. plochu.")

st.markdown("---")

# ── Export ───────────────────────────────────────────────────
st.markdown("### 💾 Export")
fname = uploaded.name.rsplit(".", 1)[0]
e1, e2, e3 = st.columns(3)

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

fig_exp = render_panels(img, annotated, zmap, maska, merged)
with e3:
    st.download_button("⬇️ Debug graf PNG",
        data=fig_to_bytes(fig_exp), file_name=f"{fname}_debug.png",
        mime="image/png", use_container_width=True)

st.caption("FV Hotspot Detektor v4  |  Z-skóre metoda  |  OpenCV + Streamlit")