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


#  Nastvení stránky

st.set_page_config(page_title="FV Hotspot Detektor", page_icon="!",
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
section[data-testid="stSidebar"] div[data-testid="column"] .stButton > button {
    height: 52px !important; min-height: 52px !important;
    white-space: normal !important; line-height: 1.2 !important;
    font-size: 0.8rem !important; padding: 4px 6px !important;
}
</style>
""", unsafe_allow_html=True)



#  Datová třída

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


#  Detekce hotspotů

def priprav(img: np.ndarray, blur_k: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if blur_k >= 3:
        k = blur_k | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def zscore_mapa(gray: np.ndarray, window: int) -> np.ndarray:
    k = window | 1
    mu = cv2.boxFilter(gray, -1, (k, k))
    sq_mu = cv2.boxFilter(gray ** 2, -1, (k, k))
    std = np.sqrt(np.clip(sq_mu - mu ** 2, 0, None)) + 1e-6
    return (gray - mu) / std


def sestav_masku(zmap: np.ndarray, thresh: float, morph_k: int) -> np.ndarray:
    maska = (zmap >= thresh).astype(np.uint8) * 255
    if morph_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k | 1, morph_k | 1))
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
    if 10 <= area <= 500: s += 1
    if circ > 0.4: s += 1
    return s


def detekuj(img_bytes: bytes, params: str):
    p = json.loads(params)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    gray = priprav(img, p["blur_k"])
    zmap = zscore_mapa(gray, p["z_window"])
    maska = sestav_masku(zmap, p["z_thresh"], p["morph_k"])
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

        cm = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px_z = zmap[cm > 0]
        px_g = gray[cm > 0]
        max_z = float(px_z.max())  if px_z.size else 0.0
        mean_z = float(px_z.mean()) if px_z.size else 0.0
        max_i = float(px_g.max())  if px_g.size else 0.0
        perim = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
        conf = spocti_confidence(max_z, area, circ, p["z_thresh"])

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


#  Anotace

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
    for i, line in enumerate([f"Hotspoty: {len(hotspoty)}",
                               f"Z >= {z_thresh}",
                               f"Okno: {z_window} px"]):
        y = 22 + i * 18
        cv2.putText(out,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1,cv2.LINE_AA)
    return out


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def render_panels(img, annotated, zmap, maska, merged):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor('#080b10')
    panels = [
        (cv2.cvtColor(img,       cv2.COLOR_BGR2RGB), "Originalni snimek", None),
        (np.clip(zmap, -2, None),                    "Z-skore mapa",      'RdYlBu_r'),
        (maska,                                       "Detekni maska",     'gray'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),  "Anotovany vysledek",None),
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


#  Sidebar

PRESETS = {
    "standard": dict(blur_k=5,  z_window=71,  z_thresh=3.0, morph_k=3, merge_dist=10,
                     min_area=10, max_area=2000, max_aspect=4.0, conf_min=0),
    "strict":   dict(blur_k=7,  z_window=91,  z_thresh=3.5, morph_k=5, merge_dist=10,
                     min_area=10, max_area=500,  max_aspect=3.5, conf_min=1),
    "sensitive":dict(blur_k=3,  z_window=51,  z_thresh=2.5, morph_k=3, merge_dist=15,
                     min_area=5,  max_area=3000, max_aspect=5.0, conf_min=0),
    "large":    dict(blur_k=9,  z_window=121, z_thresh=3.0, morph_k=5, merge_dist=20,
                     min_area=10, max_area=2000, max_aspect=4.0, conf_min=0),
}

def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[f"sl_{k}"] = v

with st.sidebar:
    st.markdown("## FV Hotspot Detektor")
    st.caption("v4 - Z-skore pristup")
    st.markdown("---")

    uploaded = st.file_uploader("Nahraj termogram",
                                 type=["jpg","jpeg","png","bmp","tif"])
    st.markdown("---")

    # Rychlé předvolby (sidebar)
    st.markdown('<div class="sec-hdr">Rychle predvolby</div>', unsafe_allow_html=True)
    st.caption("Kliknutim se parametry okamzite nastavi")
    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)
    if p1.button("Standardni",  use_container_width=True,
                 help="Dobry vychozi bod pro vetsinu snimku"):
        apply_preset("standard")
    if p2.button("Prisnejsi",   use_container_width=True,
                 help="Mene false positives, jen vyrazne hotspoty"):
        apply_preset("strict")
    if p3.button("Citlivejsi",  use_container_width=True,
                 help="Vice detekci - vhodne pro slabe hotspoty"):
        apply_preset("sensitive")
    if p4.button("Velke panely", use_container_width=True,
                 help="Vetsi okno pro snimky s velkymi panely"):
        apply_preset("large")

    st.markdown("---")

    # Předzpracování
    st.markdown('<div class="sec-hdr">Predzpracovani</div>', unsafe_allow_html=True)
    blur_k = st.slider("Gaussian blur [px]", 0, 15, 5, 2, key="sl_blur_k",
        help="Potlaci JPEG sum a texturu bunek pred vypoctem Z-skore.\n"
             "Doporuceno: 5-9. Prilis velky rozmazee skutecne hotspoty.")

    # Z-skóre
    st.markdown('<div class="sec-hdr">Z-skore parametry</div>', unsafe_allow_html=True)
    z_window = st.slider("Okno Z-skore [px]", 21, 201, 71, 10, key="sl_z_window",
        help="Velikost okoli pro vypocet lokalniho prumeru a std.\n"
             "Prilis male: zachyti mrizku bunek a ramy.\n"
             "Prilis velke: hotspoty splynou s pozadim.\n"
             "Doporuceno: 2-3x velikost solarni bunky v px.")
    z_thresh = st.slider("Z-skore prah", 1.0, 8.0, 3.0, 0.1, key="sl_z_thresh",
        help="Hotspot = pixel s Z >= prah.\n"
             "Nizky (< 2.5): vice false positives.\n"
             "Vysoky (> 4.0): jen vyrazne hotspoty.\n"
             "Start: 3.0, doladte podle vysledku.")

    # Čištění masky
    st.markdown('<div class="sec-hdr">Cisteni masky</div>', unsafe_allow_html=True)
    morph_k = st.slider("Morfolog. otevreni [px]", 0, 11, 3, 2, key="sl_morph_k",
        help="Odstrani izalovane pixely a drobny sum z masky.\n"
             "0 = vypnuto. 3-5 doporuceno.")
    merge_dist = st.slider("Slouceni fragmentu [px]", 0, 40, 10, 5, key="sl_merge_dist",
        help="Fragmenty blize nez X px se slouci.\n"
             "0 = vypnuto.")

    # Filtrování kontur
    st.markdown('<div class="sec-hdr">Filtrovani kontur</div>', unsafe_allow_html=True)
    min_area   = st.slider("Min. plocha [px2]", 5, 200, 10, 5, key="sl_min_area",
        help="Odstrani sumove body. Hotspot typicky > 10 px2.")
    max_area   = st.slider("Max. plocha [px2]", 100, 10000, 2000, 100, key="sl_max_area",
        help="Klicovy filtr: velke oblasti nejsou hotspoty.\n"
             "Nastavte na max. ocekavanou plochu hotspotu.")
    max_aspect = st.slider("Max. pomer stran", 1.5, 10.0, 4.0, 0.5, key="sl_max_aspect",
        help="Prilis protahle tvary (kabely, ramy) se odfiltrují.")
    conf_min   = st.slider("Min. spolehlivost (0-5)", 0, 4, 0, key="sl_conf_min",
        help="0 = zobraz vse\n"
             "+1 za Z>=1.5x prah, +1 za Z>=2.5x, +1 za Z>=4x\n"
             "+1 za plochu 10-500px2, +1 za kulatý tvar.")

    st.markdown("---")
    show_debug = st.toggle("Zobrazit debug panel", value=True)



#  Hlavní obsah


st.title("FV Hotspot Detektor")

if uploaded is None:
    st.info("Nahrajte termogram v levem panelu.")
    with st.expander("Jak ladit parametry – pruvodce", expanded=True):
        st.markdown("""
### Priznaky a reseni

| Problem | Pricina | Reseni |
|---|---|---|
| Maska zachycuje ramy a prechody | Okno prilis male | Zvysit **Okno** na 71-101px |
| Velke bile bloky v masce | Max. plocha prilis vysoka | Snizit **Max. plochu** na 500-1000px2 |
| Protahle oblasti (kabely) | Max. aspect ratio prilis vysoky | Snizit na 3.0 |
| Hotspot neni detekovan | Prah prilis vysoky | Snizit **Z prah** na 2.0-2.5 |
| Prilis mnoho false positives | Prah prilis nizky | Zvysit **Z prah** na 3.5-4.0 |
| Fragmentovany hotspot | Slouceni prilis male | Zvysit **Slouceni** na 15-25px |

### Postup ladeni
1. Start: **blur=5, okno=71, prah=3.0**
2. Podivej se na Z-skore mapu – hotspot by mel byt cerveny bod
3. Pokud je maska preplnena: zvyste okno nebo prah
4. Pokud hotspot chybi: snizujte prah nebo zmenste okno
5. Pokud jsou tam velke bloky: snizujte Max. plochu
""")
    st.stop()


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

with st.spinner("Pocitam Z-skore mapu..."):
    img, gray, zmap, maska, merged, hotspoty = cached(img_bytes, params)

annotated = anotuj(img, hotspoty, z_thresh, z_window)

# Metriky
n_h = sum(1 for h in hotspoty if h.confidence >= 3)
n_m = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
n_l = sum(1 for h in hotspoty if h.confidence == 0)
avg_z     = round(float(np.mean([h.max_z for h in hotspoty])), 2) if hotspoty else 0
max_z_val = round(float(max([h.max_z for h in hotspoty])), 2)     if hotspoty else 0

cols = st.columns(6)
for col, val, lbl in zip(cols,
    [len(hotspoty), n_h, n_m, n_l, max_z_val, avg_z],
    ["Hotspoty celkem", "Silne (C>=3)", "Stredni (C1-2)", "Slabe (C0)",
     "Max Z-skore", "Prum. max Z"]):
    col.markdown(f'<div class="metric-box"><div class="val">{val}</div>'
                 f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Hlavní panely
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Originalni snimek**")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Detekni maska**")
    st.image(maska, use_container_width=True)
with c3:
    st.markdown("**Anotovany vysledek**")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

# Debug
if show_debug:
    st.markdown("---")
    st.markdown("### Debug")
    fig = render_panels(img, annotated, zmap, maska, merged)
    st.image(fig_to_bytes(fig), use_container_width=True)

    px_total  = maska.size
    px_active = int((maska > 0).sum())
    pct       = 100 * px_active / px_total
    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("Aktivni px v masce", f"{px_active:,}", f"{pct:.1f} % snimku")
    col_d2.metric("Max Z-skore v snimku", f"{float(zmap.max()):.2f}")
    col_d3.metric("Z-skore prah", f"{z_thresh}")

    if pct > 20:
        st.warning(f"Maska pokryva {pct:.0f} % snimku – zvyste Z prah nebo Okno Z-skore.")
    elif px_active == 0:
        st.error("Maska je prazdna – snizujte Z prah.")

st.markdown("---")

# Tabulka hotspotů
st.markdown(f"### Nalezene hotspoty: `{len(hotspoty)}`")
if hotspoty:
    rows = []
    for h in hotspoty:
        badge = "C3+" if h.confidence >= 3 else ("C1-2" if h.confidence >= 1 else "C0")
        rows.append({
            "ID":          h.id,
            "Skore":       f"{badge}  (C{h.confidence})",
            "Max Z":       h.max_z,
            "Mean Z":      h.mean_z,
            "Max jas":     h.max_intensity,
            "Plocha [px2]":h.area,
            "Pomer stran": h.aspect_ratio,
            "Kruhovitost": h.circularity,
            "X,Y":         f"{h.x},{h.y}",
            "WxH":         f"{h.w}x{h.h}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 height=min(450, 48 + len(df) * 38))
else:
    st.warning("Zadne hotspoty. Zkuste snizit Z prah nebo Max. plochu.")

st.markdown("---")

# Export
st.markdown("### Export")
fname = uploaded.name.rsplit(".", 1)[0]
e1, e2, e3 = st.columns(3)

_, png_enc = cv2.imencode(".png", annotated)
with e1:
    st.download_button("Anotovany PNG",
        data=png_enc.tobytes(), file_name=f"{fname}_hotspoty.png",
        mime="image/png", use_container_width=True)
if hotspoty:
    with e2:
        st.download_button("CSV – hotspoty",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{fname}_hotspoty.csv",
            mime="text/csv", use_container_width=True)

fig_exp = render_panels(img, annotated, zmap, maska, merged)
with e3:
    st.download_button("Debug graf PNG",
        data=fig_to_bytes(fig_exp), file_name=f"{fname}_debug.png",
        mime="image/png", use_container_width=True)

st.caption("FV Hotspot Detektor v4  |  Z-skore metoda  |  OpenCV + Streamlit")