import io
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List


#  Nastavení stránky

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
.zbar-wrap {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 14px 16px 12px;
    margin-bottom: 8px;
}
.zbar-title {
    font-size: 0.68rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
.zbar-track {
    background: #1f2937;
    border-radius: 4px;
    height: 10px;
    width: 100%;
    position: relative;
    overflow: hidden;
}
.zbar-fill {
    height: 100%;
    border-radius: 4px;
}
.zbar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #4b5563;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.zbar-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
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
    max_z: float
    mean_z: float
    max_intensity: float
    confidence: int  # interní, nezobrazuje se uživateli

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
    def zavaznost_color(self) -> str:
        if self.confidence >= 3: return "#ef4444"
        if self.confidence >= 1: return "#f97316"
        return "#eab308"


#  Detekce

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


def detekce(img_bytes: bytes, params: str):
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
            area=round(area, 1), aspect_ratio=round(asp, 2),
            max_z=round(max_z, 2),
            mean_z=round(mean_z, 2), max_intensity=round(max_i, 1),
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
        cv2.putText(out, lbl, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(out, lbl, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, b, 1, cv2.LINE_AA)
    for i, line in enumerate([f"Hotspoty: {len(hotspoty)}"]):
        y = 22 + i * 18
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
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
        (cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "Originální snímek",  None),
        (np.clip(zmap, -2, None), "Z-skóre mapa", 'RdYlBu_r'),
        (maska, "Detekční maska", 'gray'),
        (cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Anotovaný výsledek", None),
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


# Z-skóre bar

def z_gauge_html(max_z: float, z_thresh: float, zmap_max: float) -> str:
    z_max_display = max(zmap_max, z_thresh * 4.5, 12.0)
    pct = min(100, max(0, (max_z / z_max_display) * 100))

    if max_z < z_thresh:
        color, label = "#4b5563", "Pod prahem"
    elif max_z < z_thresh * 2:
        color, label = "#eab308", "Hotspot"
    elif max_z < z_thresh * 4:
        color, label = "#f97316", "Hotspot"
    else:
        color, label = "#ef4444", "Hotspot"

    thresh_pct  = min(99, (z_thresh / z_max_display) * 100)
    thresh2_pct = min(99, (z_thresh * 2 / z_max_display) * 100)
    thresh4_pct = min(99, (z_thresh * 4 / z_max_display) * 100)

    return f"""
    <div class="zbar-wrap">
        <div class="zbar-title">Max Z-skóre ve snímku</div>
        <div class="zbar-val" style="color:{color}">{max_z:.2f}
            <span style="font-size:0.85rem; color:#6b7280; font-weight:400">&nbsp;/ práh {z_thresh}</span>
        </div>
        <div class="zbar-track">
            <div class="zbar-fill" style="width:{pct:.1f}%; background:linear-gradient(90deg,#374151,{color});"></div>
            <div style="position:absolute;top:0;left:{thresh_pct:.1f}%;width:2px;height:100%;background:#ffffff55;"></div>
            <div style="position:absolute;top:0;left:{thresh2_pct:.1f}%;width:1px;height:100%;background:#ffffff33;"></div>
            <div style="position:absolute;top:0;left:{thresh4_pct:.1f}%;width:1px;height:100%;background:#ffffff22;"></div>
        </div>
        <div class="zbar-labels">
            <span>0</span>
            <span style="color:{color}">{label}</span>
            <span>{z_max_display:.0f}</span>
        </div>
    </div>
    """


# Detekční maska bar

def maska_status_html(pct: float) -> str:
    if pct == 0:
        color, label = "#ef4444", "Prázdná – snižte Z práh"
    elif pct < 1:
        color, label = "#22c55e", "OK"
    elif pct < 5:
        color, label = "#eab308", "Mírně vysoké pokrytí"
    elif pct < 10:
        color, label = "#f97316", "Vysoké – zvyšte Z práh"
    else:
        color, label = "#ef4444", "Příliš mnoho pixelů"

    bar_pct = min(100, pct * 10)   # 10 % → 100 % baru
    return f"""
    <div class="zbar-wrap">
        <div class="zbar-title">Pokrytí detekční masky</div>
        <div class="zbar-val" style="color:{color}">{pct:.1f} %</div>
        <div class="zbar-track">
            <div class="zbar-fill" style="width:{bar_pct:.1f}%;background:{color};"></div>
            <div style="position:absolute;top:0;left:25%;width:1px;height:100%;background:#ffffff22;"></div>
        </div>
        <div class="zbar-labels"><span>0 %</span><span style="color:{color}">{label}</span><span>≥ 10 %</span></div>
    </div>
    """


# Velikost buněk na snímku

CELL_SIZE_MAP = {
    "Malé  (~20 px)": 51,
    "Střední (~35 px)": 91,
    "Velké  (~55 px)": 121,
}

# Mapování lidsky čitelného filtru závažnosti na interní conf_min hodnotu
FILTER_ZAVAZNOST_MAP = {
    "Vše (i slabé nálezy)":  0,
    "Střední a kritické":     1,
    "Pouze kritické":         3,
}

# Pouze 3 předvolby – "Velké panely" odstraněno, nahrazeno radio buttonem velikosti buněk
PRESETS = {
    "standard":  dict(blur_k=7, z_window=91,  z_thresh=3.2, morph_k=5, merge_dist=15,
                      min_area=15, max_area=800,  max_aspect=3.5, conf_min=1),
    "sensitive": dict(blur_k=3, z_window=51,  z_thresh=2.5, morph_k=3, merge_dist=15,
                      min_area=5,  max_area=3000, max_aspect=5.0, conf_min=0),
    "strict":    dict(blur_k=7, z_window=91,  z_thresh=3.5, morph_k=5, merge_dist=10,
                      min_area=10, max_area=500,  max_aspect=3.5, conf_min=1),
}

# Zpětné mapování conf_min → label pro radio button
CONF_TO_FILTER = {v: k for k, v in FILTER_ZAVAZNOST_MAP.items()}

def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[f"sl_{k}"] = v
    window_to_cell = {v: k for k, v in CELL_SIZE_MAP.items()}
    zw = PRESETS[name]["z_window"]
    if zw in window_to_cell:
        st.session_state["cell_size"] = window_to_cell[zw]
    # Synchronizuj radio button závažnosti
    cm = PRESETS[name]["conf_min"]
    st.session_state["filter_zavaznost"] = CONF_TO_FILTER.get(cm, "Střední a kritické")


#  Sidebar

with st.sidebar:
    st.markdown("## FV Hotspot Detektor")
    st.markdown("---")

    uploaded = st.file_uploader("Nahrát snímek",
                                 type=["jpg","jpeg","png","bmp","tif"])
    st.markdown("---")

    # Rychlé předvolby – 3 tlačítka přes celou šířku
    st.markdown('<div class="sec-hdr">Rychlé předvolby</div>', unsafe_allow_html=True)
    st.caption("Kliknutím se nastaví předvolené parametry")
    if st.button("Standardní",  use_container_width=True):
        apply_preset("standard")
    if st.button("Citlivější",  use_container_width=True):
        apply_preset("sensitive")
    if st.button("Přísnější",   use_container_width=True):
        apply_preset("strict")

    st.markdown("---")

    # Nastavení detekce
    st.markdown('<div class="sec-hdr">Nastavení detekce</div>', unsafe_allow_html=True)
    z_thresh = st.slider(
        "Z-skóre práh", 1.0, 8.0, 3.2, 0.1, key="sl_z_thresh")

    # Filtr závažnosti – nahrazuje neintuitivní "Min. spolehlivost (0–5)"
    st.caption("Zobrazovat hotspoty")
    filter_label = st.radio(
        "Zobrazovat hotspoty",
        options=list(FILTER_ZAVAZNOST_MAP.keys()),
        index=1,
        key="filter_zavaznost",
        label_visibility="collapsed",
    )
    conf_min = FILTER_ZAVAZNOST_MAP[filter_label]

    # Velikost buněk
    st.markdown('<div class="sec-hdr">Velikost buněk na snímku</div>', unsafe_allow_html=True)
    st.caption("Určuje velikost detekčního okna Z-skóre")
    cell_size_label = st.radio(
        "Velikost buněk",
        options=list(CELL_SIZE_MAP.keys()),
        index=1,
        key="cell_size",
        label_visibility="collapsed")
    z_window = CELL_SIZE_MAP[cell_size_label]

    # Rozměry hotspotu
    st.markdown('<div class="sec-hdr">Rozměry hotspotu</div>', unsafe_allow_html=True)
    min_area   = st.slider("Min. plocha [px²]",  5,    200,   15,  5,   key="sl_min_area")
    max_area   = st.slider("Max. plocha [px²]",  100, 3500,  800, 100, key="sl_max_area")
    max_aspect = st.slider("Max. poměr stran",   1.5,  10.0,  3.5, 0.5, key="sl_max_aspect")

    st.markdown("---")

    # Pokročilé nastavení
    with st.expander("Pokročilé nastavení"):
        blur_k     = st.slider("Gaussovské rozmazání [px]",   0, 15, 7, 2, key="sl_blur_k")
        morph_k    = st.slider("Morfologické otevření [px]",  0, 11, 5, 2, key="sl_morph_k")
        merge_dist = st.slider("Sloučení fragmentů [px]",     0, 40, 15, 5, key="sl_merge_dist")

    st.markdown("---")
    show_debug = st.toggle("Zobrazit debug panel", value=True)


#  Hlavní funkční část

st.title("FV Hotspot Detektor")

if uploaded is None:
    st.info("Nahrajte termogram v levém panelu.")
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
    return detekce(img_bytes, params)

with st.spinner("Počítám Z-skóre mapu..."):
    img, gray, zmap, maska, merged, hotspoty = cached(img_bytes, params)

annotated = anotuj(img, hotspoty, z_thresh, z_window)

# Počítání metrik
n_krit = sum(1 for h in hotspoty if h.confidence >= 3)
n_str  = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
n_slab = sum(1 for h in hotspoty if h.confidence == 0)
max_z_val = round(float(max([h.max_z for h in hotspoty])), 2) if hotspoty else 0.0
zmap_max  = float(zmap.max())
px_total  = maska.size
px_active = int((maska > 0).sum())
pct       = 100 * px_active / px_total

# Počty hotspotů
cols = st.columns(4)
for col, (val, lbl) in zip(cols, [
    (len(hotspoty), "Celkem hotspotů"),
    (n_krit, "Kritické hotspoty"),
    (n_str, "Střední hotspoty"),
    (n_slab, "Slabé hotspoty"),
]):

    custom_style = "color: #ef4444;" if lbl == "Celkem hotspotů" else ""
    col.markdown(
        f'<div class="metric-box">'
        f'<div class="val" style="{custom_style}">{val}</div>'
        f'<div class="lbl">{lbl}</div>'
        f'</div>',
        unsafe_allow_html=True)

# Z-skóre + pokrytí detekční masky bar
gc1, gc2 = st.columns(2)
with gc1:
    st.markdown(z_gauge_html(max_z_val, z_thresh, zmap_max), unsafe_allow_html=True)
with gc2:
    st.markdown(maska_status_html(pct), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Hlavní obrázky
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

# Debug panel
if show_debug:
    st.markdown("---")
    st.markdown("### Debug")
    fig = render_panels(img, annotated, zmap, maska, merged)
    st.image(fig_to_bytes(fig), use_container_width=True)

st.markdown("---")

# Tabulka hotspotů – bez C skóre, jen Závažnost
st.markdown(f"### Nalezené hotspoty: `{len(hotspoty)}`")
if hotspoty:
    rows = []
    for h in hotspoty:
        rows.append({
            "ID":               h.id,
            "Závažnost":        h.zavaznost,
            "Max Z":            h.max_z,
            "Mean Z":           h.mean_z,
            "Max jas":          h.max_intensity,
            "Plocha [px²]":     h.area,
            "Poměr stran":      h.aspect_ratio,
            "Souřadnice X":     f"{h.x}",
            "Souřadnice Y":     f"{h.y}",
            "Šířka × Výška":    f"{h.w}×{h.h}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True,
                 height=min(450, 48 + len(df) * 38))
else:
    st.warning("Žádné hotspoty nenalezeny.")

st.markdown("---")

# Export
st.markdown("### Export")
fname = uploaded.name.rsplit(".", 1)[0]
e1, e2, e3 = st.columns(3)

_, png_enc = cv2.imencode(".png", annotated)
with e1:
    st.download_button("Anotovaný PNG",
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