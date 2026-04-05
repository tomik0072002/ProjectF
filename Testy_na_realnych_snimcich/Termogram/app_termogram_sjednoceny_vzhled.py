import io
import os
import json
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import List
from PIL import Image

# Načtení obrázku loga
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
try:
    logo = Image.open(logo_path)
except FileNotFoundError:
    logo = "🔥"

# ── Nastavení stránky ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FV Hotspot Detektor",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Datová třída ───────────────────────────────────────────────────────────────
@dataclass
class Hotspot:
    id: int
    x: int
    y: int
    w: int
    h: int
    area: float
    aspect_ratio: float
    circularity: float
    max_z: float
    mean_z: float
    max_intensity: float
    confidence: int  # interní skóre

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def zavaznost(self) -> str:
        if self.confidence >= 3:
            return "Kritický"
        if self.confidence >= 1:
            return "Střední"
        return "Slabý"


# ── Detekce a zpracování ───────────────────────────────────────────────────────
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
    if dist <= 0:
        return maska
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
    img_raw = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    h_orig, w_orig = img_raw.shape[:2]
    ct, cb = p["crop_t"], p["crop_b"]
    cl, cr = p["crop_l"], p["crop_r"]
    if ct + cb < h_orig and cl + cr < w_orig:
        img = img_raw[ct: h_orig - cb, cl: w_orig - cr]
    else:
        img = img_raw

    if p["alpha"] != 1.0 or p["beta"] != 0:
        img = cv2.convertScaleAbs(img, alpha=p["alpha"], beta=p["beta"])

    gray = priprav(img, p["blur_k"])
    zmap = zscore_mapa(gray, p["z_window"])
    maska_surova = (zmap >= p["z_thresh"]).astype(np.uint8) * 255
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

        perim = cv2.arcLength(cnt, True)
        circ = (4 * np.pi * area / perim ** 2) if perim > 0 else 0.0
        if circ < p["min_circ"]:
            continue

        cm = np.zeros(gray.shape[:2], np.uint8)
        cv2.drawContours(cm, [cnt], -1, 255, -1)
        px_z = zmap[cm > 0]
        px_g = gray[cm > 0]

        max_z = float(px_z.max()) if px_z.size else 0.0
        mean_z = float(px_z.mean()) if px_z.size else 0.0
        max_i = float(px_g.max()) if px_g.size else 0.0
        conf = spocti_confidence(max_z, area, circ, p["z_thresh"])

        if conf < p["conf_min"]:
            continue

        hotspoty.append(Hotspot(
            id=len(hotspoty) + 1, x=x, y=y, w=w, h=h,
            area=round(area, 1), aspect_ratio=round(asp, 2), circularity=round(circ, 2),
            max_z=round(max_z, 2), mean_z=round(mean_z, 2), max_intensity=round(max_i, 1),
            confidence=conf,
        ))

    hotspoty.sort(key=lambda h: h.max_z, reverse=True)
    for i, h in enumerate(hotspoty):
        h.id = i + 1

    return img, gray, zmap, maska_surova, merged, hotspoty


# ── Anotace ────────────────────────────────────────────────────────────────────
CONF_BGR = {
    0: (160, 160, 160),
    1: (0, 210, 210),
    2: (0, 165, 255),
    3: (0, 80, 255),
    4: (0, 20, 220),
    5: (0, 0, 180),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def putText_outline(img, text, org, scale, color_fg, thickness=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color_fg, thickness, cv2.LINE_AA)


def anotuj(img: np.ndarray, hotspoty: List[Hotspot]) -> np.ndarray:
    out = img.copy()
    for hs in hotspoty:
        bgr = CONF_BGR.get(min(hs.confidence, 5), (0, 0, 255))
        pad = 4
        x1 = max(0, hs.x - pad)
        y1 = max(0, hs.y - pad)
        x2 = min(img.shape[1] - 1, hs.x + hs.w + pad)
        y2 = min(img.shape[0] - 1, hs.y + hs.h + pad)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        line1 = f"#{hs.id}"
        scale1 = 0.5
        (_, th1), _ = cv2.getTextSize(line1, FONT, scale1, 1)
        ty1 = max(th1 + 2, y1 - 4)
        putText_outline(out, line1, (x1, ty1), scale1, bgr)

    putText_outline(out, f"Hotspoty: {len(hotspoty)}", (8, 22), 0.5, (255, 255, 255))
    return out


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def render_zscore(zmap: np.ndarray, z_thresh: float) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    im = ax.imshow(np.clip(zmap, -3, None), cmap="RdYlBu_r", vmin=-3)
    ax.contour(zmap, levels=[z_thresh], colors="lime", linewidths=0.8, linestyles="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label("Z-skóre", color="gray")
    cb.ax.yaxis.set_tick_params(color="gray")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="gray")
    ax.axis("off")
    plt.tight_layout()
    return fig_to_bytes(fig)


# ── Konfigurace a Presets ──────────────────────────────────────────────────────
FILTER_ZAVAZNOST_MAP = {
    "Vše (i slabé nálezy)": 0,
    "Střední a kritické": 1,
    "Pouze kritické": 3,
}

PRESETS = {
    "standard": dict(alpha=1.0, beta=0, blur_k=7, z_window=91, z_thresh=3.2, morph_k=5,
                     merge_dist=15, min_area=15, max_area=800, max_aspect=3.5, min_circ=0.0, conf_min=1),
    "sensitive": dict(alpha=1.0, beta=0, blur_k=3, z_window=51, z_thresh=2.5, morph_k=3,
                      merge_dist=15, min_area=5, max_area=3000, max_aspect=5.0, min_circ=0.0, conf_min=0),
    "strict": dict(alpha=1.0, beta=0, blur_k=7, z_window=91, z_thresh=3.5, morph_k=5,
                   merge_dist=10, min_area=10, max_area=500, max_aspect=3.5, min_circ=0.3, conf_min=1),
}
CONF_TO_FILTER = {v: k for k, v in FILTER_ZAVAZNOST_MAP.items()}


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[f"sl_{k}"] = v
    cm = PRESETS[name]["conf_min"]
    st.session_state["filter_zavaznost"] = CONF_TO_FILTER.get(cm, "Střední a kritické")


# Prvotní inicializace
if "sl_alpha" not in st.session_state:
    apply_preset("standard")
    st.session_state["active_preset"] = "Standard"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### FV Hotspot Detektor")
    st.caption("Demonstrátor zpracování obrazu")
    st.markdown("---")

    uploaded = st.file_uploader("Nahrát termogram", type=["jpg", "jpeg", "png", "bmp", "tif"])
    st.markdown("---")

    st.markdown("**Rychlé předvolby**")
    cols = st.columns(3)


    # Pomocná funkce pro barvu tlačítka
    def get_btn_type(label):
        return "primary" if st.session_state.get("active_preset") == label else "secondary"


    # Zjednodušená logika tlačítek pomocí st.rerun()
    if cols[0].button("Standard", use_container_width=True, type=get_btn_type("Standard")):
        apply_preset("standard")
        st.session_state["active_preset"] = "Standard"
        st.rerun()

    if cols[1].button("Citlivé", use_container_width=True, type=get_btn_type("Citlivé")):
        apply_preset("sensitive")
        st.session_state["active_preset"] = "Citlivé"
        st.rerun()

    if cols[2].button("Přísné", use_container_width=True, type=get_btn_type("Přísné")):
        apply_preset("strict")
        st.session_state["active_preset"] = "Přísné"
        st.rerun()

    st.markdown("---")

    with st.expander("Fáze 1: Geometrie a Předzpracování", expanded=False):
        crop_t = st.number_input("Ořez shora [px]", 0, step=10, key="sl_crop_t")
        crop_b = st.number_input("Ořez zdola [px]", 0, step=10, key="sl_crop_b")
        crop_l = st.number_input("Ořez zleva [px]", 0, step=10, key="sl_crop_l")
        crop_r = st.number_input("Ořez zprava [px]", 0, step=10, key="sl_crop_r")
        st.markdown("---")
        alpha = st.slider("Kontrast (Alpha)", 0.5, 3.0, key="sl_alpha")
        beta = st.slider("Jas (Beta)", -100, 100, key="sl_beta")
        blur_k = st.slider("Gauss. filtr [px]", 0, 15, step=2, key="sl_blur_k")

    with st.expander("Fáze 2: Statistická anomálie", expanded=False):
        z_window = st.slider("Velikost okna [px]", 11, 201, step=2, key="sl_z_window")
        z_thresh = st.slider("Z-skóre práh", 1.0, 8.0, step=0.1, key="sl_z_thresh")

    with st.expander("Fáze 3: Morfologické operace", expanded=False):
        morph_k = st.slider("Morf. otevření [px]", 0, 11, step=2, key="sl_morph_k")
        merge_dist = st.slider("Sloučení [px]", 0, 40, step=5, key="sl_merge_dist")

    with st.expander("Fáze 4: Geometrická filtrace", expanded=False):
        min_area = st.slider("Min. plocha [px²]", 5, 200, step=5, key="sl_min_area")
        max_area = st.slider("Max. plocha [px²]", 100, 3500, step=100, key="sl_max_area")
        max_aspect = st.slider("Max. poměr stran", 1.0, 10.0, step=0.5, key="sl_max_aspect")
        min_circ = st.slider("Min. kruhovitost", 0.0, 1.0, step=0.05, key="sl_min_circ")
        st.markdown("---")
        filter_label = st.radio(
            "Zobrazovat ve výsledcích:",
            options=list(FILTER_ZAVAZNOST_MAP.keys()),
            key="filter_zavaznost"
        )
        conf_min = FILTER_ZAVAZNOST_MAP[filter_label]

# ── Hlavní část ────────────────────────────────────────────────────────────────
st.title("Detekce Hotspotů")

if uploaded is None:
    st.info("Nahrajte termogram v levém panelu pro spuštění analýzy.")
    st.stop()

params_dict = {
    "crop_t": st.session_state.get("sl_crop_t", 0),
    "crop_b": st.session_state.get("sl_crop_b", 0),
    "crop_l": st.session_state.get("sl_crop_l", 0),
    "crop_r": st.session_state.get("sl_crop_r", 0),
    "alpha": alpha,
    "beta": beta,
    "blur_k": blur_k,
    "z_window": z_window,
    "z_thresh": z_thresh,
    "morph_k": morph_k,
    "merge_dist": merge_dist,
    "min_area": min_area,
    "max_area": max_area,
    "max_aspect": max_aspect,
    "min_circ": min_circ,
    "conf_min": conf_min,
}
params_json = json.dumps(params_dict, sort_keys=True)
img_bytes = uploaded.getvalue()


@st.cache_data(show_spinner=False)
def cached(img_bytes, params):
    return detekce(img_bytes, params)


with st.spinner("Počítám analýzu..."):
    img, gray, zmap, maska_surova, merged, hotspoty = cached(img_bytes, params_json)

annotated = anotuj(img, hotspoty)

# ── Kontinuální rozvržení (Pipeline pro komisi pod sebou) ──────────────────────

st.markdown("---")
st.header("Vstup a Předzpracování")
c1, c2 = st.columns(2)
with c1:
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Originál (s aplikovaným ořezem)", use_container_width=True)
with c2:
    st.image(gray / 255.0, caption="Předzpracovaný snímek (Kontrast, Jas, Gauss. filtr)", use_container_width=True,
             clamp=True)

st.markdown("---")
st.header("Detekce (Z-skóre)")
c3, c4 = st.columns(2)
with c3:
    zs_bytes = render_zscore(zmap, z_thresh)
    st.image(zs_bytes, caption=f"Z-skóre mapa (práh: {z_thresh})", use_container_width=True)
with c4:
    st.image(maska_surova, caption="Surová maska (před morfologií)", use_container_width=True)

st.markdown("---")
st.header("Finální výsledek a filtrace")

n_krit = sum(1 for h in hotspoty if h.confidence >= 3)
n_str = sum(1 for h in hotspoty if 1 <= h.confidence < 3)
n_slab = sum(1 for h in hotspoty if h.confidence == 0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Celkem hotspotů", len(hotspoty))
m2.metric("Kritické nálezy", n_krit)
m3.metric("Střední nálezy", n_str)
m4.metric("Slabé nálezy", n_slab)

# Uložení finálního obrázku do prostředního ze 3 sloupců pro zmenšení jeho velikosti
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown("---")
st.markdown("#### Export výsledků")

fname = uploaded.name.rsplit(".", 1)[0]

# Převod anotovaného snímku na bajty pro stažení
_, buffer = cv2.imencode(".png", annotated)
img_dl_bytes = buffer.tobytes()

if hotspoty:
    rows = []
    for h in hotspoty:
        rows.append({
            "ID": h.id,
            "Závažnost nálezu": h.zavaznost,
            "Max Z": h.max_z,
            "Plocha [px²]": h.area,
            "Poměr stran": h.aspect_ratio,
            "Kruhovitost": h.circularity,
            "Rozměr (W×H) [px]": f"{h.w}×{h.h}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Tlačítka pro stažení vedle sebe
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "Stáhnout anotovaný snímek (PNG)",
            data=img_dl_bytes,
            file_name=f"{fname}_anotace.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "Stáhnout tabulku (CSV)",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{fname}_hotspoty.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.warning("Při aktuálním nastavení nebyly detekovány žádné hotspoty.")

    # Tlačítko pro stažení samotného snímku, i když nebyly nalezeny hotspoty
    st.download_button(
        "Stáhnout snímek (PNG)",
        data=img_dl_bytes,
        file_name=f"{fname}_bez_hotspotu.png",
        mime="image/png",
    )