import io
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Optional, Any

st.set_page_config(
    page_title="Ferografie – Analýza částic",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Kompaktní CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Zmenšení hlavního nadpisu */
h1 { font-size: 1.4rem !important; margin-bottom: 0.3rem !important; }
h3 { font-size: 1.0rem !important; margin: 0.4rem 0 0.2rem 0 !important; }

/* Kompaktnější metriky */
[data-testid="metric-container"] {
    padding: 6px 10px !important;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    border: 1px solid rgba(128,128,128,0.15);
}
[data-testid="metric-container"] label {
    font-size: 0.68rem !important;
    line-height: 1.1 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.05rem !important;
}

/* Obrázky v náhledu – omezit maximální výšku */
.preview-img img {
    max-height: 260px !important;
    object-fit: contain;
    width: 100%;
}

/* Méně mezer kolem horizontálních oddělovačů */
hr { margin: 0.5rem 0 !important; }

/* Kompaktnější popisky nad obrázky */
.img-label {
    text-align: center;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-color, #888);
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Pomocné funkce ─────────────────────────────────────────────────────────────
def auto_canny_thresholds(blurred: np.ndarray, sigma: float = 0.33):
    p_low = np.percentile(blurred, 10)
    p_high = np.percentile(blurred, 90)
    t1 = int(max(0, p_low * (1.0 - sigma)))
    t2 = int(min(255, p_high * (1.0 - sigma)))
    t1 = min(t1, 50)
    t2 = max(t2, t1 + 30)
    t2 = min(t2, 200)
    return t1, t2


def get_mask(img, blur_kernel, canny_auto, canny_t1, canny_t2,
             canny_sigma, dilate_kernel, dilate_iter, clean_kernel, clean_iter):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    if canny_auto:
        t1, t2 = auto_canny_thresholds(blurred, canny_sigma)
    else:
        t1, t2 = canny_t1, canny_t2
    canny = cv2.Canny(blurred, t1, t2)
    k_conn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    dilated = cv2.dilate(canny, k_conn, iterations=dilate_iter)
    contours_c, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edges = np.zeros_like(gray)
    cv2.drawContours(edges, contours_c, -1, 255, thickness=cv2.FILLED)
    _, mask_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(edges, mask_otsu)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel, clean_kernel))
    final = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_clean, iterations=clean_iter)
    return final, t1, t2


def analyze_shape(contour, min_area_px, px_per_mm) -> Optional[Dict[str, Any]]:
    area_px = cv2.contourArea(contour)
    if area_px < min_area_px:
        return None
    rect = cv2.minAreaRect(contour)
    center, (w, h), _ = rect
    length_px = max(w, h)
    width_px = min(w, h)
    ar = length_px / width_px if width_px > 0 else 0
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area_px / hull_area if hull_area > 0 else 0
    equiv_diam_px = np.sqrt(4 * area_px / np.pi)
    def px2(p): return p / px_per_mm
    def px2_area(p): return p / (px_per_mm ** 2)
    return {
        "area_px": area_px,
        "area_mm2": px2_area(area_px),
        "length_mm": px2(length_px),
        "width_mm": px2(width_px),
        "equiv_diam_mm": px2(equiv_diam_px),
        "ar": ar,
        "circularity": circularity,
        "convexity": convexity,
        "rect": rect,
        "center": center,
    }


def run_analysis(img, params):
    adjusted_img = cv2.convertScaleAbs(img, alpha=params["contrast"], beta=params["brightness"])
    px_per_mm = params["px_per_mm"]
    min_area_px = params["min_area_mm2"] * (px_per_mm ** 2)
    mask, t1, t2 = get_mask(
        adjusted_img,
        blur_kernel=params["blur_kernel"],
        canny_auto=params["canny_auto"],
        canny_t1=params["canny_t1"],
        canny_t2=params["canny_t2"],
        canny_sigma=params["canny_sigma"],
        dilate_kernel=params["dilate_kernel"],
        dilate_iter=params["dilate_iter"],
        clean_kernel=params["clean_kernel"],
        clean_iter=params["clean_iter"],
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    annotated = adjusted_img.copy()
    data_list = []
    u = params["unit"]
    for cnt in contours:
        s = analyze_shape(cnt, min_area_px, px_per_mm)
        if s is None or s["circularity"] < params["min_circularity"]:
            continue
        pid = len(data_list) + 1
        data_list.append({
            "ID": pid,
            f"Plocha ({u}²)": round(s["area_mm2"], 4),
            f"Délka ({u})": round(s["length_mm"], 3),
            f"Šířka ({u})": round(s["width_mm"], 3),
            f"Ekv. průměr ({u})": round(s["equiv_diam_mm"], 3),
            "Poměr stran": round(s["ar"], 2),
            "Kruhovitost": round(s["circularity"], 3),
            "Konvexnost": round(s["convexity"], 3),
        })
        cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 1)
        box = np.int64(cv2.boxPoints(s["rect"]))
        cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
        cx, cy = int(s["center"][0]), int(s["center"][1])
        cv2.putText(annotated, str(pid), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return adjusted_img, annotated, mask, pd.DataFrame(data_list), t1, t2


# ── Export helpers ─────────────────────────────────────────────────────────────
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    buf.seek(0)
    return buf.read()

def img_to_png_bytes(img_bgr): _, enc = cv2.imencode(".png", img_bgr); return enc.tobytes()
def mask_to_png_bytes(mask):   _, enc = cv2.imencode(".png", mask);    return enc.tobytes()

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Částice", index=False)
        ws = writer.sheets["Částice"]
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) for c in col_cells if c.value), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4
    buf.seek(0)
    return buf.read()


def make_histogram_fig(df: pd.DataFrame, unit: str) -> plt.Figure:
    col = f"Ekv. průměr ({unit})"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))          # ← menší výška
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.tick_params(colors="gray")
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        ax.title.set_color("gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")
    if col in df.columns and len(df) > 0:
        data = df[col]
        axes[0].hist(data, bins=min(25, len(df)), color="#4fc3f7", edgecolor="gray", linewidth=0.4, alpha=0.8)
        axes[0].axvline(data.mean(),   color="#ff7043", lw=1.5, ls="--", label=f"Průměr {data.mean():.3f}")
        axes[0].axvline(data.median(), color="#66bb6a", lw=1.5, ls=":",  label=f"Medián {data.median():.3f}")
        axes[0].set_xlabel(f"Ekvivalentní průměr ({unit})")
        axes[0].set_ylabel("Počet")
        axes[0].set_title("Distribuce velikostí")
        axes[0].legend(fontsize=7, framealpha=0.2)
        circ = df["Kruhovitost"]
        axes[1].hist(circ, bins=min(20, len(df)), color="#4fc3f7", edgecolor="gray", linewidth=0.4, alpha=0.8)
        axes[1].axvline(circ.mean(), color="#ff7043", lw=1.5, ls="--", label=f"Průměr {circ.mean():.3f}")
        axes[1].set_xlabel("Kruhovitost (0–1)")
        axes[1].set_ylabel("Počet")
        axes[1].set_title("Distribuce kruhovitosti")
        axes[1].legend(fontsize=7, framealpha=0.2)
    plt.tight_layout(pad=0.8)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Ferografie")
    st.markdown("---")
    uploaded_file = st.file_uploader("Nahrát snímek (PNG / JPG)", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    st.markdown("### Kalibrace")
    px_per_mm = st.number_input("Rozlišení (px / mm)", min_value=0.1, max_value=50000.0, value=10.0, step=0.5)
    unit = st.selectbox("Jednotka výstupu", ["mm", "µm"], index=0)
    st.markdown("---")
    st.markdown("### Předzpracování")
    brightness  = st.slider("Jas",      -100, 100,  0,   5)
    contrast    = st.slider("Kontrast",  0.5, 3.0,  1.0, 0.1)
    blur_kernel = st.slider("Gaussovo rozostření (px)", 1, 15, 5, 2)
    st.markdown("---")
    st.markdown("### Canny – detekce hran")
    canny_auto = st.toggle("Automatické prahy", value=True)
    if canny_auto:
        canny_sigma = st.slider("Citlivost (sigma)", 0.05, 0.8, 0.33, 0.01)
        canny_t1 = canny_t2 = 0
    else:
        canny_sigma = 0.33
        canny_t1 = st.slider("Canny T1", 0, 254, 10)
        canny_t2 = st.slider("Canny T2", canny_t1 + 1, 255, 100)
    st.markdown("### Dilatace")
    dilate_kernel = st.slider("Kernel dilatace (px)", 3, 25, 9, 2)
    dilate_iter   = st.slider("Iterace dilatace",     1,  5, 2)
    st.markdown("### Čištění masky")
    clean_kernel = st.slider("Kernel čištění (px)", 1, 11, 3, 2)
    clean_iter   = st.slider("Iterace čištění",     1,  5, 2)
    st.markdown("---")
    st.markdown("### Filtrace částic")
    min_area_mm2    = st.number_input("Min. plocha (mm²)", min_value=0.0001, max_value=100.0,
                                      value=0.005, step=0.001, format="%.4f")
    min_circularity = st.slider("Min. kruhovitost", 0.0, 1.0, 0.0, 0.05)
    st.markdown("---")
    st.markdown("### Zobrazení")
    show_mask          = st.toggle("Zobrazit masku",             value=True)
    show_table         = st.toggle("Zobrazit tabulku dat",       value=True)
    show_histograms    = st.toggle("Zobrazit histogramy",        value=True)
    highlight_max      = st.toggle("Zvýraznit maxima v tabulce", value=True)
    show_preprocessing = st.toggle("Předzpracování obrazu",      value=False)
    show_debug         = st.toggle("Mezikroky segmentace masky", value=False)


# ── Hlavní obsah ───────────────────────────────────────────────────────────────
st.title("Analýza ferografických snímků")

if uploaded_file is None:
    st.info("Nahrajte snímek v levém panelu pro zahájení analýzy.")
    st.stop()


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


params = dict(
    px_per_mm=px_per_mm, unit=unit,
    brightness=brightness, contrast=contrast,
    blur_kernel=_odd(blur_kernel), min_area_mm2=min_area_mm2,
    min_circularity=min_circularity,
    canny_auto=canny_auto, canny_t1=canny_t1, canny_t2=canny_t2, canny_sigma=canny_sigma,
    dilate_kernel=_odd(dilate_kernel), dilate_iter=dilate_iter,
    clean_kernel=_odd(clean_kernel),   clean_iter=clean_iter,
)


@st.cache_data(show_spinner=False)
def cached_analysis(img_bytes: bytes, params_key: str):
    arr  = np.frombuffer(img_bytes, dtype=np.uint8)
    _img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return run_analysis(_img, json.loads(params_key))


img_bytes  = uploaded_file.getvalue()
params_key = json.dumps(params, sort_keys=True)

with st.spinner("Zpracovávám obraz…"):
    adjusted_img, annotated, mask, df, used_t1, used_t2 = cached_analysis(img_bytes, params_key)

# ── Metriky ────────────────────────────────────────────────────────────────────
n = len(df)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Celkem částic", n)
ecol = f"Ekv. průměr ({unit})"
acol = f"Plocha ({unit}²)"
if n > 0:
    m2.metric(f"Prům. ekv. Ø ({unit})",  f"{df[ecol].mean():.2f}")
    m3.metric(f"Medián ekv. Ø ({unit})", f"{df[ecol].median():.2f}")
    m4.metric(f"Prům. plocha ({unit}²)", f"{df[acol].mean():.2f}")
    m5.metric("Prům. kruhovitost",        f"{df['Kruhovitost'].mean():.2f}")
    m6.metric("Canny T1 / T2",            f"{used_t1} / {used_t2}")
else:
    for col, lbl in zip([m2, m3, m4, m5, m6],
                        [f"Prům. ekv. Ø ({unit})", f"Medián ekv. Ø ({unit})",
                         f"Prům. plocha ({unit}²)", "Prům. kruhovitost", "Canny T1 / T2"]):
        col.metric(lbl, "—")

st.markdown("---")

# ── Náhledy obrazů – KOMPAKTNÍ (omezená výška přes CSS třídu) ─────────────────
def _label(text: str):
    st.markdown(f"<div class='img-label'>{text}</div>", unsafe_allow_html=True)

def _img(arr, is_gray=False):
    """Zobrazí obrázek v kontejneru s omezenou výškou."""
    st.markdown("<div class='preview-img'>", unsafe_allow_html=True)
    if is_gray:
        st.image(arr, clamp=True, use_container_width=True)
    else:
        st.image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

cols = st.columns(3 if show_mask else 2)
with cols[0]:
    _label("Upravený snímek")
    _img(adjusted_img)
if show_mask:
    with cols[1]:
        _label("Detekční maska")
        _img(mask, is_gray=True)
    with cols[2]:
        _label("Anotovaný výsledek")
        _img(annotated)
else:
    with cols[1]:
        _label("Anotovaný výsledek")
        _img(annotated)

st.markdown("---")

# ── Předzpracování (volitelné) ─────────────────────────────────────────────────
if show_preprocessing:
    st.markdown("### Vizuální pipeline: Předzpracování")
    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray_img, (params["blur_kernel"], params["blur_kernel"]), 0)
    canny_no_blur = cv2.Canny(gray_img, used_t1, used_t2)
    canny_blur    = cv2.Canny(blurred,  used_t1, used_t2)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        _label("1. Stupně šedi")
        _img(gray_img, is_gray=True)
    with p2:
        _label(f"2. Gaussův filtr ({params['blur_kernel']}×{params['blur_kernel']})")
        _img(blurred, is_gray=True)
    with p3:
        _label("3. Canny BEZ filtru")
        _img(canny_no_blur, is_gray=True)
    with p4:
        _label("4. Canny S filtrem")
        _img(canny_blur, is_gray=True)
    st.markdown("---")

# ── Debug maska (volitelné) ────────────────────────────────────────────────────
if show_debug:
    st.markdown("### Vizuální pipeline: Segmentace")
    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray_img, (params["blur_kernel"], params["blur_kernel"]), 0)
    canny_dbg = cv2.Canny(blurred, used_t1, used_t2)
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["dilate_kernel"], params["dilate_kernel"]))
    dilated = cv2.dilate(canny_dbg, k_d, iterations=params["dilate_iter"])
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    d1, d2, d3, d4 = st.columns(4)
    for col, lbl, arr in zip([d1, d2, d3, d4],
                              ["1. Canny hrany", "2. Po dilataci", "3. Otsu maska", "4. Výsledná maska"],
                              [canny_dbg, dilated, otsu, mask]):
        with col:
            _label(lbl)
            _img(arr, is_gray=True)
    st.markdown("---")

# ── Histogramy ─────────────────────────────────────────────────────────────────
if show_histograms and n > 0:
    st.markdown("### Distribuce")
    hist_fig = make_histogram_fig(df, unit)
    st.pyplot(hist_fig, use_container_width=True, clear_figure=True)
    plt.close(hist_fig)
    st.markdown("---")

# ── Tabulka ────────────────────────────────────────────────────────────────────
if show_table:
    st.markdown(f"### Naměřená data &nbsp; `{n} částic`", unsafe_allow_html=True)
    if n > 0:
        num_cols = [c for c in df.columns if c != "ID"]
        styled = df.style.format({c: "{:.4f}" for c in num_cols})
        if highlight_max:
            styled = styled.highlight_max(subset=num_cols, color="rgba(249,115,22,0.3)")
        st.dataframe(styled, use_container_width=True, height=300)   # ← nižší tabulka
    else:
        st.warning("Žádné částice nebyly detekovány. Zkuste upravit parametry v sidebaru.")
    st.markdown("---")

# ── Export ─────────────────────────────────────────────────────────────────────
st.markdown("### Export výsledků")
e1, e2, e3, e4 = st.columns(4)
fname = uploaded_file.name.rsplit(".", 1)[0]

with e1:
    if n > 0:
        st.download_button("CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{fname}_data.csv", mime="text/csv", use_container_width=True)
    else:
        st.button("CSV", disabled=True, use_container_width=True)

with e2:
    if n > 0:
        st.download_button("Excel (.xlsx)", data=df_to_excel_bytes(df),
                           file_name=f"{fname}_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    else:
        st.button("Excel (.xlsx)", disabled=True, use_container_width=True)

with e3:
    st.download_button("Anotovaný snímek (PNG)", data=img_to_png_bytes(annotated),
                       file_name=f"{fname}_anotovany.png", mime="image/png", use_container_width=True)

with e4:
    st.download_button("Maska (PNG)", data=mask_to_png_bytes(mask),
                       file_name=f"{fname}_maska.png", mime="image/png", use_container_width=True)