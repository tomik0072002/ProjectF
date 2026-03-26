import io
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import streamlit as st
from typing import Dict, Optional, Any

#  Nastavení stránky
st.set_page_config(
    page_title="Ferografie – Analýza částic",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Globální CSS pro kompaktnější UI ──────────────────────────────────────────
st.markdown("""
<style>
/* Zmenšení mezer mezi sekcemi */
.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }

/* Popisky nad obrázky – overlay styl */
.img-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 4px;
}

/* Metriky – tighter */
div[data-testid="metric-container"] { padding: 0.5rem 0.6rem !important; }
div[data-testid="metric-container"] label { font-size: 0.68rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.1rem !important;
}

/* Oddělovač */
hr { margin: 0.6rem 0 !important; border-color: rgba(148,163,184,0.15) !important; }

/* Sidebar nadpisy */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0.8rem 0 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Paleta pro grafy ──────────────────────────────────────────────────────────
CHART_BG      = "#0f172a"   # téměř černá
CHART_SURFACE = "#1e293b"   # tmavě modrošedá
GRID_COLOR    = "#334155"
TEXT_COLOR    = "#94a3b8"
ACCENT_BLUE   = "#38bdf8"
ACCENT_ORANGE = "#fb923c"
ACCENT_GREEN  = "#4ade80"
BAR_COLOR     = "#0ea5e9"
BAR_EDGE      = "#1e40af"


#  Pomocné funkce pro detekci hran a částic

def auto_canny_thresholds(blurred: np.ndarray, sigma: float = 0.33):
    p_low = np.percentile(blurred, 10)
    p_high = np.percentile(blurred, 90)
    t1 = int(max(0, p_low * (1.0 - sigma)))
    t2 = int(min(255, p_high * (1.0 - sigma)))
    t1 = min(t1, 50)
    t2 = max(t2, t1 + 30)
    t2 = min(t2, 200)
    return t1, t2


def get_mask(img: np.ndarray,
             blur_kernel: int,
             canny_auto: bool,
             canny_t1: int,
             canny_t2: int,
             canny_sigma: float,
             dilate_kernel: int,
             dilate_iter: int,
             clean_kernel: int,
             clean_iter: int) -> tuple[np.ndarray, int, int]:
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


def analyze_shape(contour: np.ndarray,
                  min_area_px: float,
                  px_per_mm: float) -> Optional[Dict[str, Any]]:
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

    def px2(p):   return p / px_per_mm
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
    alpha = params["contrast"]
    beta = params["brightness"]
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    px_per_mm = params["px_per_mm"]
    min_area_px = params["min_area_mm2"] * (px_per_mm ** 2)
    min_circularity = params["min_circularity"]

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

    for i, cnt in enumerate(contours):
        s = analyze_shape(cnt, min_area_px, px_per_mm)
        if s is None or s["circularity"] < min_circularity:
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

    df = pd.DataFrame(data_list)
    return adjusted_img, annotated, mask, df, t1, t2


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    buf.seek(0)
    return buf.read()


def img_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    _, enc = cv2.imencode(".png", img_bgr)
    return enc.tobytes()


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    _, enc = cv2.imencode(".png", mask)
    return enc.tobytes()


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


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Aplikuje jednotný dark styl na osu."""
    ax.set_facecolor(CHART_SURFACE)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if title:   ax.set_title(title, fontsize=9, fontweight="bold", pad=6, color=TEXT_COLOR)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=8)


def make_histogram_fig(df: pd.DataFrame, unit: str) -> plt.Figure:
    col_diam = f"Ekv. průměr ({unit})"
    col_circ = "Kruhovitost"
    col_ar   = "Poměr stran"

    # 3 grafy ve 2 řadách: [diam | circ] a [ar | scatter diam vs circ]
    fig = plt.figure(figsize=(12, 7), facecolor=CHART_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                           left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in (ax1, ax2, ax3, ax4):
        _style_ax(ax)

    if len(df) > 0:
        # ── 1. Distribuce ekvivalentního průměru ─────────────────────────────
        data_d = df[col_diam]
        n_bins = min(25, max(5, len(df) // 2))
        counts, edges, patches = ax1.hist(
            data_d, bins=n_bins,
            color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.4, alpha=0.85
        )
        # Gradient efekt – tmavší okraje
        for patch in patches:
            patch.set_alpha(0.85)

        mean_d = data_d.mean()
        med_d  = data_d.median()
        ymax   = counts.max() * 1.12
        ax1.axvline(mean_d, color=ACCENT_ORANGE, lw=1.5, ls="--", zorder=5)
        ax1.axvline(med_d,  color=ACCENT_GREEN,  lw=1.5, ls=":",  zorder=5)
        ax1.set_ylim(0, ymax)
        ax1.text(mean_d, ymax * 0.92, f"μ={mean_d:.3f}", color=ACCENT_ORANGE,
                 fontsize=7, ha="left", va="top", fontweight="bold")
        ax1.text(med_d,  ymax * 0.78, f"med={med_d:.3f}", color=ACCENT_GREEN,
                 fontsize=7, ha="left", va="top", fontweight="bold")
        _style_ax(ax1,
                  title="Distribuce ekvivalentního průměru",
                  xlabel=f"Ekv. průměr ({unit})",
                  ylabel="Počet částic")

        # ── 2. Distribuce kruhovitosti ────────────────────────────────────────
        circ = df[col_circ]
        ax2.hist(circ, bins=min(20, max(5, len(df) // 2)),
                 color=ACCENT_BLUE, edgecolor="#0369a1", linewidth=0.4, alpha=0.85)
        mean_c = circ.mean()
        ax2.axvline(mean_c, color=ACCENT_ORANGE, lw=1.5, ls="--")
        ax2.text(mean_c, ax2.get_ylim()[1] * 0.02 if ax2.get_ylim()[1] > 0 else 0.02,
                 f"μ={mean_c:.3f}", color=ACCENT_ORANGE, fontsize=7,
                 ha="left", va="bottom", fontweight="bold")
        _style_ax(ax2,
                  title="Distribuce kruhovitosti",
                  xlabel="Kruhovitost (0–1)",
                  ylabel="Počet částic")
        ax2.set_xlim(0, 1.05)

        # ── 3. Distribuce poměru stran ────────────────────────────────────────
        ar_data = df[col_ar]
        ax3.hist(ar_data, bins=min(20, max(5, len(df) // 2)),
                 color="#a78bfa", edgecolor="#6d28d9", linewidth=0.4, alpha=0.85)
        mean_ar = ar_data.mean()
        ax3.axvline(mean_ar, color=ACCENT_ORANGE, lw=1.5, ls="--")
        ax3.text(mean_ar, ax3.get_ylim()[1] * 0.02 if ax3.get_ylim()[1] > 0 else 0.02,
                 f"μ={mean_ar:.2f}", color=ACCENT_ORANGE, fontsize=7,
                 ha="left", va="bottom", fontweight="bold")
        _style_ax(ax3,
                  title="Distribuce poměru stran",
                  xlabel="Poměr stran (délka/šířka)",
                  ylabel="Počet částic")

        # ── 4. Scatter: průměr vs kruhovitost ─────────────────────────────────
        sc = ax4.scatter(
            data_d, circ,
            c=df[col_ar], cmap="plasma",
            s=18, alpha=0.75, linewidths=0.3, edgecolors=CHART_BG,
            zorder=3
        )
        cbar = fig.colorbar(sc, ax=ax4, pad=0.02)
        cbar.ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        cbar.set_label("Poměr stran", color=TEXT_COLOR, fontsize=7)
        cbar.outline.set_edgecolor(GRID_COLOR)
        ax4.set_xlim(left=0)
        ax4.set_ylim(0, 1.05)
        _style_ax(ax4,
                  title="Průměr vs Kruhovitost (barva = poměr stran)",
                  xlabel=f"Ekv. průměr ({unit})",
                  ylabel="Kruhovitost")

    return fig


#  Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙ Ferografie")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Nahrát snímek (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Kalibrace</p>', unsafe_allow_html=True)
    px_per_mm = st.number_input(
        "Rozlišení (px / mm)",
        min_value=0.1, max_value=50000.0,
        value=10.0, step=0.5
    )
    unit = st.selectbox("Jednotka výstupu", ["mm", "µm"], index=0)

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Předzpracování obrazu</p>', unsafe_allow_html=True)
    brightness = st.slider("Jas", -100, 100, 0, 5)
    contrast   = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1)
    blur_kernel = st.slider("Gaussovo rozostření (px)", 1, 15, 5, 2)

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Canny – detekce hran</p>', unsafe_allow_html=True)
    canny_auto = st.toggle("Automatické prahy", value=True)
    if canny_auto:
        canny_sigma = st.slider("Citlivost (sigma)", 0.05, 0.8, 0.33, 0.01)
        canny_t1, canny_t2 = 0, 0
    else:
        canny_sigma = 0.33
        canny_t1 = st.slider("Canny T1 (dolní práh)", 0, 254, 10)
        canny_t2 = st.slider("Canny T2 (horní práh)", canny_t1 + 1, 255, 100)

    st.markdown('<p class="sidebar-section">Dilatace</p>', unsafe_allow_html=True)
    dilate_kernel = st.slider("Kernel dilatace (px)", 3, 25, 9, 2)
    dilate_iter   = st.slider("Iterace dilatace", 1, 5, 2)

    st.markdown('<p class="sidebar-section">Čištění masky</p>', unsafe_allow_html=True)
    clean_kernel = st.slider("Kernel čištění (px)", 1, 11, 3, 2)
    clean_iter   = st.slider("Iterace čištění", 1, 5, 2)

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Filtrace částic</p>', unsafe_allow_html=True)
    min_area_mm2 = st.number_input(
        "Minimální plocha (mm²)",
        min_value=0.0001, max_value=100.0,
        value=0.005, step=0.001, format="%.4f"
    )
    min_circularity = st.slider("Min. kruhovitost", 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.markdown('<p class="sidebar-section">Zobrazení</p>', unsafe_allow_html=True)
    show_mask      = st.toggle("Zobrazit masku", value=True)
    show_table     = st.toggle("Zobrazit tabulku dat", value=True)
    show_histograms = st.toggle("Zobrazit histogramy", value=True)
    highlight_max  = st.toggle("Zvýraznit maxima v tabulce", value=True)
    show_debug     = st.toggle("Mezikroky výsledné masky", value=False)

#  Hlavní obsah ───────────────────────────────────────────────────────────────

st.title("Analýza ferografických snímků")

if uploaded_file is None:
    st.info("Nahrajte snímek v levém panelu pro zahájení analýzy.")
    st.stop()


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


params = dict(
    px_per_mm=px_per_mm,
    unit=unit,
    brightness=brightness,
    contrast=contrast,
    blur_kernel=_odd(blur_kernel),
    min_area_mm2=min_area_mm2,
    min_circularity=min_circularity,
    canny_auto=canny_auto,
    canny_t1=canny_t1,
    canny_t2=canny_t2,
    canny_sigma=canny_sigma,
    dilate_kernel=_odd(dilate_kernel),
    dilate_iter=dilate_iter,
    clean_kernel=_odd(clean_kernel),
    clean_iter=clean_iter,
)


@st.cache_data(show_spinner=False)
def cached_analysis(img_bytes: bytes, params_key: str):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    _img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    _p = json.loads(params_key)
    return run_analysis(_img, _p)


img_bytes  = uploaded_file.getvalue()
params_key = json.dumps(params, sort_keys=True)

with st.spinner("Zpracovávám obraz…"):
    adjusted_img, annotated, mask, df, used_t1, used_t2 = cached_analysis(img_bytes, params_key)

# ── Metriky ───────────────────────────────────────────────────────────────────
st.markdown("#### Přehled výsledků")
m1, m2, m3, m4, m5, m6 = st.columns(6)

n = len(df)
m1.metric("Celkem částic", n)
if n > 0:
    ecol = f"Ekv. průměr ({unit})"
    acol = f"Plocha ({unit}²)"
    m2.metric(f"Průměr Ø ({unit})",    f"{df[ecol].mean():.3f}")
    m3.metric(f"Medián Ø ({unit})",    f"{df[ecol].median():.3f}")
    m4.metric(f"Prům. plocha ({unit}²)", f"{df[acol].mean():.4f}")
    m5.metric("Prům. kruhovitost",      f"{df['Kruhovitost'].mean():.3f}")
    m6.metric("Canny T1 / T2",          f"{used_t1} / {used_t2}")
else:
    for col, lbl in zip([m2, m3, m4, m5, m6],
                        [f"Průměr Ø ({unit})", f"Medián Ø ({unit})",
                         f"Prům. plocha ({unit}²)", "Prům. kruhovitost", "Canny T1 / T2"]):
        col.metric(lbl, "—")

st.markdown("---")

# ── Snímky – kompaktní zobrazení ─────────────────────────────────────────────
IMG_HEIGHT = 280   # px – výška náhledu; lze změnit dle potřeby

if show_mask:
    c1, c2, c3 = st.columns(3)
    cols_imgs = [(c1, cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), "Upravený snímek"),
                 (c2, mask,                                           "Detekční maska"),
                 (c3, cv2.cvtColor(annotated,    cv2.COLOR_BGR2RGB), "Anotovaný výsledek")]
else:
    c1, c2 = st.columns(2)
    cols_imgs = [(c1, cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), "Upravený snímek"),
                 (c2, cv2.cvtColor(annotated,    cv2.COLOR_BGR2RGB), "Anotovaný výsledek")]

for col, img_data, label in cols_imgs:
    with col:
        st.markdown(f'<p class="img-label">{label}</p>', unsafe_allow_html=True)
        # Zmenšení obrázku na fixní výšku pro kompaktnost
        if isinstance(img_data, np.ndarray):
            h_orig, w_orig = img_data.shape[:2]
            scale  = IMG_HEIGHT / h_orig
            w_new  = int(w_orig * scale)
            resized = cv2.resize(img_data, (w_new, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            st.image(resized, use_container_width=True)
        else:
            st.image(img_data, use_container_width=True)

st.markdown("---")

# ── Debug mezikroky ───────────────────────────────────────────────────────────
if show_debug:
    st.markdown("#### Mezikroky vytvoření výsledné masky")
    gray_img  = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray_img, (params["blur_kernel"], params["blur_kernel"]), 0)
    canny_dbg = cv2.Canny(blurred, used_t1, used_t2)
    k_d       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (params["dilate_kernel"], params["dilate_kernel"]))
    dilated   = cv2.dilate(canny_dbg, k_d, iterations=params["dilate_iter"])
    _, otsu   = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    d1, d2, d3, d4 = st.columns(4)
    for col_d, img_d, lbl_d in [
        (d1, canny_dbg, "1. Canny hrany"),
        (d2, dilated,   "2. Po dilataci"),
        (d3, otsu,      "3. Otsu maska"),
        (d4, mask,      "4. Výsledná maska"),
    ]:
        with col_d:
            st.markdown(f'<p class="img-label">{lbl_d}</p>', unsafe_allow_html=True)
            st.image(img_d, clamp=True, use_container_width=True)
    st.markdown("---")

# ── Histogramy ────────────────────────────────────────────────────────────────
if show_histograms and n > 0:
    st.markdown("#### Distribuce")
    hist_fig = make_histogram_fig(df, unit)
    st.pyplot(hist_fig, use_container_width=True, clear_figure=True)
    plt.close(hist_fig)
    st.markdown("---")

# ── Tabulka ───────────────────────────────────────────────────────────────────
if show_table:
    st.markdown(f"#### Naměřená data  `{n} částic`")
    if n > 0:
        num_cols = [c for c in df.columns if c != "ID"]
        styled = df.style.format({c: "{:.4f}" for c in num_cols})
        if highlight_max:
            styled = styled.highlight_max(subset=num_cols, color="rgba(249, 115, 22, 0.25)")
        st.dataframe(styled, use_container_width=True, height=380)
    else:
        st.warning("Žádné částice nebyly detekovány. Zkuste upravit parametry v sidebaru.")

st.markdown("---")

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("#### Export výsledků")
e1, e2, e3, e4 = st.columns(4)
fname = uploaded_file.name.rsplit(".", 1)[0]

with e1:
    if n > 0:
        st.download_button("CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{fname}_data.csv", mime="text/csv",
                           use_container_width=True)
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
                       file_name=f"{fname}_anotovany.png", mime="image/png",
                       use_container_width=True)

with e4:
    st.download_button("Maska (PNG)", data=mask_to_png_bytes(mask),
                       file_name=f"{fname}_maska.png", mime="image/png",
                       use_container_width=True)