import io
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Dict, Optional, Any

#  Nastavení stránky
st.set_page_config(
    page_title="Ferografie – Analýza částic",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def make_histogram_fig(df: pd.DataFrame, unit: str) -> plt.Figure:
    col = f"Ekv. průměr ({unit})"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Průhledné pozadí grafu pro přizpůsobení se Streamlit tématu
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor("none")
        # Neutrální barva textu os, aby byla vidět na světlém i tmavém pozadí
        ax.tick_params(colors="gray")
        ax.xaxis.label.set_color("gray")
        ax.yaxis.label.set_color("gray")
        ax.title.set_color("gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    if col in df.columns and len(df) > 0:
        data = df[col]
        # Barvy sloupců přizpůsobené pro dobrou viditelnost
        axes[0].hist(data, bins=min(25, len(df)), color="#4fc3f7", edgecolor="gray", linewidth=0.4, alpha=0.8)
        axes[0].axvline(data.mean(), color="#ff7043", lw=1.5, ls="--", label=f"Průměr {data.mean():.3f}")
        axes[0].axvline(data.median(), color="#66bb6a", lw=1.5, ls=":", label=f"Medián {data.median():.3f}")
        axes[0].set_xlabel(f"Ekvivalentní průměr ({unit})")
        axes[0].set_ylabel("Počet")
        axes[0].set_title("Distribuce velikostí")
        axes[0].legend(fontsize=8, framealpha=0.2)

        circ = df["Kruhovitost"]
        axes[1].hist(circ, bins=min(20, len(df)), color="#4fc3f7", edgecolor="gray", linewidth=0.4, alpha=0.8)
        axes[1].axvline(circ.mean(), color="#ff7043", lw=1.5, ls="--", label=f"Průměr {circ.mean():.3f}")
        axes[1].set_xlabel("Kruhovitost (0–1)")
        axes[1].set_ylabel("Počet")
        axes[1].set_title("Distribuce kruhovitosti")
        axes[1].legend(fontsize=8, framealpha=0.2)

    plt.tight_layout()
    return fig


#  Sidebar

with st.sidebar:
    st.markdown("##  Ferografie")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        " Nahrát snímek (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

    st.markdown("---")
    st.markdown("###  Kalibrace")
    px_per_mm = st.number_input(
        "Rozlišení (px / mm)",
        min_value=0.1, max_value=50000.0,
        value=10.0, step=0.5
    )
    unit = st.selectbox("Jednotka výstupu", ["mm", "µm"], index=0)

    st.markdown("---")
    st.markdown("###  Předzpracování obrazu")
    brightness = st.slider("Jas (Brightness)", -100, 100, 0, 5)
    contrast = st.slider("Kontrast (Contrast)", 0.5, 3.0, 1.0, 0.1)
    blur_kernel = st.slider("Gaussovo rozostření (px)", 1, 15, 5, 2,
                            help="Vyšší hodnota odstraní šum, ale může smazat drobné částice.")

    st.markdown("---")
    st.markdown("###  Canny – detekce hran")
    canny_auto = st.toggle("Automatické prahy", value=True)
    if canny_auto:
        canny_sigma = st.slider(
            "Citlivost (sigma)", 0.05, 0.8, 0.33, 0.01,
            help="Menší hodnota = přísnější detekce hran"
        )
        canny_t1, canny_t2 = 0, 0
    else:
        canny_sigma = 0.33
        canny_t1 = st.slider("Canny T1 (dolní práh)", 0, 254, 10)
        canny_t2 = st.slider("Canny T2 (horní práh)", canny_t1 + 1, 255, 100)

    st.markdown("###  Dilatace (propojení hran)")
    dilate_kernel = st.slider("Velikost kernelu dilatace (px)", 3, 25, 9, 2)
    dilate_iter = st.slider("Počet iterací dilatace", 1, 5, 2)

    st.markdown("###  Čištění masky")
    clean_kernel = st.slider("Velikost kernelu čištění (px)", 1, 11, 3, 2)
    clean_iter = st.slider("Počet iterací čištění", 1, 5, 2)

    st.markdown("---")
    st.markdown("###  Filtrace částic")
    min_area_mm2 = st.number_input(
        "Minimální plocha (mm²)",
        min_value=0.0001, max_value=100.0,
        value=0.005, step=0.001, format="%.4f"
    )
    min_circularity = st.slider("Minimální kruhovitost", 0.0, 1.0, 0.0, 0.05,
                                help="0 = všechny tvary, 1 = pouze dokonalé kruhy")

    st.markdown("---")
    st.markdown("###  Zobrazení")
    show_mask = st.toggle("Zobrazit masku", value=True)
    show_table = st.toggle("Zobrazit tabulku dat", value=True)
    show_histograms = st.toggle("Zobrazit histogramy", value=True)
    highlight_max = st.toggle("Zvýraznit maxima v tabulce", value=True)
    show_debug = st.toggle(" Mezikroky výsledné masky", value=False)

#  Hlavní obsah aplikace

st.title(" Analýza ferografických snímků")

if uploaded_file is None:
    st.info("  Nahrajte snímek v levém panelu pro zahájení analýzy.")
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


img_bytes = uploaded_file.getvalue()
params_key = json.dumps(params, sort_keys=True)

with st.spinner(" Zpracovávám obraz…"):
    adjusted_img, annotated, mask, df, used_t1, used_t2 = cached_analysis(img_bytes, params_key)

st.markdown("###  Přehled výsledků")
m1, m2, m3, m4, m5, m6 = st.columns(6)

n = len(df)
m1.metric("Celkem částic", n)
if n > 0:
    ecol = f"Ekv. průměr ({unit})"
    acol = f"Plocha ({unit}²)"
    m2.metric(f"Průměr ekv. Ø ({unit})", f"{df[ecol].mean():.2f}")
    m3.metric(f"Medián ekv. Ø ({unit})", f"{df[ecol].median():.2f}")
    m4.metric(f"Průměr plochy ({unit}²)", f"{df[acol].mean():.2f}")
    m5.metric("Průměr kruhovitosti", f"{df['Kruhovitost'].mean():.2f}")
    m6.metric("Canny T1 / T2", f"{used_t1} / {used_t2}")
else:
    for col, lbl in zip([m2, m3, m4, m5, m6],
                        [f"Průměr ekv. Ø ({unit})", f"Medián ekv. Ø ({unit})", f"Průměr plochy ({unit}²)",
                         "Průměr kruhovitosti", "Canny T1 / T2"]):
        col.metric(lbl, "—")

st.markdown("---")

if show_mask:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 10px;'>Upravený snímek</div>",
                    unsafe_allow_html=True)
        st.image(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 10px;'>Detekční maska</div>",
                    unsafe_allow_html=True)
        st.image(mask, clamp=True, use_container_width=True)
    with c3:
        st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 10px;'>Anotovaný výsledek</div>",
                    unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 10px;'>Upravený snímek</div>",
                    unsafe_allow_html=True)
        st.image(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with c2:
        st.markdown("<div style='text-align: center; font-weight: bold; margin-bottom: 10px;'>Anotovaný výsledek</div>",
                    unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

st.markdown("---")

if show_debug:
    st.markdown("###  Mezikroky vytvoření výsledné masky")
    gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (params["blur_kernel"], params["blur_kernel"]), 0)
    canny_dbg = cv2.Canny(blurred, used_t1, used_t2)
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (params["dilate_kernel"], params["dilate_kernel"]))
    dilated = cv2.dilate(canny_dbg, k_d, iterations=params["dilate_iter"])
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"**1. Canny hrany**")
        st.image(canny_dbg, clamp=True, use_container_width=True)
    with d2:
        st.markdown(f"**2. Po dilataci**")
        st.image(dilated, clamp=True, use_container_width=True)
    with d3:
        st.markdown("**3. Otsu maska**")
        st.image(otsu, clamp=True, use_container_width=True)
    with d4:
        st.markdown("**4. Výsledná maska**")
        st.image(mask, clamp=True, use_container_width=True)
    st.markdown("---")

if show_histograms and n > 0:
    st.markdown("###  Distribuce")
    hist_fig = make_histogram_fig(df, unit)
    st.pyplot(hist_fig, use_container_width=True, clear_figure=True)
    plt.close(hist_fig)
    st.markdown("---")

if show_table:
    st.markdown(f"###  Naměřená data  `{n} částic`")
    if n > 0:
        num_cols = [c for c in df.columns if c != "ID"]
        styled = df.style.format({c: "{:.4f}" for c in num_cols})
        if highlight_max:
            # Neutrální průhledná barva pro zvýraznění max hodnot, ať ladí se vším
            styled = styled.highlight_max(subset=num_cols, color="rgba(249, 115, 22, 0.3)")
        st.dataframe(styled, use_container_width=True, height=420)
    else:
        st.warning(" Žádné částice nebyly detekovány. Zkuste upravit parametry v sidebaru.")

st.markdown("---")

st.markdown("###  Export výsledků")
e1, e2, e3, e4 = st.columns(4)

fname = uploaded_file.name.rsplit(".", 1)[0]

with e1:
    if n > 0:
        st.download_button(
            " CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{fname}_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.button(" CSV", disabled=True, use_container_width=True)

with e2:
    if n > 0:
        st.download_button(
            " Excel (.xlsx)",
            data=df_to_excel_bytes(df),
            file_name=f"{fname}_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    else:
        st.button(" Excel (.xlsx)", disabled=True, use_container_width=True)

with e3:
    st.download_button(
        " Anotovaný snímek (PNG)",
        data=img_to_png_bytes(annotated),
        file_name=f"{fname}_anotovany.png",
        mime="image/png",
        use_container_width=True,
    )

with e4:
    st.download_button(
        " Maska (PNG)",
        data=mask_to_png_bytes(mask),
        file_name=f"{fname}_maska.png",
        mime="image/png",
        use_container_width=True,
    )