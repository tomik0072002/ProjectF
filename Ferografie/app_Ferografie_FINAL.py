import io
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from typing import Any
from PIL import Image

# Načtení loga VUT
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
logo = Image.open(logo_path)

# Nastavení stránky
st.set_page_config(
    page_title="Ferografie - analyza otěrových částic",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)


# Pomocné funkce

# Automatické prahy pro Cannyho hranový detektor
def auto_canny_thresholds(blurred: np.ndarray, sigma: float = 0.33) -> tuple[int, int]:

    p_low = np.percentile(blurred, 10)
    p_high = np.percentile(blurred, 90)

    t1 = int(max(0, p_low * (1.0 - sigma)))
    t2 = int(min(255, p_high * (1.0 - sigma)))

    # Zajistíme minimální odstup mezi prahy
    t1 = min(t1, 50)
    t2 = max(t2, t1 + 30)
    t2 = min(t2, 200)

    return t1, t2

def create_canny_edges(blurred: np.ndarray, canny_auto: bool, t1: int, t2: int, sigma: float) -> tuple[np.ndarray, int, int]:
    if canny_auto:
        t1, t2 = auto_canny_thresholds(blurred, sigma)
    edges = cv2.Canny(blurred, t1, t2)
    return edges, t1, t2

# Morfoilogické operace
def morf_operate(edges: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(edges, k, iterations=iterations)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled

# Binární maska pro detekované částice
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

    # Převod do stupňů šedi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Detekce hran a morfologické operace (dilatace + vyplnění)
    canny_edges, used_t1, used_t2 = create_canny_edges(
        blurred, canny_auto, canny_t1, canny_t2, canny_sigma
    )
    edge_filled = morf_operate(canny_edges, dilate_kernel, dilate_iter)

    # Otsuovo prahování
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kombinace obou masek přes logické OR
    combined = cv2.bitwise_or(edge_filled, otsu_mask)

    # Morfologické čištění
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel, clean_kernel))
    final_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_clean, iterations=clean_iter)

    return final_mask, used_t1, used_t2


# Analýza tvarů nalezených částic

def analyze_shape_of_particle(contour: np.ndarray,
                  min_area_px: float,
                  px_per_mm: float) -> dict[str, Any]:

    area_px = cv2.contourArea(contour)
    if area_px < min_area_px:
        return None

    rect = cv2.minAreaRect(contour)
    _, (w, h), _ = rect
    length_px = max(w, h)
    width_px = min(w, h)


    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 1e-6 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area_px / hull_area if hull_area > 1e-6 else 0.0

    equiv_diam_px = np.sqrt(4 * area_px / np.pi)

    scale = 1.0 / px_per_mm          # px → mm
    scale2 = scale ** 2               # px² → mm²

    center = rect[0]

    return {
        "area_px": area_px,
        "area_mm2": area_px * scale2,
        "length_mm": length_px * scale,
        "width_mm": width_px * scale,
        "equiv_diam_mm": equiv_diam_px * scale,
        "circularity": min(circularity, 1.0),
        "ar": length_px / width_px if width_px > 0 else 0.0,
        "rect": rect,
        "center": center,
    }


# Hlavní analýza/výpočet

def run_analysis(img: np.ndarray, params: dict):

    # Úprava jasu a kontrastu
    adjusted = cv2.convertScaleAbs(img, alpha=params["contrast"], beta=params["brightness"])

    # Převod px na mm
    px_per_mm = params["px_per_mm"]
    min_area_px = params["min_area_mm2"] * px_per_mm ** 2

    mask, t1, t2 = get_mask(
        adjusted,
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

    # Hledání kontur částic v binární masce
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Anotace nalezených částic
    annotated = adjusted.copy()
    rows = []
    u = params["unit"]

    for cnt in contours:
        s = analyze_shape_of_particle(cnt, min_area_px, px_per_mm)
        if s is None or s["circularity"] < params["min_circularity"]:
            continue

        pid = len(rows) + 1
        rows.append({
            "ID": pid,
            f"Plocha ({u}²)": round(s["area_mm2"], 4),
            f"Délka ({u})": round(s["length_mm"], 3),
            f"Šířka ({u})": round(s["width_mm"], 3),
            f"Ekv. průměr ({u})": round(s["equiv_diam_mm"], 3),
            "Poměr stran": round(s["ar"], 2),
            "Kruhovitost": round(s["circularity"], 3),
        })

        # Červený obrys částice + zelený obdélník
        cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 1)
        box = np.int64(cv2.boxPoints(s["rect"]))
        cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
        cx, cy = int(s["center"][0]), int(s["center"][1])
        cv2.putText(annotated, str(pid), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    return adjusted, annotated, mask, pd.DataFrame(rows), t1, t2


# Převod OpenCV obrazu do PNG pro export
def to_image(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Nepodařilo se zakódovat obraz do PNG.")
    return buf.tobytes()

# Převod grafu do PNG pro export
def fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    buf.seek(0)
    return buf.read()

# Převod DaraFrame do xlsx
def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Částice", index=False)
        ws = writer.sheets["Částice"]
        for col_cells in ws.columns:
            max_len = max((len(str(c.value)) for c in col_cells if c.value), default=8)
            ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4
    buf.seek(0)
    return buf.read()

def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1

# Boční panel - sidebar

with st.sidebar:
    st.markdown("## Ferografie")
    st.markdown("---")

    # Testovací snímky
    st.markdown("### Testovací snímky")
    test_files = ["test_fero_01.png", "test_fero_02.png", "test_fero_03.png"]
    found_any = False

    for name in test_files:
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            found_any = True
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Stáhnout {name}",
                    data=f,
                    file_name=name,
                    mime="image/png",
                    use_container_width=True,
                )

    if not found_any:
        st.warning("Testovací snímky nebyly nalezeny.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Nahrát snímek (PNG / JPG)", type=["png", "jpg", "jpeg"])

    st.markdown("---")

    # Kalibrace - převod px na mm
    st.markdown("### Kalibrace")
    px_per_mm = st.number_input(
        "Rozlišení (px / mm)",
        min_value=0.1, max_value=50_000.0,
        value=10.0, step=0.5,
    )
    unit = "mm"

    st.markdown("---")

    # Předzpracování
    st.markdown("### Předzpracování obrazu")
    brightness = st.slider("Jas", -100, 100, 0, 5)
    contrast = st.slider("Kontrast", 0.5, 3.0, 1.0, 0.1)
    blur_kernel = st.slider("Gaussovo rozostření (px)", 1, 15, 5, 2,)

    st.markdown("---")

    # Canny
    st.markdown("### Canny – detekce hran")
    canny_auto = st.toggle("Automatické prahy", value=True)
    if canny_auto:
        canny_sigma = st.slider("Citlivost detektoru", 0.05, 0.8, 0.33, 0.01,)
        canny_t1 = canny_t2 = 0
    else:
        canny_sigma = 0.33
        canny_t1 = st.slider("Dolní práh)", 0, 254, 10)
        canny_t2 = st.slider("Horní práh)", canny_t1 + 1, 255, 100)

    st.markdown("### Dilatace (propojení hran)")
    dilate_kernel = st.slider("Velikost kernelu (px)", 3, 25, 9, 2)
    dilate_iter = st.slider("Počet iterací", 1, 5, 2, key="dilate_iter_slider")

    st.markdown("### Čištění masky")
    clean_kernel = st.slider("Velikost kernelu (px)", 1, 11, 3, 2)
    clean_iter = st.slider("Počet itercí", 1, 5, 2, key="clean_iter_slider")

    st.markdown("---")

    # Filtrace
    st.markdown("### Filtrace částic")
    min_area_mm2 = st.number_input(
        "Minimální plocha (mm²)",
        min_value=0.0001, max_value=100.0,
        value=0.005, step=0.001, format="%.4f",
    )
    min_circularity = st.slider(
        "Minimální kruhovitost", 0.0, 1.0, 0.0, 0.05,
        help="0 = všechny tvary, 1 = dokonalé kruhy.",
    )

    st.markdown("---")

    # Zobrazení
    st.markdown("### Zobrazení")
    show_mask = st.toggle("Zobrazit masku", value=True)
    show_table = st.toggle("Zobrazit tabulku dat", value=True)
    highlight_max = st.toggle("Zvýraznit maxima v tabulce", value=True)
    show_debug = st.toggle("Mezikroky výsledné masky", value=True)


# Hlavní stránka

st.title("Analýza ferografických snímků")

if uploaded_file is None:
    st.info("Nahrajte snímek v levém panelu pro zahájení analýzy.")
    st.stop()

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

# Načtení obrazu do aplikace
@st.cache_data(show_spinner=False)
def cached_analysis(img_bytes: bytes, params_key: str):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Nepodařilo se načíst obraz – zkontrolujte formát souboru.")
        st.stop()
    return run_analysis(img, json.loads(params_key))

img_bytes = uploaded_file.getvalue()
params_key = json.dumps(params, sort_keys=True)

with st.spinner("Zpracovávám obraz…"):
    adjusted_img, annotated, mask, df, used_t1, used_t2 = cached_analysis(img_bytes, params_key)

# Statistické hodnoty
st.markdown("### Přehled výsledků")
col_metrics = st.columns(5)
n = len(df)
col_metrics[0].metric("Celkem částic", n)

if n > 0:
    ecol = f"Ekv. průměr ({unit})"
    acol = f"Plocha ({unit}²)"
    col_metrics[1].metric(f"Průměr plochy ({unit}²)", f"{df[acol].mean():.2f}")
    col_metrics[2].metric("Průměr kruhovitosti", f"{df['Kruhovitost'].mean():.2f}")

st.markdown("---")

# Snímky
if show_mask:
    c1, c2, c3 = st.columns(3)
    panels = [
        (c1, "Upravený snímek", cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)),
        (c2, "Detekční maska", mask),
        (c3, "Anotovaný výsledek", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)),
    ]
else:
    c1, c2 = st.columns(2)
    panels = [
        (c1, "Upravený snímek", cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB)),
        (c2, "Anotovaný výsledek", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)),
    ]

for col, title, image in panels:
    with col:
        st.markdown(
            f"<div style='text-align:center;font-weight:bold;margin-bottom:10px'>{title}</div>",
            unsafe_allow_html=True,
        )
        st.image(image, clamp=True, use_container_width=True)

st.markdown("---")

# Mezikroky masky
if show_debug:
    st.markdown("#### Mezikroky vytvoření výsledné masky")
    gray_dbg = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    blurred_dbg = cv2.GaussianBlur(gray_dbg, (params["blur_kernel"], params["blur_kernel"]), 0)
    canny_dbg = cv2.Canny(blurred_dbg, used_t1, used_t2)
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params["dilate_kernel"], params["dilate_kernel"]))
    dilated_dbg = cv2.dilate(canny_dbg, k_d, iterations=params["dilate_iter"])
    _, otsu_dbg = cv2.threshold(blurred_dbg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    d1, d2, d3, d4 = st.columns(4)
    for col, label, img_dbg in [
        (d1, "Cannyho detektor", canny_dbg),
        (d2, "Po dilataci", dilated_dbg),
        (d3, "Otsuova maska", otsu_dbg),
        (d4, "Výsledná maska", mask),
    ]:
        with col:
            st.markdown(f"**{label}**")
            st.image(img_dbg, clamp=True, use_container_width=True)

    st.markdown("---")


# Tabulka
if show_table:
    st.markdown(f"### Naměřená data  `{n} částic`")
    if n > 0:
        num_cols = [c for c in df.columns if c != "ID"]
        styled = df.style.format({c: "{:.4f}" for c in num_cols})
        if highlight_max:
            styled = styled.highlight_max(subset=num_cols, color="rgba(249, 115, 22, 0.3)")
        st.dataframe(styled, use_container_width=True, height=420)
    else:
        st.warning("Žádné částice nebyly detekovány. Zkuste upravit parametry v sidebaru.")

st.markdown("---")

# Export
st.markdown("### Export výsledků")
fname = uploaded_file.name.rsplit(".", 1)[0]
e1, e2, e3, e4 = st.columns(4)

with e1:
    if n > 0:
        st.download_button("CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{fname}_data.csv", mime="text/csv", use_container_width=True,)
    else:
        st.button("CSV", disabled=True, use_container_width=True)

with e2:
    if n > 0:
        st.download_button("Excel (.xlsx)", data=df_to_excel(df), file_name=f"{fname}_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,)
    else:
        st.button("Excel (.xlsx)", disabled=True, use_container_width=True)

with e3:
    st.download_button("Anotovaný snímek (PNG)", data=to_image(annotated), file_name=f"{fname}_anotovany.png", mime="image/png", use_container_width=True,)

with e4:
    st.download_button("Maska (PNG)", data=to_image(mask), file_name=f"{fname}_maska.png", mime="image/png", use_container_width=True,)