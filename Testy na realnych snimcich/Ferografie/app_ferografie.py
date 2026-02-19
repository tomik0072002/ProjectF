import streamlit as st
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


def get_mask(img: np.ndarray) -> np.ndarray:

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny s pevnými hodnotami
    canny = cv2.Canny(blurred, 10, 100)
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    canny_connected = cv2.dilate(canny, kernel_connect, iterations=2)

    contours_c, _ = cv2.findContours(canny_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_metal = np.zeros_like(gray)
    cv2.drawContours(mask_metal, contours_c, -1, 255, thickness=cv2.FILLED)

    # Adaptivní Otsu threshold
    _, mask_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Sloučení
    combined = cv2.bitwise_or(mask_metal, mask_dark)

    # Finální začištění
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    return final_mask


def analyze_exact_shape(contour: np.ndarray, min_area: float) -> Optional[Dict[str, Any]]:
    exact_area = cv2.contourArea(contour)

    if exact_area < min_area:
        return None

    rect = cv2.minAreaRect(contour)
    (center, (w, h), angle) = rect
    length = max(w, h)
    width = min(w, h)
    aspect_ratio = length / width if width > 0 else 0

    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * exact_area) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        "area": exact_area, "length": length, "width": width,
        "ar": aspect_ratio, "circularity": circularity,
        "rect": rect, "center": center
    }



####### Streamlit aplikace #######

st.set_page_config(page_title="Ferografie Analýza", layout="wide")
st.title("Analýza ferografických snímků")
st.write("Nahrajte snímek a vyfiltrujte částice podle velikosti.")

st.sidebar.header("Nastavení")
uploaded_file = st.sidebar.file_uploader("Vyberte obrázek (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

st.sidebar.subheader("Filtrace")
min_area_val = st.sidebar.number_input("Minimální plocha částice (px)", min_value=1, value=50, step=10)

# Hlavní část apliakce
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Originální snímek")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", width=800)

    with st.spinner('Zpracovávám obraz...'):
        mask = get_mask(img)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        annotated_img = img.copy()
        data_list = []

        for i, cnt in enumerate(contours):
            stats = analyze_exact_shape(cnt, min_area=min_area_val)
            if stats is None: continue

            data_list.append({
                "ID": i + 1, "Plocha (px)": int(stats["area"]),
                "Délka (px)": round(stats["length"], 1), "Šířka (px)": round(stats["width"], 1),
                "Poměr stran": round(stats["ar"], 2), "Kruhovitost": round(stats["circularity"], 3)
            })

            cv2.drawContours(annotated_img, [cnt], -1, (0, 0, 255), 1)
            box = np.int64(cv2.boxPoints(stats["rect"]))
            cv2.drawContours(annotated_img, [box], 0, (0, 255, 0), 2)
            cx, cy = stats["center"]
            cv2.putText(annotated_img, str(i + 1), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detekční maska")
        st.image(mask, channels="GRAY", width=600)

    with col2:
        st.subheader("Výsledek detekce")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), channels="RGB", width=600)

    st.write("---")
    st.subheader(f"Naměřená data (Nalezeno: {len(data_list)} částic)")

    if data_list:
        df = pd.DataFrame(data_list)

        sloupce_pro_zvyrazneni = ["Plocha (px)", "Délka (px)", "Šířka (px)", "Kruhovitost"]

        st.dataframe(
            df.style.highlight_max(axis=0, subset=sloupce_pro_zvyrazneni, color='lightgreen'),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Stáhnout data jako CSV", data=csv, file_name='ferografie.csv', mime='text/csv')
    else:
        st.warning("Žádné částice nebyly detekovány.")
else:
    st.info("Pro zahájení analýzy nahrajte obrázek vlevo.")