import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
import io
import time
from datetime import datetime
import pandas as pd
from PIL import Image
# Načtení obrázku loga
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
logo = Image.open(logo_path)
# Nastavení stránky
st.set_page_config(
    page_title="Optický tok",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)
# Inicializace stavu
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
_ss("img_frame1", None)
_ss("img_frame2", None)
_ss("img_file1_id", None)
_ss("img_file2_id", None)
# Hlavní funkce
# Výpočet hustého toku
def OF_flow_dense(gray1, gray2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma):
    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=0
    )
# Výpočet řídkého toku
def OF_flow_sparse(gray1, gray2, max_corners, quality, min_dist, block_size, lk_winsize, lk_levels):
    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    # Hledání výrazných bodů
    pts1 = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist, blockSize=block_size)
    if pts1 is None or len(pts1) == 0:
        h, w = gray1.shape
        return np.zeros((h, w, 2), dtype=np.float32), np.array([]), np.array([])
    pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        g1, g2, pts1, None,
        winSize=(lk_winsize, lk_winsize), maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    good = (status.ravel() == 1)
    p1g = pts1[good].reshape(-1, 2)
    p2g = pts2[good].reshape(-1, 2)
    h, w = gray1.shape
    flow = np.zeros((h, w, 2), dtype=np.float32)
    for (x1, y1), (x2, y2) in zip(p1g, p2g):
        xi, yi = int(round(x1)), int(round(y1))
        if 0 <= xi < w and 0 <= yi < h:
            flow[yi, xi, 0] = x2 - x1
            flow[yi, xi, 1] = y2 - y1
    return flow, p1g, p2g
# Vytvoření panelu vizualizace výsledků
def make_panels(frame1_bgr, frame2_bgr, flow, threshold_factor=2.0, arrow_step=25, method="dense", sparse_pts1=None, sparse_pts2=None):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    med = np.median(magnitude)
    std = np.std(magnitude)
    motion_threshold = med + threshold_factor * std
    motion_mask = magnitude > motion_threshold
    moving_mags = magnitude[motion_mask]
    typical_motion = np.percentile(moving_mags, 50) if len(moving_mags) > 0 else 1.0
    arrow_scale = 40.0 / max(typical_motion, 1.0)
    p1 = frame2_bgr.copy()
    hsv = np.zeros_like(frame1_bgr)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    p2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    p3 = frame2_bgr.copy()
    highlight = np.zeros_like(frame2_bgr)
    highlight[motion_mask] = (0, 0, 255)
    p3 = cv2.addWeighted(p3, 0.65, highlight, 0.35, 0)
    p4 = frame2_bgr.copy()
    h, w = frame2_bgr.shape[:2]
    if method == "sparse" and sparse_pts1 is not None and len(sparse_pts1) > 0:
        for (x1, y1), (x2, y2) in zip(sparse_pts1, sparse_pts2):
            mag = np.hypot(x2 - x1, y2 - y1)
            speed_norm = min(mag / max(motion_threshold * 2, 1e-6), 1.0)
            t = speed_norm * 2 if speed_norm < 0.5 else (speed_norm - 0.5) * 2
            color = (int(255 * (1 - t)), 255, int(255 * t)) if speed_norm < 0.5 else (0, int(255 * (1 - t)), 255)
            cv2.arrowedLine(p4, (int(round(x1)), int(round(y1))),
                            (int(round(x2)), int(round(y2))), color, 2, tipLength=0.3, line_type=cv2.LINE_AA)
            cv2.circle(p4, (int(round(x1)), int(round(y1))), 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        for y in range(0, h, arrow_step):
            for x in range(0, w, arrow_step):
                if motion_mask[y, x]:
                    fx, fy = flow[y, x]
                    end = (int(x + fx * arrow_scale), int(y + fy * arrow_scale))
                    speed_norm = min(magnitude[y, x] / max(motion_threshold * 2, 1e-6), 1.0)
                    t = speed_norm * 2 if speed_norm < 0.5 else (speed_norm - 0.5) * 2
                    color = (int(255 * (1 - t)), 255, int(255 * t)) if speed_norm < 0.5 else (0, int(255 * (1 - t)),
                                                                                                255)
                    cv2.arrowedLine(p4, (x, y), end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)
    stats = {
        "moving_pixels": int(motion_mask.sum()),
        "coverage_pct": float(motion_mask.sum() / motion_mask.size * 100),
        "avg_magnitude": float(moving_mags.mean()) if len(moving_mags) > 0 else 0.0,
        "max_magnitude": float(magnitude.max()),
    }
    return p1, p2, p3, p4, stats, magnitude
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def encode_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""
def make_combined(panels):
    h = min(p.shape[0] for p in panels)
    w = min(p.shape[1] for p in panels)
    resized = [cv2.resize(p, (w, h)) for p in panels]
    top = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    return np.vstack([top, bottom])
def results_to_dataframe(results):
    rows = []
    for i, r in enumerate(results):
        s = r["stats"]
        rows.append({
            "segment": i + 1,
            "pohybujici_pixely": s["moving_pixels"],
            "pokryti_pct": round(s["coverage_pct"], 2),
            "avg_magnituda": round(s["avg_magnitude"], 4),
            "max_magnituda": round(s["max_magnitude"], 4),
        })
    return pd.DataFrame(rows)
def downscale(img, max_w=1280):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img
# Boční panel - sidebar
with st.sidebar:
    st.markdown("## Detekce pohybu")
    st.markdown("Analýza optického toku")
    st.markdown("---")
    # Testovací snímky
    st.markdown("### Testovací snímky")
    test_files = ["test_flow_01a.jpg", "test_flow_01b.jpg", "test_flow_02a.jpg", "test_flow_02b.jpg",
                  "test_flow_03a.jpg", "test_flow_03b.jpg"]
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
                    mime="image/jpg",
                    use_container_width=True,
                )
    st.markdown("**Metoda výpočtu**")
    flow_method = st.radio(
        "flow_method_radio",
        ["Hustý tok (Dense Flow - Farneback)", "Řídký tok (Sparse Flow - Lucas-Kanade)"],
        label_visibility="collapsed",
    )
    is_dense = "Dense" in flow_method
    st.markdown("---")
    with st.expander("Parametry výpočtu", expanded=True):
        if is_dense:
            st.caption("Parametry hustého toku")
            pyr_scale = st.slider("Měřítko pyramid", 0.1, 0.9, 0.5, 0.05)
            levels = st.slider("Úrovně", 1, 10, 5)
            winsize = st.slider("Velikost okna", 5, 50, 25, 2)
            iterations = st.slider("Počet iterací", 1, 10, 6)
            poly_n = st.slider("Velikost okna N-tého polynomu", 5, 9, 7, 2)
            poly_sigma = st.slider("Směrodatná odchylka", 1.0, 2.5, 1.5, 0.1)
            lk_max_corners = 500; lk_quality = 0.01; lk_min_dist = 7
            lk_block = 7; lk_winsize = 21; lk_levels = 3
        else:
            st.caption("Parametry řídkého toku")
            lk_max_corners = st.slider("Max. počet rohů", 50, 2000, 200, 50)
            lk_quality = st.slider("Kvalita rohů", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            lk_min_dist = st.slider("Min. vzdálenost rohů", 3, 30, 8)
            lk_block = st.slider("Velikost sledovaného okna", 3, 15, 9, 2)
            lk_winsize = st.slider("Velikost vyhledávacího okna", 5, 51, 32, 2)
            lk_levels = st.slider("Počet úrovní pyramid", 1, 6, 4)
            pyr_scale = 0.5; levels = 6; winsize = 25
            iterations = 5; poly_n = 7; poly_sigma = 1.5
    with st.expander("Detekce a zobrazení pohybu", expanded=True):
        st.caption("Nastavení prahů a vizualizace")
        threshold_factor = st.slider("Práh pohybu (násobek směrodatné odchylky)", 0.5, 5.0, 2.3, 0.1)
        arrow_step = st.slider("HUstota vykreslených šipek", 10, 60, 25, 5, disabled=not is_dense)
# Hlavní stránka
st.title("OPTICKÝ TOK")
st.markdown("Nahrajte dva po sobě jdoucí snímky pro detekci pohybu mezi nimi.")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Snímek 1 (referenční)**")
    f1 = st.file_uploader("Snímek 1", type=["png", "jpg", "jpeg",], key="img1", label_visibility="collapsed")
    if f1 is not None and f1.file_id != st.session_state.img_file1_id:
            img1_np = np.frombuffer(f1.read(), np.uint8)
            decoded = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
            st.session_state.img_frame1 = downscale(decoded)
            st.session_state.img_file1_id = f1.file_id
    if st.session_state.img_frame1 is not None:
            st.image(bgr_to_rgb(st.session_state.img_frame1), use_container_width=True)
with c2:
    st.markdown("**Snímek 2 (cílový)**")
    f2 = st.file_uploader("Snímek 2", type=["png", "jpg", "jpeg",], key="img2", label_visibility="collapsed")
    if f2 is not None and f2.file_id != st.session_state.img_file2_id:
            img2_np = np.frombuffer(f2.read(), np.uint8)
            decoded = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
            st.session_state.img_frame2 = downscale(decoded)
            st.session_state.img_file2_id = f2.file_id
    if st.session_state.img_frame2 is not None:
            st.image(bgr_to_rgb(st.session_state.img_frame2), use_container_width=True)
frame1 = st.session_state.get("img_frame1")
frame2 = st.session_state.get("img_frame2")
if frame1 is not None and frame2 is not None:
    if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    if is_dense:
        flow = OF_flow_dense(g1, g2, pyr_scale, levels, winsize,
                                      iterations, poly_n, poly_sigma)
        p1, p2, p3, p4, stats, magnitude = make_panels(frame1, frame2, flow,
                                                           threshold_factor, arrow_step, "dense")
    else:
        flow, pts1, pts2 = OF_flow_sparse(g1, g2, lk_max_corners, lk_quality,
                                                   lk_min_dist, lk_block, lk_winsize, lk_levels)
        p1, p2, p3, p4, stats, magnitude = make_panels(frame1, frame2, flow,
                                                           threshold_factor, arrow_step, "sparse",
                                                           sparse_pts1=pts1, sparse_pts2=pts2)
    elapsed = time.perf_counter() - t0
    st.markdown("---")
    st.markdown("### Statistické hodnoty")
    st.caption(f"Doba výpočtu: {elapsed * 1000:.0f} ms")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Pohybující se pixely", f"{stats['moving_pixels']:,}")
    mc2.metric("Pokrytí pohybem", f"{stats['coverage_pct']:.1f} %")
    mc3.metric("Průměrný pohyb", f"{stats['avg_magnitude']:.2f} [px]")
    mc4.metric("Max. pohyb", f"{stats['max_magnitude']:.2f} [px]")
    st.markdown("---")
    st.markdown("### Výsledky analýzy")
    panels_list = [p1, p2, p3, p4]
    panels_label = ["Originál", "Flow mapa (HSV)", "Detekce pohybu", "Vektory pohybu"]
    cols = st.columns(4)
    for i, (panel, label) in enumerate(zip(panels_list, panels_label)):
        with cols[i]:
            st.markdown(f"**{label}**")
            st.image(bgr_to_rgb(panel), use_container_width=True)
    combined = make_combined([p1, p2, p3, p4])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for panel, lbl in zip([p1, p2, p3, p4],
                                ["original", "flow_mapa", "pohyb_oblasti", "sipky"]):
            zf.writestr(f"{ts}_{lbl}.png", encode_png(panel))
        zf.writestr(f"{ts}_combined.png", encode_png(combined))
    zip_buf.seek(0)
    st.markdown("---")
    st.markdown("### Export")
    st.download_button("Stáhnout výsledky obrazové analýzy (ZIP)", data=zip_buf, file_name=f"opticky_tok_snimky_{ts}.zip", mime="application/zip")