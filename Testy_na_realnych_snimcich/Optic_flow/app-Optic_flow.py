import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
import io
from datetime import datetime


# Nastavení stránky
st.set_page_config(
    page_title="Optical Flow Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }

  /* Dark industrial theme */
  .stApp {
      background: #0e0e12;
      color: #e2e2e8;
  }

  section[data-testid="stSidebar"] {
      background: #141418 !important;
      border-right: 1px solid #2a2a35;
  }

  .block-container {
      padding-top: 2rem;
      padding-bottom: 2rem;
  }

  /* Header */
  .app-header {
      font-family: 'Space Mono', monospace;
      font-size: 2.2rem;
      font-weight: 700;
      letter-spacing: -1px;
      background: linear-gradient(135deg, #00e5ff 0%, #7b61ff 60%, #ff4fcf 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0;
  }

  .app-subtitle {
      font-size: 0.85rem;
      color: #5a5a72;
      letter-spacing: 3px;
      text-transform: uppercase;
      font-family: 'Space Mono', monospace;
      margin-top: 0.2rem;
  }

  /* Cards */
  .panel-card {
      background: #18181f;
      border: 1px solid #2a2a38;
      border-radius: 12px;
      padding: 1.2rem;
      margin-bottom: 1rem;
  }

  .panel-title {
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #5a5a78;
      margin-bottom: 0.6rem;
  }

  /* Metric pills */
  .metric-row {
      display: flex;
      gap: 0.6rem;
      flex-wrap: wrap;
      margin-top: 0.8rem;
  }

  .metric-pill {
      background: #1f1f2c;
      border: 1px solid #2e2e42;
      border-radius: 20px;
      padding: 0.25rem 0.8rem;
      font-family: 'Space Mono', monospace;
      font-size: 0.72rem;
      color: #9090b8;
  }

  .metric-pill span {
      color: #00e5ff;
      font-weight: 700;
  }

  /* Streamlit button override */
  .stButton > button {
      background: linear-gradient(135deg, #00e5ff22, #7b61ff22);
      border: 1px solid #7b61ff88;
      color: #c4b5fd;
      font-family: 'Space Mono', monospace;
      font-size: 0.8rem;
      letter-spacing: 1px;
      border-radius: 8px;
      padding: 0.5rem 1.2rem;
      transition: all 0.2s;
  }

  .stButton > button:hover {
      background: linear-gradient(135deg, #00e5ff44, #7b61ff44);
      border-color: #00e5ff;
      color: #fff;
  }

  /* Progress & sliders */
  .stSlider > div { color: #7b61ff; }

  /* Divider */
  hr { border-color: #2a2a38; margin: 1.5rem 0; }

  /* Upload area */
  .uploadedFile { background: #1a1a24 !important; border-color: #2e2e42 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: #141418;
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      color: #5a5a78;
  }
  .stTabs [aria-selected="true"] {
      background: #1f1f2c !important;
      color: #00e5ff !important;
  }

  /* Info boxes */
  .stInfo { background: #0a1628; border-color: #00e5ff44; }
  .stSuccess { background: #0a1f14; border-color: #00ff9444; }
  .stWarning { background: #1f1800; border-color: #ffb30044; }
</style>
""", unsafe_allow_html=True)


# Optic flow funkce (flow dense, flow sparse)

def compute_flow_dense(gray1, gray2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma):
    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=0
    )


def compute_flow_sparse(gray1, gray2, max_corners=500, quality=0.01,
                         min_dist=7, block_size=7, lk_winsize=21, lk_levels=3):

    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    pts1 = cv2.goodFeaturesToTrack(
        g1, maxCorners=max_corners, qualityLevel=quality,
        minDistance=min_dist, blockSize=block_size
    )

    if pts1 is None or len(pts1) == 0:
        h, w = gray1.shape
        return np.zeros((h, w, 2), dtype=np.float32), np.array([]), np.array([]), np.array([])

    pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        g1, g2, pts1,
        None,
        winSize=(lk_winsize, lk_winsize),
        maxLevel=lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    good = (status.ravel() == 1)
    p1_good = pts1[good].reshape(-1, 2)
    p2_good = pts2[good].reshape(-1, 2)

    # Vytvoření proudového pole rozptylem vektorů
    h, w = gray1.shape
    flow = np.zeros((h, w, 2), dtype=np.float32)
    for (x1, y1), (x2, y2) in zip(p1_good, p2_good):
        xi, yi = int(round(x1)), int(round(y1))
        if 0 <= xi < w and 0 <= yi < h:
            flow[yi, xi, 0] = x2 - x1
            flow[yi, xi, 1] = y2 - y1

    return flow, p1_good, p2_good, good


def make_panels(frame1_bgr, frame2_bgr, flow, motion_threshold_factor=2.0, arrow_step=25,
                method="dense", sparse_pts1=None, sparse_pts2=None):
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    med = np.median(magnitude)
    std = np.std(magnitude)
    motion_threshold = med + motion_threshold_factor * std
    motion_mask = magnitude > motion_threshold

    moving_mags = magnitude[motion_mask]
    typical_motion = np.percentile(moving_mags, 50) if len(moving_mags) > 0 else 1.0
    arrow_scale = 40.0 / max(typical_motion, 1.0)

    # Panel 1 – odriginální druhý obraz
    p1 = frame2_bgr.copy()

    # Panel 2 – HSV mapa toku
    hsv = np.zeros_like(frame1_bgr)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    p2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Panel 3 – překrytí pohybové masky
    p3 = frame2_bgr.copy()
    highlight = np.zeros_like(frame2_bgr)
    highlight[motion_mask] = (0, 0, 255)
    p3 = cv2.addWeighted(p3, 0.65, highlight, 0.35, 0)

    # Panel 4 – šipky
    p4 = frame2_bgr.copy()
    h, w = frame2_bgr.shape[:2]

    if method == "sparse" and sparse_pts1 is not None and len(sparse_pts1) > 0:
        # Nakreslení šipek na sledovaných bodech
        for (x1, y1), (x2, y2) in zip(sparse_pts1, sparse_pts2):
            fx, fy = x2 - x1, y2 - y1
            mag = np.hypot(fx, fy)
            speed_norm = min(mag / (motion_threshold * 2), 1.0)
            if speed_norm < 0.5:
                t = speed_norm * 2
                color = (int(255 * (1 - t)), 255, int(255 * t))
            else:
                t = (speed_norm - 0.5) * 2
                color = (0, int(255 * (1 - t)), 255)
            start = (int(round(x1)), int(round(y1)))
            end   = (int(round(x2)), int(round(y2)))
            cv2.arrowedLine(p4, start, end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)
            cv2.circle(p4, start, 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        # Šipky husté mřížky
        for y in range(0, h, arrow_step):
            for x in range(0, w, arrow_step):
                if motion_mask[y, x]:
                    fx, fy = flow[y, x]
                    end = (int(x + fx * arrow_scale), int(y + fy * arrow_scale))
                    speed_norm = min(magnitude[y, x] / (motion_threshold * 2), 1.0)
                    if speed_norm < 0.5:
                        t = speed_norm * 2
                        color = (int(255 * (1 - t)), 255, int(255 * t))
                    else:
                        t = (speed_norm - 0.5) * 2
                        color = (0, int(255 * (1 - t)), 255)
                    cv2.arrowedLine(p4, (x, y), end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    # Statisktiky
    moving_px = int(motion_mask.sum())
    total_px = motion_mask.size
    coverage = moving_px / total_px * 100
    avg_mag = float(moving_mags.mean()) if len(moving_mags) > 0 else 0.0
    max_mag = float(magnitude.max())

    stats = {
        "moving_pixels": moving_px,
        "coverage_pct": coverage,
        "avg_magnitude": avg_mag,
        "max_magnitude": max_mag,
        "motion_threshold": float(motion_threshold),
    }

    return p1, p2, p3, p4, stats


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def make_combined(panels, target_h=None):
    if target_h is None:
        target_h = min(p.shape[0] for p in panels)
    target_w = min(p.shape[1] for p in panels)
    resized = [cv2.resize(p, (target_w, target_h)) for p in panels]
    return np.hstack(resized)



# Hlavní text stránky

col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="app-header">OPTICAL FLOW</div>', unsafe_allow_html=True)
    method_label = "Farneback Dense Flow" if "is_dense" not in dir() or is_dense else "Lucas-Kanade Sparse Flow"
    st.markdown(f'<div class="app-subtitle">Motion Analysis Engine · {method_label}</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar

with st.sidebar:
    st.markdown("### Parametry")

    # Výběr metody
    st.markdown("**Metoda výpočtu**")
    flow_method = st.radio(
        "",
        [" Dense Flow (Farneback)", " Sparse Flow (Lucas-Kanade)"],
        key="flow_method",
        label_visibility="collapsed",
    )
    is_dense = flow_method.startswith("")

    st.markdown("---")

    if is_dense:
        st.markdown("**Farneback parametry**")
        pyr_scale   = st.slider("Pyramid scale",   0.1, 0.9, 0.5, 0.05)
        levels      = st.slider("Levels",          1, 10, 6)
        winsize     = st.slider("Window size",     5, 51, 25, 2)
        iterations  = st.slider("Iterations",      1, 10, 5)
        poly_n      = st.slider("Poly N",          5, 9,  7, 2)
        poly_sigma  = st.slider("Poly sigma",      1.0, 2.5, 1.5, 0.1)
        # LK defaults (not used)
        lk_max_corners = 500; lk_quality = 0.01; lk_min_dist = 7
        lk_block = 7; lk_winsize = 21; lk_levels = 3
    else:
        st.markdown("**Lucas-Kanade parametry**")
        lk_max_corners = st.slider("Max. rohů (features)", 50, 2000, 500, 50)
        lk_quality     = st.slider("Kvalita rohů",         0.001, 0.1, 0.01, 0.001, format="%.3f")
        lk_min_dist    = st.slider("Min. vzdálenost rohů", 3, 30, 7)
        lk_block       = st.slider("Block size",           3, 15, 7, 2)
        lk_winsize     = st.slider("LK window size",       5, 51, 21, 2)
        lk_levels      = st.slider("LK pyramid levels",    1, 6, 3)
        # Dense defaults (not used)
        pyr_scale = 0.5; levels = 6; winsize = 25
        iterations = 5; poly_n = 7; poly_sigma = 1.5

    st.markdown("---")
    st.markdown("**Detekce pohybu**")
    threshold_factor = st.slider("Motion threshold (σ násobek)", 0.5, 5.0, 2.0, 0.1)
    arrow_step       = st.slider("Hustota šipek (dense)",        10, 60, 25, 5,
                                  disabled=not is_dense)

    st.markdown("---")
    st.markdown("**Video nastavení**")
    video_step = st.slider("Analyzovat každý N-tý frame", 1, 10, 1)
    max_frames = st.slider("Max. počet framů", 10, 500, 100)

# Záložky - snímky / video

tab_img, tab_vid = st.tabs(["🖼️  Snímky", "🎬  Video"])

# Záložka 1: Snímky

with tab_img:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-title">📷 Snímek 1 (referenční)</div>', unsafe_allow_html=True)
        f1 = st.file_uploader("", type=["png","jpg","jpeg","bmp","tiff"], key="img1")
        if f1:
            img1_np = np.frombuffer(f1.read(), np.uint8)
            frame1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
            st.image(bgr_to_rgb(frame1), use_container_width=True)

    with c2:
        st.markdown('<div class="panel-title">📷 Snímek 2 (cílový)</div>', unsafe_allow_html=True)
        f2 = st.file_uploader("", type=["png","jpg","jpeg","bmp","tiff"], key="img2")
        if f2:
            img2_np = np.frombuffer(f2.read(), np.uint8)
            frame2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
            st.image(bgr_to_rgb(frame2), use_container_width=True)

    if f1 and f2:
        if st.button("Analyzovat pohyb", key="btn_img"):
            with st.spinner("Počítám optical flow…"):
                if frame1.shape != frame2.shape:
                    frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

                g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

                if is_dense:
                    flow = compute_flow_dense(g1, g2, pyr_scale, levels, winsize,
                                              iterations, poly_n, poly_sigma)
                    p1, p2, p3, p4, stats = make_panels(frame1, frame2, flow,
                                                         threshold_factor, arrow_step,
                                                         method="dense")
                else:
                    flow, pts1, pts2, _ = compute_flow_sparse(
                        g1, g2, lk_max_corners, lk_quality, lk_min_dist,
                        lk_block, lk_winsize, lk_levels)
                    p1, p2, p3, p4, stats = make_panels(frame1, frame2, flow,
                                                         threshold_factor, arrow_step,
                                                         method="sparse",
                                                         sparse_pts1=pts1,
                                                         sparse_pts2=pts2)

            # Statistiky
            st.markdown("---")
            st.markdown('<div class="panel-title">Statistiky pohybu</div>', unsafe_allow_html=True)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Pohybující se pixely", f"{stats['moving_pixels']:,}")
            mc2.metric("Pokrytí pohybem", f"{stats['coverage_pct']:.1f}%")
            mc3.metric("Průměrná magnituda", f"{stats['avg_magnitude']:.2f}")
            mc4.metric("Max. magnituda", f"{stats['max_magnitude']:.2f}")

            st.markdown("---")
            st.markdown('<div class="panel-title">Výsledky analýzy</div>', unsafe_allow_html=True)

            labels = ["Originál", "Flow mapa (HSV)", "Pohybující se oblasti", "Šipky pohybu"]
            panels_out = [p1, p2, p3, p4]

            row1, row2 = st.columns(2), st.columns(2)
            rows = [row1, row2]
            for i, (panel, label) in enumerate(zip(panels_out, labels)):
                with rows[i // 2][i % 2]:
                    st.markdown(f'<div class="panel-title">{label}</div>', unsafe_allow_html=True)
                    st.image(bgr_to_rgb(panel), use_container_width=True)

            # Stažení výsledků - ZIP
            combined = make_combined([p1, p2, p3, p4])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for panel, lbl in zip(panels_out, ["original","flow_mapa","pohyb_oblasti","sipky"]):
                    zf.writestr(f"{ts}_{lbl}.png", encode_png(panel))
                zf.writestr(f"{ts}_combined.png", encode_png(combined))
            zip_buf.seek(0)

            st.markdown("---")
            st.download_button(
                "Stáhnout výsledky (ZIP)",
                data=zip_buf,
                file_name=f"optical_flow_{ts}.zip",
                mime="application/zip",
            )

# Záložka 2: video

# Ukazatel progressu
if "vid_results" not in st.session_state:
    st.session_state.vid_results = None
if "vid_browser_idx" not in st.session_state:
    st.session_state.vid_browser_idx = 0

with tab_vid:
    st.markdown('<div class="panel-title">Nahraj video soubor</div>', unsafe_allow_html=True)
    vid_file = st.file_uploader("", type=["mp4","avi","mov","mkv","webm"], key="vid")

    if vid_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid_file.read())
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Celkem framů", total)
        vc2.metric("FPS", f"{fps:.1f}")
        vc3.metric("Rozlišení", f"{w_vid}×{h_vid}")
        vc4.metric("Délka", f"{total/fps:.1f}s")

        analyze_btn = st.button("Analyzovat video", key="btn_vid")

        if analyze_btn:
            cap = cv2.VideoCapture(tmp_path)
            placeholder = st.empty()
            prog_bar    = st.progress(0)
            status_txt  = st.empty()

            frames_to_process = min(max_frames, total // max(video_step, 1))
            results   = []
            prev_gray = None
            prev_bgr  = None
            fi = 0
            pi = 0

            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and (fi % video_step == 0):
                    if is_dense:
                        flow = compute_flow_dense(prev_gray, gray, pyr_scale, levels, winsize,
                                                   iterations, poly_n, poly_sigma)
                        p1, p2, p3, p4, stats = make_panels(prev_bgr, bgr, flow,
                                                              threshold_factor, arrow_step,
                                                              method="dense")
                    else:
                        flow, pts1, pts2, _ = compute_flow_sparse(
                            prev_gray, gray, lk_max_corners, lk_quality, lk_min_dist,
                            lk_block, lk_winsize, lk_levels)
                        p1, p2, p3, p4, stats = make_panels(prev_bgr, bgr, flow,
                                                              threshold_factor, arrow_step,
                                                              method="sparse",
                                                              sparse_pts1=pts1,
                                                              sparse_pts2=pts2)
                    combined = make_combined([p1, p2, p3, p4])
                    placeholder.image(bgr_to_rgb(combined), caption=f"Frame {fi}",
                                      use_container_width=True)
                    results.append({
                        "frame_idx": fi,
                        "time_s":    fi / fps,
                        "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                        "combined": combined,
                        "stats": stats,
                    })
                    pi += 1
                    prog_bar.progress(min(pi / frames_to_process, 1.0))
                    status_txt.markdown(
                        f'<div class="panel-title">Zpracováno: {pi}/{frames_to_process} '
                        f'segmentů · Frame {fi}/{total}</div>',
                        unsafe_allow_html=True,
                    )
                    if pi >= frames_to_process:
                        break
                prev_gray = gray
                prev_bgr  = bgr
                fi += 1

            cap.release()
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

            placeholder.empty()
            prog_bar.empty()
            status_txt.empty()

            st.session_state.vid_results = results
            st.session_state.vid_browser_idx = 0
            st.success(f"Zpracováno {len(results)} segmentů.")

    # Výsledky a jejich prohlížeč
    results = st.session_state.vid_results
    if results:
        import pandas as pd
        n = len(results)
        frame_indices = [r["frame_idx"] for r in results]
        all_coverage  = [r["stats"]["coverage_pct"]  for r in results]
        all_avg_mag   = [r["stats"]["avg_magnitude"]  for r in results]
        all_max_mag   = [r["stats"]["max_magnitude"]  for r in results]

        # Grafy
        st.markdown("---")
        st.markdown('<div class="panel-title">Časový průběh pohybu</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame({
            "Frame": frame_indices,
            "Pokrytí pohybem (%)": all_coverage,
            "Průměrná magnituda": all_avg_mag,
            "Max. magnituda": all_max_mag,
        })
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.line_chart(df.set_index("Frame")[["Pokrytí pohybem (%)"]])
        with chart_col2:
            st.line_chart(df.set_index("Frame")[["Průměrná magnituda", "Max. magnituda"]])
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Průměrné pokrytí",   f"{np.mean(all_coverage):.1f}%")
        sc2.metric("Průměrná magnituda",  f"{np.mean(all_avg_mag):.2f}")
        sc3.metric("Maximální magnituda", f"{np.max(all_max_mag):.2f}")

        # Prohlížeč snímků
        st.markdown("---")
        st.markdown('<div class="panel-title">🔍 Prohlížeč snímků</div>',
                    unsafe_allow_html=True)

        # Tlačítka pro pohyb ve snímcích
        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 4, 1, 1])
        with nav1:
            if st.button("⏮", help="První segment"):
                st.session_state.vid_browser_idx = 0
        with nav2:
            if st.button("◀", help="Předchozí"):
                st.session_state.vid_browser_idx = max(0, st.session_state.vid_browser_idx - 1)
        with nav4:
            if st.button("▶", help="Následující"):
                st.session_state.vid_browser_idx = min(n - 1, st.session_state.vid_browser_idx + 1)
        with nav5:
            if st.button("⏭", help="Poslední segment"):
                st.session_state.vid_browser_idx = n - 1

        # Posuvník
        slider_val = st.slider(
            "Přejít na segment",
            min_value=0, max_value=n - 1,
            value=st.session_state.vid_browser_idx,
            key="browser_slider",
        )
        if slider_val != st.session_state.vid_browser_idx:
            st.session_state.vid_browser_idx = slider_val

        cur = st.session_state.vid_browser_idx
        seg = results[cur]

        # Informační pruh
        st.markdown(
            f'<div class="panel-title">'
            f'Segment <span style="color:#00e5ff">{cur + 1}</span> / {n}'
            f'&nbsp;·&nbsp;Frame <span style="color:#00e5ff">{seg["frame_idx"]}</span>'
            f'&nbsp;·&nbsp;Čas <span style="color:#00e5ff">{seg["time_s"]:.2f}s</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Náhlednutí
        view_mode = st.radio(
            "Zobrazení",
            ["Kombinovaný pohled", "Originál", "Flow mapa",
             "Pohybující se oblasti", "Šipky pohybu"],
            horizontal=True,
            key="view_mode_radio",
        )
        panel_map = {
            "Kombinovaný pohled":    seg["combined"],
            "Originál":              seg["p1"],
            "Flow mapa":             seg["p2"],
            "Pohybující se oblasti": seg["p3"],
            "Šipky pohybu":          seg["p4"],
        }

        # Kombinovaný pohled
        if view_mode == "Kombinovaný pohled":
            g1, g2 = st.columns(2)
            g3, g4 = st.columns(2)
            for col, panel, lbl in [
                (g1, seg["p1"], "Originál"),
                (g2, seg["p2"], "Flow mapa"),
                (g3, seg["p3"], "Pohybující se oblasti"),
                (g4, seg["p4"], "Šipky pohybu"),
            ]:
                with col:
                    st.markdown(f'<div class="panel-title">{lbl}</div>', unsafe_allow_html=True)
                    st.image(bgr_to_rgb(panel), use_container_width=True)
        else:
            st.image(bgr_to_rgb(panel_map[view_mode]), use_container_width=True)

        # Statistiky za frame
        s = seg["stats"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Pohybující se pixely", f"{s['moving_pixels']:,}")
        fc2.metric("Pokrytí pohybem",      f"{s['coverage_pct']:.1f}%")
        fc3.metric("Průměrná magnituda",   f"{s['avg_magnitude']:.2f}")
        fc4.metric("Max. magnituda",       f"{s['max_magnitude']:.2f}")

        # Stahování
        st.markdown("---")
        dl1, dl2 = st.columns(2)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fi_cur = seg["frame_idx"]

        with dl1:
            chosen_panel = panel_map[view_mode] if view_mode != "Kombinovaný pohled" else seg["combined"]
            st.download_button(
                "⬇Stáhnout aktuální snímek (PNG)",
                data=encode_png(chosen_panel),
                file_name=f"frame_{fi_cur:05d}_{view_mode.replace(' ','_')}.png",
                mime="image/png",
                key="dl_single",
            )

        with dl2:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    fi_r = r["frame_idx"]
                    zf.writestr(f"frame_{fi_r:05d}_combined.png",  encode_png(r["combined"]))
                    zf.writestr(f"frame_{fi_r:05d}_original.png",  encode_png(r["p1"]))
                    zf.writestr(f"frame_{fi_r:05d}_flow.png",      encode_png(r["p2"]))
                    zf.writestr(f"frame_{fi_r:05d}_pohyb.png",     encode_png(r["p3"]))
                    zf.writestr(f"frame_{fi_r:05d}_sipky.png",     encode_png(r["p4"]))
            zip_buf.seek(0)
            st.download_button(
                "⬇Stáhnout všechny výsledky (ZIP)",
                data=zip_buf,
                file_name=f"video_flow_{ts}.zip",
                mime="application/zip",
                key="dl_all",
            )