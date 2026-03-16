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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optical Flow Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .stApp { background: #0e0e12; color: #e2e2e8; }
  section[data-testid="stSidebar"] { background: #141418 !important; border-right: 1px solid #2a2a35; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  .app-header {
      font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700;
      letter-spacing: -1px;
      background: linear-gradient(135deg, #00e5ff 0%, #7b61ff 60%, #ff4fcf 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0;
  }
  .app-subtitle {
      font-size: 0.85rem; color: #5a5a72; letter-spacing: 3px; text-transform: uppercase;
      font-family: 'Space Mono', monospace; margin-top: 0.2rem;
  }
  .panel-title {
      font-family: 'Space Mono', monospace; font-size: 0.72rem; letter-spacing: 2px;
      text-transform: uppercase; color: #5a5a78; margin-bottom: 0.6rem;
  }
  .stButton > button {
      background: linear-gradient(135deg, #00e5ff22, #7b61ff22);
      border: 1px solid #7b61ff88; color: #c4b5fd;
      font-family: 'Space Mono', monospace; font-size: 0.8rem;
      letter-spacing: 1px; border-radius: 8px; padding: 0.5rem 1.2rem; transition: all 0.2s;
  }
  .stButton > button:hover { background: linear-gradient(135deg, #00e5ff44, #7b61ff44); border-color: #00e5ff; color: #fff; }
  hr { border-color: #2a2a38; margin: 1.5rem 0; }
  .stTabs [data-baseweb="tab-list"] { background: #141418; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #5a5a78; }
  .stTabs [aria-selected="true"] { background: #1f1f2c !important; color: #00e5ff !important; }
  .stInfo { background: #0a1628; border-color: #00e5ff44; }
  .stSuccess { background: #0a1f14; border-color: #00ff9444; }
  .scene-badge {
      display: inline-block; background: #ff4fcf33; border: 1px solid #ff4fcf88;
      color: #ff4fcf; font-family: 'Space Mono', monospace; font-size: 0.65rem;
      letter-spacing: 1px; padding: 2px 8px; border-radius: 4px; margin-left: 8px;
  }
  .stat-delta-up   { color: #ff4fcf; font-size: 0.7rem; }
  .stat-delta-down { color: #00e5a0; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("vid_results",       None)
_ss("vid_tmp_path",      None)
_ss("vid_file_id",       None)
_ss("vid_analyzing",     False)
_ss("vid_slider",        0)
_ss("selected_frames",   set())
_ss("img_frame1",        None)
_ss("img_frame2",        None)
_ss("img_file1_id",      None)
_ss("img_file2_id",      None)
_ss("heatmap_cache",     None)    # kumulativní heatmapa
_ss("scene_cuts",        [])      # indexy detekovaných střihů


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_dense(gray1, gray2, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma):
    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    return cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=pyr_scale, levels=levels, winsize=winsize,
        iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma, flags=0
    )


def compute_flow_sparse(gray1, gray2, max_corners, quality, min_dist, block_size, lk_winsize, lk_levels):
    g1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    g2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    pts1 = cv2.goodFeaturesToTrack(g1, maxCorners=max_corners, qualityLevel=quality,
                                    minDistance=min_dist, blockSize=block_size)
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


def make_panels(frame1_bgr, frame2_bgr, flow, threshold_factor=2.0, arrow_step=25,
                method="dense", sparse_pts1=None, sparse_pts2=None):
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
            color = (int(255*(1-t)), 255, int(255*t)) if speed_norm < 0.5 else (0, int(255*(1-t)), 255)
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
                    color = (int(255*(1-t)), 255, int(255*t)) if speed_norm < 0.5 else (0, int(255*(1-t)), 255)
                    cv2.arrowedLine(p4, (x, y), end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    stats = {
        "moving_pixels": int(motion_mask.sum()),
        "coverage_pct":  float(motion_mask.sum() / motion_mask.size * 100),
        "avg_magnitude": float(moving_mags.mean()) if len(moving_mags) > 0 else 0.0,
        "max_magnitude": float(magnitude.max()),
    }
    return p1, p2, p3, p4, stats, magnitude   # ← magnitude navíc pro heatmapu


def build_heatmap(results, ref_shape):
    """Kumulativní heatmapa pohybu přes všechny zpracované framy."""
    h, w = ref_shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)
    for r in results:
        mag = r.get("magnitude")
        if mag is None:
            continue
        resized = cv2.resize(mag, (w, h))
        accum += resized
    accum_norm = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(accum_norm, cv2.COLORMAP_INFERNO)
    return heatmap_bgr


def detect_scene_cuts(results, jump_factor=3.0):
    """Detekuje náhlé skoky v průměrné magnitudě — potenciální střihy."""
    if len(results) < 3:
        return []
    mags = np.array([r["stats"]["avg_magnitude"] for r in results])
    diffs = np.abs(np.diff(mags))
    threshold = np.median(diffs) + jump_factor * np.std(diffs)
    cuts = [i + 1 for i, d in enumerate(diffs) if d > threshold]
    return cuts


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def encode_jpg(img_bgr, quality=85):
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def make_combined(panels):
    """Složí 4 panely do 2×2 gridu (šetří paměť vs hstack celého videa)."""
    h = min(p.shape[0] for p in panels)
    w = min(p.shape[1] for p in panels)
    resized = [cv2.resize(p, (w, h)) for p in panels]
    top    = np.hstack(resized[:2])
    bottom = np.hstack(resized[2:])
    return np.vstack([top, bottom])


def results_to_dataframe(results):
    rows = []
    for i, r in enumerate(results):
        s = r["stats"]
        rows.append({
            "segment":          i + 1,
            "frame":            r["frame_idx"],
            "cas_s":            round(r["time_s"], 3),
            "pohybujici_pixely": s["moving_pixels"],
            "pokryti_pct":      round(s["coverage_pct"], 2),
            "avg_magnituda":    round(s["avg_magnitude"], 4),
            "max_magnituda":    round(s["max_magnitude"], 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
_, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="app-header">OPTICAL FLOW</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Motion Analysis Engine · Dense &amp; Sparse Flow · v2.0</div>',
                unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry")

    st.markdown("**Metoda výpočtu**")
    flow_method = st.radio(
        "flow_method_radio",
        ["🌊  Dense Flow (Farneback)", "✨  Sparse Flow (Lucas-Kanade)"],
        label_visibility="collapsed",
    )
    is_dense = flow_method.startswith("🌊")

    st.markdown("---")

    if is_dense:
        st.markdown("**Farneback parametry**")
        pyr_scale  = st.slider("Pyramid scale",  0.1, 0.9, 0.5, 0.05)
        levels     = st.slider("Levels",         1, 10, 6)
        winsize    = st.slider("Window size",    5, 51, 25, 2)
        iterations = st.slider("Iterations",     1, 10, 5)
        poly_n     = st.slider("Poly N",         5, 9,  7, 2)
        poly_sigma = st.slider("Poly sigma",     1.0, 2.5, 1.5, 0.1)
        lk_max_corners = 500; lk_quality = 0.01; lk_min_dist = 7
        lk_block = 7; lk_winsize = 21; lk_levels = 3
    else:
        st.markdown("**Lucas-Kanade parametry**")
        lk_max_corners = st.slider("Max. rohů",             50, 2000, 500, 50)
        lk_quality     = st.slider("Kvalita rohů",          0.001, 0.1, 0.01, 0.001, format="%.3f")
        lk_min_dist    = st.slider("Min. vzdálenost rohů",  3, 30, 7)
        lk_block       = st.slider("Block size",            3, 15, 7, 2)
        lk_winsize     = st.slider("LK window size",        5, 51, 21, 2)
        lk_levels      = st.slider("LK pyramid levels",     1, 6, 3)
        pyr_scale = 0.5; levels = 6; winsize = 25
        iterations = 5; poly_n = 7; poly_sigma = 1.5

    st.markdown("---")
    st.markdown("**Detekce pohybu**")
    threshold_factor = st.slider("Motion threshold (σ násobek)", 0.5, 5.0, 2.0, 0.1)
    arrow_step       = st.slider("Hustota šipek (dense)", 10, 60, 25, 5, disabled=not is_dense)

    st.markdown("---")
    st.markdown("**Video nastavení**")
    video_step = st.slider("Analyzovat každý N-tý frame", 1, 10, 1)
    max_frames = st.slider("Max. počet framů",            10, 500, 100)

    st.markdown("---")
    st.markdown("**Pokročilé funkce**")
    enable_heatmap     = st.toggle("🌡️  Kumulativní heatmapa",    value=True)
    enable_scene_det   = st.toggle("🎬  Detekce střihů scén",     value=True)
    scene_jump_factor  = st.slider("Citlivost detekce střihů", 1.0, 6.0, 3.0, 0.5,
                                    disabled=not enable_scene_det)

    st.markdown("---")
    # ── RAM odhad ─────────────────────────────────────────────────────────────
    st.markdown("**ℹ️ Odhad paměti**")
    est_mb = max_frames * 4 * (1920 * 1080 * 3) / 1e6   # worst-case Full HD
    est_mb_real = max_frames * 4 * (640 * 360 * 3) / 1e6  # typické
    st.caption(f"Při 640×360: ~{est_mb_real:.0f} MB  |  FullHD: ~{est_mb:.0f} MB")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["🖼️  OBRAZY", "🎬  VIDEO"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – IMAGES
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel-title">📷 Snímek 1 (referenční)</div>', unsafe_allow_html=True)
        f1 = st.file_uploader("Snímek 1", type=["png","jpg","jpeg","bmp","tiff"],
                               key="img1", label_visibility="collapsed")
        # BUG FIX: načítej snímek jen při nové nahrávce, ne při každém renderu
        if f1 is not None and f1.file_id != st.session_state.img_file1_id:
            img1_np = np.frombuffer(f1.read(), np.uint8)
            st.session_state.img_frame1   = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
            st.session_state.img_file1_id = f1.file_id
        if st.session_state.img_frame1 is not None:
            st.image(bgr_to_rgb(st.session_state.img_frame1), use_container_width=True)

    with c2:
        st.markdown('<div class="panel-title">📷 Snímek 2 (cílový)</div>', unsafe_allow_html=True)
        f2 = st.file_uploader("Snímek 2", type=["png","jpg","jpeg","bmp","tiff"],
                               key="img2", label_visibility="collapsed")
        if f2 is not None and f2.file_id != st.session_state.img_file2_id:
            img2_np = np.frombuffer(f2.read(), np.uint8)
            st.session_state.img_frame2   = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
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
            flow = compute_flow_dense(g1, g2, pyr_scale, levels, winsize,
                                      iterations, poly_n, poly_sigma)
            p1, p2, p3, p4, stats, magnitude = make_panels(frame1, frame2, flow,
                                                 threshold_factor, arrow_step, "dense")
        else:
            flow, pts1, pts2 = compute_flow_sparse(g1, g2, lk_max_corners, lk_quality,
                                                    lk_min_dist, lk_block, lk_winsize, lk_levels)
            p1, p2, p3, p4, stats, magnitude = make_panels(frame1, frame2, flow,
                                                 threshold_factor, arrow_step, "sparse",
                                                 sparse_pts1=pts1, sparse_pts2=pts2)
        elapsed = time.perf_counter() - t0

        st.markdown("---")
        st.markdown(
            f'<div class="panel-title">📊 Statistiky pohybu '
            f'<span style="color:#5a5a78">· výpočet: {elapsed*1000:.0f} ms</span></div>',
            unsafe_allow_html=True,
        )
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Pohybující se pixely", f"{stats['moving_pixels']:,}")
        mc2.metric("Pokrytí pohybem",      f"{stats['coverage_pct']:.1f}%")
        mc3.metric("Průměrná magnituda",   f"{stats['avg_magnitude']:.2f}")
        mc4.metric("Max. magnituda",        f"{stats['max_magnitude']:.2f}")

        # ── Výsledkové panely ─────────────────────────────────────────────────
        st.markdown("---")
        panels_list  = [p1, p2, p3, p4]
        panels_label = ["Originál", "Flow mapa (HSV)", "Pohybující se oblasti", "Šipky pohybu"]

        if enable_heatmap:
            hm_single = build_heatmap([{"magnitude": magnitude}], frame1.shape)
            panels_list.append(hm_single)
            panels_label.append("Heatmapa pohybu")

        st.markdown('<div class="panel-title">🔬 Výsledky analýzy</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(panels_list), 4))
        for i, (panel, label) in enumerate(zip(panels_list, panels_label)):
            with cols[i % 4]:
                st.markdown(f'<div class="panel-title">{label}</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb(panel), use_container_width=True)
            if (i + 1) % 4 == 0 and i + 1 < len(panels_list):
                cols = st.columns(min(len(panels_list) - i - 1, 4))

        # ── Export ────────────────────────────────────────────────────────────
        combined = make_combined([p1, p2, p3, p4])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for panel, lbl in zip([p1, p2, p3, p4],
                                   ["original", "flow_mapa", "pohyb_oblasti", "sipky"]):
                zf.writestr(f"{ts}_{lbl}.png", encode_png(panel))
            if enable_heatmap:
                zf.writestr(f"{ts}_heatmapa.png", encode_png(hm_single))
            zf.writestr(f"{ts}_combined.png", encode_png(combined))
        zip_buf.seek(0)
        st.markdown("---")
        st.download_button("⬇️  Stáhnout výsledky (ZIP)", data=zip_buf,
                            file_name=f"optical_flow_{ts}.zip", mime="application/zip")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    st.markdown('<div class="panel-title">🎬 Nahraj video soubor</div>', unsafe_allow_html=True)
    vid_file = st.file_uploader("Video soubor", type=["mp4","avi","mov","mkv","webm"],
                                 key="vid", label_visibility="collapsed")

    if vid_file is not None:
        file_id = vid_file.file_id
        if file_id != st.session_state.vid_file_id:
            st.session_state.vid_file_id     = file_id
            st.session_state.vid_results     = None
            st.session_state.vid_slider      = 0
            st.session_state.vid_analyzing   = False
            st.session_state.selected_frames = set()
            st.session_state.heatmap_cache   = None
            st.session_state.scene_cuts      = []
            # Smazat starý temp soubor
            old = st.session_state.vid_tmp_path
            if old and os.path.exists(old):
                try: os.unlink(old)
                except: pass
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vid_file.read())
            tmp.flush(); tmp.close()
            st.session_state.vid_tmp_path = tmp.name

    if vid_file is not None and st.session_state.vid_tmp_path:
        tmp_path = st.session_state.vid_tmp_path
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

        # ── Výběr rozsahu ─────────────────────────────────────────────────────
        st.markdown('<div class="panel-title">🎯 Rozsah analýzy</div>', unsafe_allow_html=True)
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_frame = st.number_input(
                "Od framu", min_value=0, max_value=total - 2,
                value=0, step=1, key="range_start",
            )
        with range_col2:
            end_frame = st.number_input(
                "Do framu", min_value=int(start_frame) + 1, max_value=total - 1,
                value=min(total - 1, int(start_frame) + max_frames * video_step + 1),
                step=1, key="range_end",
            )

        start_frame  = int(start_frame)
        end_frame    = int(end_frame)
        range_frames = end_frame - start_frame
        est_segments = min(max_frames, max(range_frames // max(video_step, 1), 1))

        st.markdown(
            f'<div class="panel-title">'
            f'Rozsah: <span style="color:#00e5ff">{start_frame}</span> – '
            f'<span style="color:#00e5ff">{end_frame}</span>'
            f'&nbsp;·&nbsp; {range_frames} framů'
            f'&nbsp;·&nbsp; {start_frame/fps:.1f}s – {end_frame/fps:.1f}s'
            f'&nbsp;·&nbsp; odhadované segmenty: <span style="color:#00e5ff">{est_segments}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        frac_start = start_frame / max(total - 1, 1)
        frac_end   = end_frame   / max(total - 1, 1)
        st.markdown(
            f'<div style="background:#1a1a24;border-radius:6px;height:10px;position:relative;margin:6px 0 14px">'
            f'<div style="position:absolute;left:{frac_start*100:.1f}%;width:{(frac_end-frac_start)*100:.1f}%;'
            f'height:100%;background:linear-gradient(90deg,#7b61ff,#00e5ff);border-radius:6px"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        analyze_btn = st.button("🔍  ANALYZOVAT VIDEO", key="btn_vid",
                                 disabled=st.session_state.vid_analyzing)

        if analyze_btn:
            st.session_state.vid_analyzing   = True
            st.session_state.vid_results     = None
            st.session_state.vid_slider      = 0
            st.session_state.selected_frames = set()
            st.session_state.heatmap_cache   = None
            st.session_state.scene_cuts      = []

            cap = cv2.VideoCapture(tmp_path)
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            preview_ph = st.empty()
            prog_bar   = st.progress(0)
            status_ph  = st.empty()
            time_ph    = st.empty()

            results   = []
            prev_gray = None
            prev_bgr  = None
            fi  = start_frame
            pi  = 0
            t_start = time.perf_counter()

            try:
                while True:
                    ret, bgr = cap.read()
                    if not ret or fi > end_frame:
                        break
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

                    if prev_gray is not None and ((fi - start_frame) % video_step == 0):
                        if is_dense:
                            flow = compute_flow_dense(prev_gray, gray, pyr_scale, levels, winsize,
                                                       iterations, poly_n, poly_sigma)
                            p1, p2, p3, p4, stats, mag = make_panels(prev_bgr, bgr, flow,
                                                                       threshold_factor, arrow_step, "dense")
                        else:
                            flow, pts1, pts2 = compute_flow_sparse(
                                prev_gray, gray, lk_max_corners, lk_quality, lk_min_dist,
                                lk_block, lk_winsize, lk_levels)
                            p1, p2, p3, p4, stats, mag = make_panels(prev_bgr, bgr, flow,
                                                                       threshold_factor, arrow_step, "sparse",
                                                                       sparse_pts1=pts1, sparse_pts2=pts2)

                        combined = make_combined([p1, p2, p3, p4])
                        # BUG FIX: use_container_width místo width="stretch"
                        preview_ph.image(bgr_to_rgb(combined), caption=f"Frame {fi}",
                                         use_container_width=True)
                        results.append({
                            "frame_idx": fi,
                            "time_s":    fi / fps,
                            "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                            "combined": combined,
                            "stats":    stats,
                            "magnitude": mag,   # pro heatmapu
                        })
                        pi += 1

                        # ── Odhadovaný zbývající čas ───────────────────────
                        elapsed = time.perf_counter() - t_start
                        fps_proc = pi / max(elapsed, 1e-6)
                        remaining = (est_segments - pi) / max(fps_proc, 1e-6)

                        prog_bar.progress(min(pi / est_segments, 1.0))
                        status_ph.markdown(
                            f'<div class="panel-title">Zpracováno: {pi}/{est_segments} '
                            f'segmentů · Frame {fi}/{end_frame}</div>',
                            unsafe_allow_html=True,
                        )
                        time_ph.markdown(
                            f'<div class="panel-title">⏱ Uplynulo: {elapsed:.1f}s '
                            f'· Zbývá: ~{remaining:.0f}s '
                            f'· {fps_proc:.1f} segmentů/s</div>',
                            unsafe_allow_html=True,
                        )
                        if pi >= est_segments:
                            break

                    prev_gray = gray
                    prev_bgr  = bgr
                    fi += 1

            except BaseException:
                cap.release()
                raise
            finally:
                cap.release()

            preview_ph.empty()
            prog_bar.empty()
            time_ph.empty()

            st.session_state.vid_results   = results
            st.session_state.vid_analyzing = False

            # Post-processing
            if results:
                if enable_heatmap:
                    st.session_state.heatmap_cache = build_heatmap(results, results[0]["p1"].shape)
                if enable_scene_det:
                    st.session_state.scene_cuts = detect_scene_cuts(results, scene_jump_factor)
                total_time = time.perf_counter() - t_start
                status_ph.success(
                    f"✅ Hotovo! Zpracováno {len(results)} segmentů za {total_time:.1f}s"
                    + (f" · Detekováno {len(st.session_state.scene_cuts)} střihů" if enable_scene_det else "")
                )
            st.rerun()

    # ── Results browser ────────────────────────────────────────────────────────
    results    = st.session_state.vid_results
    scene_cuts = st.session_state.scene_cuts
    heatmap    = st.session_state.heatmap_cache

    if results and not st.session_state.vid_analyzing:
        n = len(results)
        frame_indices = [r["frame_idx"] for r in results]
        all_coverage  = [r["stats"]["coverage_pct"]  for r in results]
        all_avg_mag   = [r["stats"]["avg_magnitude"]  for r in results]
        all_max_mag   = [r["stats"]["max_magnitude"]  for r in results]

        # ── Grafy ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="panel-title">📈 Časový průběh pohybu</div>', unsafe_allow_html=True)
        df = pd.DataFrame({
            "Frame":               frame_indices,
            "Pokrytí pohybem (%)": all_coverage,
            "Průměrná magnituda":  all_avg_mag,
            "Max. magnituda":      all_max_mag,
        })
        ch1, ch2 = st.columns(2)
        with ch1:
            st.line_chart(df.set_index("Frame")[["Pokrytí pohybem (%)"]])
        with ch2:
            st.line_chart(df.set_index("Frame")[["Průměrná magnituda", "Max. magnituda"]])

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Průměrné pokrytí",    f"{np.mean(all_coverage):.1f}%")
        sc2.metric("Průměrná magnituda",   f"{np.mean(all_avg_mag):.2f}")
        sc3.metric("Maximální magnituda",  f"{np.max(all_max_mag):.2f}")
        sc4.metric("Detekované střihy",    len(scene_cuts))

        # ── Export CSV ────────────────────────────────────────────────────────
        csv_df  = results_to_dataframe(results)
        csv_buf = csv_df.to_csv(index=False).encode("utf-8")
        ts_now  = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "📊  Stáhnout statistiky (CSV)",
            data=csv_buf,
            file_name=f"stats_{ts_now}.csv",
            mime="text/csv",
            key="dl_csv",
        )

        # ── Heatmapa ──────────────────────────────────────────────────────────
        if heatmap is not None:
            st.markdown("---")
            st.markdown('<div class="panel-title">🌡️ Kumulativní heatmapa pohybu</div>', unsafe_allow_html=True)
            hm_col1, hm_col2 = st.columns([3, 1])
            with hm_col1:
                st.image(bgr_to_rgb(heatmap), use_container_width=True,
                         caption="Světlejší = více pohybu, tmavší = méně pohybu")
            with hm_col2:
                st.download_button(
                    "⬇️  Heatmapa (PNG)",
                    data=encode_png(heatmap),
                    file_name=f"heatmapa_{ts_now}.png",
                    mime="image/png",
                    key="dl_heatmap",
                )

        # ── Detekované střihy ─────────────────────────────────────────────────
        if scene_cuts:
            st.markdown("---")
            st.markdown(
                f'<div class="panel-title">🎬 Detekované střihy scén '
                f'<span style="color:#ff4fcf">({len(scene_cuts)})</span></div>',
                unsafe_allow_html=True,
            )
            cut_cols = st.columns(min(len(scene_cuts), 6))
            for col, ci in zip(cut_cols, scene_cuts[:6]):
                r = results[ci]
                with col:
                    st.image(bgr_to_rgb(r["p1"]), use_container_width=True)
                    st.markdown(
                        f'<div class="panel-title" style="text-align:center">'
                        f'F{r["frame_idx"]} · {r["time_s"]:.1f}s</div>',
                        unsafe_allow_html=True,
                    )

        # ── Frame browser ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="panel-title">🔍 Prohlížeč snímků</div>', unsafe_allow_html=True)

        _ss("vid_slider", 0)
        st.session_state.vid_slider = int(np.clip(st.session_state.vid_slider, 0, n - 1))

        def _go(delta=None, absolute=None):
            cur_val = st.session_state.vid_slider
            if absolute is not None:
                st.session_state.vid_slider = absolute
            else:
                st.session_state.vid_slider = int(np.clip(cur_val + delta, 0, n - 1))

        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 5, 1, 1])
        with nav1:
            st.button("⏮", key="nav_first", on_click=_go, kwargs={"absolute": 0},     help="První")
        with nav2:
            st.button("◀",  key="nav_prev",  on_click=_go, kwargs={"delta": -1},       help="Předchozí")
        with nav3:
            st.slider("Segment", 0, n - 1, key="vid_slider", label_visibility="collapsed")
        with nav4:
            st.button("▶",  key="nav_next",  on_click=_go, kwargs={"delta": 1},        help="Následující")
        with nav5:
            st.button("⏭", key="nav_last",  on_click=_go, kwargs={"absolute": n - 1}, help="Poslední")

        cur = st.session_state.vid_slider
        seg = results[cur]

        # Indikátor střihu
        is_cut = cur in scene_cuts
        cut_badge = '<span class="scene-badge">✂️ STŘIH</span>' if is_cut else ""

        # Info bar + výběr
        info_col, sel_col = st.columns([5, 2])
        with info_col:
            st.markdown(
                f'<div class="panel-title">'
                f'Segment <span style="color:#00e5ff">{cur + 1} / {n}</span>'
                f'&nbsp;·&nbsp; Frame <span style="color:#00e5ff">{seg["frame_idx"]}</span>'
                f'&nbsp;·&nbsp; Čas <span style="color:#00e5ff">{seg["time_s"]:.2f}s</span>'
                f'{cut_badge}</div>',
                unsafe_allow_html=True,
            )
        with sel_col:
            is_selected = cur in st.session_state.selected_frames
            label = "✅  Vybráno" if is_selected else "☐  Vybrat tento snímek"
            # BUG FIX: closure přes kwargs, ne default argument
            def _toggle_frame(idx):
                if idx in st.session_state.selected_frames:
                    st.session_state.selected_frames.discard(idx)
                else:
                    st.session_state.selected_frames.add(idx)
            st.button(label, key=f"sel_btn_{cur}", on_click=_toggle_frame, kwargs={"idx": cur})

        # Metriky aktuálního snímku s delta oproti předchozímu
        s = seg["stats"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        if cur > 0:
            prev_s = results[cur - 1]["stats"]
            fc1.metric("Pohybující se pixely", f"{s['moving_pixels']:,}",
                       delta=f"{s['moving_pixels'] - prev_s['moving_pixels']:+,}")
            fc2.metric("Pokrytí pohybem",      f"{s['coverage_pct']:.1f}%",
                       delta=f"{s['coverage_pct'] - prev_s['coverage_pct']:+.1f}%")
            fc3.metric("Průměrná magnituda",   f"{s['avg_magnitude']:.2f}",
                       delta=f"{s['avg_magnitude'] - prev_s['avg_magnitude']:+.2f}")
            fc4.metric("Max. magnituda",        f"{s['max_magnitude']:.2f}",
                       delta=f"{s['max_magnitude'] - prev_s['max_magnitude']:+.2f}")
        else:
            fc1.metric("Pohybující se pixely", f"{s['moving_pixels']:,}")
            fc2.metric("Pokrytí pohybem",      f"{s['coverage_pct']:.1f}%")
            fc3.metric("Průměrná magnituda",   f"{s['avg_magnitude']:.2f}")
            fc4.metric("Max. magnituda",        f"{s['max_magnitude']:.2f}")

        # View mode
        view_mode = st.radio(
            "Zobrazení", key="vid_view_mode",
            options=["Kombinovaný pohled", "Originál", "Flow mapa",
                     "Pohybující se oblasti", "Šipky pohybu"],
            horizontal=True,
            label_visibility="collapsed",
        )
        panel_map = {
            "Kombinovaný pohled":    seg["combined"],
            "Originál":              seg["p1"],
            "Flow mapa":             seg["p2"],
            "Pohybující se oblasti": seg["p3"],
            "Šipky pohybu":          seg["p4"],
        }

        if view_mode == "Kombinovaný pohled":
            g1c, g2c = st.columns(2)
            g3c, g4c = st.columns(2)
            for col, panel, lbl in [
                (g1c, seg["p1"], "Originál"),
                (g2c, seg["p2"], "Flow mapa"),
                (g3c, seg["p3"], "Pohybující se oblasti"),
                (g4c, seg["p4"], "Šipky pohybu"),
            ]:
                with col:
                    st.markdown(f'<div class="panel-title">{lbl}</div>', unsafe_allow_html=True)
                    # BUG FIX: use_container_width místo width="stretch"
                    st.image(bgr_to_rgb(panel), use_container_width=True)
        else:
            st.image(bgr_to_rgb(panel_map[view_mode]), use_container_width=True)

        # ── Vybrané snímky ────────────────────────────────────────────────────
        st.markdown("---")
        sel = sorted(st.session_state.selected_frames)

        if sel:
            st.markdown(
                f'<div class="panel-title">🗂️ Vybrané snímky '
                f'<span style="color:#00e5ff">({len(sel)})</span></div>',
                unsafe_allow_html=True,
            )
            cols_per_row = 5
            rows_sel = [sel[i:i+cols_per_row] for i in range(0, len(sel), cols_per_row)]
            for row in rows_sel:
                thumb_cols = st.columns(cols_per_row)
                for col, idx in zip(thumb_cols, row):
                    r = results[idx]
                    with col:
                        st.image(bgr_to_rgb(r["p1"]), use_container_width=True)
                        fi_r = r["frame_idx"]
                        t_r  = r["time_s"]
                        st.markdown(
                            f'<div class="panel-title" style="text-align:center">'
                            f'F{fi_r} · {t_r:.1f}s</div>',
                            unsafe_allow_html=True,
                        )
                        # BUG FIX: kwargs místo default arg closure
                        def _remove(i):
                            st.session_state.selected_frames.discard(i)
                        st.button("✕", key=f"rm_{idx}", on_click=_remove, kwargs={"i": idx},
                                   help="Odebrat z výběru")

            act1, act2, act3 = st.columns(3)
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")

            with act1:
                zip_sel = io.BytesIO()
                with zipfile.ZipFile(zip_sel, "w", zipfile.ZIP_DEFLATED) as zf:
                    for idx in sel:
                        r    = results[idx]
                        fi_r = r["frame_idx"]
                        zf.writestr(f"frame_{fi_r:05d}_combined.png", encode_png(r["combined"]))
                        zf.writestr(f"frame_{fi_r:05d}_original.png", encode_png(r["p1"]))
                        zf.writestr(f"frame_{fi_r:05d}_flow.png",     encode_png(r["p2"]))
                        zf.writestr(f"frame_{fi_r:05d}_pohyb.png",    encode_png(r["p3"]))
                        zf.writestr(f"frame_{fi_r:05d}_sipky.png",    encode_png(r["p4"]))
                zip_sel.seek(0)
                st.download_button(
                    f"⬇️  Stáhnout vybrané ({len(sel)}) ZIP",
                    data=zip_sel,
                    file_name=f"vybrane_snimky_{ts_now}.zip",
                    mime="application/zip",
                    key="dl_selected",
                )
            with act2:
                if st.button("🗑️  Zrušit výběr", key="clear_sel"):
                    st.session_state.selected_frames.clear()
            with act3:
                if st.button("✅  Vybrat vše", key="sel_all"):
                    for i in range(n):
                        st.session_state.selected_frames.add(i)
        else:
            st.markdown(
                '<div class="panel-title" style="color:#3a3a52">Žádné snímky nejsou vybrány — '
                'použij tlačítko ☐ Vybrat tento snímek</div>',
                unsafe_allow_html=True,
            )

        # ── Stažení všech ─────────────────────────────────────────────────────
        st.markdown("---")
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dl1, dl2 = st.columns(2)
        with dl1:
            chosen = panel_map[view_mode] if view_mode != "Kombinovaný pohled" else seg["combined"]
            st.download_button(
                "⬇️  Stáhnout aktuální snímek (PNG)",
                data=encode_png(chosen),
                file_name=f"frame_{seg['frame_idx']:05d}_{view_mode.replace(' ','_')}.png",
                mime="image/png",
                key="dl_single",
            )
        with dl2:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    fi_r = r["frame_idx"]
                    zf.writestr(f"frame_{fi_r:05d}_combined.png", encode_png(r["combined"]))
                    zf.writestr(f"frame_{fi_r:05d}_original.png", encode_png(r["p1"]))
                    zf.writestr(f"frame_{fi_r:05d}_flow.png",     encode_png(r["p2"]))
                    zf.writestr(f"frame_{fi_r:05d}_pohyb.png",    encode_png(r["p3"]))
                    zf.writestr(f"frame_{fi_r:05d}_sipky.png",    encode_png(r["p4"]))
                if heatmap is not None:
                    zf.writestr(f"heatmapa_kumulativni.png", encode_png(heatmap))
            zip_buf.seek(0)
            st.download_button(
                "⬇️  Stáhnout všechny výsledky (ZIP)",
                data=zip_buf,
                file_name=f"video_flow_{ts_now}.zip",
                mime="application/zip",
                key="dl_all",
            )