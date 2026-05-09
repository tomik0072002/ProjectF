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
    page_title="Zpracování obrazu - Optický tok",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inicializace stavu
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


_ss("vid_results", None)
_ss("vid_tmp_path", None)
_ss("vid_file_id", None)
_ss("vid_analyzing", False)
_ss("vid_slider", 0)
_ss("selected_frames", set())
_ss("img_frame1", None)
_ss("img_frame2", None)
_ss("img_file1_id", None)
_ss("img_file2_id", None)


# Hlavní funkce
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
    """Složí 4 panely do 2×2 gridu (šetří paměť vs hstack celého videa)."""
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
            "frame": r["frame_idx"],
            "cas_s": round(r["time_s"], 3),
            "pohybujici_pixely": s["moving_pixels"],
            "pokryti_pct": round(s["coverage_pct"], 2),
            "avg_magnituda": round(s["avg_magnitude"], 4),
            "max_magnituda": round(s["max_magnitude"], 4),
        })
    return pd.DataFrame(rows)


# Boční panel - sidebar
with st.sidebar:
    st.markdown("## Detekce pohybu")
    st.markdown("Analýza optického toku")
    st.markdown("---")

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
            st.caption("Parametry algoritmu Gunner Farneback")
            pyr_scale = st.slider("Pyramid scale", 0.1, 0.9, 0.5, 0.05)
            levels = st.slider("Levels", 1, 10, 6)
            winsize = st.slider("Window size", 5, 51, 25, 2)
            iterations = st.slider("Iterations", 1, 10, 5)
            poly_n = st.slider("Poly N", 5, 9, 7, 2)
            poly_sigma = st.slider("Poly sigma", 1.0, 2.5, 1.5, 0.1)
            lk_max_corners = 500; lk_quality = 0.01; lk_min_dist = 7
            lk_block = 7; lk_winsize = 21; lk_levels = 3
        else:
            st.caption("Parametry algoritmu Lucas-Kanade")
            lk_max_corners = st.slider("Max. počet rohů", 50, 2000, 500, 50)
            lk_quality = st.slider("Kvalita rohů", 0.001, 0.1, 0.01, 0.001, format="%.3f")
            lk_min_dist = st.slider("Min. vzdálenost rohů", 3, 30, 7)
            lk_block = st.slider("Block size", 3, 15, 7, 2)
            lk_winsize = st.slider("LK window size", 5, 51, 21, 2)
            lk_levels = st.slider("LK pyramid levels", 1, 6, 3)
            pyr_scale = 0.5; levels = 6; winsize = 25
            iterations = 5; poly_n = 7; poly_sigma = 1.5

    with st.expander("Detekce a zobrazení pohybu", expanded=False):
        st.caption("Nastavení prahů a vizualizace")
        threshold_factor = st.slider("Práh pohybu (násobek směrodatné odchylky)", 0.5, 5.0, 2.0, 0.1)
        arrow_step = st.slider("Hustota vykreslených šipek", 10, 60, 25, 5, disabled=not is_dense)

    with st.expander("Nastavení zpracování videa", expanded=False):
        st.caption("Omezení pro efektivnější analýzu")
        video_step = st.slider("Analyzovat každý N-tý snímek", 1, 10, 1)
        max_frames = st.slider("Maximální počet zpracovaných snímků", 10, 500, 100)


st.title("OPTICKÝ TOK")
tab_img, tab_vid = st.tabs(["ANALÝZA OBRAZŮ", "ANALÝZA VIDEA"])


# Záložka 1 - obrazy
with tab_img:
    st.markdown("Nahrajte dva po sobě jdoucí snímky pro detekci pohybu mezi nimi.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Snímek 1 (referenční)**")
        f1 = st.file_uploader("Snímek 1", type=["png", "jpg", "jpeg", "bmp", "tiff"],
                              key="img1", label_visibility="collapsed")
        if f1 is not None and f1.file_id != st.session_state.img_file1_id:
            img1_np = np.frombuffer(f1.read(), np.uint8)
            st.session_state.img_frame1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
            st.session_state.img_file1_id = f1.file_id
        if st.session_state.img_frame1 is not None:
            st.image(bgr_to_rgb(st.session_state.img_frame1), use_container_width=True)

    with c2:
        st.markdown("**Snímek 2 (cílový)**")
        f2 = st.file_uploader("Snímek 2", type=["png", "jpg", "jpeg", "bmp", "tiff"],
                              key="img2", label_visibility="collapsed")
        if f2 is not None and f2.file_id != st.session_state.img_file2_id:
            img2_np = np.frombuffer(f2.read(), np.uint8)
            st.session_state.img_frame2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
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
        st.markdown("### Statistické hodnoty")
        st.caption(f"Doba výpočtu: {elapsed * 1000:.0f} ms")

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Pohybující se pixely", f"{stats['moving_pixels']:,}")
        mc2.metric("Pokrytí pohybem", f"{stats['coverage_pct']:.1f} %")
        mc3.metric("Průměrná magnituda", f"{stats['avg_magnitude']:.2f}")
        mc4.metric("Max. magnituda", f"{stats['max_magnitude']:.2f}")

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
        st.download_button("Stáhnout výsledky obrazové analýzy (ZIP)", data=zip_buf,
                           file_name=f"optical_flow_snimky_{ts}.zip", mime="application/zip")

# Záložka 2 - video
with tab_vid:
    st.markdown("**Vstupní soubor**")
    vid_file = st.file_uploader("Video soubor", type=["mp4", "avi", "mov", "mkv", "webm"],
                                key="vid", label_visibility="collapsed")

    if vid_file is not None:
        file_id = vid_file.file_id
        if file_id != st.session_state.vid_file_id:
            st.session_state.vid_file_id = file_id
            st.session_state.vid_results = None
            st.session_state.vid_slider = 0
            st.session_state.vid_analyzing = False
            st.session_state.selected_frames = set()

            old = st.session_state.vid_tmp_path
            if old and os.path.exists(old):
                try:
                    os.unlink(old)
                except:
                    pass
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(vid_file.read())
            tmp.flush(); tmp.close()
            st.session_state.vid_tmp_path = tmp.name

    if vid_file is not None and st.session_state.vid_tmp_path:
        tmp_path = st.session_state.vid_tmp_path
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.info(
            f"**Načteno video:** {vid_file.name} | **Celkem snímků:** {total} | **Rozlišení:** {w_vid}×{h_vid} | **FPS:** {fps:.1f} | **Délka:** {total / fps:.1f} s")

        st.markdown("**Rozsah analýzy**")
        range_col1, range_col2 = st.columns(2)
        with range_col1:
            start_frame = st.number_input(
                "Od snímku", min_value=0, max_value=total - 2,
                value=0, step=1, key="range_start",
            )
        with range_col2:
            end_frame = st.number_input(
                "Do snímku", min_value=int(start_frame) + 1, max_value=total - 1,
                value=min(total - 1, int(start_frame) + max_frames * video_step + 1),
                step=1, key="range_end",
            )

        start_frame = int(start_frame)
        end_frame = int(end_frame)
        range_frames = end_frame - start_frame
        est_segments = min(max_frames, max(range_frames // max(video_step, 1), 1))

        st.caption(
            f"Bude analyzováno {range_frames} snímků (čas úseku: {start_frame / fps:.1f} s – {end_frame / fps:.1f} s). Při zvoleném kroku {video_step} se zpracuje maximálně {est_segments} segmentů.")

        st.markdown("---")
        analyze_btn = st.button("SPUSTIT ANALÝZU VIDEA", key="btn_vid", use_container_width=True,
                                disabled=st.session_state.vid_analyzing)

        if analyze_btn:
            st.session_state.vid_analyzing = True
            st.session_state.vid_results = None
            st.session_state.vid_slider = 0
            st.session_state.selected_frames = set()

            cap = cv2.VideoCapture(tmp_path)
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            preview_ph = st.empty()
            prog_bar = st.progress(0)
            status_ph = st.empty()

            results = []
            prev_gray = None
            prev_bgr = None
            fi = start_frame
            pi = 0
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
                        preview_ph.image(bgr_to_rgb(combined), caption=f"Zpracovávám snímek {fi}/{end_frame}",
                                         use_container_width=True)
                        results.append({
                            "frame_idx": fi,
                            "time_s": fi / fps,
                            "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                            "combined": combined,
                            "stats": stats,
                            "magnitude": mag,
                        })
                        pi += 1

                        elapsed = time.perf_counter() - t_start
                        fps_proc = pi / max(elapsed, 1e-6)
                        remaining = (est_segments - pi) / max(fps_proc, 1e-6)

                        prog_bar.progress(min(pi / est_segments, 1.0))
                        status_ph.caption(
                            f"Uplynulo: {elapsed:.1f} s | Zbývá: ~{remaining:.0f} s | Zpracováno: {pi}/{est_segments} segmentů")

                        if pi >= est_segments:
                            break

                    prev_gray = gray
                    prev_bgr = bgr
                    fi += 1

            except BaseException:
                cap.release()
                raise
            finally:
                cap.release()

            preview_ph.empty()
            prog_bar.empty()
            status_ph.empty()

            st.session_state.vid_results = results
            st.session_state.vid_analyzing = False

            if results:
                total_time = time.perf_counter() - t_start
                st.success(f"Analýza dokončena. Zpracováno {len(results)} segmentů za {total_time:.1f} s.")
            st.rerun()

    results = st.session_state.vid_results

    if results and not st.session_state.vid_analyzing:
        n = len(results)
        frame_indices = [r["frame_idx"] for r in results]
        all_coverage = [r["stats"]["coverage_pct"] for r in results]
        all_avg_mag = [r["stats"]["avg_magnitude"] for r in results]
        all_max_mag = [r["stats"]["max_magnitude"] for r in results]

        st.markdown("### Agregované statistiky")

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Průměrné pokrytí", f"{np.mean(all_coverage):.1f} %")
        sc2.metric("Průměrná magnituda", f"{np.mean(all_avg_mag):.2f}")
        sc3.metric("Maximální magnituda", f"{np.max(all_max_mag):.2f}")

        df = pd.DataFrame({
            "Snímek": frame_indices,
            "Pokrytí pohybem (%)": all_coverage,
            "Průměrná magnituda": all_avg_mag,
            "Max. magnituda": all_max_mag,
        })
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Časový průběh plošného pokrytí pohybem**")
            st.line_chart(df.set_index("Snímek")[["Pokrytí pohybem (%)"]])
        with ch2:
            st.markdown("**Časový průběh intenzity pohybu (magnituda)**")
            st.line_chart(df.set_index("Snímek")[["Průměrná magnituda", "Max. magnituda"]])

        # Prohlížení snímků
        st.markdown("---")
        st.markdown("### Prohlížení segmentů videa")

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
            st.button("První", key="nav_first", on_click=_go, kwargs={"absolute": 0}, use_container_width=True)
        with nav2:
            st.button("Předchozí", key="nav_prev", on_click=_go, kwargs={"delta": -1}, use_container_width=True)
        with nav3:
            st.slider("Segment", 0, n - 1, key="vid_slider", label_visibility="collapsed")
        with nav4:
            st.button("Další", key="nav_next", on_click=_go, kwargs={"delta": 1}, use_container_width=True)
        with nav5:
            st.button("Poslední", key="nav_last", on_click=_go, kwargs={"absolute": n - 1}, use_container_width=True)

        cur = st.session_state.vid_slider
        seg = results[cur]

        st.info(
            f"**Zobrazen segment:** {cur + 1} / {n} | "
            f"**Číslo snímku:** {seg['frame_idx']} | "
            f"**Čas záznamu:** {seg['time_s']:.2f} s"
        )

        info_col, sel_col = st.columns([5, 2])
        with sel_col:
            is_selected = cur in st.session_state.selected_frames
            label = "✓ Vybráno pro export" if is_selected else "Vybrat pro export"

            def _toggle_frame(idx):
                if idx in st.session_state.selected_frames:
                    st.session_state.selected_frames.discard(idx)
                else:
                    st.session_state.selected_frames.add(idx)

            st.button(label, key=f"sel_btn_{cur}", on_click=_toggle_frame, kwargs={"idx": cur},
                      use_container_width=True)

        s = seg["stats"]
        fc1, fc2, fc3, fc4 = st.columns(4)
        if cur > 0:
            prev_s = results[cur - 1]["stats"]
            fc1.metric("Pohybující se pixely", f"{s['moving_pixels']:,}",
                       delta=f"{s['moving_pixels'] - prev_s['moving_pixels']:+,}")
            fc2.metric("Pokrytí pohybem", f"{s['coverage_pct']:.1f} %",
                       delta=f"{s['coverage_pct'] - prev_s['coverage_pct']:+.1f}%")
            fc3.metric("Průměrná magnituda", f"{s['avg_magnitude']:.2f}",
                       delta=f"{s['avg_magnitude'] - prev_s['avg_magnitude']:+.2f}")
            fc4.metric("Max. magnituda", f"{s['max_magnitude']:.2f}",
                       delta=f"{s['max_magnitude'] - prev_s['max_magnitude']:+.2f}")
        else:
            fc1.metric("Pohybující se pixely", f"{s['moving_pixels']:,}")
            fc2.metric("Pokrytí pohybem", f"{s['coverage_pct']:.1f} %")
            fc3.metric("Průměrná magnituda", f"{s['avg_magnitude']:.2f}")
            fc4.metric("Max. magnituda", f"{s['max_magnitude']:.2f}")

        view_mode = st.radio(
            "Režim zobrazení", key="vid_view_mode",
            options=["Kombinovaný pohled", "Originál", "Flow mapa",
                     "Detekce pohybu", "Vektory pohybu"],
            horizontal=True,
            label_visibility="collapsed",
        )
        panel_map = {
            "Kombinovaný pohled": seg["combined"],
            "Originál": seg["p1"],
            "Flow mapa": seg["p2"],
            "Detekce pohybu": seg["p3"],
            "Vektory pohybu": seg["p4"],
        }

        if view_mode == "Kombinovaný pohled":
            g1c, g2c = st.columns(2)
            g3c, g4c = st.columns(2)
            for col, panel, lbl in [
                (g1c, seg["p1"], "Originál"),
                (g2c, seg["p2"], "Flow mapa (HSV)"),
                (g3c, seg["p3"], "Detekce pohybu"),
                (g4c, seg["p4"], "Vektory pohybu"),
            ]:
                with col:
                    st.markdown(f"**{lbl}**")
                    st.image(bgr_to_rgb(panel), use_container_width=True)
        else:
            st.image(bgr_to_rgb(panel_map[view_mode]), use_container_width=True)

        # Export
        st.markdown("---")
        st.markdown("### Export")

        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        sel = sorted(st.session_state.selected_frames)

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            csv_df = results_to_dataframe(results)
            csv_buf = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Stáhnout časová data (CSV)",
                data=csv_buf,
                file_name=f"optical_flow_data_{ts_now}.csv",
                mime="text/csv",
                key="dl_csv",
                use_container_width=True
            )
        with dl2:
            if sel:
                zip_sel = io.BytesIO()
                with zipfile.ZipFile(zip_sel, "w", zipfile.ZIP_DEFLATED) as zf:
                    for idx in sel:
                        r = results[idx]
                        fi_r = r["frame_idx"]
                        zf.writestr(f"frame_{fi_r:05d}_combined.png", encode_png(r["combined"]))
                        zf.writestr(f"frame_{fi_r:05d}_original.png", encode_png(r["p1"]))
                        zf.writestr(f"frame_{fi_r:05d}_flow.png", encode_png(r["p2"]))
                        zf.writestr(f"frame_{fi_r:05d}_pohyb.png", encode_png(r["p3"]))
                zip_sel.seek(0)
                st.download_button(
                    f"Stáhnout vybrané snímky ZIP ({len(sel)})",
                    data=zip_sel,
                    file_name=f"optical_flow_vybrane_{ts_now}.zip",
                    mime="application/zip",
                    key="dl_sel",
                    use_container_width=True,
                )
            else:
                st.button("Stáhnout vybrané snímky ZIP (0)", disabled=True, use_container_width=True,
                          key="dl_sel_disabled")