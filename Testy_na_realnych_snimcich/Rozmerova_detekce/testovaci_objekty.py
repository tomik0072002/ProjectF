import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import math

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VizioMeter – Měření tvarů",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --accent:   #00FFB3;
    --accent2:  #FF6B35;
    --accent3:  #00BFFF;
    --bg:       #0D0F14;
    --surface:  #151820;
    --surface2: #1E2230;
    --border:   #2A2F40;
    --text:     #E8EAF0;
    --muted:    #6B7280;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
h1,h2,h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; margin: 6px 0;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content:''; position:absolute; left:0; top:0; bottom:0;
    width:3px; background:var(--accent);
}
.metric-label {
    font-size:11px; color:var(--muted);
    text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;
}
.metric-value {
    font-family:'Space Mono',monospace; font-size:22px;
    font-weight:700; color:var(--text);
}
.metric-unit { font-size:12px; color:var(--muted); margin-left:4px; }

.section-title {
    font-family:'Space Mono',monospace; font-size:12px; color:var(--accent);
    text-transform:uppercase; letter-spacing:2px;
    border-bottom:1px solid var(--border); padding-bottom:8px; margin:20px 0 12px 0;
}
.tol-box {
    background:rgba(0,191,255,0.05); border:1px solid rgba(0,191,255,0.2);
    border-radius:8px; padding:10px 14px; font-size:13px; color:var(--muted); margin:6px 0;
}
.pass-row {
    background:rgba(0,255,179,0.07); border-left:3px solid #00FFB3;
    border-radius:0 8px 8px 0; padding:8px 14px; font-size:13px; margin:4px 0;
}
.fail-row {
    background:rgba(255,68,68,0.07); border-left:3px solid #FF4444;
    border-radius:0 8px 8px 0; padding:8px 14px; font-size:13px; margin:4px 0; color:#FF8888;
}
.skip-row {
    background:rgba(107,114,128,0.07); border-left:3px solid #4B5563;
    border-radius:0 8px 8px 0; padding:8px 14px; font-size:13px; margin:4px 0; color:#6B7280;
}

div[data-testid="stMetric"] {
    background:var(--surface2) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; padding:12px !important;
}
div[data-testid="stMetricValue"] { color:var(--accent) !important; font-family:'Space Mono',monospace !important; }

.stButton > button {
    background:var(--accent) !important; color:#0D0F14 !important;
    font-family:'Space Mono',monospace !important; font-weight:700 !important;
    border:none !important; border-radius:6px !important;
    padding:8px 20px !important; transition:all 0.2s !important;
}
.stButton > button:hover { opacity:.85 !important; transform:translateY(-1px) !important; }

[data-testid="stExpander"] {
    background:var(--surface2) !important; border:1px solid var(--border) !important;
    border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Utility ──────────────────────────────────────────────────────────────────

def cv_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─── Calibration ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def calibrate(img_bytes, rows, cols, square_mm):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern = (cols, rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm

    flags_list = [
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS,
        0,
    ]
    ret, corners, used_gray, found_scale = False, None, gray, 1.0
    for scale in [1.0, 0.75, 0.5]:
        h, w = gray.shape
        gs = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale != 1.0 else gray
        for pre in [gs, cv2.equalizeHist(gs), cv2.GaussianBlur(gs, (5, 5), 0)]:
            for fl in flags_list:
                ret, corners = cv2.findChessboardCorners(pre, pattern, fl)
                if ret:
                    used_gray, found_scale = gs, scale
                    break
            if ret:
                break
        if ret:
            break

    if not ret:
        return None, None, None, None, "Rohy šachovnice nebyly nalezeny."

    corners_ref = cv2.cornerSubPix(
        used_gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria
    )
    if found_scale != 1.0:
        corners_ref = corners_ref / found_scale

    h, w = gray.shape
    _, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners_ref], (w, h), None, None
    )
    proj, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], cam_mtx, dist)
    proj = proj.reshape(-1, 2)

    dx, dy = [], []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            dx.append(np.linalg.norm(proj[i + 1] - proj[i]))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            dy.append(np.linalg.norm(proj[i + cols] - proj[i]))
    ppm = ((np.mean(dx) + np.mean(dy)) / 2.0) / square_mm

    debug = img.copy()
    pts = corners_ref.reshape(-1, 2)
    for i, (cx, cy) in enumerate(pts):
        color = (int(255 * i / len(pts)), int(255 * (1 - i / len(pts))), 180)
        cv2.circle(debug, (int(cx), int(cy)), 12, color, -1)
        cv2.circle(debug, (int(cx), int(cy)), 12, (255, 255, 255), 2)
    return cam_mtx, dist, ppm, debug, None


# ─── Geometry ────────────────────────────────────────────────────────────────

def get_edges(cnt):
    """Approximate contour to polygon and return edge list."""
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2).astype(np.float32)
    n = len(approx)
    edges = []
    for i in range(n):
        p1 = approx[i]
        p2 = approx[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        mid = ((p1 + p2) / 2).astype(int)
        edges.append({"p1": p1, "p2": p2, "angle": angle, "length": length, "mid": mid})
    return edges


def parallelism_deg(edges, ia, ib):
    a1, a2 = edges[ia]["angle"], edges[ib]["angle"]
    diff = abs(a1 - a2)
    return round(min(diff, 180.0 - diff), 3)


def perpendicularity_deg(edges, ia, ib):
    a1, a2 = edges[ia]["angle"], edges[ib]["angle"]
    diff = abs(a1 - a2)
    diff = min(diff, 180.0 - diff)
    return round(abs(diff - 90.0), 3)


def circularity_pct(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return round(min((4 * math.pi * area) / (peri ** 2), 1.0) * 100.0, 2)


# ─── Detection ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def detect_objects(img_bytes, _cam_mtx, _dist, canny_low, canny_high,
                   min_area, max_obj, merge, merge_dist):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if _cam_mtx is not None:
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(_cam_mtx, _dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, _cam_mtx, _dist, None, new_mtx)
        x, y, rw, rh = roi
        if all(v > 0 for v in (x, y, rw, rh)):
            img = img[y:y + rh, x:x + rw]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    k = max(5, merge_dist) if merge else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    if merge:
        closed = cv2.dilate(closed, kernel, iterations=2)
        closed = cv2.erode(closed, kernel, iterations=2)

    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if max_obj > 0:
        contours = contours[:max_obj]

    # serialize contours (numpy arrays aren't hashable for cache)
    cnts_serialized = [c.tobytes() for c in contours]
    shapes = [c.shape for c in contours]
    edge_vis_bytes = cv2.imencode(".png", cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))[1].tobytes()
    img_bytes_out = cv2.imencode(".png", img)[1].tobytes()
    return img_bytes_out, edge_vis_bytes, cnts_serialized, shapes


def deserialize_contours(cnts_s, shapes):
    return [np.frombuffer(s, dtype=np.int32).reshape(sh) for s, sh in zip(cnts_s, shapes)]


def draw_edge_map(base_img, edges, highlights):
    """highlights: dict {idx: (B,G,R)}
    Draws edges with high-contrast glow effect and numbered badges.
    Non-highlighted edges are drawn as dashed grey lines so they don't compete.
    """
    out = base_img.copy()
    h_img, w_img = out.shape[:2]

    # ── 1. Non-highlighted edges first (dashed, dim) ────────────────────────
    for i, e in enumerate(edges):
        if i in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        # dashed line
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            continue
        dash_len, gap_len = 12, 8
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        pos = 0
        drawing = True
        while pos < dist:
            seg_end = min(pos + (dash_len if drawing else gap_len), dist)
            if drawing:
                sx = int(p1[0] + pos * dx)
                sy = int(p1[1] + pos * dy)
                ex = int(p1[0] + seg_end * dx)
                ey = int(p1[1] + seg_end * dy)
                cv2.line(out, (sx, sy), (ex, ey), (130, 145, 170), 2, cv2.LINE_AA)
            pos = seg_end
            drawing = not drawing

    # ── 2. Highlighted edges with glow ──────────────────────────────────────
    for i, e in enumerate(edges):
        if i not in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        color_bgr = highlights[i]
        # outer glow (wide, semi-transparent via blending)
        glow_layer = out.copy()
        cv2.line(glow_layer, p1, p2, color_bgr, 18, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.25, out, 0.75, 0, out)
        # mid glow
        cv2.line(out, p1, p2, color_bgr, 8, cv2.LINE_AA)
        # bright core
        bright = tuple(min(255, int(c * 1.4)) for c in color_bgr)
        cv2.line(out, p1, p2, bright, 3, cv2.LINE_AA)

    # ── 3. Edge number badges (all edges) ───────────────────────────────────
    for i, e in enumerate(edges):
        mid = tuple(e["mid"].astype(int))
        is_hi = i in highlights
        badge_color = highlights[i] if is_hi else (90, 100, 120)
        badge_r = 16 if is_hi else 13

        # drop shadow
        cv2.circle(out, (mid[0]+2, mid[1]+2), badge_r, (0, 0, 0), -1)
        # badge fill
        cv2.circle(out, mid, badge_r, badge_color, -1)
        # badge border
        cv2.circle(out, mid, badge_r, (255, 255, 255), 2 if is_hi else 1)

        label = str(i + 1)
        fs = 0.55 if is_hi else 0.45
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        cv2.putText(out, label,
                    (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label,
                    (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_legend_pil(cv_img, legend_defs):
    """Render legend with full UTF-8 text using PIL (avoids cv2 encoding issues).
    legend_defs: list of (bgr_color, text_str)
    Returns BGR numpy image.
    """
    from PIL import ImageFont, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    # try to load a decent font, fall back to default
    font_size = 20
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size - 4)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    if not legend_defs:
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # measure text to size the panel
    pad = 12
    swatch = 18
    gap = 8
    line_h = font_size + 6
    panel_w = max(
        cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + swatch + gap + pad * 2
        for _, txt in legend_defs
    ) + 60  # PIL text is slightly wider, add margin
    panel_h = len(legend_defs) * line_h + pad * 2 + 4

    # draw semi-transparent dark panel
    x0, y0 = 10, 10
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + panel_w, y0 + panel_h],
        radius=8,
        fill=(15, 18, 28, 210),
        outline=(60, 70, 90, 255),
        width=1,
    )
    pil = pil.convert("RGBA")
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)

    for idx, (bgr, txt) in enumerate(legend_defs):
        rgb = (bgr[2], bgr[1], bgr[0])
        ty = y0 + pad + idx * line_h
        # colour swatch with rounded feel (draw as rectangle + inner highlight)
        sx = x0 + pad
        draw.rectangle([sx, ty + 2, sx + swatch, ty + 2 + swatch - 4], fill=rgb)
        # white text
        draw.text((sx + swatch + gap, ty), txt, font=font, fill=(230, 235, 245))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def tol_row(label, measured, lo, hi, unit, extra="", skip=False):
    if skip:
        return f'<div class="skip-row">⏭ <b>{label}</b> — přeskočeno</div>'
    ok = lo <= measured <= hi
    cls = "pass-row" if ok else "fail-row"
    icon = "✅" if ok else "❌"
    m_str = f"{measured:.3f}".rstrip("0").rstrip(".")
    return (f'<div class="{cls}">{icon} <b>{label}</b>: '
            f'<span style="font-family:Space Mono,monospace;color:#E8EAF0">{m_str} {unit}</span>'
            f' &nbsp;∈&nbsp; '
            f'<span style="font-family:Space Mono,monospace;color:#6B7280">[{lo:.2f} – {hi:.2f} {unit}]</span>'
            f'{extra}</div>')


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='font-family:Space Mono,monospace;font-size:28px;margin-bottom:0;
    background:linear-gradient(90deg,#00FFB3,#00BFFF);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;letter-spacing:2px;'>
  🔬 VIZIOMETER
</h1>
<p style='color:#6B7280;font-size:13px;margin-top:2px;letter-spacing:1px;'>
  INTERAKTIVNÍ ANALÝZA GEOMETRICKÝCH TOLERANCÍ
</p>
""", unsafe_allow_html=True)
st.divider()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Kalibrace</div>', unsafe_allow_html=True)
    calib_file = st.file_uploader("Šachovnicový snímek",
                                   type=["jpg", "jpeg", "png", "bmp"], key="calib")
    c1, c2 = st.columns(2)
    with c1:
        cb_rows = st.number_input("Řádky rohů", 3, 40, 23)
    with c2:
        cb_cols = st.number_input("Sloupce rohů", 3, 50, 32)
    sq_mm = st.number_input("Čtverec [mm]", 0.1, 100.0, 5.0, step=0.1)

    st.markdown('<div class="section-title">🎛 Detekce</div>', unsafe_allow_html=True)
    canny_low  = st.slider("Canny spodní",  0, 200,  50)
    canny_high = st.slider("Canny horní",  50, 500, 150)
    min_area   = st.slider("Min. plocha [px]", 50, 5000, 500)
    max_obj    = st.slider("Max. objektů (0=vše)", 0, 20, 1)
    merge      = st.toggle("Slučovat kontury", value=True)
    merge_dist = st.slider("Vzdálenost slučování [px]", 5, 100, 40, disabled=not merge)

    st.markdown('<div class="section-title">📏 Ruční px/mm</div>', unsafe_allow_html=True)
    manual_ppm = st.number_input("Vlastní px/mm (0=auto)", 0.0, 500.0, 0.0, step=0.1)

# ─── Calibration ──────────────────────────────────────────────────────────────
cam_mtx, dist_c, px_per_mm = None, None, None

if calib_file:
    with st.spinner("Kalibrace…"):
        cam_mtx, dist_c, px_per_mm_cal, debug_img, err = calibrate(
            calib_file.read(), cb_rows, cb_cols, sq_mm)
    if err:
        st.error(f"❌ {err}")
    else:
        px_per_mm = px_per_mm_cal
        st.success(f"✅ Kalibrace OK — {px_per_mm:.4f} px/mm")
        with st.expander("Kalibrační snímek"):
            st.image(cv_to_pil(debug_img), use_container_width=True)

if manual_ppm and manual_ppm > 0:
    px_per_mm = manual_ppm
    st.info(f"ℹ️ Ruční px/mm: **{px_per_mm:.4f}**")

if not px_per_mm:
    px_per_mm = 5.0
    if not calib_file:
        st.markdown('<div class="tol-box">💡 Nahrajte šachovnici nebo zadejte vlastní px/mm.'
                    ' Výchozí: 5.0 px/mm</div>', unsafe_allow_html=True)

# ─── Object upload ────────────────────────────────────────────────────────────
obj_file = st.file_uploader("📷 Snímek měřeného objektu",
                              type=["jpg", "jpeg", "png", "bmp"])

if not obj_file:
    st.markdown("""<div style='text-align:center;padding:60px 20px;color:#4B5563;'>
        <div style='font-size:60px;'>📷</div>
        <div style='font-family:Space Mono,monospace;font-size:16px;margin-top:16px;'>
            Nahrajte snímek objektu</div>
        <div style='font-size:13px;margin-top:8px;'>
            Volitelně přidejte kalibrační šachovnici pro přesné mm</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

img_bytes = obj_file.read()
orig_pil  = Image.open(io.BytesIO(img_bytes))

# ─── Detect ───────────────────────────────────────────────────────────────────
with st.spinner("Detekuji objekty…"):
    img_out_bytes, edge_vis_bytes, cnts_s, shapes = detect_objects(
        img_bytes, cam_mtx, dist_c,
        canny_low, canny_high, min_area, max_obj, merge, merge_dist)

base_arr  = np.frombuffer(img_out_bytes,  np.uint8)
edge_arr  = np.frombuffer(edge_vis_bytes, np.uint8)
base_img  = cv2.imdecode(base_arr,  cv2.IMREAD_COLOR)
edge_img  = cv2.imdecode(edge_arr,  cv2.IMREAD_COLOR)
contours  = deserialize_contours(cnts_s, shapes)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="section-title">ORIGINÁL</div>', unsafe_allow_html=True)
    st.image(orig_pil, use_container_width=True)
with col_b:
    st.markdown('<div class="section-title">HRANOVÝ DETEKTOR</div>', unsafe_allow_html=True)
    st.image(cv_to_pil(edge_img), use_container_width=True)

if not contours:
    st.warning("⚠️ Žádné objekty. Upravte prahy Canny nebo min. plochu.")
    st.stop()

st.divider()
st.markdown(
    f"<h3 style='font-family:Space Mono,monospace;'>Nalezeno: {len(contours)} objekt(ů)</h3>",
    unsafe_allow_html=True)

# ─── Per-object tabs ──────────────────────────────────────────────────────────
HCOLORS = {
    "par_a":  (0,   255, 179),
    "par_b":  (255, 107,  53),
    "perp_a": (0,   191, 255),
    "perp_b": (255, 215,   0),
    "len_e":  (220,   0, 220),
}

for obj_i, cnt in enumerate(contours):
    # basic measurements
    area_px = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw_px, rh_px), angle = rect
    rw_mm = max(rw_px, rh_px) / px_per_mm
    rh_mm = min(rw_px, rh_px) / px_per_mm
    bw_mm = bw / px_per_mm
    bh_mm = bh / px_per_mm
    area_mm2  = area_px / (px_per_mm ** 2)
    peri_mm   = cv2.arcLength(cnt, True) / px_per_mm
    circ      = circularity_pct(cnt)
    edges     = get_edges(cnt)
    n_edges   = len(edges)

    with st.expander(
        f"🔷 Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  {n_edges} hran",
        expanded=True,
    ):
        # ── Dimensions ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Rozměry</div>', unsafe_allow_html=True)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Šířka",  f"{rw_mm:.2f}", "mm")
        d2.metric("Výška",  f"{rh_mm:.2f}", "mm")
        d3.metric("Plocha", f"{area_mm2:.1f}", "mm²")
        d4.metric("Obvod",  f"{peri_mm:.1f}", "mm")
        d5.metric("Úhel",   f"{angle:.1f}", "°")

        st.divider()

        # ── Edge map (no tolerances yet) ──────────────────────────────────────
        st.markdown('<div class="section-title">🗺 Mapa hran</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="tol-box">Detekováno <b>{n_edges} hran</b>. '
            'Čísla hran jsou vykreslena v náhledu. Vyberte níže, které hrany zkoumat '
            'a zadejte toleranci pro každou kontrolu.</div>',
            unsafe_allow_html=True)

        # Assign a distinct colour to every edge in the preview map
        PALETTE = [
            (0, 255, 179),   # cyan-green
            (255, 107, 53),  # orange
            (0, 191, 255),   # sky blue
            (255, 215, 0),   # yellow
            (220, 0, 220),   # magenta
            (0, 255, 80),    # lime
            (255, 60, 120),  # pink-red
            (80, 200, 255),  # light blue
            (255, 160, 0),   # amber
            (160, 255, 60),  # yellow-green
        ]
        preview_highlights = {i: PALETTE[i % len(PALETTE)] for i in range(n_edges)}
        edge_preview = draw_edge_map(base_img, edges, preview_highlights)
        cv2.rectangle(edge_preview, (x, y), (x + bw, y + bh), (50, 60, 80), 1)
        preview_legend = [
            (PALETTE[i % len(PALETTE)],
             f"H{i+1}  {edges[i]['length']/px_per_mm:.1f} mm  {edges[i]['angle']:.1f} deg")
            for i in range(n_edges)
        ]
        edge_preview = draw_legend_pil(edge_preview, preview_legend)
        st.image(cv_to_pil(edge_preview), use_container_width=True)

        # Edge table
        with st.expander("Tabulka hran"):
            header = "| # | Délka [mm] | Úhel [°] |\n|---|---|---|\n"
            rows = "".join(
                f"| {i+1} | {e['length']/px_per_mm:.2f} | {e['angle']:.1f} |\n"
                for i, e in enumerate(edges)
            )
            st.markdown(header + rows)

        st.divider()

        # ── Tolerance configuration ───────────────────────────────────────────
        st.markdown('<div class="section-title">⚖ Konfigurace tolerancí</div>',
                    unsafe_allow_html=True)

        nums = list(range(1, n_edges + 1))

        # helper for edge selectbox label
        def edge_label(i):
            e = edges[i - 1]
            return f"H{i}  ({e['length']/px_per_mm:.1f} mm, {e['angle']:.1f}°)"

        cfg = {}

        # 1) Rovnoběžnost
        with st.container():
            st.markdown("#### ⬌ Rovnoběžnost dvou hran")
            pc = st.columns([1, 1, 1, 1])
            cfg["par_en"] = pc[0].checkbox("Aktivní", value=True,
                                            key=f"par_en_{obj_i}")
            cfg["par_a"]  = pc[1].selectbox("Hrana A", nums, 0, key=f"par_a_{obj_i}",
                                             disabled=not cfg["par_en"],
                                             format_func=edge_label) - 1
            cfg["par_b"]  = pc[2].selectbox("Hrana B", nums,
                                             min(1, n_edges - 1),
                                             key=f"par_b_{obj_i}",
                                             disabled=not cfg["par_en"],
                                             format_func=edge_label) - 1
            cfg["par_tol"] = pc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 5.0,
                                                  step=0.5, key=f"par_tol_{obj_i}",
                                                  disabled=not cfg["par_en"])

        st.markdown("---")

        # 2) Kolmost
        with st.container():
            st.markdown("#### ⊾ Kolmost dvou hran")
            qc = st.columns([1, 1, 1, 1])
            cfg["perp_en"] = qc[0].checkbox("Aktivní", value=True,
                                              key=f"perp_en_{obj_i}")
            cfg["perp_a"]  = qc[1].selectbox("Hrana A", nums, 0, key=f"perp_a_{obj_i}",
                                               disabled=not cfg["perp_en"],
                                               format_func=edge_label) - 1
            cfg["perp_b"]  = qc[2].selectbox("Hrana B", nums,
                                               min(1, n_edges - 1),
                                               key=f"perp_b_{obj_i}",
                                               disabled=not cfg["perp_en"],
                                               format_func=edge_label) - 1
            cfg["perp_tol"] = qc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 5.0,
                                                    step=0.5, key=f"perp_tol_{obj_i}",
                                                    disabled=not cfg["perp_en"])

        st.markdown("---")

        # 3) Kruhovitost
        with st.container():
            st.markdown("#### ◯ Kruhovitost")
            cc = st.columns([1, 2, 2])
            cfg["circ_en"]  = cc[0].checkbox("Aktivní", value=False,
                                               key=f"circ_en_{obj_i}")
            cfg["circ_min"] = cc[1].number_input("Min. kruhovitost [%]", 0.0, 100.0, 70.0,
                                                    step=1.0, key=f"circ_min_{obj_i}",
                                                    disabled=not cfg["circ_en"])
            if cfg["circ_en"]:
                cc[2].markdown(
                    f'<div style="padding-top:28px;font-size:13px;color:#6B7280;">'
                    f'Naměřeno: <b style="color:#E8EAF0">{circ:.1f}%</b></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # 4) Délka hrany
        with st.container():
            st.markdown("#### ↔ Délka vybrané hrany")
            lc = st.columns([1, 1, 1, 1, 1])
            cfg["len_en"]    = lc[0].checkbox("Aktivní", value=False,
                                                key=f"len_en_{obj_i}")
            cfg["len_edge"]  = lc[1].selectbox("Hrana", nums, 0, key=f"len_edge_{obj_i}",
                                                 disabled=not cfg["len_en"],
                                                 format_func=edge_label) - 1
            cfg["len_nom"]   = lc[2].number_input("Jmenovitá [mm]", 0.0, 2000.0, 10.0,
                                                    step=0.5, key=f"len_nom_{obj_i}",
                                                    disabled=not cfg["len_en"])
            cfg["len_plus"]  = lc[3].number_input("Tol. + [mm]", 0.0, 100.0, 1.0,
                                                    step=0.1, key=f"len_plus_{obj_i}",
                                                    disabled=not cfg["len_en"])
            cfg["len_minus"] = lc[4].number_input("Tol. − [mm]", 0.0, 100.0, 1.0,
                                                    step=0.1, key=f"len_minus_{obj_i}",
                                                    disabled=not cfg["len_en"])

        st.markdown("---")

        # 5) Rozměrová tolerance
        with st.container():
            st.markdown("#### 📐 Rozměr objektu (šířka / výška)")
            dc = st.columns([1, 1, 1, 1, 1, 1])
            cfg["dim_en"]    = dc[0].checkbox("Aktivní", value=True,
                                               key=f"dim_en_{obj_i}")
            cfg["dim_axis"]  = dc[1].selectbox("Osa", ["Šířka", "Výška", "Obě"],
                                                 key=f"dim_axis_{obj_i}",
                                                 disabled=not cfg["dim_en"])
            cfg["dim_wnom"]  = dc[2].number_input("Jmenovitá šířka [mm]",
                                                    0.0, 2000.0, round(rw_mm, 1),
                                                    step=0.5, key=f"dim_wnom_{obj_i}",
                                                    disabled=not cfg["dim_en"])
            cfg["dim_hnom"]  = dc[3].number_input("Jmenovitá výška [mm]",
                                                    0.0, 2000.0, round(rh_mm, 1),
                                                    step=0.5, key=f"dim_hnom_{obj_i}",
                                                    disabled=not cfg["dim_en"])
            cfg["dim_plus"]  = dc[4].number_input("Tol. + [mm]", 0.0, 100.0, 1.0,
                                                    step=0.1, key=f"dim_plus_{obj_i}",
                                                    disabled=not cfg["dim_en"])
            cfg["dim_minus"] = dc[5].number_input("Tol. − [mm]", 0.0, 100.0, 1.0,
                                                    step=0.1, key=f"dim_minus_{obj_i}",
                                                    disabled=not cfg["dim_en"])

        st.divider()

        # ── Evaluation ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Výsledky</div>', unsafe_allow_html=True)

        rows_html   = []
        pass_list   = []
        highlights  = {}

        # Parallelism
        if cfg["par_en"]:
            ia, ib = cfg["par_a"], cfg["par_b"]
            dev = parallelism_deg(edges, ia, ib)
            ok  = dev <= cfg["par_tol"]
            pass_list.append(ok)
            rows_html.append(tol_row(
                f"Rovnoběžnost (H{ia+1} ∥ H{ib+1})",
                dev, 0.0, cfg["par_tol"], "°",
                extra=f" &nbsp;—&nbsp; odchylka <b>{dev:.3f}°</b>"
            ))
            highlights[ia] = HCOLORS["par_a"]
            highlights[ib] = HCOLORS["par_b"]
        else:
            rows_html.append(tol_row("Rovnoběžnost", 0, 0, 0, "°", skip=True))

        # Perpendicularity
        if cfg["perp_en"]:
            ia, ib = cfg["perp_a"], cfg["perp_b"]
            dev = perpendicularity_deg(edges, ia, ib)
            ok  = dev <= cfg["perp_tol"]
            pass_list.append(ok)
            rows_html.append(tol_row(
                f"Kolmost (H{ia+1} ⊾ H{ib+1})",
                dev, 0.0, cfg["perp_tol"], "°",
                extra=f" &nbsp;—&nbsp; odchylka od 90°: <b>{dev:.3f}°</b>"
            ))
            highlights[ia] = HCOLORS["perp_a"]
            highlights[ib] = HCOLORS["perp_b"]
        else:
            rows_html.append(tol_row("Kolmost", 0, 0, 0, "°", skip=True))

        # Circularity
        if cfg["circ_en"]:
            ok = circ >= cfg["circ_min"]
            pass_list.append(ok)
            rows_html.append(tol_row(
                "Kruhovitost", circ, cfg["circ_min"], 100.0, "%",
                extra=f" &nbsp;—&nbsp; naměřeno <b>{circ:.1f}%</b>"
            ))
        else:
            rows_html.append(tol_row("Kruhovitost", 0, 0, 100, "%", skip=True))

        # Edge length
        if cfg["len_en"]:
            ie = cfg["len_edge"]
            if ie < n_edges:
                elen = edges[ie]["length"] / px_per_mm
                lo   = cfg["len_nom"] - cfg["len_minus"]
                hi   = cfg["len_nom"] + cfg["len_plus"]
                ok   = lo <= elen <= hi
                pass_list.append(ok)
                rows_html.append(tol_row(
                    f"Délka H{ie+1}", elen, lo, hi, "mm",
                    extra=f" &nbsp;—&nbsp; jmenovitá <b>{cfg['len_nom']:.1f} mm</b>"
                ))
                highlights[ie] = HCOLORS["len_e"]
            else:
                rows_html.append(
                    '<div class="fail-row">❌ Délka: hrana neexistuje</div>')
        else:
            rows_html.append(tol_row("Délka hrany", 0, 0, 0, "mm", skip=True))

        # Dimensions
        if cfg["dim_en"]:
            check_w = cfg["dim_axis"] in ("Šířka", "Obě")
            check_h = cfg["dim_axis"] in ("Výška", "Obě")
            if check_w:
                lo = cfg["dim_wnom"] - cfg["dim_minus"]
                hi = cfg["dim_wnom"] + cfg["dim_plus"]
                ok = lo <= rw_mm <= hi
                pass_list.append(ok)
                rows_html.append(tol_row("Šířka objektu", rw_mm, lo, hi, "mm"))
            if check_h:
                lo = cfg["dim_hnom"] - cfg["dim_minus"]
                hi = cfg["dim_hnom"] + cfg["dim_plus"]
                ok = lo <= rh_mm <= hi
                pass_list.append(ok)
                rows_html.append(tol_row("Výška objektu", rh_mm, lo, hi, "mm"))
        else:
            rows_html.append(tol_row("Rozměry", 0, 0, 0, "mm", skip=True))

        for r in rows_html:
            st.markdown(r, unsafe_allow_html=True)

        # Overall verdict
        if pass_list:
            n_ok    = sum(pass_list)
            n_total = len(pass_list)
            all_ok  = n_ok == n_total
            verdict_color  = "#00FFB3" if all_ok else "#FF4444"
            verdict_border = "rgba(0,255,179,0.4)" if all_ok else "rgba(255,68,68,0.4)"
            verdict_bg     = "rgba(0,255,179,0.08)" if all_ok else "rgba(255,68,68,0.08)"
            verdict_icon   = "✅ DÍLEK V TOLERANCI" if all_ok else "❌ MIMO TOLERANCI"
            st.markdown(f"""
            <div style='background:{verdict_bg};border:2px solid {verdict_border};
            border-radius:10px;padding:16px 20px;margin:16px 0;text-align:center;'>
                <span style='font-family:Space Mono,monospace;font-size:20px;color:{verdict_color};'>
                {verdict_icon} &nbsp;|&nbsp; {n_ok}/{n_total} kontrol prošlo
                </span>
            </div>""", unsafe_allow_html=True)

        # Annotated result
        st.markdown('<div class="section-title">Vizualizace vybraných hran</div>',
                    unsafe_allow_html=True)
        annotated = draw_edge_map(base_img, edges, highlights)

        # rotated rect outline
        box_pts = cv2.boxPoints(rect).astype(np.intp)
        cv2.drawContours(annotated, [box_pts], 0, (50, 65, 90), 2)

        # legend — built as list of (bgr_color, utf8_text) and rendered via PIL
        legend_defs = []
        if cfg["par_en"]:
            ia, ib = cfg["par_a"], cfg["par_b"]
            e_a = edges[ia]
            e_b = edges[ib]
            legend_defs += [
                (HCOLORS["par_a"],
                 f"H{ia+1}  rovnobeznost A  ({e_a['length']/px_per_mm:.1f} mm, {e_a['angle']:.1f}°)"),
                (HCOLORS["par_b"],
                 f"H{ib+1}  rovnobeznost B  ({e_b['length']/px_per_mm:.1f} mm, {e_b['angle']:.1f}°)"),
            ]
        if cfg["perp_en"]:
            ia, ib = cfg["perp_a"], cfg["perp_b"]
            e_a = edges[ia]
            e_b = edges[ib]
            legend_defs += [
                (HCOLORS["perp_a"],
                 f"H{ia+1}  kolmost A  ({e_a['length']/px_per_mm:.1f} mm, {e_a['angle']:.1f}°)"),
                (HCOLORS["perp_b"],
                 f"H{ib+1}  kolmost B  ({e_b['length']/px_per_mm:.1f} mm, {e_b['angle']:.1f}°)"),
            ]
        if cfg["len_en"] and cfg["len_edge"] < n_edges:
            ie = cfg["len_edge"]
            e_l = edges[ie]
            legend_defs.append(
                (HCOLORS["len_e"],
                 f"H{ie+1}  delka  ({e_l['length']/px_per_mm:.1f} mm, {e_l['angle']:.1f}°)")
            )

        annotated = draw_legend_pil(annotated, legend_defs)

        st.image(cv_to_pil(annotated), use_container_width=True)

        buf = io.BytesIO()
        cv_to_pil(annotated).save(buf, format="JPEG", quality=92)
        st.download_button(
            f"⬇ Stáhnout výsledek objektu #{obj_i + 1}",
            data=buf.getvalue(),
            file_name=f"viziometer_obj{obj_i + 1}.jpg",
            mime="image/jpeg",
            key=f"dl_{obj_i}",
        )