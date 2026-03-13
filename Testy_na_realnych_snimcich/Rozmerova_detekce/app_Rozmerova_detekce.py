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
    """Approximate contour to polygon and return edge list.

    Strategy:
    - Compute raw circularity of the contour.
    - For circular/curved shapes (circ > 0.7) use a tight epsilon so curved
      segments are preserved as individual short edges rather than collapsed
      into a coarse polygon.
    - For clearly polygonal shapes (circ <= 0.7) use a progressively larger
      epsilon so tiny noise bumps are merged into clean straight sides.
    - After approximation, merge collinear consecutive edges (angle diff < 4°)
      so a "wall" that was split by noise appears as one edge.
    """
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    if raw_circ >= 0.75:
        # Curved / circular: very fine approximation → many short segments
        # that faithfully trace the arc.  We want enough segments to be useful
        # but not thousands — aim for roughly 12-36 segments.
        epsilon = 0.005 * peri          # very tight
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # if still too many (>40) bump epsilon slightly
        for factor in [0.008, 0.012, 0.018, 0.025]:
            if len(approx) <= 40:
                break
            approx = cv2.approxPolyDP(cnt, factor * peri, True)
    else:
        # Polygonal: use a tighter-than-default epsilon first, then try to
        # find a stable result where edge count matches the visual polygon.
        # We iterate epsilon from tight to loose and stop once edge count
        # stabilises (changes < 2 edges between steps).
        best = None
        prev_n = None
        for factor in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]:
            cand = cv2.approxPolyDP(cnt, factor * peri, True)
            n_cand = len(cand)
            if prev_n is not None and abs(n_cand - prev_n) <= 1 and n_cand >= 3:
                best = cand
                break
            best = cand
            prev_n = n_cand
        approx = best if best is not None else cv2.approxPolyDP(cnt, 0.02 * peri, True)

    pts = approx.reshape(-1, 2).astype(np.float32)
    n = len(pts)

    # Build raw edge list
    raw_edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        mid = ((p1 + p2) / 2).astype(int)
        raw_edges.append({"p1": p1, "p2": p2, "angle": angle, "length": length, "mid": mid})

    # For polygonal shapes merge collinear consecutive edges (< 4° diff)
    if raw_circ < 0.75 and len(raw_edges) >= 3:
        merged = []
        used = [False] * len(raw_edges)
        i = 0
        while i < len(raw_edges):
            if used[i]:
                i += 1
                continue
            e = raw_edges[i]
            j = (i + 1) % len(raw_edges)
            while j != i and not used[j]:
                diff = abs(raw_edges[j]["angle"] - e["angle"])
                diff = min(diff, 180.0 - diff)
                if diff < 4.0:
                    # extend edge to j's endpoint
                    e = {
                        "p1": e["p1"],
                        "p2": raw_edges[j]["p2"],
                        "angle": e["angle"],
                        "length": 0.0,  # recalculated below
                        "mid": None,
                    }
                    used[j] = True
                    j = (j + 1) % len(raw_edges)
                else:
                    break
            vec = e["p2"] - e["p1"]
            e["length"] = float(np.linalg.norm(vec))
            e["mid"] = ((e["p1"] + e["p2"]) / 2).astype(int)
            # recompute angle from actual endpoints
            e["angle"] = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
            merged.append(e)
            used[i] = True
            i += 1
        raw_edges = merged if len(merged) >= 3 else raw_edges

    return raw_edges


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
    """ISO 4291 inspired: 4pi*A/P2  (raw contour, not approximation)."""
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return round(min((4 * math.pi * area) / (peri ** 2), 1.0) * 100.0, 2)


def classify_shape(cnt):
    """Return 'circle', 'ellipse', or 'polygon' based on raw contour metrics.

    Strategy uses two independent signals:
    1. Circularity (4pi*A/P2) — high for smooth round shapes
    2. Polygon edge count at coarse epsilon — few edges = polygon, many = curved

    A hexagon has circ~83% but only 6 coarse edges → polygon.
    An ellipse has circ~77% but 12+ coarse edges → ellipse.
    A rectangle has circ~74% and 4 edges → polygon.
    """
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    # Coarse polygon approximation to count "sides"
    coarse = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    n_coarse = len(coarse)

    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 1.0

    # Perfect circle: high circularity + nearly square bounding box
    if circ >= 0.85 and aspect >= 0.88:
        # extra guard: if coarse approx gives <=5 sides it's probably a
        # rendered polygon (pentagon/hexagon), not a real circle
        if n_coarse >= 7:
            return "circle"
    # Ellipse / oval: moderate-high circularity with smooth boundary
    if circ >= 0.70 and n_coarse >= 7:
        return "ellipse"
    # Fallback: truly high circularity even with few coarse sides
    # (e.g. a very smooth generated ellipse approximated as 4-gon)
    if circ >= 0.90:
        return "circle"
    if circ >= 0.78:
        return "ellipse"
    # Everything else (rectangles, L, T, hexagons, polygons)
    return "polygon"


def circle_metrics(cnt, px_per_mm):
    """
    For circular/elliptical contours compute:
    - fitted circle centre + radius (least-squares Kasa method)
    - max/min radius from centre to each raw contour point
    - roundness deviation = (R_max - R_min) / R_mean
    - radial profile: list of (angle_deg, radius_mm)
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    r_fit = math.sqrt(max(result[2] + cx ** 2 + cy ** 2, 0))

    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    r_max, r_min, r_mean = float(radii.max()), float(radii.min()), float(radii.mean())
    roundness_dev = (r_max - r_min) / r_mean if r_mean > 0 else 0.0

    angles_rad = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles_rad)
    profile = [(math.degrees(float(angles_rad[i])), float(radii[i]) / px_per_mm)
               for i in order]

    return {
        "cx_px": float(cx), "cy_px": float(cy),
        "r_fit_mm":  r_fit  / px_per_mm,
        "r_max_mm":  r_max  / px_per_mm,
        "r_min_mm":  r_min  / px_per_mm,
        "r_mean_mm": r_mean / px_per_mm,
        "dev_mm":    (r_max - r_min) / px_per_mm,
        "roundness_dev": roundness_dev,
        "profile": profile,
    }


def draw_circle_overlay(base_img, cm, px_per_mm):
    """Draw fitted circle + min/max bands on image."""
    out = base_img.copy()
    cx, cy = int(cm["cx_px"]), int(cm["cy_px"])
    r_fit_px = int(cm["r_fit_mm"] * px_per_mm)
    r_max_px = int(cm["r_max_mm"] * px_per_mm)
    r_min_px = int(cm["r_min_mm"] * px_per_mm)
    # fitted — dashed cyan
    for a in range(0, 360, 3):
        s, e = math.radians(a), math.radians(a + 2)
        cv2.line(out,
                 (cx + int(r_fit_px*math.cos(s)), cy + int(r_fit_px*math.sin(s))),
                 (cx + int(r_fit_px*math.cos(e)), cy + int(r_fit_px*math.sin(e))),
                 (0, 255, 220), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_max_px, (60,  60, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_min_px, (60, 220,  60), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 255, 220), cv2.MARKER_CROSS, 20, 2)
    return out


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


def draw_edge_map(base_img, edges, highlights, raw_cnt=None):
    """highlights: dict {idx: (B,G,R)}
    raw_cnt: optional raw contour drawn as a thin white guide line so the
             true shape is always visible regardless of polygon approximation.
    """
    out = base_img.copy()
    if raw_cnt is not None:
        cv2.drawContours(out, [raw_cnt], 0, (200, 210, 225), 1, cv2.LINE_AA)
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
    # ── Basic measurements (always from raw contour) ──────────────────────────
    area_px  = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect     = cv2.minAreaRect(cnt)
    (cx_r, cy_r), (rw_px, rh_px), angle = rect
    rw_mm    = max(rw_px, rh_px) / px_per_mm
    rh_mm    = min(rw_px, rh_px) / px_per_mm
    area_mm2 = area_px / (px_per_mm ** 2)
    peri_mm  = cv2.arcLength(cnt, True) / px_per_mm
    circ     = circularity_pct(cnt)           # always from raw contour
    shape    = classify_shape(cnt)            # 'circle' | 'ellipse' | 'polygon'
    edges    = get_edges(cnt)
    n_edges  = len(edges)

    shape_icons   = {"circle": "⭕", "ellipse": "🥚", "polygon": "🔷"}
    shape_labels  = {"circle": "Kruh", "ellipse": "Elipsa / Oválný", "polygon": "Mnohoúhelník / Hranaté"}
    auto_idx      = {"circle": 0, "ellipse": 1, "polygon": 2}[shape]

    with st.expander(
        f"🔍 Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  kruhovitost {circ:.0f}%"
        f"  |  auto: {shape_icons[shape]} {shape}",
        expanded=True,
    ):

        # ── Basic dims ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Rozměry</div>', unsafe_allow_html=True)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Šířka",  f"{rw_mm:.2f}", "mm")
        d2.metric("Výška",  f"{rh_mm:.2f}", "mm")
        d3.metric("Plocha", f"{area_mm2:.1f}", "mm²")
        d4.metric("Obvod",  f"{peri_mm:.1f}", "mm")
        d5.metric("Kruhovitost (raw)", f"{circ:.1f}", "%")

        # ── Mode selector ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔧 Typ analýzy</div>', unsafe_allow_html=True)
        sel_cols = st.columns([3, 2])
        with sel_cols[0]:
            mode_choice = st.radio(
                "Vyberte typ analýzy:",
                options=["polygon", "circle"],
                index=0 if shape == "polygon" else 1,
                format_func=lambda x: {
                    "polygon": "🔷 Hranaté — výběr hran, rovnoběžnost, kolmost",
                    "circle":  "⭕ Kruhové — průměr, radiální odchylka, kruhovitost",
                }[x],
                key=f"mode_{obj_i}",
                horizontal=True,
            )
        with sel_cols[1]:
            st.markdown(
                f'<div class="tol-box" style="margin-top:8px">' 
                f'Auto-detekce: <b>{shape_icons[shape]} {shape_labels[shape]}</b><br>' 
                f'Kruhovitost (4π·A/P²): <b>{circ:.1f}%</b></div>',
                unsafe_allow_html=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # MODE A — CIRCULAR / ELLIPTICAL
        # Kruhovitost se počítá z RAW kontury pomocí fitovaného kruhu.
        # Hrany polygonu jsou k ničemu — nabídneme místo nich kruhové metriky.
        # ══════════════════════════════════════════════════════════════════════
        if mode_choice == "circle":
            cm = circle_metrics(cnt, px_per_mm)

            st.markdown('<div class="section-title">📐 Přesné kruhové metriky (z raw kontury)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="tol-box">'
                'Tyto hodnoty jsou počítány přímo z každého bodu raw kontury — <b>bez aproximace polygony</b>. '
                'Fitovaný kruh = metoda nejmenších čtverců (Kasa). '
                'Radiální odchylka = rozdíl největšího a nejmenšího poloměru od středu.</div>',
                unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Průměr (fit)",    f"{cm['r_fit_mm']*2:.3f}", "mm")
            m2.metric("R max",           f"{cm['r_max_mm']:.3f}", "mm")
            m3.metric("R min",           f"{cm['r_min_mm']:.3f}", "mm")
            m4.metric("Radiální odchylka", f"{cm['dev_mm']:.3f}", "mm")

            roundness_pct = max(0.0, 100.0 * (1.0 - cm["roundness_dev"]))
            st.metric("Kulatost (1 − (Rmax−Rmin)/Rstř)", f"{roundness_pct:.1f}", "%")

            # ── Visualisation: fitted circle overlay ─────────────────────────
            st.markdown('<div class="section-title">Vizualizace — fitovaný kruh</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="tol-box">'
                        '<b style="color:#00FFD0">—— Fitovaný kruh</b> &nbsp;&nbsp;'
                        '<b style="color:#3C3CFF">— R max</b> &nbsp;&nbsp;'
                        '<b style="color:#3CDC3C">— R min</b></div>',
                        unsafe_allow_html=True)
            circ_vis = draw_circle_overlay(base_img, cm, px_per_mm)
            cv2.drawContours(circ_vis, [cnt], 0, (200, 200, 210), 1, cv2.LINE_AA)
            st.image(cv_to_pil(circ_vis), use_container_width=True)

            st.divider()

            # ── Radial deviation chart (mini bar chart via markdown) ──────────
            with st.expander("Radiální profil (úhel → poloměr)"):
                profile = cm["profile"]
                if profile:
                    # downsample to ~72 points for display
                    step = max(1, len(profile) // 72)
                    sampled = profile[::step]
                    r_vals = [r for _, r in sampled]
                    r_fit  = cm["r_fit_mm"]
                    rows_p = []
                    for ang, r in sampled:
                        dev_r = r - r_fit
                        bar_len = int(abs(dev_r) / cm["dev_mm"] * 20) if cm["dev_mm"] > 0 else 0
                        bar_len = min(bar_len, 20)
                        bar = ("+" if dev_r >= 0 else "-") * bar_len
                        color = "#FF6B6B" if dev_r > 0 else "#6BFF9E"
                        rows_p.append(
                            f"<tr><td style='width:50px;color:#6B7280;font-size:11px'>{ang:+.0f}°</td>"
                            f"<td style='font-family:Space Mono;font-size:11px'>{r:.3f}</td>"
                            f"<td style='font-family:Space Mono;font-size:11px;color:{color}'>{bar}</td>"
                            f"<td style='color:#6B7280;font-size:11px'>{dev_r:+.3f} mm</td></tr>"
                        )
                    st.markdown(
                        "<table style='width:100%;border-collapse:collapse;'>"
                        "<tr><th>Úhel</th><th>R [mm]</th><th>Odchylka</th><th>Δ [mm]</th></tr>"
                        + "".join(rows_p) + "</table>",
                        unsafe_allow_html=True
                    )

            st.divider()

            # ── Circular tolerances ───────────────────────────────────────────
            st.markdown('<div class="section-title">⚖ Tolerance kruhu</div>',
                        unsafe_allow_html=True)

            cfg = {}
            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("##### ◯ Kruhovitost (4π·A/P²)")
                cfg["circ_en"]  = st.checkbox("Aktivní", value=True,  key=f"circ_en_{obj_i}")
                cfg["circ_min"] = st.number_input("Min. [%]", 0.0, 100.0, 80.0, step=1.0,
                                                   key=f"circ_min_{obj_i}",
                                                   disabled=not cfg["circ_en"])

            with tc2:
                st.markdown("##### ⊙ Průměr (fitovaný)")
                cfg["diam_en"]    = st.checkbox("Aktivní", value=True,  key=f"diam_en_{obj_i}")
                cfg["diam_nom"]   = st.number_input("Jmenovitý průměr [mm]", 0.0, 2000.0,
                                                     round(cm["r_fit_mm"]*2, 2), step=0.1,
                                                     key=f"diam_nom_{obj_i}",
                                                     disabled=not cfg["diam_en"])
                cfg["diam_plus"]  = st.number_input("Tol. + [mm]", 0.0, 100.0, 0.5, step=0.05,
                                                     key=f"diam_plus_{obj_i}",
                                                     disabled=not cfg["diam_en"])
                cfg["diam_minus"] = st.number_input("Tol. − [mm]", 0.0, 100.0, 0.5, step=0.05,
                                                     key=f"diam_minus_{obj_i}",
                                                     disabled=not cfg["diam_en"])

            with tc3:
                st.markdown("##### 〰 Radiální odchylka")
                cfg["rad_en"]  = st.checkbox("Aktivní", value=True,  key=f"rad_en_{obj_i}")
                _rad_default   = round(cm["dev_mm"] * 1.5 + 0.1, 2)
                _rad_max_limit = max(100.0, _rad_default * 2)
                cfg["rad_max"] = st.number_input("Max. odchylka [mm]", 0.0, _rad_max_limit,
                                                  _rad_default,
                                                  step=0.05,
                                                  key=f"rad_max_{obj_i}",
                                                  disabled=not cfg["rad_en"])

            st.divider()

            # ── Evaluate circular ─────────────────────────────────────────────
            st.markdown('<div class="section-title">📊 Výsledky</div>', unsafe_allow_html=True)

            rows_html, pass_list = [], []

            if cfg["circ_en"]:
                ok = circ >= cfg["circ_min"]
                pass_list.append(ok)
                rows_html.append(tol_row("Kruhovitost (4π·A/P²)", circ,
                                          cfg["circ_min"], 100.0, "%",
                                          extra=f" — naměřeno <b>{circ:.2f}%</b>"))
            else:
                rows_html.append(tol_row("Kruhovitost", 0, 0, 100, "%", skip=True))

            if cfg["diam_en"]:
                d_fit = cm["r_fit_mm"] * 2
                lo = cfg["diam_nom"] - cfg["diam_minus"]
                hi = cfg["diam_nom"] + cfg["diam_plus"]
                ok = lo <= d_fit <= hi
                pass_list.append(ok)
                rows_html.append(tol_row("Průměr (fitovaný)", d_fit, lo, hi, "mm",
                                          extra=f" — jmenovitý <b>{cfg['diam_nom']:.2f} mm</b>"))
            else:
                rows_html.append(tol_row("Průměr", 0, 0, 0, "mm", skip=True))

            if cfg["rad_en"]:
                ok = cm["dev_mm"] <= cfg["rad_max"]
                pass_list.append(ok)
                rows_html.append(tol_row("Radiální odchylka", cm["dev_mm"],
                                          0.0, cfg["rad_max"], "mm",
                                          extra=f" — Rmax−Rmin = <b>{cm['dev_mm']:.3f} mm</b>"))
            else:
                rows_html.append(tol_row("Radiální odchylka", 0, 0, 0, "mm", skip=True))

            for r in rows_html:
                st.markdown(r, unsafe_allow_html=True)

            if pass_list:
                n_ok = sum(pass_list)
                all_ok = n_ok == len(pass_list)
                vc = "#00FFB3" if all_ok else "#FF4444"
                vb = "rgba(0,255,179,0.4)" if all_ok else "rgba(255,68,68,0.4)"
                vg = "rgba(0,255,179,0.08)" if all_ok else "rgba(255,68,68,0.08)"
                vi = "✅ DÍLEK V TOLERANCI" if all_ok else "❌ MIMO TOLERANCI"
                st.markdown(f"""<div style='background:{vg};border:2px solid {vb};
                    border-radius:10px;padding:16px 20px;margin:16px 0;text-align:center;'>
                    <span style='font-family:Space Mono,monospace;font-size:20px;color:{vc};'>
                    {vi} &nbsp;|&nbsp; {n_ok}/{len(pass_list)} kontrol prošlo
                    </span></div>""", unsafe_allow_html=True)

            # download
            buf = io.BytesIO()
            cv_to_pil(circ_vis).save(buf, format="JPEG", quality=92)
            st.download_button(f"⬇ Stáhnout výsledek objektu #{obj_i + 1}",
                               data=buf.getvalue(),
                               file_name=f"viziometer_obj{obj_i+1}.jpg",
                               mime="image/jpeg", key=f"dl_{obj_i}")

        # ══════════════════════════════════════════════════════════════════════
        # MODE B — POLYGONAL
        # ══════════════════════════════════════════════════════════════════════
        else:  # mode_choice == 'polygon'
            PALETTE = [
                (0, 255, 179), (255, 107, 53), (0, 191, 255), (255, 215, 0),
                (220, 0, 220), (0, 255, 80),   (255, 60, 120),(80, 200, 255),
                (255, 160, 0), (160, 255, 60),
            ]

            # ── Edge map ─────────────────────────────────────────────────────
            st.markdown('<div class="section-title">🗺 Mapa hran</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="tol-box">Detekováno <b>{n_edges} hran</b>. '
                'Čísla hran jsou označena v náhledu. Vyberte níže, které hrany zkoumat.</div>',
                unsafe_allow_html=True)

            preview_highlights = {i: PALETTE[i % len(PALETTE)] for i in range(n_edges)}
            edge_preview = draw_edge_map(base_img, edges, preview_highlights, raw_cnt=cnt)
            cv2.rectangle(edge_preview, (x, y), (x + bw, y + bh), (50, 60, 80), 1)
            preview_legend = [
                (PALETTE[i % len(PALETTE)],
                 f"H{i+1}  {edges[i]['length']/px_per_mm:.1f} mm  {edges[i]['angle']:.1f} deg")
                for i in range(n_edges)
            ]
            edge_preview = draw_legend_pil(edge_preview, preview_legend)
            st.image(cv_to_pil(edge_preview), use_container_width=True)

            with st.expander("Tabulka hran"):
                header = "| # | Délka [mm] | Úhel [°] |\n|---|---|---|\n"
                rows_t = "".join(
                    f"| {{i+1}} | {{e['length']/px_per_mm:.2f}} | {{e['angle']:.1f}} |\n"
                    for i, e in enumerate(edges)
                )
                st.markdown(header + rows_t)

            st.divider()

            # ── Tolerance config ──────────────────────────────────────────────
            st.markdown('<div class="section-title">⚖ Konfigurace tolerancí</div>',
                        unsafe_allow_html=True)
            nums = list(range(1, n_edges + 1))

            def edge_label(i):
                e = edges[i - 1]
                return f"H{i}  ({e['length']/px_per_mm:.1f} mm, {e['angle']:.1f}°)"

            cfg = {}

            with st.container():
                st.markdown("#### ⬌ Rovnoběžnost dvou hran")
                pc = st.columns([1, 1, 1, 1])
                cfg["par_en"]  = pc[0].checkbox("Aktivní", value=True, key=f"par_en_{obj_i}")
                cfg["par_a"]   = pc[1].selectbox("Hrana A", nums, 0, key=f"par_a_{obj_i}",
                                                  disabled=not cfg["par_en"],
                                                  format_func=edge_label) - 1
                cfg["par_b"]   = pc[2].selectbox("Hrana B", nums, min(1, n_edges-1),
                                                  key=f"par_b_{obj_i}",
                                                  disabled=not cfg["par_en"],
                                                  format_func=edge_label) - 1
                cfg["par_tol"] = pc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 5.0,
                                                     step=0.5, key=f"par_tol_{obj_i}",
                                                     disabled=not cfg["par_en"])
            st.markdown("---")

            with st.container():
                st.markdown("#### ⊾ Kolmost dvou hran")
                qc = st.columns([1, 1, 1, 1])
                cfg["perp_en"]  = qc[0].checkbox("Aktivní", value=True, key=f"perp_en_{obj_i}")
                cfg["perp_a"]   = qc[1].selectbox("Hrana A", nums, 0, key=f"perp_a_{obj_i}",
                                                   disabled=not cfg["perp_en"],
                                                   format_func=edge_label) - 1
                cfg["perp_b"]   = qc[2].selectbox("Hrana B", nums, min(1, n_edges-1),
                                                   key=f"perp_b_{obj_i}",
                                                   disabled=not cfg["perp_en"],
                                                   format_func=edge_label) - 1
                cfg["perp_tol"] = qc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 5.0,
                                                      step=0.5, key=f"perp_tol_{obj_i}",
                                                      disabled=not cfg["perp_en"])
            st.markdown("---")

            with st.container():
                st.markdown("#### ↔ Délka vybrané hrany")
                lc = st.columns([1, 1, 1, 1, 1])
                cfg["len_en"]    = lc[0].checkbox("Aktivní", value=False, key=f"len_en_{obj_i}")
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

            with st.container():
                st.markdown("#### 📐 Rozměr objektu (šířka / výška)")
                dc = st.columns([1, 1, 1, 1, 1, 1])
                cfg["dim_en"]    = dc[0].checkbox("Aktivní", value=True,  key=f"dim_en_{obj_i}")
                cfg["dim_axis"]  = dc[1].selectbox("Osa", ["Šířka", "Výška", "Obě"],
                                                    key=f"dim_axis_{obj_i}",
                                                    disabled=not cfg["dim_en"])
                cfg["dim_wnom"]  = dc[2].number_input("Jmenovitá šířka [mm]", 0.0, 2000.0,
                                                       round(rw_mm, 1), step=0.5,
                                                       key=f"dim_wnom_{obj_i}",
                                                       disabled=not cfg["dim_en"])
                cfg["dim_hnom"]  = dc[3].number_input("Jmenovitá výška [mm]", 0.0, 2000.0,
                                                       round(rh_mm, 1), step=0.5,
                                                       key=f"dim_hnom_{obj_i}",
                                                       disabled=not cfg["dim_en"])
                cfg["dim_plus"]  = dc[4].number_input("Tol. + [mm]", 0.0, 100.0, 1.0,
                                                       step=0.1, key=f"dim_plus_{obj_i}",
                                                       disabled=not cfg["dim_en"])
                cfg["dim_minus"] = dc[5].number_input("Tol. − [mm]", 0.0, 100.0, 1.0,
                                                       step=0.1, key=f"dim_minus_{obj_i}",
                                                       disabled=not cfg["dim_en"])

            st.divider()

            # ── Evaluate polygon ──────────────────────────────────────────────
            st.markdown('<div class="section-title">📊 Výsledky</div>', unsafe_allow_html=True)
            rows_html, pass_list, highlights = [], [], {}

            if cfg["par_en"]:
                ia, ib = cfg["par_a"], cfg["par_b"]
                dev = parallelism_deg(edges, ia, ib)
                ok  = dev <= cfg["par_tol"]
                pass_list.append(ok)
                rows_html.append(tol_row(f"Rovnoběžnost (H{ia+1} ∥ H{ib+1})",
                                          dev, 0.0, cfg["par_tol"], "°",
                                          extra=f" — odchylka <b>{dev:.3f}°</b>"))
                highlights[ia] = HCOLORS["par_a"]
                highlights[ib] = HCOLORS["par_b"]
            else:
                rows_html.append(tol_row("Rovnoběžnost", 0, 0, 0, "°", skip=True))

            if cfg["perp_en"]:
                ia, ib = cfg["perp_a"], cfg["perp_b"]
                dev = perpendicularity_deg(edges, ia, ib)
                ok  = dev <= cfg["perp_tol"]
                pass_list.append(ok)
                rows_html.append(tol_row(f"Kolmost (H{ia+1} ⊾ H{ib+1})",
                                          dev, 0.0, cfg["perp_tol"], "°",
                                          extra=f" — odchylka od 90°: <b>{dev:.3f}°</b>"))
                highlights[ia] = HCOLORS["perp_a"]
                highlights[ib] = HCOLORS["perp_b"]
            else:
                rows_html.append(tol_row("Kolmost", 0, 0, 0, "°", skip=True))

            if cfg["len_en"]:
                ie = cfg["len_edge"]
                if ie < n_edges:
                    elen = edges[ie]["length"] / px_per_mm
                    lo   = cfg["len_nom"] - cfg["len_minus"]
                    hi   = cfg["len_nom"] + cfg["len_plus"]
                    ok   = lo <= elen <= hi
                    pass_list.append(ok)
                    rows_html.append(tol_row(f"Délka H{ie+1}", elen, lo, hi, "mm",
                                              extra=f" — jmenovitá <b>{cfg['len_nom']:.1f} mm</b>"))
                    highlights[ie] = HCOLORS["len_e"]
                else:
                    rows_html.append('<div class="fail-row">❌ Délka: hrana neexistuje</div>')
            else:
                rows_html.append(tol_row("Délka hrany", 0, 0, 0, "mm", skip=True))

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

            if pass_list:
                n_ok   = sum(pass_list)
                all_ok = n_ok == len(pass_list)
                vc = "#00FFB3" if all_ok else "#FF4444"
                vb = "rgba(0,255,179,0.4)" if all_ok else "rgba(255,68,68,0.4)"
                vg = "rgba(0,255,179,0.08)" if all_ok else "rgba(255,68,68,0.08)"
                vi = "✅ DÍLEK V TOLERANCI" if all_ok else "❌ MIMO TOLERANCI"
                st.markdown(f"""<div style='background:{vg};border:2px solid {vb};
                    border-radius:10px;padding:16px 20px;margin:16px 0;text-align:center;'>
                    <span style='font-family:Space Mono,monospace;font-size:20px;color:{vc};'>
                    {vi} &nbsp;|&nbsp; {n_ok}/{len(pass_list)} kontrol prošlo
                    </span></div>""", unsafe_allow_html=True)

            # Annotated visualisation
            st.markdown('<div class="section-title">Vizualizace vybraných hran</div>',
                        unsafe_allow_html=True)
            annotated = draw_edge_map(base_img, edges, highlights, raw_cnt=cnt)
            box_pts   = cv2.boxPoints(rect).astype(np.intp)
            cv2.drawContours(annotated, [box_pts], 0, (50, 65, 90), 2)

            legend_defs = []
            if cfg["par_en"]:
                ia, ib = cfg["par_a"], cfg["par_b"]
                legend_defs += [
                    (HCOLORS["par_a"], f"H{ia+1}  rovnobeznost A  ({edges[ia]['length']/px_per_mm:.1f} mm, {edges[ia]['angle']:.1f} deg)"),
                    (HCOLORS["par_b"], f"H{ib+1}  rovnobeznost B  ({edges[ib]['length']/px_per_mm:.1f} mm, {edges[ib]['angle']:.1f} deg)"),
                ]
            if cfg["perp_en"]:
                ia, ib = cfg["perp_a"], cfg["perp_b"]
                legend_defs += [
                    (HCOLORS["perp_a"], f"H{ia+1}  kolmost A  ({edges[ia]['length']/px_per_mm:.1f} mm, {edges[ia]['angle']:.1f} deg)"),
                    (HCOLORS["perp_b"], f"H{ib+1}  kolmost B  ({edges[ib]['length']/px_per_mm:.1f} mm, {edges[ib]['angle']:.1f} deg)"),
                ]
            if cfg["len_en"] and cfg["len_edge"] < n_edges:
                ie = cfg["len_edge"]
                legend_defs.append(
                    (HCOLORS["len_e"], f"H{ie+1}  delka  ({edges[ie]['length']/px_per_mm:.1f} mm, {edges[ie]['angle']:.1f} deg)")
                )
            annotated = draw_legend_pil(annotated, legend_defs)
            st.image(cv_to_pil(annotated), use_container_width=True)

            buf = io.BytesIO()
            cv_to_pil(annotated).save(buf, format="JPEG", quality=92)
            st.download_button(f"⬇ Stáhnout výsledek objektu #{obj_i + 1}",
                               data=buf.getvalue(),
                               file_name=f"viziometer_obj{obj_i+1}.jpg",
                               mime="image/jpeg", key=f"dl_{obj_i}")