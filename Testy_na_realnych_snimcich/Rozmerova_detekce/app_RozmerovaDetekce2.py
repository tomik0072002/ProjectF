import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
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
.warn-box {
    background:rgba(255,165,0,0.08); border:1px solid rgba(255,165,0,0.35);
    border-radius:8px; padding:10px 14px; font-size:13px; color:#FFA500; margin:6px 0;
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


# ─── PIL font loader ──────────────────────────────────────────────────────────
@st.cache_resource
def _load_pil_fonts():
    bold_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    reg_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    font = font_sm = ImageFont.load_default()
    for p in bold_paths:
        try:
            font = ImageFont.truetype(p, 20)
            break
        except Exception:
            pass
    for p in reg_paths:
        try:
            font_sm = ImageFont.truetype(p, 16)
            break
        except Exception:
            pass
    return font, font_sm


# ─── Utility ─────────────────────────────────────────────────────────────────
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
    # px/mm pocitame primo z detekce rohu v pixelech (corners_ref)
    # — ne pres projectPoints, ktere zavisi na cam_mtx a dava nesmyslne hodnoty
    pts_flat = corners_ref.reshape(-1, 2)
    dx, dy = [], []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            dx.append(np.linalg.norm(pts_flat[i + 1] - pts_flat[i]))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            dy.append(np.linalg.norm(pts_flat[i + cols] - pts_flat[i]))
    ppm = ((np.mean(dx) + np.mean(dy)) / 2.0) / square_mm

    debug = img.copy()
    pts = corners_ref.reshape(-1, 2)
    for i, (cx, cy) in enumerate(pts):
        color = (int(255 * i / len(pts)), int(255 * (1 - i / len(pts))), 180)
        cv2.circle(debug, (int(cx), int(cy)), 12, color, -1)
        cv2.circle(debug, (int(cx), int(cy)), 12, (255, 255, 255), 2)
    return cam_mtx, dist, ppm, debug, None


# ─── Geometry helpers ─────────────────────────────────────────────────────────
def _rect_to_4_edges(rect, px_per_mm):
    box = cv2.boxPoints(rect).astype(np.float32)
    pts = sorted(box.tolist(), key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0], reverse=True)
    ordered = [np.array(p, dtype=np.float32) for p in (top[0], top[1], bot[0], bot[1])]

    labels = ["Top", "Right", "Bottom", "Left"]
    edges = []
    n = len(ordered)
    for i in range(n):
        p1 = ordered[i]
        p2 = ordered[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        mid = ((p1 + p2) / 2).astype(int)
        edges.append({
            "p1": p1, "p2": p2,
            "angle": angle,
            "length": length,
            "mid": mid,
            "label": labels[i],
        })
    return edges


def get_edges(cnt):
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    if raw_circ >= 0.75:
        epsilon = 0.005 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for factor in [0.008, 0.012, 0.018, 0.025]:
            if len(approx) <= 40:
                break
            approx = cv2.approxPolyDP(cnt, factor * peri, True)
    else:
        best, prev_n = None, None
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

    raw_edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        mid = ((p1 + p2) / 2).astype(int)
        raw_edges.append({"p1": p1, "p2": p2, "angle": angle, "length": length, "mid": mid})

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
                    e = {"p1": e["p1"], "p2": raw_edges[j]["p2"],
                         "angle": e["angle"], "length": 0.0, "mid": None}
                    used[j] = True
                    j = (j + 1) % len(raw_edges)
                else:
                    break
            vec = e["p2"] - e["p1"]
            e["length"] = float(np.linalg.norm(vec))
            e["mid"] = ((e["p1"] + e["p2"]) / 2).astype(int)
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
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return round(min((4 * math.pi * area) / (peri ** 2), 1.0) * 100.0, 2)


def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    coarse = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    n_coarse = len(coarse)

    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    w_major = max(rw, rh)
    w_minor = min(rw, rh)
    aspect = w_minor / w_major if w_major > 0 else 1.0
    is_elongated = aspect < 0.4

    if circ >= 0.85 and aspect >= 0.88 and n_coarse >= 7:
        return "circle", False
    if circ >= 0.70 and n_coarse >= 7:
        return "ellipse", is_elongated
    if circ >= 0.90:
        return "circle", False
    if circ >= 0.78:
        return "ellipse", is_elongated
    if is_elongated:
        return "rectangle", True
    return "polygon", False


def circle_metrics(cnt, px_per_mm):
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
        "r_fit_mm": r_fit / px_per_mm,
        "r_max_mm": r_max / px_per_mm,
        "r_min_mm": r_min / px_per_mm,
        "r_mean_mm": r_mean / px_per_mm,
        "dev_mm": (r_max - r_min) / px_per_mm,
        "roundness_dev": roundness_dev,
        "profile": profile,
    }


def draw_circle_overlay(base_img, cm, px_per_mm):
    out = base_img.copy()
    cx, cy = int(cm["cx_px"]), int(cm["cy_px"])
    r_fit_px = int(cm["r_fit_mm"] * px_per_mm)
    r_max_px = int(cm["r_max_mm"] * px_per_mm)
    r_min_px = int(cm["r_min_mm"] * px_per_mm)
    for a in range(0, 360, 3):
        s, e = math.radians(a), math.radians(a + 2)
        cv2.line(out,
                 (cx + int(r_fit_px * math.cos(s)), cy + int(r_fit_px * math.sin(s))),
                 (cx + int(r_fit_px * math.cos(e)), cy + int(r_fit_px * math.sin(e))),
                 (0, 255, 220), 2, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_max_px, (60, 60, 255), 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_min_px, (60, 220, 60), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 255, 220), cv2.MARKER_CROSS, 20, 2)
    return out


# ─── Detection ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_objects(img_bytes, cam_mtx_bytes, dist_bytes, canny_low, canny_high,
                   min_area, max_obj, merge, merge_dist):
    cam_mtx = np.frombuffer(cam_mtx_bytes, dtype=np.float64).reshape(3, 3) if cam_mtx_bytes else None
    dist    = np.frombuffer(dist_bytes,    dtype=np.float64)               if dist_bytes    else None

    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # ── OPRAVA: sleduj škálovací faktor po undistort+crop ────────────────────
    ppm_scale = 1.0
    if cam_mtx is not None and dist is not None:
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, cam_mtx, dist, None, new_mtx)
        x, y, rw, rh = roi
        if all(v > 0 for v in (x, y, rw, rh)):
            img = img[y:y + rh, x:x + rw]

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, canny_low, canny_high)

    k      = max(5, merge_dist) if merge else 5
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

    cnts_serialized  = [c.tobytes() for c in contours]
    shapes           = [c.shape for c in contours]
    edge_vis_bytes   = cv2.imencode(".png", cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))[1].tobytes()
    img_bytes_out    = cv2.imencode(".png", img)[1].tobytes()

    # ── OPRAVA: vracíme také ppm_scale ───────────────────────────────────────
    return img_bytes_out, edge_vis_bytes, cnts_serialized, shapes, ppm_scale


def deserialize_contours(cnts_s, shapes):
    return [np.frombuffer(s, dtype=np.int32).reshape(sh) for s, sh in zip(cnts_s, shapes)]


RECT_EDGE_COLORS = [
    (0, 255, 179),
    (0, 191, 255),
    (255, 107, 53),
    (255, 215, 0),
]


def draw_rect_overlay(base_img, rect, edges, px_per_mm):
    out = base_img.copy()
    box = cv2.boxPoints(rect).astype(np.intp)

    glow = out.copy()
    cv2.drawContours(glow, [box], 0, (0, 255, 179), 10)
    cv2.addWeighted(glow, 0.18, out, 0.82, 0, out)
    cv2.drawContours(out, [box], 0, (0, 255, 179), 2, cv2.LINE_AA)

    for i, e in enumerate(edges):
        color    = RECT_EDGE_COLORS[i % len(RECT_EDGE_COLORS)]
        p1       = tuple(e["p1"].astype(int))
        p2       = tuple(e["p2"].astype(int))
        cv2.line(out, p1, p2, color, 4, cv2.LINE_AA)

        mid       = tuple(e["mid"].astype(int))
        label_txt = f"H{i + 1}"
        r         = 14
        cv2.circle(out, (mid[0] + 2, mid[1] + 2), r, (0, 0, 0), -1)
        cv2.circle(out, mid, r, color, -1)
        cv2.circle(out, mid, r, (255, 255, 255), 2)
        tw = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0][0]
        cv2.putText(out, label_txt, (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label_txt, (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_edge_map(base_img, edges, highlights, raw_cnt=None):
    out = base_img.copy()
    if raw_cnt is not None:
        cv2.drawContours(out, [raw_cnt], 0, (200, 210, 225), 1, cv2.LINE_AA)

    for i, e in enumerate(edges):
        if i in highlights:
            continue
        p1   = tuple(e["p1"].astype(int))
        p2   = tuple(e["p2"].astype(int))
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            continue
        dash_len, gap_len = 12, 8
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        pos, drawing = 0, True
        while pos < dist:
            seg_end = min(pos + (dash_len if drawing else gap_len), dist)
            if drawing:
                sx = int(p1[0] + pos * dx)
                sy = int(p1[1] + pos * dy)
                ex = int(p1[0] + seg_end * dx)
                ey = int(p1[1] + seg_end * dy)
                cv2.line(out, (sx, sy), (ex, ey), (130, 145, 170), 2, cv2.LINE_AA)
            pos     = seg_end
            drawing = not drawing

    for i, e in enumerate(edges):
        if i not in highlights:
            continue
        p1        = tuple(e["p1"].astype(int))
        p2        = tuple(e["p2"].astype(int))
        color_bgr = highlights[i]
        glow_layer = out.copy()
        cv2.line(glow_layer, p1, p2, color_bgr, 18, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.25, out, 0.75, 0, out)
        cv2.line(out, p1, p2, color_bgr, 8, cv2.LINE_AA)
        bright = tuple(min(255, int(c * 1.4)) for c in color_bgr)
        cv2.line(out, p1, p2, bright, 3, cv2.LINE_AA)

    for i, e in enumerate(edges):
        mid      = tuple(e["mid"].astype(int))
        is_hi    = i in highlights
        badge_color = highlights[i] if is_hi else (90, 100, 120)
        badge_r  = 16 if is_hi else 13
        cv2.circle(out, (mid[0] + 2, mid[1] + 2), badge_r, (0, 0, 0), -1)
        cv2.circle(out, mid, badge_r, badge_color, -1)
        cv2.circle(out, mid, badge_r, (255, 255, 255), 2 if is_hi else 1)
        label = str(i + 1)
        fs    = 0.55 if is_hi else 0.45
        tw    = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0][0]
        cv2.putText(out, label, (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (mid[0] - tw // 2, mid[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_legend_pil(cv_img, legend_defs):
    font, font_sm = _load_pil_fonts()
    pil  = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    if not legend_defs:
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    pad, swatch, gap, line_h = 12, 18, 8, 26
    panel_w = 280
    panel_h = len(legend_defs) * line_h + pad * 2 + 4

    x0, y0  = 10, 10
    overlay  = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ov_draw  = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + panel_w, y0 + panel_h],
        radius=8, fill=(15, 18, 28, 210), outline=(60, 70, 90, 255), width=1,
    )
    pil  = pil.convert("RGBA")
    pil  = Image.alpha_composite(pil, overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)

    for idx, (bgr, txt) in enumerate(legend_defs):
        rgb = (bgr[2], bgr[1], bgr[0])
        ty  = y0 + pad + idx * line_h
        sx  = x0 + pad
        draw.rectangle([sx, ty + 2, sx + swatch, ty + 2 + swatch - 4], fill=rgb)
        draw.text((sx + swatch + gap, ty), txt, font=font_sm, fill=(230, 235, 245))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def tol_row(label, measured, lo, hi, unit, extra="", skip=False):
    if skip:
        return f'<div class="skip-row">⏭ <b>{label}</b> — přeskočeno</div>'
    ok  = lo <= measured <= hi
    cls = "pass-row" if ok else "fail-row"
    icon = "✅" if ok else "❌"
    m_str = f"{measured:.3f}".rstrip("0").rstrip(".")
    return (f'<div class="{cls}">{icon} <b>{label}</b>: '
            f'<span style="font-family:Space Mono,monospace;color:#E8EAF0">{m_str} {unit}</span>'
            f' &nbsp;∈&nbsp; '
            f'<span style="font-family:Space Mono,monospace;color:#6B7280">[{lo:.2f} – {hi:.2f} {unit}]</span>'
            f'{extra}</div>')


def render_verdict(pass_list):
    if not pass_list:
        return
    n_ok  = sum(pass_list)
    all_ok = n_ok == len(pass_list)
    vc = "#00FFB3" if all_ok else "#FF4444"
    vb = "rgba(0,255,179,0.4)" if all_ok else "rgba(255,68,68,0.4)"
    vg = "rgba(0,255,179,0.08)" if all_ok else "rgba(255,68,68,0.08)"
    vi = "✅ DÍLEK V TOLERANCI" if all_ok else "❌ MIMO TOLERANCI"
    st.markdown(
        f"""<div style='background:{vg};border:2px solid {vb};
            border-radius:10px;padding:16px 20px;margin:16px 0;text-align:center;'>
            <span style='font-family:Space Mono,monospace;font-size:20px;color:{vc};'>
            {vi} &nbsp;|&nbsp; {n_ok}/{len(pass_list)} kontrol prošlo
            </span></div>""",
        unsafe_allow_html=True,
    )


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

# ─── Sidebar ─────────────────────────────────────────────────────────────────
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
    canny_low  = st.slider("Canny spodní", 0, 200, 50)
    canny_high = st.slider("Canny horní", 50, 500, 150)
    min_area   = st.slider("Min. plocha [px]", 50, 5000, 500)
    max_obj    = st.slider("Max. objektů (0=vše)", 0, 20, 1)
    merge      = st.toggle("Slučovat kontury", value=True)
    merge_dist = st.slider("Vzdálenost slučování [px]", 5, 100, 40, disabled=not merge)

    st.markdown('<div class="section-title">📏 Ruční px/mm</div>', unsafe_allow_html=True)
    manual_ppm = st.number_input("Vlastní px/mm (0=auto)", 0.0, 500.0, 0.0, step=0.1)

    st.markdown("---")
    if st.button("Vymazat cache (reload)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── Calibration ─────────────────────────────────────────────────────────────
cam_mtx, dist_c, px_per_mm = None, None, None

if calib_file:
    # ── OPRAVA: getvalue() místo read() – lze volat opakovaně bez vyčerpání ──
    calib_bytes = calib_file.getvalue()
    with st.spinner("Kalibrace…"):
        cam_mtx, dist_c, px_per_mm_cal, debug_img, err = calibrate(
            calib_bytes, cb_rows, cb_cols, sq_mm)
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
        st.markdown(
            '<div class="warn-box">&#9888; <b>Neni kalibrace ani rucni px/mm!</b><br>'
            'Vsechny rozmery v mm budou <b>nespravne</b> (pouzit vychozi 5.0 px/mm).<br>'
            'Zadejte <b>Vlastni px/mm</b> v levem panelu nebo nahrajte sachovnicovy snimek.'
            '</div>', unsafe_allow_html=True)

# Vzdy zobraz aktivni px/mm – uzivatel vi jaka hodnota se pouziva
_calib_src = ("(z kalibrace)" if (calib_file and not (manual_ppm and manual_ppm > 0))
              else "(rucni)" if (manual_ppm and manual_ppm > 0)
              else "– VYCHOZI, nekalibrováno")
st.info(f"Aktivni kalibrace: {px_per_mm:.4f} px/mm  {_calib_src}")

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

# ── OPRAVA: getvalue() místo read() ──────────────────────────────────────────
img_bytes = obj_file.getvalue()
orig_pil  = Image.open(io.BytesIO(img_bytes))

cam_mtx_bytes = cam_mtx.astype(np.float64).tobytes() if cam_mtx is not None else b""
dist_bytes    = dist_c.astype(np.float64).tobytes()  if dist_c  is not None else b""

# ─── Detect ──────────────────────────────────────────────────────────────────
with st.spinner("Detekuji objekty…"):
    # ── OPRAVA: rozbalíme také ppm_scale ─────────────────────────────────────
    img_out_bytes, edge_vis_bytes, cnts_s, shapes, ppm_scale = detect_objects(
        img_bytes, cam_mtx_bytes, dist_bytes,
        canny_low, canny_high, min_area, max_obj, merge, merge_dist)

# ── OPRAVA: efektivní px/mm po undistort (škálovací faktor z new_mtx) ────────
px_per_mm_eff = px_per_mm * ppm_scale

base_arr = np.frombuffer(img_out_bytes, np.uint8)
edge_arr = np.frombuffer(edge_vis_bytes, np.uint8)
base_img = cv2.imdecode(base_arr, cv2.IMREAD_COLOR)
edge_img = cv2.imdecode(edge_arr, cv2.IMREAD_COLOR)
contours = deserialize_contours(cnts_s, shapes)

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
    "par_a":  (0, 255, 179),
    "par_b":  (255, 107, 53),
    "perp_a": (0, 191, 255),
    "perp_b": (255, 215, 0),
    "len_e":  (220, 0, 220),
}
PALETTE = [
    (0, 255, 179), (255, 107, 53), (0, 191, 255), (255, 215, 0),
    (220, 0, 220), (0, 255, 80), (255, 60, 120), (80, 200, 255),
    (255, 160, 0), (160, 255, 60),
]

for obj_i, cnt in enumerate(contours):
    # ── OPRAVA: všude px_per_mm_eff místo px_per_mm ──────────────────────────
    ppm = px_per_mm_eff if px_per_mm_eff and px_per_mm_eff > 0 else px_per_mm

    area_px   = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect      = cv2.minAreaRect(cnt)
    (cx_r, cy_r), (rw_px, rh_px), angle = rect
    rw_mm     = max(rw_px, rh_px) / ppm
    rh_mm     = min(rw_px, rh_px) / ppm
    area_mm2  = area_px / (ppm ** 2)
    peri_mm   = cv2.arcLength(cnt, True) / ppm
    circ      = circularity_pct(cnt)
    shape, is_elongated = classify_shape(cnt)

    shape_icons  = {"circle": "⭕", "ellipse": "🥚", "polygon": "🔷", "rectangle": "▬"}
    shape_labels = {
        "circle":    "Kruh",
        "ellipse":   "Elipsa / Oválný",
        "polygon":   "Mnohoúhelník",
        "rectangle": "Obdélník / Tyčinka",
    }

    with st.expander(
            f"🔍 Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  kruhovitost {circ:.0f}%"
            f"  |  auto: {shape_icons[shape]} {shape}",
            expanded=True,
    ):
        # Debug řádek – vždy viditelný, pomáhá odhalit kalibrační problém
        st.caption(f"Použito: {ppm:.4f} px/mm  (px velikost kontury: {max(rw_px, rh_px):.0f} × {min(rw_px, rh_px):.0f} px)")
        st.markdown('<div class="section-title">Rozměry</div>', unsafe_allow_html=True)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Délka",      f"{rw_mm:.2f}",   "mm")
        d2.metric("Šířka",      f"{rh_mm:.2f}",   "mm")
        d3.metric("Plocha",     f"{area_mm2:.1f}", "mm²")
        d4.metric("Obvod",      f"{peri_mm:.1f}",  "mm")
        d5.metric("Kruhovitost",f"{circ:.1f}",     "%")

        st.markdown('<div class="section-title">🔧 Typ analýzy</div>', unsafe_allow_html=True)

        default_mode = "rectangle" if shape in ("rectangle", "polygon") else "circle"

        sel_cols = st.columns([3, 2])
        with sel_cols[0]:
            mode_choice = st.radio(
                "Vyberte typ analýzy:",
                options=["rectangle", "circle"],
                index=0 if default_mode == "rectangle" else 1,
                format_func=lambda x: {
                    "rectangle": "▬ Obdélník — přesný fit minAreaRect, 4 čisté hrany",
                    "circle":    "⭕ Kruhové — průměr, radiální odchylka, kruhovitost",
                }[x],
                key=f"mode_{obj_i}",
                horizontal=True,
            )
        with sel_cols[1]:
            aspect_ratio = rw_mm / rh_mm if rh_mm > 0 else 1.0
            st.markdown(
                f'<div class="tol-box" style="margin-top:8px">'
                f'Auto-detekce: <b>{shape_icons[shape]} {shape_labels[shape]}</b><br>'
                f'Aspect ratio: <b>{aspect_ratio:.1f}:1</b> &nbsp;|&nbsp; '
                f'Kruhovitost: <b>{circ:.1f}%</b></div>',
                unsafe_allow_html=True)

        if mode_choice == "circle" and shape in ("rectangle", "polygon") and circ < 60:
            st.markdown(
                '<div class="warn-box">⚠️ <b>Upozornění:</b> Tvar nevypadá jako kruh/elipsa '
                '(kruhovitost < 60 %). Kruhové metriky nemusí dávat smysl.</div>',
                unsafe_allow_html=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # RECTANGLE MODE
        # ══════════════════════════════════════════════════════════════════════
        if mode_choice == "rectangle":
            rect_edges = _rect_to_4_edges(rect, ppm)
            n_rect     = len(rect_edges)

            st.markdown('<div class="section-title">📐 Obdélníkový fit (minAreaRect)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="tol-box">'
                'Kontury jsou fitovány jako <b>minimální ohraničující obdélník</b> (cv2.minAreaRect). '
                'Výsledkem jsou vždy přesně <b>4 rovné hrany</b> bez ohledu na zaoblené rohy nebo výřezy. '
                'Ideální pro tyčinky, lišty, profily a obdélníkové díly.</div>',
                unsafe_allow_html=True)

            rect_vis = draw_rect_overlay(base_img, rect, rect_edges, ppm)
            cv2.drawContours(rect_vis, [cnt], 0, (160, 170, 190), 1, cv2.LINE_AA)

            vis_col, leg_col = st.columns([1, 1])
            with vis_col:
                st.image(cv_to_pil(rect_vis), use_container_width=True)
            with leg_col:
                st.markdown("**Hrany obdélníku:**")
                for i, e in enumerate(rect_edges):
                    color_hex = ["#00FFB3", "#00BFFF", "#FF6B35", "#FFD700"][i]
                    lbl       = e.get("label", f"H{i + 1}")
                    length_mm = e["length"] / ppm
                    angle_deg = e["angle"]
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:10px;'
                        f'padding:10px 14px;margin:6px 0;background:#1E2230;'
                        f'border-radius:8px;border-left:4px solid {color_hex};">'
                        f'<span style="font-family:Space Mono,monospace;font-size:18px;'
                        f'font-weight:700;color:{color_hex};min-width:36px;">H{i + 1}</span>'
                        f'<span style="color:#9CA3AF;font-size:13px;min-width:60px;">{lbl}</span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:20px;'
                        f'font-weight:700;color:#E8EAF0;">{length_mm:.2f}'
                        f'<span style="font-size:13px;color:#6B7280;margin-left:3px;">mm</span></span>'
                        f'<span style="font-family:Space Mono,monospace;font-size:14px;'
                        f'color:#6B7280;margin-left:auto;">{angle_deg:.1f}°</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("Tabulka hran (minAreaRect)"):
                header = "| # | Název | Délka [mm] | Úhel [°] |\n|---|---|---|---|\n"
                rows_t = "".join(
                    f"| {i + 1} | {e.get('label', '—')} | {e['length'] / ppm:.2f} | {e['angle']:.1f} |\n"
                    for i, e in enumerate(rect_edges)
                )
                st.markdown(header + rows_t)

            st.divider()

            st.markdown('<div class="section-title">⚖ Konfigurace tolerancí</div>',
                        unsafe_allow_html=True)
            nums_r = [1, 2, 3, 4]

            def rect_edge_label(i):
                e = rect_edges[i - 1]
                return f"H{i} {e.get('label', '?')}  ({e['length'] / ppm:.1f} mm)"

            cfg = {}

            with st.container():
                st.markdown("#### ⬌ Rovnoběžnost dvou hran")
                pc = st.columns([1, 1, 1, 1])
                cfg["par_en"]  = pc[0].checkbox("Aktivní", value=True, key=f"rpar_en_{obj_i}")
                cfg["par_a"]   = pc[1].selectbox("Hrana A", nums_r, 0, key=f"rpar_a_{obj_i}",
                                                 disabled=not cfg["par_en"],
                                                 format_func=rect_edge_label) - 1
                cfg["par_b"]   = pc[2].selectbox("Hrana B", nums_r, 2, key=f"rpar_b_{obj_i}",
                                                 disabled=not cfg["par_en"],
                                                 format_func=rect_edge_label) - 1
                cfg["par_tol"] = pc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 2.0,
                                                    step=0.5, key=f"rpar_tol_{obj_i}",
                                                    disabled=not cfg["par_en"])
            st.markdown("---")

            with st.container():
                st.markdown("#### ⊾ Kolmost dvou hran")
                qc = st.columns([1, 1, 1, 1])
                cfg["perp_en"]  = qc[0].checkbox("Aktivní", value=True, key=f"rperp_en_{obj_i}")
                cfg["perp_a"]   = qc[1].selectbox("Hrana A", nums_r, 0, key=f"rperp_a_{obj_i}",
                                                  disabled=not cfg["perp_en"],
                                                  format_func=rect_edge_label) - 1
                cfg["perp_b"]   = qc[2].selectbox("Hrana B", nums_r, 1, key=f"rperp_b_{obj_i}",
                                                  disabled=not cfg["perp_en"],
                                                  format_func=rect_edge_label) - 1
                cfg["perp_tol"] = qc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 2.0,
                                                     step=0.5, key=f"rperp_tol_{obj_i}",
                                                     disabled=not cfg["perp_en"])
            st.markdown("---")

            with st.container():
                st.markdown("#### 📐 Rozměry objektu (délka / šířka)")
                dc = st.columns([1, 1, 1, 1, 1, 1])
                cfg["dim_en"]    = dc[0].checkbox("Aktivní", value=True, key=f"rdim_en_{obj_i}")
                cfg["dim_axis"]  = dc[1].selectbox("Osa", ["Délka", "Šířka", "Obě"],
                                                   key=f"rdim_axis_{obj_i}",
                                                   disabled=not cfg["dim_en"])
                cfg["dim_wnom"]  = dc[2].number_input("Jmenovitá délka [mm]", 0.0, 2000.0,
                                                      float(round(rw_mm, 1)), step=0.5,
                                                      key=f"rdim_wnom_{obj_i}",
                                                      disabled=not cfg["dim_en"])
                cfg["dim_hnom"]  = dc[3].number_input("Jmenovitá šířka [mm]", 0.0, 2000.0,
                                                      float(round(rh_mm, 1)), step=0.5,
                                                      key=f"rdim_hnom_{obj_i}",
                                                      disabled=not cfg["dim_en"])
                cfg["dim_plus"]  = dc[4].number_input("Tol. + [mm]", 0.0, 100.0, 0.5,
                                                      step=0.1, key=f"rdim_plus_{obj_i}",
                                                      disabled=not cfg["dim_en"])
                cfg["dim_minus"] = dc[5].number_input("Tol. − [mm]", 0.0, 100.0, 0.5,
                                                      step=0.1, key=f"rdim_minus_{obj_i}",
                                                      disabled=not cfg["dim_en"])
            st.markdown("---")

            with st.container():
                st.markdown("#### ↔ Délka vybrané hrany")
                lc = st.columns([1, 1, 1, 1, 1])
                cfg["len_en"]    = lc[0].checkbox("Aktivní", value=False, key=f"rlen_en_{obj_i}")
                cfg["len_edge"]  = lc[1].selectbox("Hrana", nums_r, 0, key=f"rlen_edge_{obj_i}",
                                                   disabled=not cfg["len_en"],
                                                   format_func=rect_edge_label) - 1
                cfg["len_nom"]   = lc[2].number_input("Jmenovitá [mm]", 0.0, 2000.0,
                                                      float(round(rw_mm, 1)), step=0.5,
                                                      key=f"rlen_nom_{obj_i}",
                                                      disabled=not cfg["len_en"])
                cfg["len_plus"]  = lc[3].number_input("Tol. + [mm]", 0.0, 100.0, 0.5,
                                                      step=0.1, key=f"rlen_plus_{obj_i}",
                                                      disabled=not cfg["len_en"])
                cfg["len_minus"] = lc[4].number_input("Tol. − [mm]", 0.0, 100.0, 0.5,
                                                      step=0.1, key=f"rlen_minus_{obj_i}",
                                                      disabled=not cfg["len_en"])

            st.divider()

            st.markdown('<div class="section-title">📊 Výsledky</div>', unsafe_allow_html=True)
            rows_html, pass_list, highlights = [], [], {}

            if cfg["par_en"]:
                ia, ib = cfg["par_a"], cfg["par_b"]
                dev    = parallelism_deg(rect_edges, ia, ib)
                ok     = dev <= cfg["par_tol"]
                pass_list.append(ok)
                rows_html.append(tol_row(f"Rovnoběžnost (H{ia + 1} ∥ H{ib + 1})",
                                         dev, 0.0, cfg["par_tol"], "°",
                                         extra=f" — odchylka <b>{dev:.3f}°</b>"))
                highlights[ia] = HCOLORS["par_a"]
                highlights[ib] = HCOLORS["par_b"]
            else:
                rows_html.append(tol_row("Rovnoběžnost", 0, 0, 0, "°", skip=True))

            if cfg["perp_en"]:
                ia, ib = cfg["perp_a"], cfg["perp_b"]
                dev    = perpendicularity_deg(rect_edges, ia, ib)
                ok     = dev <= cfg["perp_tol"]
                pass_list.append(ok)
                rows_html.append(tol_row(f"Kolmost (H{ia + 1} ⊾ H{ib + 1})",
                                         dev, 0.0, cfg["perp_tol"], "°",
                                         extra=f" — odchylka od 90°: <b>{dev:.3f}°</b>"))
                highlights[ia] = HCOLORS["perp_a"]
                highlights[ib] = HCOLORS["perp_b"]
            else:
                rows_html.append(tol_row("Kolmost", 0, 0, 0, "°", skip=True))

            if cfg["dim_en"]:
                check_w = cfg["dim_axis"] in ("Délka", "Obě")
                check_h = cfg["dim_axis"] in ("Šířka", "Obě")
                if check_w:
                    lo = cfg["dim_wnom"] - cfg["dim_minus"]
                    hi = cfg["dim_wnom"] + cfg["dim_plus"]
                    ok = lo <= rw_mm <= hi
                    pass_list.append(ok)
                    rows_html.append(tol_row("Délka objektu", rw_mm, lo, hi, "mm"))
                if check_h:
                    lo = cfg["dim_hnom"] - cfg["dim_minus"]
                    hi = cfg["dim_hnom"] + cfg["dim_plus"]
                    ok = lo <= rh_mm <= hi
                    pass_list.append(ok)
                    rows_html.append(tol_row("Šířka objektu", rh_mm, lo, hi, "mm"))
            else:
                rows_html.append(tol_row("Rozměry", 0, 0, 0, "mm", skip=True))

            if cfg["len_en"]:
                ie   = cfg["len_edge"]
                elen = rect_edges[ie]["length"] / ppm
                lo   = cfg["len_nom"] - cfg["len_minus"]
                hi   = cfg["len_nom"] + cfg["len_plus"]
                ok   = lo <= elen <= hi
                pass_list.append(ok)
                rows_html.append(tol_row(f"Délka H{ie + 1}", elen, lo, hi, "mm",
                                         extra=f" — jmenovitá <b>{cfg['len_nom']:.1f} mm</b>"))
                highlights[ie] = HCOLORS["len_e"]
            else:
                rows_html.append(tol_row("Délka hrany", 0, 0, 0, "mm", skip=True))

            for r in rows_html:
                st.markdown(r, unsafe_allow_html=True)

            render_verdict(pass_list)

            st.markdown('<div class="section-title">Vizualizace vybraných hran</div>',
                        unsafe_allow_html=True)
            final_vis = draw_rect_overlay(base_img, rect, rect_edges, ppm)
            cv2.drawContours(final_vis, [cnt], 0, (160, 170, 190), 1, cv2.LINE_AA)

            fv_col, fl_col = st.columns([1, 1])
            with fv_col:
                st.image(cv_to_pil(final_vis), use_container_width=True)
            with fl_col:
                if highlights:
                    st.markdown("**Měřené hrany:**")
                    role_names = {
                        HCOLORS["par_a"]:  "rovnoběžnost A",
                        HCOLORS["par_b"]:  "rovnoběžnost B",
                        HCOLORS["perp_a"]: "kolmost A",
                        HCOLORS["perp_b"]: "kolmost B",
                        HCOLORS["len_e"]:  "délka",
                    }
                    for hi_idx, color_bgr in highlights.items():
                        e         = rect_edges[hi_idx]
                        r, g, b   = color_bgr[2], color_bgr[1], color_bgr[0]
                        color_hex = f"#{r:02X}{g:02X}{b:02X}"
                        role      = role_names.get(color_bgr, "hrana")
                        length_mm = e["length"] / ppm
                        lbl       = e.get("label", "")
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'padding:10px 14px;margin:6px 0;background:#1E2230;'
                            f'border-radius:8px;border-left:4px solid {color_hex};">'
                            f'<span style="font-family:Space Mono,monospace;font-size:18px;'
                            f'font-weight:700;color:{color_hex};min-width:36px;">H{hi_idx + 1}</span>'
                            f'<div style="display:flex;flex-direction:column;">'
                            f'<span style="color:#9CA3AF;font-size:12px;">{lbl} — {role}</span>'
                            f'<span style="font-family:Space Mono,monospace;font-size:22px;'
                            f'font-weight:700;color:#E8EAF0;line-height:1.2;">{length_mm:.2f}'
                            f'<span style="font-size:13px;color:#6B7280;margin-left:3px;">mm</span></span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            buf = io.BytesIO()
            cv_to_pil(final_vis).save(buf, format="JPEG", quality=92)
            st.download_button(f"⬇ Stáhnout výsledek objektu #{obj_i + 1}",
                               data=buf.getvalue(),
                               file_name=f"viziometer_obj{obj_i + 1}_rect.jpg",
                               mime="image/jpeg", key=f"dl_{obj_i}")

        # ══════════════════════════════════════════════════════════════════════
        # CIRCLE MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            # ── OPRAVA: předáváme ppm (= px_per_mm_eff) ──────────────────────
            cm = circle_metrics(cnt, ppm)

            st.markdown('<div class="section-title">📐 Přesné kruhové metriky (z raw kontury)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="tol-box">'
                'Hodnoty počítány přímo z každého bodu raw kontury — <b>bez aproximace polygony</b>. '
                'Fitovaný kruh = metoda nejmenších čtverců (Kasa). '
                'Radiální odchylka = rozdíl největšího a nejmenšího poloměru od středu.</div>',
                unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Průměr (fit)",      f"{cm['r_fit_mm'] * 2:.3f}", "mm")
            m2.metric("R max",             f"{cm['r_max_mm']:.3f}",     "mm")
            m3.metric("R min",             f"{cm['r_min_mm']:.3f}",     "mm")
            m4.metric("Radiální odchylka", f"{cm['dev_mm']:.3f}",       "mm")

            roundness_pct = max(0.0, 100.0 * (1.0 - cm["roundness_dev"]))
            st.metric("Kulatost (1 − (Rmax−Rmin)/Rstř)", f"{roundness_pct:.1f}", "%")

            st.markdown('<div class="section-title">Vizualizace — fitovaný kruh</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="tol-box">'
                        '<b style="color:#00FFD0">—— Fitovaný kruh</b> &nbsp;&nbsp;'
                        '<b style="color:#3C3CFF">— R max</b> &nbsp;&nbsp;'
                        '<b style="color:#3CDC3C">— R min</b></div>',
                        unsafe_allow_html=True)
            # ── OPRAVA: předáváme ppm ─────────────────────────────────────────
            circ_vis = draw_circle_overlay(base_img, cm, ppm)
            cv2.drawContours(circ_vis, [cnt], 0, (200, 200, 210), 1, cv2.LINE_AA)
            st.image(cv_to_pil(circ_vis), use_container_width=True)

            st.divider()

            with st.expander("Radiální profil (úhel → poloměr)"):
                profile = cm["profile"]
                if profile:
                    step    = max(1, len(profile) // 72)
                    sampled = profile[::step]
                    r_fit   = cm["r_fit_mm"]
                    rows_p  = []
                    for ang, r in sampled:
                        dev_r   = r - r_fit
                        bar_len = int(abs(dev_r) / cm["dev_mm"] * 20) if cm["dev_mm"] > 0 else 0
                        bar_len = min(bar_len, 20)
                        bar     = ("+" if dev_r >= 0 else "−") * bar_len
                        color   = "#FF6B6B" if dev_r > 0 else "#6BFF9E"
                        rows_p.append(
                            f"<tr>"
                            f"<td style='width:50px;color:#6B7280;font-size:11px'>{ang:+.0f}°</td>"
                            f"<td style='font-family:Space Mono;font-size:11px'>{r:.3f}</td>"
                            f"<td style='font-family:Space Mono;font-size:11px;color:{color}'>{bar}</td>"
                            f"<td style='color:#6B7280;font-size:11px'>{dev_r:+.3f} mm</td>"
                            f"</tr>"
                        )
                    st.markdown(
                        "<table style='width:100%;border-collapse:collapse;'>"
                        "<tr><th>Úhel</th><th>R [mm]</th><th>Odchylka</th><th>Δ [mm]</th></tr>"
                        + "".join(rows_p) + "</table>",
                        unsafe_allow_html=True,
                    )

            st.divider()

            st.markdown('<div class="section-title">⚖ Tolerance kruhu</div>',
                        unsafe_allow_html=True)
            cfg = {}
            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("##### ◯ Kruhovitost (4π·A/P²)")
                cfg["circ_en"]  = st.checkbox("Aktivní", value=True, key=f"circ_en_{obj_i}")
                cfg["circ_min"] = st.number_input("Min. [%]", 0.0, 100.0, 80.0, step=1.0,
                                                  key=f"circ_min_{obj_i}",
                                                  disabled=not cfg["circ_en"])
            with tc2:
                st.markdown("##### ⊙ Průměr (fitovaný)")
                cfg["diam_en"]    = st.checkbox("Aktivní", value=True, key=f"diam_en_{obj_i}")
                cfg["diam_nom"]   = st.number_input("Jmenovitý průměr [mm]", 0.0, 2000.0,
                                                    float(round(cm["r_fit_mm"] * 2, 2)), step=0.1,
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
                cfg["rad_en"]  = st.checkbox("Aktivní", value=True, key=f"rad_en_{obj_i}")
                _rad_default   = float(round(cm["dev_mm"] * 1.5 + 0.1, 2))
                _rad_max_limit = float(max(100.0, _rad_default * 2))
                cfg["rad_max"] = st.number_input("Max. odchylka [mm]", 0.0, _rad_max_limit,
                                                 _rad_default, step=0.05,
                                                 key=f"rad_max_{obj_i}",
                                                 disabled=not cfg["rad_en"])

            st.divider()
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
                lo    = cfg["diam_nom"] - cfg["diam_minus"]
                hi    = cfg["diam_nom"] + cfg["diam_plus"]
                ok    = lo <= d_fit <= hi
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

            render_verdict(pass_list)

            buf = io.BytesIO()
            cv_to_pil(circ_vis).save(buf, format="JPEG", quality=92)
            st.download_button(f"⬇ Stáhnout výsledek objektu #{obj_i + 1}",
                               data=buf.getvalue(),
                               file_name=f"viziometer_obj{obj_i + 1}_circle.jpg",
                               mime="image/jpeg", key=f"dl_{obj_i}")