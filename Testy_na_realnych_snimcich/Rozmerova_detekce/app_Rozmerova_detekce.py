import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import math
import os

# Načtení obrázku loga
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
try:
    logo = Image.open(logo_path)
except:
    logo = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rozměrová detekce",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.info-box {
    background: rgba(128, 128, 128, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.8rem;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Pomocné funkce pro ADAPTIVNÍ MĚŘÍTKO ─────────────────────────────────────
def get_drawing_scale(img_h, img_w, min_scale=0.1):
    """Vypočítá koeficient měřítka přímo úměrný rozlišení obrázku (nebo výřezu)."""
    diagonal = math.sqrt(img_h ** 2 + img_w ** 2)
    reference_diag = 1500.0
    return max(min_scale, diagonal / reference_diag)


def cv_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ─── Calibration ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calibrate(img_bytes, rows, cols, square_mm):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    d_scale = get_drawing_scale(gray.shape[0], gray.shape[1])
    dot_r = max(3, int(12 * d_scale))
    dot_th = max(1, int(2 * d_scale))

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
        return None, None, None, None, "Rohy šachovnice nebyly nalezeny.", None

    corners_ref = cv2.cornerSubPix(
        used_gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria
    )
    if found_scale != 1.0:
        corners_ref = corners_ref / found_scale

    h, w = gray.shape
    _, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        [objp], [corners_ref], (w, h), None, None
    )

    corners_undist = cv2.undistortPoints(corners_ref, cam_mtx, dist, P=cam_mtx)
    pts = corners_undist.reshape(-1, 2)

    dx, dy = [], []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            dx.append(np.linalg.norm(pts[i + 1] - pts[i]))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            dy.append(np.linalg.norm(pts[i + cols] - pts[i]))
    ppm = ((np.mean(dx) + np.mean(dy)) / 2.0) / square_mm

    debug = cv2.undistort(img, cam_mtx, dist, None, cam_mtx)
    for i, (cx, cy) in enumerate(pts):
        color = (int(255 * i / len(pts)), int(255 * (1 - i / len(pts))), 180)
        cv2.circle(debug, (int(cx), int(cy)), dot_r, color, -1)
        cv2.circle(debug, (int(cx), int(cy)), dot_r, (255, 255, 255), dot_th)

    return cam_mtx, dist, ppm, debug, None, (w, h)


# ─── Geometry ────────────────────────────────────────────────────────────────

def clean_polygon_vertices(pts, peri, angle_merge_tol=6.0):
    pts = list(pts)
    changed = True
    min_len_px = max(3.0, 0.015 * peri)
    max_iters = 100
    iters = 0
    while changed and len(pts) > 3 and iters < max_iters:
        changed = False
        iters += 1
        n = len(pts)
        for i in range(n):
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]
            dist = np.linalg.norm(p_next - p_curr)
            if dist < min_len_px:
                pts.pop((i + 1) % n)
                changed = True
                break
        if changed: continue
        n = len(pts)
        for i in range(n):
            p_prev = pts[(i - 1) % n]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n]
            v1 = p_curr - p_prev
            v2 = p_next - p_curr
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_theta = np.dot(v1, v2) / (n1 * n2)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_diff = math.degrees(math.acos(cos_theta))
                if angle_diff < angle_merge_tol:
                    pts.pop(i)
                    changed = True
                    break
    return np.array(pts)


def fit_sharp_corners(cnt_pts, rough_pts, peri, angle_merge_tol):
    """Proloží matematické přímky středy stěn a najde jejich dokonalý průsečík (ignoruje rohy)."""
    rough_pts = clean_polygon_vertices(rough_pts, peri, angle_merge_tol)
    n_edges = len(rough_pts)
    if n_edges < 3:
        return rough_pts

    edge_points = [[] for _ in range(n_edges)]

    for pt in cnt_pts:
        min_dist = float('inf')
        best_edge = -1
        best_t = 0.0

        for i in range(n_edges):
            p1 = rough_pts[i]
            p2 = rough_pts[(i + 1) % n_edges]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0: continue

            pt_vec = pt - p1
            t = np.dot(pt_vec, line_vec) / (line_len ** 2)
            t_clamped = max(0.0, min(1.0, t))
            closest_pt = p1 + t_clamped * line_vec
            dist = np.linalg.norm(pt - closest_pt)

            if dist < min_dist:
                min_dist = dist
                best_edge = i
                best_t = t_clamped

        if best_edge != -1:
            edge_points[best_edge].append((pt, best_t))

    fitted_lines = []
    for i in range(n_edges):
        pts_with_t = edge_points[i]
        pts_with_t.sort(key=lambda x: x[1])

        n_pts = len(pts_with_t)
        if n_pts >= 10:
            start_idx = int(0.20 * n_pts)
            end_idx = int(0.80 * n_pts)
            valid_pts = [p[0] for p in pts_with_t[start_idx:end_idx]]
        elif n_pts >= 3:
            valid_pts = [p[0] for p in pts_with_t]
        else:
            valid_pts = []

        if len(valid_pts) >= 2:
            valid_pts_arr = np.array(valid_pts, dtype=np.float32)
            line = cv2.fitLine(valid_pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            fitted_lines.append((float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])))
        else:
            p1 = rough_pts[i]
            p2 = rough_pts[(i + 1) % n_edges]
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            norm = math.hypot(vx, vy)
            if norm == 0: norm = 1
            fitted_lines.append((vx / norm, vy / norm, float(p1[0]), float(p1[1])))

    sharp_pts = []
    for i in range(n_edges):
        l1 = fitted_lines[i]
        l2 = fitted_lines[(i + 1) % n_edges]

        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2

        denom = vx1 * vy2 - vy1 * vx2
        if abs(denom) > 1e-6:
            t1 = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
            ix = x1 + t1 * vx1
            iy = y1 + t1 * vy1

            rough_p = rough_pts[(i + 1) % n_edges]
            if math.hypot(ix - rough_p[0], iy - rough_p[1]) > 0.15 * peri:
                sharp_pts.append(rough_p)
            else:
                sharp_pts.append(np.array([ix, iy], dtype=np.float32))
        else:
            sharp_pts.append(rough_pts[(i + 1) % n_edges])

    return np.array(sharp_pts, dtype=np.float32)


def get_edges(cnt, angle_merge_tol=6.0):
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    if raw_circ >= 0.75:
        epsilon = 0.005 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        for factor in [0.008, 0.012, 0.018, 0.025]:
            if len(approx) <= 40: break
            approx = cv2.approxPolyDP(cnt, factor * peri, True)
    else:
        best = None
        prev_n = None
        for factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            cand = cv2.approxPolyDP(cnt, factor * peri, True)
            n_cand = len(cand)
            if prev_n is not None and abs(n_cand - prev_n) <= 1 and n_cand >= 3:
                best = cand
                break
            best = cand
            prev_n = n_cand
        approx = best if best is not None else cv2.approxPolyDP(cnt, 0.02 * peri, True)

    pts = approx.reshape(-1, 2).astype(np.float32)

    if raw_circ < 0.75:
        pts = fit_sharp_corners(cnt.reshape(-1, 2).astype(np.float32), pts, peri, angle_merge_tol)

    n = len(pts)
    raw_edges = []
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length == 0: continue
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        mid = ((p1 + p2) / 2).astype(int)
        raw_edges.append({"p1": p1, "p2": p2, "angle": angle, "length": length, "mid": mid})

    return raw_edges, pts


def draw_holes_with_labels(out, holes, angle_merge_tol):
    h_img, w_img = out.shape[:2]
    d_scale = get_drawing_scale(h_img, w_img)

    fs = max(0.4, 0.7 * d_scale)
    th = max(1, int(2 * d_scale))
    line_th = max(1, int(2 * d_scale))
    badge_r = max(10, int(22 * d_scale))
    text_y_offset = int(7 * d_scale)

    for i, h_cnt in enumerate(holes):
        peri = cv2.arcLength(h_cnt, True)
        area = cv2.contourArea(h_cnt)
        raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

        if raw_circ < 0.75:
            edges, sharp_pts = get_edges(h_cnt, angle_merge_tol)
            if sharp_pts is not None and len(sharp_pts) >= 3:
                cv2.polylines(out, [sharp_pts.astype(np.int32)], True, (0, 0, 255), line_th, cv2.LINE_AA)
                for pt in sharp_pts:
                    cv2.circle(out, tuple(pt.astype(int)), max(1, line_th // 2), (0, 0, 255), -1, cv2.LINE_AA)
            else:
                cv2.drawContours(out, [h_cnt], -1, (0, 0, 255), line_th, cv2.LINE_AA)
        else:
            cv2.drawContours(out, [h_cnt], -1, (0, 0, 255), line_th, cv2.LINE_AA)

        M = cv2.moments(h_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(h_cnt[0][0][0]), int(h_cnt[0][0][1])

        label = f"#{i + 1}"
        cv2.circle(out, (cx, cy), badge_r, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), badge_r, (0, 0, 255), line_th, cv2.LINE_AA)

        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
        cv2.putText(out, label, (cx - tw // 2, cy + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)


def draw_edge_map(base_img, edges, highlights, raw_cnt=None, sharp_pts=None):
    out = base_img.copy()
    h_img, w_img = out.shape[:2]

    d_scale = get_drawing_scale(h_img, w_img)

    th_dash = max(1, int(1.5 * d_scale))
    th_highlight = max(1, int(3 * d_scale))
    th_badge_outline = max(1, int(2 * d_scale))
    th_raw_cnt = max(1, int(1.5 * d_scale))
    dash_len = max(2, int(10 * d_scale))
    gap_len = max(2, int(10 * d_scale))

    # Proměnné pro velikost fontu přidány zpět!
    fs_edge = max(0.4, 0.7 * d_scale)
    fs_edge_hi = max(0.4, 0.8 * d_scale)
    th_edge_text = max(1, int(2 * d_scale))

    if sharp_pts is not None and len(sharp_pts) >= 3:
        cv2.polylines(out, [sharp_pts.astype(np.int32)], True, (160, 160, 160), th_raw_cnt, cv2.LINE_AA)
    elif raw_cnt is not None:
        cv2.drawContours(out, [raw_cnt], 0, (160, 160, 160), th_raw_cnt, cv2.LINE_AA)

    for i, e in enumerate(edges):
        if i in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            continue
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        pos = 0
        drawing = True
        while pos < dist:
            seg_end = min(pos + (dash_len if drawing else gap_len), dist)
            if drawing:
                sx, sy = int(p1[0] + pos * dx), int(p1[1] + pos * dy)
                ex, ey = int(p1[0] + seg_end * dx), int(p1[1] + seg_end * dy)
                cv2.line(out, (sx, sy), (ex, ey), (150, 150, 150), th_dash, cv2.LINE_AA)
            pos = seg_end
            drawing = not drawing

    for i, e in enumerate(edges):
        if i not in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        color_bgr = highlights[i]

        cv2.line(out, p1, p2, color_bgr, th_highlight, cv2.LINE_AA)
        cv2.circle(out, p1, max(1, th_highlight // 2), color_bgr, -1, cv2.LINE_AA)
        cv2.circle(out, p2, max(1, th_highlight // 2), color_bgr, -1, cv2.LINE_AA)

    badge_r_highlight = max(10, int(24 * d_scale))
    badge_r_normal = max(8, int(20 * d_scale))
    text_y_offset = int(6 * d_scale)

    for i, e in enumerate(edges):
        mid = tuple(e["mid"].astype(int))
        is_hi = i in highlights
        badge_color = highlights[i] if is_hi else (120, 120, 120)
        curr_badge_r = badge_r_highlight if is_hi else badge_r_normal

        cv2.circle(out, mid, curr_badge_r, (255, 255, 255), -1)
        curr_badge_th = th_badge_outline if is_hi else max(1, int(1 * d_scale))
        cv2.circle(out, mid, curr_badge_r, badge_color, curr_badge_th, cv2.LINE_AA)

        label = str(i + 1)
        curr_fs = fs_edge_hi if is_hi else fs_edge

        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, curr_fs, th_edge_text)[0][0]
        cv2.putText(out, label, (mid[0] - tw // 2, mid[1] + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, curr_fs, (0, 0, 0), th_edge_text, cv2.LINE_AA)

    return out


def draw_legend_cv(cv_img, legend_defs):
    if not legend_defs:
        return cv_img

    out = cv_img.copy()
    h_img, w_img = out.shape[:2]
    d_scale = get_drawing_scale(h_img, w_img)

    fs = max(0.4, 0.7 * d_scale)
    th = max(1, int(1.5 * d_scale))
    pad = max(10, int(20 * d_scale))
    swatch = max(12, int(25 * d_scale))
    gap = max(6, int(15 * d_scale))
    line_spacing = max(6, int(15 * d_scale))

    max_tw = 0
    max_th = 0
    for _, txt in legend_defs:
        (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        max_tw = max(max_tw, tw)
        max_th = max(max_th, th_txt)

    row_h = max(swatch, max_th)

    panel_w = pad * 2 + swatch + gap + max_tw
    panel_h = pad * 2 + (len(legend_defs) * row_h) + ((len(legend_defs) - 1) * line_spacing)

    x0, y0 = max(10, int(20 * d_scale)), max(10, int(20 * d_scale))

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)

    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (100, 100, 100), max(1, int(1 * d_scale)))

    curr_y = y0 + pad
    for bgr, txt in legend_defs:
        cv2.rectangle(out, (x0 + pad, curr_y), (x0 + pad + swatch, curr_y + swatch), bgr, -1)
        cv2.rectangle(out, (x0 + pad, curr_y), (x0 + pad + swatch, curr_y + swatch), (50, 50, 50),
                      max(1, int(1 * d_scale)))

        (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        text_y = curr_y + (swatch + th_txt) // 2
        cv2.putText(out, txt, (x0 + pad + swatch + gap, text_y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th,
                    cv2.LINE_AA)

        curr_y += row_h + line_spacing

    return out


def analyze_edge_straightness(cnt, edges, px_per_mm):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    for e in edges:
        e["dev_max_mm"] = 0.0
        e["dev_mean_mm"] = 0.0
        e["_raw_dists"] = []

    for pt in pts:
        min_dist = float('inf')
        best_edge_idx = -1
        for i, e in enumerate(edges):
            p1, p2 = e["p1"], e["p2"]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0: continue
            pt_vec = pt - p1
            t = np.dot(pt_vec, line_vec) / (line_len ** 2)
            t = max(0.0, min(1.0, t))
            closest_pt = p1 + t * line_vec
            dist_to_segment = np.linalg.norm(pt - closest_pt)
            if dist_to_segment < min_dist:
                min_dist = dist_to_segment
                best_edge_idx = i
        if best_edge_idx != -1:
            edges[best_edge_idx]["_raw_dists"].append(min_dist)

    for e in edges:
        dists = e["_raw_dists"]
        if dists:
            e["dev_max_mm"] = max(dists) / px_per_mm
            e["dev_mean_mm"] = np.mean(dists) / px_per_mm
        del e["_raw_dists"]
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


def angle_between_edges(edges, ia, ib):
    a1, a2 = edges[ia]["angle"], edges[ib]["angle"]
    diff = abs(a1 - a2)
    return round(min(diff, 180.0 - diff), 3)


def circularity_pct(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return round(min((4 * math.pi * area) / (peri ** 2), 1.0) * 100.0, 2)


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
    h_img, w_img = out.shape[:2]
    d_scale = get_drawing_scale(h_img, w_img)

    cx, cy = int(cm["cx_px"]), int(cm["cy_px"])
    r_fit_px = int(cm["r_fit_mm"] * px_per_mm)
    r_max_px = int(cm["r_max_mm"] * px_per_mm)
    r_min_px = int(cm["r_min_mm"] * px_per_mm)

    c_fit = (178, 114, 0)
    c_max = (0, 94, 213)
    c_min = (0, 158, 115)

    line_th = max(1, int(2 * d_scale))
    marker_size = max(5, int(15 * d_scale))

    for a in range(0, 360, 4):
        s, e = math.radians(a), math.radians(a + 2)
        cv2.line(out,
                 (cx + int(r_fit_px * math.cos(s)), cy + int(r_fit_px * math.sin(s))),
                 (cx + int(r_fit_px * math.cos(e)), cy + int(r_fit_px * math.sin(e))),
                 c_fit, line_th, cv2.LINE_AA)

    cv2.circle(out, (cx, cy), r_max_px, c_max, max(1, int(1 * d_scale)), cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_min_px, c_min, max(1, int(1 * d_scale)), cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, marker_size, max(1, int(1 * d_scale)))
    return out


# ─── Detection ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_objects(img_bytes, _cam_mtx, _dist, _calib_size, canny_low, canny_high,
                   min_area, max_obj, merge, merge_dist, detect_holes, det_method, invert_thresh, min_hole_area,
                   apply_undistort):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    scale_ratio = 1.0

    if _cam_mtx is not None and apply_undistort:
        h, w = img.shape[:2]
        scaled_mtx = _cam_mtx.copy()

        if _calib_size is not None:
            cal_w, cal_h = _calib_size
            scale_x = w / cal_w
            scale_y = h / cal_h
            scale_ratio = (scale_x + scale_y) / 2.0

            scaled_mtx[0, 0] *= scale_x
            scaled_mtx[1, 1] *= scale_y
            scaled_mtx[0, 2] *= scale_x
            scaled_mtx[1, 2] *= scale_y

        img = cv2.undistort(img, scaled_mtx, _dist, None, scaled_mtx)

        mask = np.ones((h, w), dtype=np.uint8) * 255
        mask = cv2.undistort(mask, scaled_mtx, _dist, None, scaled_mtx)

        kernel_mask = np.ones((9, 9), np.uint8)
        mask = cv2.erode(mask, kernel_mask, iterations=1)

        bg_color = (255, 255, 255) if invert_thresh else (0, 0, 0)
        img[mask < 255] = bg_color

    elif _cam_mtx is not None and not apply_undistort:
        if _calib_size is not None:
            h, w = img.shape[:2]
            cal_w, cal_h = _calib_size
            scale_x = w / cal_w
            scale_y = h / cal_h
            scale_ratio = (scale_x + scale_y) / 2.0

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_type = cv2.THRESH_BINARY_INV if invert_thresh else cv2.THRESH_BINARY

    if "Otsu" in det_method:
        _, processed = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
        if merge:
            k = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    else:  # Canny
        processed = cv2.Canny(blurred, canny_low, canny_high)
        if merge:
            k = max(5, merge_dist)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            processed = cv2.dilate(processed, kernel, iterations=1)
            processed = cv2.erode(processed, kernel, iterations=1)

    retrieval_mode = cv2.RETR_TREE if detect_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(processed.copy(), retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    parts_serialized = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        raw_parts = []
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] == -1:
                outer_area = cv2.contourArea(cnt)
                if outer_area > 0.98 * total_img_area:
                    continue
                if outer_area >= min_area:
                    holes = []
                    if detect_holes:
                        child_idx = hierarchy[i][2]
                        while child_idx != -1:
                            child_cnt = contours[child_idx]
                            child_area = cv2.contourArea(child_cnt)
                            if child_area < 0.90 * outer_area:
                                if child_area >= min_hole_area:
                                    holes.append(child_cnt)
                            else:
                                gc_idx = hierarchy[child_idx][2]
                                while gc_idx != -1:
                                    gc_cnt = contours[gc_idx]
                                    if cv2.contourArea(gc_cnt) >= min_hole_area:
                                        holes.append(gc_cnt)
                                    gc_idx = hierarchy[gc_idx][0]
                            child_idx = hierarchy[child_idx][0]
                    raw_parts.append({
                        "outer": cnt,
                        "holes": holes,
                        "area": outer_area
                    })

        raw_parts = sorted(raw_parts, key=lambda x: x["area"], reverse=True)
        if max_obj > 0:
            raw_parts = raw_parts[:max_obj]

        for part in raw_parts:
            holes_s = [(h.tobytes(), h.shape) for h in part["holes"]]
            parts_serialized.append({
                "outer_bytes": part["outer"].tobytes(),
                "outer_shape": part["outer"].shape,
                "holes": holes_s
            })

    d_scale_map = get_drawing_scale(img_h, img_w)
    map_th = max(1, int(1 * d_scale_map))
    clean_map = np.zeros_like(img)
    for p in deserialize_parts(parts_serialized):
        cv2.drawContours(clean_map, [p["outer"]], -1, (255, 255, 255), map_th, cv2.LINE_AA)
        if p["holes"]:
            cv2.drawContours(clean_map, p["holes"], -1, (255, 255, 255), map_th, cv2.LINE_AA)

    edge_vis_bytes = cv2.imencode(".png", clean_map)[1].tobytes()
    img_bytes_out = cv2.imencode(".png", img)[1].tobytes()
    return img_bytes_out, edge_vis_bytes, parts_serialized, scale_ratio


def deserialize_parts(parts_s):
    parts = []
    for p in parts_s:
        outer = np.frombuffer(p["outer_bytes"], dtype=np.int32).reshape(p["outer_shape"])
        holes = [np.frombuffer(hb, dtype=np.int32).reshape(hs) for hb, hs in p["holes"]]
        parts.append({"outer": outer, "holes": holes})
    return parts


def tol_row(label, measured, lo, hi, unit, extra=""):
    ok = lo <= measured <= hi
    icon = "[OK]" if ok else "[NOK]"
    m_str = f"{measured:.3f}".rstrip("0").rstrip(".")
    if m_str == "": m_str = "0"
    md = f"**{icon} {label}**: `{m_str} {unit}` ∈ `[{lo:.2f} – {hi:.2f} {unit}]` {extra}"
    data = {
        "Veličina": label, "Naměřeno": round(measured, 3), "Minimum": round(lo, 3),
        "Maximum": round(hi, 3), "Jednotka": unit, "Výsledek": "OK" if ok else "NOK"
    }
    return md, data


# --- ROZŠÍŘENÁ ANALÝZA OTVORŮ S OKAMŽITÝM VÝPISEM ---
def render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol, base_img, export_data_list):
    if len(holes) == 0:
        return False, export_data_list

    is_holes_nok = False
    rows_html_holes = []

    st.markdown("---")
    st.markdown(f"**Analýza vnitřních otvorů (Nalezeno: {len(holes)})**")

    hole_mode = st.segmented_control(
        "Typ analýzy otvorů", options=["circle", "polygon"], default="circle",
        format_func=lambda x: "Kruhové (Průměr, Kruhovitost)" if x == "circle" else "Hranaté (Detailní geometrie)",
        key=f"hole_mode_{obj_i}_{mode_choice}"
    )

    holes_data = []
    cfg = {}

    if hole_mode == "circle":
        for h_idx, hole_cnt in enumerate(holes):
            h_cm = circle_metrics(hole_cnt, px_per_mm)
            holes_data.append({
                "Otvor": f"#{h_idx + 1}",
                "Průměr (fit) [mm]": round(h_cm["r_fit_mm"] * 2, 3),
                "Kruhovitost [%]": circularity_pct(hole_cnt),
                "Radiální odchylka [mm]": round(h_cm["dev_mm"], 3)
            })
        st.dataframe(pd.DataFrame(holes_data), use_container_width=True)

        st.markdown("**Tolerance kruhových otvorů**")
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        cfg["h_en_c"] = hc1.checkbox("Průměr", key=f"h_en_{obj_i}_{mode_choice}_c")

        default_diam = float(holes_data[0]["Průměr (fit) [mm]"]) if holes_data else 10.0
        max_diam_limit = float(max(500.0, default_diam * 2.0))

        cfg["h_nom"] = hc2.number_input("Jmen. [mm]", 0.0, max_diam_limit, default_diam, step=0.1,
                                        key=f"h_nom_{obj_i}_{mode_choice}_c", disabled=not cfg["h_en_c"])
        cfg["h_tol"] = hc3.number_input("Tol. ± [mm]", 0.0, 50.0, 0.5, step=0.1, key=f"h_tol_{obj_i}_{mode_choice}_c",
                                        disabled=not cfg["h_en_c"])
        cfg["h_en_circ"] = hc4.checkbox("Kruhovitost", key=f"h_en_{obj_i}_{mode_choice}_circ")
        cfg["h_min_circ"] = hc5.number_input("Min. [%]", 0.0, 100.0, 85.0, step=1.0,
                                             key=f"h_min_circ_{obj_i}_{mode_choice}", disabled=not cfg["h_en_circ"])

        if cfg["h_en_c"] or cfg["h_en_circ"]:
            for h_idx, h_row in enumerate(holes_data):
                if cfg["h_en_c"]:
                    d_fit = h_row["Průměr (fit) [mm]"]
                    lo, hi = cfg["h_nom"] - cfg["h_tol"], cfg["h_nom"] + cfg["h_tol"]
                    ok = lo <= d_fit <= hi
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{h_idx + 1} průměr", d_fit, lo, hi, "mm",
                                           extra=f" — naměřeno **{d_fit:.3f} mm**")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

                if cfg["h_en_circ"]:
                    circ_val = h_row["Kruhovitost [%]"]
                    ok = circ_val >= cfg["h_min_circ"]
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{h_idx + 1} kruhovitost", circ_val, cfg["h_min_circ"], 100.0, "%")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

    else:  # POLYGON HOLES
        for h_idx, hole_cnt in enumerate(holes):
            h_edges, _ = get_edges(hole_cnt, angle_merge_tol=angle_merge_tol)
            hx, hy, hw, hh = cv2.boundingRect(hole_cnt)
            holes_data.append({
                "Otvor": f"#{h_idx + 1}",
                "Počet hran": len(h_edges),
                "Šířka X [mm]": round(hw / px_per_mm, 3),
                "Výška Y [mm]": round(hh / px_per_mm, 3),
                "Obvod [mm]": round(cv2.arcLength(hole_cnt, True) / px_per_mm, 3)
            })
        st.dataframe(pd.DataFrame(holes_data), use_container_width=True)

        for h_row in holes_data:
            otvor_nazev = h_row["Otvor"]
            for klic, hodnota in h_row.items():
                if klic == "Otvor": continue
                jednotka = "mm" if "[mm]" in klic else "%" if "[%]" in klic else ""
                export_data_list.append({
                    "Veličina": f"Díra {otvor_nazev}: {klic.split(' [')[0]}",
                    "Naměřeno": hodnota, "Minimum": None, "Maximum": None, "Jednotka": jednotka, "Výsledek": "Info"
                })

        st.markdown("---")
        st.markdown("**Detailní geometrické tolerance konkrétního otvoru**")
        sel_h_idx = st.selectbox("Vyberte otvor pro detailní analýzu", range(len(holes)),
                                 format_func=lambda x: f"Otvor #{x + 1}", key=f"sel_h_{obj_i}_{mode_choice}")

        sel_h_cnt = holes[sel_h_idx]
        sel_h_edges, sel_sharp_pts = get_edges(sel_h_cnt, angle_merge_tol=angle_merge_tol)
        n_h_edges = len(sel_h_edges)

        st.markdown(f"*Detekováno {n_h_edges} hran u Otvoru #{sel_h_idx + 1}. Čísla hran viz náhled.*")

        if n_h_edges >= 2:
            hx, hy, hw, hh = cv2.boundingRect(sel_h_cnt)

            pad = max(10, int(0.15 * max(hw, hh)))
            y1, y2 = max(0, hy - pad), min(base_img.shape[0], hy + hh + pad)
            x1, x2 = max(0, hx - pad), min(base_img.shape[1], hx + hw + pad)

            h_cropped_base = base_img[y1:y2, x1:x2].copy()

            crop_h, crop_w = h_cropped_base.shape[:2]
            target_dim = 800.0
            resize_ratio = target_dim / max(crop_w, crop_h)

            if resize_ratio > 1.0:
                h_cropped_base = cv2.resize(h_cropped_base, (int(crop_w * resize_ratio), int(crop_h * resize_ratio)),
                                            interpolation=cv2.INTER_CUBIC)
            else:
                resize_ratio = 1.0

            shifted_edges = []
            for e in sel_h_edges:
                shifted_edges.append({
                    "p1": (e["p1"] - np.array([x1, y1])) * resize_ratio,
                    "p2": (e["p2"] - np.array([x1, y1])) * resize_ratio,
                    "mid": (e["mid"] - np.array([x1, y1])) * resize_ratio,
                    "angle": e["angle"],
                    "length": e["length"]
                })

            shifted_sharp_pts = (sel_sharp_pts - np.array([x1, y1])) * resize_ratio
            shifted_sharp_pts = np.round(shifted_sharp_pts).astype(np.int32)

            h_cropped = draw_edge_map(h_cropped_base, shifted_edges, {}, sharp_pts=shifted_sharp_pts)

            hc_1, hc_2, hc_3 = st.columns([1, 1, 1])
            with hc_2:
                if h_cropped.size > 0:
                    st.image(cv_to_pil(h_cropped), caption=f"Detail hran Otvoru #{sel_h_idx + 1}",
                             use_container_width=True)

            h_nums = list(range(1, n_h_edges + 1))

            def h_edge_label(i):
                if i - 1 < 0 or i - 1 >= len(sel_h_edges): return f"h{i}"
                e = sel_h_edges[i - 1]
                return f"h{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"

            idx_a = 0
            idx_b = min(1, n_h_edges - 1)

            with st.container():
                st.markdown("**Rovnoběžnost uvnitř otvoru**")
                hpc = st.columns([1, 1, 1, 1])
                cfg["h_par_en"] = hpc[0].checkbox("Aktivní", key=f"h_par_en_{obj_i}")
                cfg["h_par_a"] = hpc[1].selectbox("Hrana A", h_nums, idx_a, key=f"h_par_a_{obj_i}",
                                                  disabled=not cfg["h_par_en"], format_func=h_edge_label) - 1
                cfg["h_par_b"] = hpc[2].selectbox("Hrana B", h_nums, idx_b, key=f"h_par_b_{obj_i}",
                                                  disabled=not cfg["h_par_en"], format_func=h_edge_label) - 1
                cfg["h_par_tol"] = hpc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 2.0, step=0.5,
                                                       key=f"h_par_tol_{obj_i}", disabled=not cfg["h_par_en"])

                if cfg["h_par_en"]:
                    ia, ib = cfg["h_par_a"], cfg["h_par_b"]
                    dev = parallelism_deg(sel_h_edges, ia, ib)
                    ok = dev <= cfg["h_par_tol"]
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{sel_h_idx + 1} Rovnoběžnost (h{ia + 1}|h{ib + 1})", dev, 0.0,
                                           cfg["h_par_tol"], "°")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Kolmost uvnitř otvoru**")
                hqc = st.columns([1, 1, 1, 1])
                cfg["h_perp_en"] = hqc[0].checkbox("Aktivní", key=f"h_perp_en_{obj_i}")
                cfg["h_perp_a"] = hqc[1].selectbox("Hrana A ", h_nums, idx_a, key=f"h_perp_a_{obj_i}",
                                                   disabled=not cfg["h_perp_en"], format_func=h_edge_label) - 1
                cfg["h_perp_b"] = hqc[2].selectbox("Hrana B ", h_nums, idx_b, key=f"h_perp_b_{obj_i}",
                                                   disabled=not cfg["h_perp_en"], format_func=h_edge_label) - 1
                cfg["h_perp_tol"] = hqc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 2.0, step=0.5,
                                                        key=f"h_perp_tol_{obj_i}", disabled=not cfg["h_perp_en"])

                if cfg["h_perp_en"]:
                    ia, ib = cfg["h_perp_a"], cfg["h_perp_b"]
                    dev = perpendicularity_deg(sel_h_edges, ia, ib)
                    ok = dev <= cfg["h_perp_tol"]
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{sel_h_idx + 1} Kolmost (h{ia + 1}|h{ib + 1})", dev, 0.0,
                                           cfg["h_perp_tol"], "°")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Libovolný úhel uvnitř otvoru**")
                hac = st.columns([1, 1, 1, 1, 1])
                cfg["h_ang_en"] = hac[0].checkbox("Aktivní", key=f"h_ang_en_{obj_i}")
                cfg["h_ang_a"] = hac[1].selectbox("Hrana A  ", h_nums, idx_a, key=f"h_ang_a_{obj_i}",
                                                  disabled=not cfg["h_ang_en"], format_func=h_edge_label) - 1
                cfg["h_ang_b"] = hac[2].selectbox("Hrana B  ", h_nums, idx_b, key=f"h_ang_b_{obj_i}",
                                                  disabled=not cfg["h_ang_en"], format_func=h_edge_label) - 1
                cfg["h_ang_nom"] = hac[3].number_input("Jmen. [°]", 0.0, 90.0, 45.0, step=0.5, key=f"h_ang_nom_{obj_i}",
                                                       disabled=not cfg["h_ang_en"])
                cfg["h_ang_tol"] = hac[4].number_input("Max. odch. [°]", 0.0, 45.0, 2.0, step=0.5,
                                                       key=f"h_ang_tol_{obj_i}", disabled=not cfg["h_ang_en"])

                if cfg["h_ang_en"]:
                    ia, ib = cfg["h_ang_a"], cfg["h_ang_b"]
                    act_ang = angle_between_edges(sel_h_edges, ia, ib)
                    lo, hi = cfg["h_ang_nom"] - cfg["h_ang_tol"], cfg["h_ang_nom"] + cfg["h_ang_tol"]
                    ok = lo <= act_ang <= hi
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{sel_h_idx + 1} Úhel (h{ia + 1}|h{ib + 1})", act_ang, lo, hi, "°")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Délka hrany otvoru**")
                hlc = st.columns([1, 1, 1, 1, 1])
                cfg["h_len_en"] = hlc[0].checkbox("Aktivní", key=f"h_len_en_{obj_i}")
                cfg["h_len_edge"] = hlc[1].selectbox("Vybrat hranu", h_nums, 0, key=f"h_len_e_{obj_i}",
                                                     disabled=not cfg["h_len_en"], format_func=h_edge_label) - 1
                cfg["h_len_nom"] = hlc[2].number_input("Jmenovitá [mm]", 0.0, 2000.0, 10.0, step=0.5,
                                                       key=f"h_len_nom_{obj_i}", disabled=not cfg["h_len_en"])
                cfg["h_len_plus"] = hlc[3].number_input("Tol. +", 0.0, 100.0, 1.0, step=0.1, key=f"h_len_p_{obj_i}",
                                                        disabled=not cfg["h_len_en"])
                cfg["h_len_minus"] = hlc[4].number_input("Tol. −", 0.0, 100.0, 1.0, step=0.1, key=f"h_len_m_{obj_i}",
                                                         disabled=not cfg["h_len_en"])

                if cfg["h_len_en"]:
                    ie = cfg["h_len_edge"]
                    if ie < len(sel_h_edges):
                        elen = sel_h_edges[ie]["length"] / px_per_mm
                        lo, hi = cfg["h_len_nom"] - cfg["h_len_minus"], cfg["h_len_nom"] + cfg["h_len_plus"]
                        ok = lo <= elen <= hi
                        if not ok: is_holes_nok = True
                        r_md, r_data = tol_row(f"Otvor #{sel_h_idx + 1} Délka hrany h{ie + 1}", elen, lo, hi, "mm")
                        rows_html_holes.append(r_md)
                        export_data_list.append(r_data)

    if rows_html_holes:
        st.markdown("---")
        st.markdown("**Výsledky tolerance otvorů**")
        for r in rows_html_holes: st.markdown(r)

    return is_holes_nok, export_data_list


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Rozměrová detekce")
st.markdown("Interaktivní analýza rozměrů a geometrických tolerancí s adaptivním měřítkem")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Nastavení")
    st.markdown("---")

    with st.expander("Kalibrace", expanded=True):
        st.caption("Nastavení px/mm a šachovnice")
        calib_file = st.file_uploader("Šachovnicový snímek", type=["jpg", "jpeg", "png", "bmp"], key="calib")
        c1, c2 = st.columns(2)
        with c1: cb_rows = st.number_input("Řádky rohů", 3, 40, 23)
        with c2: cb_cols = st.number_input("Sloupce rohů", 3, 50, 32)
        sq_mm = st.number_input("Čtverec [mm]", 0.1, 100.0, 5.0, step=0.1)
        st.markdown("---")
        apply_undistort = st.toggle("Korigovat zkreslení objektivu", value=True,
                                    help="Narovná sférické zkreslení čočky (Undistort) podle šachovnice.")

    with st.expander("Ruční px/mm", expanded=False):
        st.caption("Vlastní kalibrační hodnota")
        manual_ppm = st.number_input("Hodnota (0 = auto)", 0.0, 500.0, 0.0, step=0.1)

    with st.expander("Detekce", expanded=False):
        st.caption("Parametry segmentace a kontur")

        det_method = st.radio("Metoda segmentace", ["Prahování (Otsu) - plošné", "Hrany (Canny) - liniové"], index=0)
        invert_thresh = st.toggle("Tmavý objekt na světlém pozadí", value=True)

        st.markdown("---")
        canny_low = st.slider("Canny spodní", 0, 200, 50)
        canny_high = st.slider("Canny horní", 50, 500, 150)
        min_area = st.slider("Min. plocha [px]", 50, 50000, 5000)
        min_hole_area = st.slider("Min. plocha díry [px]", 10, 5000, 500)
        max_obj = st.slider("Max. objektů (0=vše)", 0, 20, 0)
        merge = st.toggle("Slučovat kontury", value=False)
        merge_dist = st.slider("Vzdálenost slučování [px]", 5, 100, 40, disabled=not merge)
        detect_holes = st.toggle("Detekovat vnitřní díry (otvory)", value=True)

    with st.expander("Detekce hran", expanded=False):
        st.caption("Parametry rozpoznání hran polygonu")
        angle_merge_tol = st.slider("Tolerance úhlu pro slučování hran [°]", 1, 20, 6)

# ─── Calibration ──────────────────────────────────────────────────────────────
cam_mtx, dist_c, px_per_mm = None, None, None
calib_size = None

if calib_file:
    with st.spinner("Probíhá kalibrace..."):
        calib_file.seek(0)
        res = calibrate(calib_file.read(), cb_rows, cb_cols, sq_mm)
        if len(res) == 6:
            cam_mtx, dist_c, px_per_mm_cal, debug_img, err, calib_size = res
        else:
            cam_mtx, dist_c, px_per_mm_cal, debug_img, err = res

    if err:
        st.error(err)
    else:
        px_per_mm = px_per_mm_cal
        st.success(f"Kalibrace úspěšná — {px_per_mm:.4f} px/mm")
        with st.expander("Kalibrační snímek", expanded=False):
            if debug_img is not None: st.image(cv_to_pil(debug_img), use_container_width=True)

if manual_ppm and manual_ppm > 0:
    px_per_mm = manual_ppm
    st.markdown(f'<div class="info-box">Ruční px/mm aktivní: {px_per_mm:.4f}</div>', unsafe_allow_html=True)

if not px_per_mm:
    px_per_mm = 5.0
    if not calib_file:
        st.markdown('<div class="info-box">Nahrajte šachovnici nebo zadejte vlastní px/mm. Výchozí: 5.0 px/mm</div>',
                    unsafe_allow_html=True)

# ─── Object upload ────────────────────────────────────────────────────────────
obj_file = st.file_uploader("Snímek měřeného objektu", type=["jpg", "jpeg", "png", "bmp"])

if not obj_file:
    st.info("Nahrajte snímek objektu ke zpracování.")
    st.stop()

obj_file.seek(0)
img_bytes = obj_file.read()
orig_pil = Image.open(io.BytesIO(img_bytes))

# ─── Detect ───────────────────────────────────────────────────────────────────
with st.spinner("Detekuji objekty..."):
    img_out_bytes, edge_vis_bytes, parts_s, scale_ratio = detect_objects(
        img_bytes, cam_mtx, dist_c, calib_size, canny_low, canny_high, min_area, max_obj, merge, merge_dist,
        detect_holes, det_method, invert_thresh, min_hole_area, apply_undistort
    )

# Ochrana ppm pro různé velikosti obrázků (kdy se liší fotka z telefonu oproti kalibraci)
if scale_ratio != 1.0 and not (manual_ppm and manual_ppm > 0):
    px_per_mm = px_per_mm * scale_ratio

base_arr = np.frombuffer(img_out_bytes, np.uint8)
edge_arr = np.frombuffer(edge_vis_bytes, np.uint8)
base_img = cv2.imdecode(base_arr, cv2.IMREAD_COLOR)
edge_img = cv2.imdecode(edge_arr, cv2.IMREAD_COLOR)
parts = deserialize_parts(parts_s)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Originál**")
    st.image(orig_pil, use_container_width=True)
with col_b:
    st.markdown("**Segmentační mapa**")
    if edge_img is not None: st.image(cv_to_pil(edge_img), use_container_width=True)

if not parts:
    st.warning("Žádné objekty nebyly nalezeny. Zkuste upravit segmentaci nebo min. plochu v postranním panelu.")
    st.stop()

st.markdown("---")
st.markdown(f"**Nalezeno:** {len(parts)} hlavních objektů")

HCOLORS = {
    "par_a": (0, 158, 230), "par_b": (0, 94, 213), "perp_a": (0, 158, 115), "perp_b": (167, 121, 204),
    "len_e": (0, 114, 178), "ang_a": (230, 158, 0), "ang_b": (213, 94, 0), "straight": (0, 0, 255),
}

# ─── Per-object tabs ──────────────────────────────────────────────────────────

# Proměnná spouštějící překreslení na konci skriptu
needs_rerun = False

for obj_i, part in enumerate(parts):
    cnt = part["outer"]
    holes = part["holes"]
    if cnt is None or cnt.size < 6: continue

    area_px = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if area_px == 0 or bw == 0 or bh == 0: continue

    rect = cv2.minAreaRect(cnt)
    (cx_r, cy_r), (rw_px, rh_px), angle = rect
    if px_per_mm == 0: px_per_mm = 1.0

    rw_mm = max(rw_px, rh_px) / px_per_mm
    rh_mm = min(rw_px, rh_px) / px_per_mm
    edges, sharp_pts = get_edges(cnt, angle_merge_tol=angle_merge_tol)
    n_edges = len(edges)

    # Zjištění předchozího stavu tolerance (aby byl viditelný ihned v nadpisu expanderu)
    prev_nok = st.session_state.get(f"is_nok_{obj_i}", None)
    if prev_nok is None:
        status_badge = "⏳ Počítám..."
    elif prev_nok:
        status_badge = "🔴 NOK"
    else:
        status_badge = "🟢 OK"

    # Z expanderu odebrána proměnná s tvarem a vložen status
    with st.expander(
            f"Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  Stav: {status_badge}",
            expanded=True):

        # Pevně nastaven defaultní tvar na polygon
        mode_choice = st.segmented_control(
            label="", label_visibility="collapsed", options=["polygon", "circle"],
            default="polygon",
            format_func=lambda x: {"polygon": "Polygon — výběr hran, rovnoběžnost, kolmost, úhly",
                                   "circle": "Kruh — průměr, radiální odchylka, kruhovitost"}[x],
            key=f"mode_{obj_i}"
        )

        st.markdown("---")

        if mode_choice == "circle":
            cm = circle_metrics(cnt, px_per_mm)
            is_nok = False
            export_data_list = []
            rows_html_main = []

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Průměr (fit)", f"{cm['r_fit_mm'] * 2:.3f} mm")
            m2.metric("R max", f"{cm['r_max_mm']:.3f} mm")
            m3.metric("R min", f"{cm['r_min_mm']:.3f} mm")
            m4.metric("Radiální odchylka", f"{cm['dev_mm']:.3f} mm")

            circ_vis = draw_circle_overlay(base_img, cm, px_per_mm)
            th_cnt = max(1, int(1 * get_drawing_scale(base_img.shape[0], base_img.shape[1])))
            cv2.drawContours(circ_vis, [cnt], 0, (180, 180, 180), th_cnt, cv2.LINE_AA)
            if holes: draw_holes_with_labels(circ_vis, holes, angle_merge_tol)

            c_v1, c_v2, c_v3 = st.columns([1, 1, 1])
            with c_v2:
                if circ_vis is not None: st.image(cv_to_pil(circ_vis), use_container_width=True)

            st.markdown("---")
            with st.expander("Radiální profil (úhel → poloměr)"):
                if cm["profile"]:
                    step = max(1, len(cm["profile"]) // 72)
                    rows_p = []
                    for ang, r in cm["profile"][::step]:
                        dev_r = r - cm["r_fit_mm"]
                        bar = ("+" if dev_r >= 0 else "-") * min(
                            int(abs(dev_r) / (cm["dev_mm"] if cm["dev_mm"] > 0 else 1.0) * 20), 20)
                        rows_p.append(f"| {ang:+.0f}° | {r:.3f} | {bar} | {dev_r:+.3f} mm |")
                    st.markdown("| Úhel | R [mm] | Odchylka | Δ [mm] |\n|---|---|---|---|\n" + "\n".join(rows_p))

            st.markdown("---")
            st.markdown("**Tolerance kruhu**")
            cfg = {}
            tc1, tc2, tc3 = st.columns(3)
            circ = circularity_pct(cnt)

            with tc1:
                cfg["circ_en"] = st.checkbox("Kruhovitost", value=True, key=f"circ_en_{obj_i}")
                cfg["circ_min"] = st.number_input("Min. [%]", 0.0, 100.0, 85.0, step=1.0, key=f"circ_min_{obj_i}",
                                                  disabled=not cfg["circ_en"])
                if cfg["circ_en"]:
                    ok = circ >= cfg["circ_min"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Kruhovitost", circ, cfg["circ_min"], 100.0, "%")
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            with tc2:
                cfg["diam_en"] = st.checkbox("Průměr", value=True, key=f"diam_en_{obj_i}_diam")

                d_fit = round(cm["r_fit_mm"] * 2, 2)
                max_diam_limit = float(max(2000.0, d_fit * 2.0))

                cfg["diam_nom"] = st.number_input("Jmenovitý [mm]", 0.0, max_diam_limit, d_fit, step=0.1,
                                                  key=f"diam_nom_{obj_i}", disabled=not cfg["diam_en"])
                cfg["diam_plus"] = st.number_input("Tol. +", 0.0, 100.0, 0.5, step=0.05, key=f"diam_plus_{obj_i}",
                                                   disabled=not cfg["diam_en"])
                cfg["diam_minus"] = st.number_input("Tol. −", 0.0, 100.0, 0.5, step=0.05, key=f"diam_minus_{obj_i}",
                                                    disabled=not cfg["diam_en"])
                if cfg["diam_en"]:
                    lo, hi = cfg["diam_nom"] - cfg["diam_minus"], cfg["diam_nom"] + cfg["diam_plus"]
                    ok = lo <= d_fit <= hi
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Průměr", d_fit, lo, hi, "mm")
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            with tc3:
                cfg["rad_en"] = st.checkbox("Radiální odchylka", value=True, key=f"rad_en_{obj_i}_rad")

                _rad_default = round(cm["dev_mm"] * 1.5 + 0.1, 2)
                _rad_max_limit = float(max(100.0, _rad_default * 2.0))

                cfg["rad_max"] = st.number_input("Max. [mm]", 0.0, _rad_max_limit, _rad_default, step=0.05,
                                                 key=f"rad_max_{obj_i}", disabled=not cfg["rad_en"])
                if cfg["rad_en"]:
                    ok = cm["dev_mm"] <= cfg["rad_max"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Radiální odchylka", cm["dev_mm"], 0.0, cfg["rad_max"], "mm")
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            if rows_html_main:
                st.markdown("---")
                for r in rows_html_main: st.markdown(r)

            is_holes_nok, export_data_list = render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol,
                                                                  base_img, export_data_list)
            if is_holes_nok: is_nok = True

            st.markdown("---")
            if is_nok:
                st.error("MIMO TOLERANCI | Alespoň jedna kontrola byla mimo toleranci.")
            else:
                st.success("V TOLERANCI | Všechny provedené kontroly jsou v tolerancích.")

            df_export = pd.DataFrame(export_data_list)
            img_buf = io.BytesIO()
            (cv_to_pil(circ_vis) if circ_vis is not None else orig_pil).save(img_buf, format="PNG")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(f"Stáhnout snímek", data=img_buf.getvalue(), file_name=f"obj{obj_i + 1}.png",
                                   mime="image/png", use_container_width=True)
            with dl_col2:
                st.download_button(f"Stáhnout CSV", data=df_export.to_csv(index=False).encode('utf-8-sig'),
                                   file_name=f"obj{obj_i + 1}_data.csv", mime="text/csv", use_container_width=True)

        else:  # polygon mode
            is_nok = False
            highlights = {}


            def assign_highlight(idx, color):
                highlights[idx] = color


            PALETTE = [(0, 158, 230), (0, 114, 178), (0, 158, 115), (10, 194, 213), (0, 94, 213), (167, 121, 204),
                       (0, 0, 0)]
            export_data_list, rows_html_main = [], []

            edges = analyze_edge_straightness(cnt, edges, px_per_mm)

            st.markdown("**Vizualizace nalezených hran**")
            preview_highlights = {i: PALETTE[i % len(PALETTE)] for i in range(n_edges)}

            # Předáváme jen ostré body pro dokonalý polygon v náhledu (hrubá šedá linka z findContours zmizí)
            edge_preview = draw_edge_map(base_img, edges, preview_highlights, sharp_pts=sharp_pts)
            th_rect = max(1, int(1 * get_drawing_scale(base_img.shape[0], base_img.shape[1])))
            cv2.rectangle(edge_preview, (x, y), (x + bw, y + bh), (150, 150, 150), th_rect)
            if holes: draw_holes_with_labels(edge_preview, holes, angle_merge_tol)

            preview_legend = [(PALETTE[i % len(PALETTE)],
                               f"H{i + 1}  {edges[i]['length'] / px_per_mm:.1f} mm  {edges[i]['angle']:.1f} deg") for i
                              in range(n_edges)]
            edge_preview_annotated = draw_legend_cv(edge_preview, preview_legend)

            c_p1, c_p2, c_p3 = st.columns([1, 1, 1])
            with c_p2:
                if edge_preview_annotated is not None: st.image(cv_to_pil(edge_preview_annotated),
                                                                use_container_width=True)

            edges_df_data = []
            for i, e in enumerate(edges):
                length_mm = e['length'] / px_per_mm
                edges_df_data.append(
                    {"Hrana": f"H{i + 1}", "Délka [mm]": round(length_mm, 3), "Úhel [°]": round(e['angle'], 1),
                     "Max. odchylka přímosti [mm]": round(e['dev_max_mm'], 3)})
                export_data_list.append(
                    {"Veličina": f"Délka hrany H{i + 1}", "Naměřeno": round(length_mm, 3), "Minimum": None,
                     "Maximum": None, "Jednotka": "mm", "Výsledek": "Info"})

            st.dataframe(pd.DataFrame(edges_df_data), use_container_width=True)
            st.markdown("---")
            st.markdown("**Konfigurace tolerancí hlavního objektu**")

            nums = list(range(1, n_edges + 1))


            def edge_label(i):
                if i - 1 < 0 or i - 1 >= len(edges): return f"H{i}"
                e = edges[i - 1]
                return f"H{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"


            cfg, idx_a, idx_b = {}, 0, min(1, n_edges - 1)

            with st.container():
                pc = st.columns([1, 1, 1, 1])
                cfg["par_en"] = pc[0].checkbox("Rovnoběžnost", value=True, key=f"par_en_{obj_i}")
                cfg["par_a"] = pc[1].selectbox("Hrana A", nums, idx_a, key=f"par_a_{obj_i}", disabled=not cfg["par_en"],
                                               format_func=edge_label) - 1
                cfg["par_b"] = pc[2].selectbox("Hrana B", nums, idx_b, key=f"par_b_{obj_i}", disabled=not cfg["par_en"],
                                               format_func=edge_label) - 1
                cfg["par_tol"] = pc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 2.0, step=0.5,
                                                    key=f"par_tol_{obj_i}", disabled=not cfg["par_en"])
                if cfg["par_en"]:
                    ia, ib = cfg["par_a"], cfg["par_b"]
                    dev = parallelism_deg(edges, ia, ib)
                    ok = dev <= cfg["par_tol"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Rovnoběžnost (H{ia + 1} | H{ib + 1})", dev, 0.0, cfg["par_tol"], "°")
                    assign_highlight(ia, HCOLORS["par_a"]);
                    assign_highlight(ib, HCOLORS["par_b"])
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            with st.container():
                qc = st.columns([1, 1, 1, 1])
                cfg["perp_en"] = qc[0].checkbox("Kolmost", value=True, key=f"perp_en_{obj_i}")
                cfg["perp_a"] = qc[1].selectbox("Hrana A ", nums, idx_a, key=f"perp_a_{obj_i}",
                                                disabled=not cfg["perp_en"], format_func=edge_label) - 1
                cfg["perp_b"] = qc[2].selectbox("Hrana B ", nums, idx_b, key=f"perp_b_{obj_i}",
                                                disabled=not cfg["perp_en"], format_func=edge_label) - 1
                cfg["perp_tol"] = qc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 2.0, step=0.5,
                                                     key=f"perp_tol_{obj_i}", disabled=not cfg["perp_en"])
                if cfg["perp_en"]:
                    ia, ib = cfg["perp_a"], cfg["perp_b"]
                    dev = perpendicularity_deg(edges, ia, ib)
                    ok = dev <= cfg["perp_tol"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Kolmost (H{ia + 1} | H{ib + 1})", dev, 0.0, cfg["perp_tol"], "°")
                    assign_highlight(ia, HCOLORS["perp_a"]);
                    assign_highlight(ib, HCOLORS["perp_b"])
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            with st.container():
                ac = st.columns([1, 1, 1, 1, 1])
                cfg["ang_en"] = ac[0].checkbox("Libovolný úhel", value=False, key=f"ang_en_{obj_i}")
                cfg["ang_a"] = ac[1].selectbox("Hrana A  ", nums, idx_a, key=f"ang_a_{obj_i}",
                                               disabled=not cfg["ang_en"], format_func=edge_label) - 1
                cfg["ang_b"] = ac[2].selectbox("Hrana B  ", nums, idx_b, key=f"ang_b_{obj_i}",
                                               disabled=not cfg["ang_en"], format_func=edge_label) - 1
                cfg["ang_nom"] = ac[3].number_input("Jmenovitý [°]", 0.0, 90.0, 45.0, step=0.5, key=f"ang_nom_{obj_i}",
                                                    disabled=not cfg["ang_en"])
                cfg["ang_tol"] = ac[4].number_input("Odchylka [°] ", 0.0, 45.0, 2.0, step=0.5, key=f"ang_tol_{obj_i}",
                                                    disabled=not cfg["ang_en"])
                if cfg["ang_en"]:
                    ia, ib = cfg["ang_a"], cfg["ang_b"]
                    act_ang = angle_between_edges(edges, ia, ib)
                    lo, hi = cfg["ang_nom"] - cfg["ang_tol"], cfg["ang_nom"] + cfg["ang_tol"]
                    ok = lo <= act_ang <= hi
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Úhel (H{ia + 1} | H{ib + 1})", act_ang, lo, hi, "°")
                    assign_highlight(ia, HCOLORS["ang_a"]);
                    assign_highlight(ib, HCOLORS["ang_b"])
                    rows_html_main.append(r_md);
                    export_data_list.append(r_data)

            with st.container():
                lc = st.columns([1, 1, 1, 1, 1])
                cfg["len_en"] = lc[0].checkbox("Délka hrany", value=False, key=f"len_en_{obj_i}")
                cfg["len_edge"] = lc[1].selectbox("Zkoumaná hrana", nums, 0, key=f"len_edge_{obj_i}",
                                                  disabled=not cfg["len_en"], format_func=edge_label) - 1

                _len_default = 10.0
                _len_max_limit = float(max(2000.0, _len_default * 2.0))

                cfg["len_nom"] = lc[2].number_input("Jmenovitá [mm]", 0.0, _len_max_limit, _len_default, step=0.5,
                                                    key=f"len_nom_{obj_i}", disabled=not cfg["len_en"])
                cfg["len_plus"] = lc[3].number_input("Tol. +", 0.0, 100.0, 1.0, step=0.1, key=f"len_plus_{obj_i}",
                                                     disabled=not cfg["len_en"])
                cfg["len_minus"] = lc[4].number_input("Tol. −", 0.0, 100.0, 1.0, step=0.1, key=f"len_minus_{obj_i}",
                                                      disabled=not cfg["len_en"])
                if cfg["len_en"]:
                    ie = cfg["len_edge"]
                    if ie < len(sel_h_edges):
                        elen = sel_h_edges[ie]["length"] / px_per_mm
                        lo, hi = cfg["len_nom"] - cfg["len_minus"], cfg["len_nom"] + cfg["len_plus"]
                        ok = lo <= elen <= hi
                        if not ok: is_holes_nok = True
                        r_md, r_data = tol_row(f"Délka H{ie + 1}", elen, lo, hi, "mm")
                        assign_highlight(ie, HCOLORS["len_e"])
                        rows_html_main.append(r_md);
                        export_data_list.append(r_data)

            with st.container():
                rc = st.columns([1, 1, 1, 1, 1])
                cfg["straight_en"] = rc[0].checkbox("Přímost (Roztřesenost)", value=False, key=f"straight_en_{obj_i}")
                cfg["straight_edge"] = rc[1].selectbox("Zkoumaná hrana ", nums, 0, key=f"straight_edge_{obj_i}",
                                                       disabled=not cfg["straight_en"], format_func=edge_label) - 1
                cfg["straight_max"] = rc[2].number_input("Max. odchylka [mm]", 0.0, 50.0, 0.5, step=0.1,
                                                         key=f"straight_max_{obj_i}", disabled=not cfg["straight_en"])
                if cfg["straight_en"]:
                    ie = cfg["straight_edge"]
                    if ie < n_edges:
                        dev = edges[ie]["dev_max_mm"]
                        ok = dev <= cfg["straight_max"]
                        if not ok: is_nok = True
                        r_md, r_data = tol_row(f"Přímost H{ie + 1}", dev, 0.0, cfg["straight_max"], "mm")
                        assign_highlight(ie, HCOLORS["straight"])
                        rows_html_main.append(r_md);
                        export_data_list.append(r_data)

            if rows_html_main:
                st.markdown("---")
                for r in rows_html_main: st.markdown(r)

            is_holes_nok, export_data_list = render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol,
                                                                  base_img, export_data_list)
            if is_holes_nok: is_nok = True

            st.markdown("---")
            st.markdown("**Vizualizace vybraných hran hlavního tvaru a děr**")

            annotated = draw_edge_map(base_img, edges, highlights, sharp_pts=sharp_pts)
            if holes: draw_holes_with_labels(annotated, holes, angle_merge_tol)

            legend_defs = []
            if cfg["par_en"]:
                ia, ib = cfg["par_a"], cfg["par_b"]
                legend_defs += [(HCOLORS["par_a"], f"H{ia + 1} rovnobeznost"),
                                (HCOLORS["par_b"], f"H{ib + 1} rovnobeznost")]
            if cfg["perp_en"]:
                ia, ib = cfg["perp_a"], cfg["perp_b"]
                legend_defs += [(HCOLORS["perp_a"], f"H{ia + 1} kolmost"), (HCOLORS["perp_b"], f"H{ib + 1} kolmost")]
            if cfg.get("ang_en", False):
                ia, ib = cfg["ang_a"], cfg["ang_b"]
                legend_defs += [(HCOLORS["ang_a"], f"H{ia + 1} uhel"), (HCOLORS["ang_b"], f"H{ib + 1} uhel")]
            if cfg.get("len_en", False) and cfg["len_edge"] < n_edges:
                legend_defs.append((HCOLORS["len_e"], f"H{cfg['len_edge'] + 1} delka"))
            if cfg.get("straight_en", False) and cfg["straight_edge"] < n_edges:
                legend_defs.append((HCOLORS["straight"], f"H{cfg['straight_edge'] + 1} primost"))

            annotated_legend = draw_legend_cv(annotated, legend_defs)

            c_a1, c_a2, c_a3 = st.columns([1, 1, 1])
            with c_a2:
                if annotated_legend is not None: st.image(cv_to_pil(annotated_legend), use_container_width=True)

            st.markdown("---")
            if is_nok:
                st.error("MIMO TOLERANCI")
            else:
                st.success("V TOLERANCI")

            df_export = pd.DataFrame(export_data_list)
            annotated_img = annotated_legend if annotated_legend is not None else annotated
            img_buf = io.BytesIO()
            cv_to_pil(annotated_img).save(img_buf, format="PNG")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(f"Stáhnout snímek", data=img_buf.getvalue(), file_name=f"obj{obj_i + 1}.png",
                                   mime="image/png", use_container_width=True)
            with dl_col2:
                st.download_button(f"Stáhnout CSV", data=df_export.to_csv(index=False).encode('utf-8-sig'),
                                   file_name=f"obj{obj_i + 1}_data.csv", mime="text/csv", use_container_width=True)

    if prev_nok != is_nok:
        st.session_state[f"is_nok_{obj_i}"] = is_nok
        needs_rerun = True

# Spuštění překreslení pokud se změní stav OK/NOK
if needs_rerun:
    st.rerun()