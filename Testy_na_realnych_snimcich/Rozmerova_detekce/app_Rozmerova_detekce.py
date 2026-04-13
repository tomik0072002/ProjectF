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

def draw_holes_with_labels(out, holes):
    """Vykreslí obrysy děr a označí je štítky (#1, #2...)."""
    cv2.drawContours(out, holes, -1, (0, 0, 255), 2, cv2.LINE_AA)
    for i, h_cnt in enumerate(holes):
        M = cv2.moments(h_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(h_cnt[0][0][0]), int(h_cnt[0][0][1])

        label = f"#{i + 1}"
        cv2.circle(out, (cx, cy), 12, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), 12, (0, 0, 255), 1, cv2.LINE_AA)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        cv2.putText(out, label, (cx - tw // 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


def get_edges(cnt, angle_merge_tol=6.0):
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
        merged = _merge_collinear_edges(raw_edges, angle_merge_tol)
        if len(merged) >= 3:
            raw_edges = merged

    return raw_edges


def _merge_collinear_edges(raw_edges, tol_deg=6.0):
    edges = list(raw_edges)
    changed = True
    max_iter = len(edges)
    itr = 0
    while changed and itr < max_iter:
        changed = False
        itr += 1
        new_edges = []
        used = [False] * len(edges)
        i = 0
        while i < len(edges):
            if used[i]:
                i += 1
                continue
            e = dict(edges[i])
            j = (i + 1) % len(edges)
            while j != i and not used[j]:
                diff = abs(edges[j]["angle"] - e["angle"])
                diff = min(diff, 180.0 - diff)
                if diff < tol_deg:
                    e["p2"] = edges[j]["p2"]
                    used[j] = True
                    changed = True
                    j = (j + 1) % len(edges)
                    vec = e["p2"] - e["p1"]
                    e["length"] = float(np.linalg.norm(vec))
                    e["angle"] = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
                    e["mid"] = ((e["p1"] + e["p2"]) / 2).astype(int)
                else:
                    break
            vec = e["p2"] - e["p1"]
            e["length"] = float(np.linalg.norm(vec))
            e["angle"] = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
            e["mid"] = ((e["p1"] + e["p2"]) / 2).astype(int)
            new_edges.append(e)
            used[i] = True
            i += 1
        edges = new_edges

    if len(edges) >= 2:
        first, last = edges[0], edges[-1]
        diff = abs(first["angle"] - last["angle"])
        diff = min(diff, 180.0 - diff)
        if diff < tol_deg:
            merged = dict(last)
            merged["p2"] = first["p2"]
            vec = merged["p2"] - merged["p1"]
            merged["length"] = float(np.linalg.norm(vec))
            merged["angle"] = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
            merged["mid"] = ((merged["p1"] + merged["p2"]) / 2).astype(int)
            edges = [merged] + edges[1:-1]

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


def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    coarse = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    n_coarse = len(coarse)

    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    aspect = min(rw, rh) / max(rw, rh) if max(rw, rh) > 0 else 1.0

    if circ >= 0.85 and aspect >= 0.88:
        if n_coarse >= 7:
            return "circle"
    if circ >= 0.70 and n_coarse >= 7:
        return "ellipse"
    if circ >= 0.90:
        return "circle"
    if circ >= 0.78:
        return "ellipse"
    return "polygon"


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

    c_fit = (178, 114, 0)
    c_max = (0, 94, 213)
    c_min = (0, 158, 115)

    for a in range(0, 360, 4):
        s, e = math.radians(a), math.radians(a + 2)
        cv2.line(out,
                 (cx + int(r_fit_px * math.cos(s)), cy + int(r_fit_px * math.sin(s))),
                 (cx + int(r_fit_px * math.cos(e)), cy + int(r_fit_px * math.sin(e))),
                 c_fit, 2, cv2.LINE_AA)

    cv2.circle(out, (cx, cy), r_max_px, c_max, 1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), r_min_px, c_min, 1, cv2.LINE_AA)
    cv2.drawMarker(out, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 15, 1)
    return out


# ─── Detection ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def detect_objects(img_bytes, _cam_mtx, _dist, canny_low, canny_high,
                   min_area, max_obj, merge, merge_dist, detect_holes, det_method, invert_thresh):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if _cam_mtx is not None:
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(_cam_mtx, _dist, (w, h), 1, (w, h))
        img = cv2.undistort(img, _cam_mtx, _dist, None, new_mtx)
        x, y, rw, rh = roi
        if all(v > 0 for v in (x, y, rw, rh)):
            img = img[y:y + rh, x:x + rw]

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if "Otsu" in det_method:
        thresh_type = cv2.THRESH_BINARY_INV if invert_thresh else cv2.THRESH_BINARY
        _, processed = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

        if merge:
            k = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    else:
        processed = cv2.Canny(blurred, canny_low, canny_high)
        k = max(5, merge_dist) if merge else 5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        if merge:
            processed = cv2.dilate(processed, kernel, iterations=2)
            processed = cv2.erode(processed, kernel, iterations=2)

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
                                if child_area >= 20:
                                    holes.append(child_cnt)
                            else:
                                gc_idx = hierarchy[child_idx][2]
                                while gc_idx != -1:
                                    gc_cnt = contours[gc_idx]
                                    if cv2.contourArea(gc_cnt) >= 20:
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

    edge_vis_bytes = cv2.imencode(".png", cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR))[1].tobytes()
    img_bytes_out = cv2.imencode(".png", img)[1].tobytes()
    return img_bytes_out, edge_vis_bytes, parts_serialized


def deserialize_parts(parts_s):
    parts = []
    for p in parts_s:
        outer = np.frombuffer(p["outer_bytes"], dtype=np.int32).reshape(p["outer_shape"])
        holes = [np.frombuffer(hb, dtype=np.int32).reshape(hs) for hb, hs in p["holes"]]
        parts.append({"outer": outer, "holes": holes})
    return parts


def draw_edge_map(base_img, edges, highlights, raw_cnt=None):
    out = base_img.copy()

    if raw_cnt is not None:
        cv2.drawContours(out, [raw_cnt], 0, (160, 160, 160), 1, cv2.LINE_AA)

    for i, e in enumerate(edges):
        if i in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        dist = int(np.linalg.norm(np.array(p2) - np.array(p1)))
        if dist == 0:
            continue

        dash_len, gap_len = 8, 8
        dx = (p2[0] - p1[0]) / dist
        dy = (p2[1] - p1[1]) / dist
        pos = 0
        drawing = True
        while pos < dist:
            seg_end = min(pos + (dash_len if drawing else gap_len), dist)
            if drawing:
                sx, sy = int(p1[0] + pos * dx), int(p1[1] + pos * dy)
                ex, ey = int(p1[0] + seg_end * dx), int(p1[1] + seg_end * dy)
                cv2.line(out, (sx, sy), (ex, ey), (150, 150, 150), 1, cv2.LINE_AA)
            pos = seg_end
            drawing = not drawing

    for i, e in enumerate(edges):
        if i not in highlights:
            continue
        p1 = tuple(e["p1"].astype(int))
        p2 = tuple(e["p2"].astype(int))
        color_bgr = highlights[i]
        cv2.line(out, p1, p2, color_bgr, 3, cv2.LINE_AA)

    for i, e in enumerate(edges):
        mid = tuple(e["mid"].astype(int))
        is_hi = i in highlights
        badge_color = highlights[i] if is_hi else (120, 120, 120)
        badge_r = 14 if is_hi else 12

        cv2.circle(out, mid, badge_r, (255, 255, 255), -1)
        cv2.circle(out, mid, badge_r, badge_color, 2 if is_hi else 1, cv2.LINE_AA)

        label = str(i + 1)
        fs = 0.5 if is_hi else 0.4
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
        cv2.putText(out, label, (mid[0] - tw // 2, mid[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)

    return out


def draw_legend_pil(cv_img, legend_defs):
    from PIL import ImageFont, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    font_size = 20
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    if not legend_defs:
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    pad = 12
    swatch = 18
    gap = 8
    line_h = font_size + 6
    panel_w = max(
        cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] + swatch + gap + pad * 2
        for _, txt in legend_defs
    ) + 60
    panel_h = len(legend_defs) * line_h + pad * 2 + 4

    x0, y0 = 10, 10
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rounded_rectangle(
        [x0, y0, x0 + panel_w, y0 + panel_h],
        radius=8,
        fill=(255, 255, 255, 230),
        outline=(100, 100, 100, 255),
        width=1,
    )
    pil = pil.convert("RGBA")
    pil = Image.alpha_composite(pil, overlay).convert("RGB")
    draw = ImageDraw.Draw(pil)

    for idx, (bgr, txt) in enumerate(legend_defs):
        rgb = (bgr[2], bgr[1], bgr[0])
        ty = y0 + pad + idx * line_h
        sx = x0 + pad
        draw.rectangle([sx, ty + 2, sx + swatch, ty + 2 + swatch - 4], fill=rgb)
        draw.text((sx + swatch + gap, ty), txt, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def tol_row(label, measured, lo, hi, unit, extra="", skip=False):
    if skip:
        md = f"**{label}** — přeskočeno"
        data = {
            "Veličina": label,
            "Naměřeno": None,
            "Minimum": None,
            "Maximum": None,
            "Jednotka": unit,
            "Výsledek": "Přeskočeno"
        }
        return md, data

    ok = lo <= measured <= hi
    icon = "[OK]" if ok else "[NOK]"
    m_str = f"{measured:.3f}".rstrip("0").rstrip(".")
    if m_str == "": m_str = "0"

    md = f"**{icon} {label}**: `{m_str} {unit}` ∈ `[{lo:.2f} – {hi:.2f} {unit}]` {extra}"
    data = {
        "Veličina": label,
        "Naměřeno": round(measured, 3),
        "Minimum": round(lo, 3),
        "Maximum": round(hi, 3),
        "Jednotka": unit,
        "Výsledek": "OK" if ok else "NOK"
    }
    return md, data


# --- ROZŠÍŘENÁ ANALÝZA OTVORŮ ---
def render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol, base_img, export_data_list):
    if len(holes) == 0:
        return False, export_data_list

    is_holes_nok = False
    rows_html_holes = []  # Lokální seznam pro výsledky otvorů

    st.markdown("---")
    st.markdown(f"**Analýza otvorů (Nalezeno: {len(holes)})**")

    hole_mode = st.segmented_control(
        "Typ analýzy otvorů",
        options=["circle", "polygon"],
        default="circle",
        format_func=lambda x: "Kruhový otvor (Průměr, Kruhovitost)" if x == "circle" else "Hranatý otvor (Detailní geometrie)",
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
        cfg["h_nom"] = hc2.number_input("Jmen. [mm]", 0.0, 500.0, float(holes_data[0]["Průměr (fit) [mm]"]), step=0.1,
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
            h_edges = get_edges(hole_cnt, angle_merge_tol=angle_merge_tol)
            hx, hy, hw, hh = cv2.boundingRect(hole_cnt)
            holes_data.append({
                "Otvor": f"#{h_idx + 1}",
                "Počet hran": len(h_edges),
                "Šířka X [mm]": round(hw / px_per_mm, 3),
                "Výška Y [mm]": round(hh / px_per_mm, 3),
                "Obvod [mm]": round(cv2.arcLength(hole_cnt, True) / px_per_mm, 3)
            })
        st.dataframe(pd.DataFrame(holes_data), use_container_width=True)

        st.markdown("---")
        st.markdown("**Detailní geometrické tolerance konkrétního otvoru**")
        sel_h_idx = st.selectbox("Vyberte otvor pro detailní analýzu", range(len(holes)),
                                 format_func=lambda x: f"Otvor #{x + 1}", key=f"sel_h_{obj_i}_{mode_choice}")

        sel_h_cnt = holes[sel_h_idx]
        sel_h_edges = get_edges(sel_h_cnt, angle_merge_tol=angle_merge_tol)
        n_h_edges = len(sel_h_edges)

        st.markdown(f"*Detekováno {n_h_edges} hran u Otvoru #{sel_h_idx + 1}. Čísla hran viz náhled.*")

        if n_h_edges >= 2:
            h_preview = draw_edge_map(base_img.copy(), sel_h_edges, {}, raw_cnt=sel_h_cnt)
            hx, hy, hw, hh = cv2.boundingRect(sel_h_cnt)
            pad = 30
            y1, y2 = max(0, hy - pad), min(base_img.shape[0], hy + hh + pad)
            x1, x2 = max(0, hx - pad), min(base_img.shape[1], hx + hw + pad)
            h_cropped = h_preview[y1:y2, x1:x2]

            hc_1, hc_2, hc_3 = st.columns([1, 2, 1])
            with hc_2:
                st.image(cv_to_pil(h_cropped), caption=f"Detail hran Otvoru #{sel_h_idx + 1}", use_container_width=True)

            h_nums = list(range(1, n_h_edges + 1))

            def h_edge_label(i):
                e = sel_h_edges[i - 1]
                return f"h{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"

            with st.container():
                st.markdown("**Rovnoběžnost uvnitř otvoru**")
                hpc = st.columns([1, 1, 1, 1])
                cfg["h_par_en"] = hpc[0].checkbox("Aktivní", key=f"h_par_en_{obj_i}")
                cfg["h_par_a"] = hpc[1].selectbox("Hrana A", h_nums, 0, key=f"h_par_a_{obj_i}",
                                                  disabled=not cfg["h_par_en"], format_func=h_edge_label) - 1
                cfg["h_par_b"] = hpc[2].selectbox("Hrana B", h_nums, min(1, n_h_edges - 1), key=f"h_par_b_{obj_i}",
                                                  disabled=not cfg["h_par_en"], format_func=h_edge_label) - 1
                cfg["h_par_tol"] = hpc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 5.0, step=0.5,
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
                cfg["h_perp_a"] = hqc[1].selectbox("Hrana A ", h_nums, 0, key=f"h_perp_a_{obj_i}",
                                                   disabled=not cfg["h_perp_en"], format_func=h_edge_label) - 1
                cfg["h_perp_b"] = hqc[2].selectbox("Hrana B ", h_nums, min(1, n_h_edges - 1), key=f"h_perp_b_{obj_i}",
                                                   disabled=not cfg["h_perp_en"], format_func=h_edge_label) - 1
                cfg["h_perp_tol"] = hqc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 5.0, step=0.5,
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
                cfg["h_ang_a"] = hac[1].selectbox("Hrana A  ", h_nums, 0, key=f"h_ang_a_{obj_i}",
                                                  disabled=not cfg["h_ang_en"], format_func=h_edge_label) - 1
                cfg["h_ang_b"] = hac[2].selectbox("Hrana B  ", h_nums, min(1, n_h_edges - 1), key=f"h_ang_b_{obj_i}",
                                                  disabled=not cfg["h_ang_en"], format_func=h_edge_label) - 1
                cfg["h_ang_nom"] = hac[3].number_input("Jmen. [°]", 0.0, 90.0, 45.0, step=0.5, key=f"h_ang_nom_{obj_i}",
                                                       disabled=not cfg["h_ang_en"])
                cfg["h_ang_tol"] = hac[4].number_input("Max. odch. [°]", 0.0, 45.0, 2.0, step=0.5,
                                                       key=f"h_ang_tol_{obj_i}", disabled=not cfg["h_ang_en"])

                if cfg["h_ang_en"]:
                    ia, ib = cfg["h_ang_a"], cfg["h_ang_b"]
                    act_ang = angle_between_edges(sel_h_edges, ia, ib)
                    lo = cfg["h_ang_nom"] - cfg["h_ang_tol"]
                    hi = cfg["h_ang_nom"] + cfg["h_ang_tol"]
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
                    elen = sel_h_edges[ie]["length"] / px_per_mm
                    lo = cfg["h_len_nom"] - cfg["h_len_minus"]
                    hi = cfg["h_len_nom"] + cfg["h_len_plus"]
                    ok = lo <= elen <= hi
                    if not ok: is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{sel_h_idx + 1} Délka hrany h{ie + 1}", elen, lo, hi, "mm")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)

    # Výpis výsledků otvorů najednou pod konfigurací
    if rows_html_holes:
        st.markdown("---")
        st.markdown("**Výsledky tolerance otvorů**")
        for r in rows_html_holes:
            st.markdown(r)

    return is_holes_nok, export_data_list


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("Rozměrová detekce")
st.markdown("Interaktivní analýza rozměrů a geometrických tolerancí")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Nastavení")
    st.markdown("---")

    with st.expander("Kalibrace", expanded=True):
        st.caption("Nastavení px/mm a šachovnice")
        calib_file = st.file_uploader("Šachovnicový snímek",
                                      type=["jpg", "jpeg", "png", "bmp"], key="calib")
        c1, c2 = st.columns(2)
        with c1:
            cb_rows = st.number_input("Řádky rohů", 3, 40, 23)
        with c2:
            cb_cols = st.number_input("Sloupce rohů", 3, 50, 32)
        sq_mm = st.number_input("Čtverec [mm]", 0.1, 100.0, 5.0, step=0.1)

    with st.expander("Ruční px/mm", expanded=False):
        st.caption("Vlastní kalibrační hodnota")
        manual_ppm = st.number_input("Hodnota (0 = auto)", 0.0, 500.0, 0.0, step=0.1)

    with st.expander("Detekce", expanded=False):
        st.caption("Parametry segmentace a kontur")
        det_method = st.radio("Metoda segmentace", ["Prahování (Otsu) - plošné", "Hrany (Canny) - liniové"])
        invert_thresh = st.toggle("Tmavý objekt na světlém pozadí", value=True)
        st.markdown("---")
        canny_low = st.slider("Canny spodní", 0, 200, 50)
        canny_high = st.slider("Canny horní", 50, 500, 150)
        min_area = st.slider("Min. plocha [px]", 50, 5000, 500)
        max_obj = st.slider("Max. objektů (0=vše)", 0, 20, 1)
        merge = st.toggle("Slučovat kontury", value=False)
        merge_dist = st.slider("Vzdálenost slučování [px]", 5, 100, 40, disabled=not merge)
        detect_holes = st.toggle("Detekovat vnitřní díry (otvory)", value=False)

    with st.expander("Detekce hran", expanded=False):
        st.caption("Parametry rozpoznání hran polygonu")
        angle_merge_tol = st.slider(
            "Tolerance úhlu pro slučování hran [°]", 1, 20, 6,
            help="Hrany jejichž úhel se liší méně než tato hodnota budou sloučeny."
        )

# ─── Calibration ──────────────────────────────────────────────────────────────
cam_mtx, dist_c, px_per_mm = None, None, None

if calib_file:
    with st.spinner("Probíhá kalibrace..."):
        cam_mtx, dist_c, px_per_mm_cal, debug_img, err = calibrate(
            calib_file.read(), cb_rows, cb_cols, sq_mm)
    if err:
        st.error(err)
    else:
        px_per_mm = px_per_mm_cal
        st.success(f"Kalibrace úspěšná — {px_per_mm:.4f} px/mm")
        with st.expander("Kalibrační snímek", expanded=False):
            st.image(cv_to_pil(debug_img), use_container_width=True)

if manual_ppm and manual_ppm > 0:
    px_per_mm = manual_ppm
    st.markdown(
        f'<div class="info-box">Ruční px/mm aktivní: {px_per_mm:.4f}</div>',
        unsafe_allow_html=True
    )

if not px_per_mm:
    px_per_mm = 5.0
    if not calib_file:
        st.markdown(
            '<div class="info-box">Nahrajte šachovnici nebo zadejte vlastní px/mm. Aktuální výchozí: 5.0 px/mm</div>',
            unsafe_allow_html=True
        )

# ─── Object upload ────────────────────────────────────────────────────────────
obj_file = st.file_uploader("Snímek měřeného objektu", type=["jpg", "jpeg", "png", "bmp"])

if not obj_file:
    st.info("Nahrajte snímek objektu ke zpracování.")
    st.stop()

img_bytes = obj_file.read()
orig_pil = Image.open(io.BytesIO(img_bytes))

# ─── Detect ───────────────────────────────────────────────────────────────────
with st.spinner("Detekuji objekty..."):
    img_out_bytes, edge_vis_bytes, parts_s = detect_objects(
        img_bytes, cam_mtx, dist_c,
        canny_low, canny_high, min_area, max_obj, merge, merge_dist, detect_holes,
        det_method, invert_thresh)

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
    st.image(cv_to_pil(edge_img), use_container_width=True)

if not parts:
    st.warning("Žádné objekty nebyly nalezeny. Zkuste upravit segmentaci nebo min. plochu v postranním panelu.")
    st.stop()

st.markdown("---")
st.markdown(f"**Nalezeno:** {len(parts)} hlavních objektů")

HCOLORS = {
    "par_a": (0, 158, 230),
    "par_b": (0, 94, 213),
    "perp_a": (0, 158, 115),
    "perp_b": (167, 121, 204),
    "len_e": (0, 114, 178),
    "ang_a": (230, 158, 0),
    "ang_b": (213, 94, 0),
}

# ─── Per-object tabs ──────────────────────────────────────────────────────────
for obj_i, part in enumerate(parts):
    cnt = part["outer"]
    holes = part["holes"]
    n_holes = len(holes)

    area_px = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    (cx_r, cy_r), (rw_px, rh_px), angle = rect
    rw_mm = max(rw_px, rh_px) / px_per_mm
    rh_mm = min(rw_px, rh_px) / px_per_mm
    area_mm2 = area_px / (px_per_mm ** 2)
    peri_mm = cv2.arcLength(cnt, True) / px_per_mm
    circ = circularity_pct(cnt)
    shape = classify_shape(cnt)
    edges = get_edges(cnt, angle_merge_tol=angle_merge_tol)
    n_edges = len(edges)

    shape_labels = {"circle": "Kruh", "ellipse": "Elipsa / Ovál", "polygon": "Polygon"}

    with st.expander(
            f"Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  detekce: {shape_labels[shape]}",
            expanded=True,
    ):
        st.markdown("**Typ hlavního objektu**")
        mode_choice = st.segmented_control(
            label="",
            label_visibility="collapsed",
            options=["polygon", "circle"],
            default="polygon" if shape == "polygon" else "circle",
            format_func=lambda x: {
                "polygon": "Polygon — výběr hran, rovnoběžnost, kolmost, úhly",
                "circle": "Kruh — průměr, radiální odchylka, kruhovitost",
            }[x],
            key=f"mode_{obj_i}"
        )

        st.markdown("---")

        if mode_choice == "circle":
            cm = circle_metrics(cnt, px_per_mm)
            is_nok = False
            export_data_list = []
            rows_html_main = []  # Ukládání výsledků pro kruh

            st.markdown("**Přesné kruhové metriky**")
            st.markdown(
                '<div class="info-box">Hodnoty jsou počítány přímo z raw kontury. Fitovaný kruh = metoda nejmenších čtverců. Radiální odchylka = rozdíl max a min poloměru od středu.</div>',
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Průměr (fit)", f"{cm['r_fit_mm'] * 2:.3f} mm")
            m2.metric("R max", f"{cm['r_max_mm']:.3f} mm")
            m3.metric("R min", f"{cm['r_min_mm']:.3f} mm")
            m4.metric("Radiální odchylka", f"{cm['dev_mm']:.3f} mm")

            st.markdown("**Vizualizace — fitovaný kruh**")
            circ_vis = draw_circle_overlay(base_img, cm, px_per_mm)
            cv2.drawContours(circ_vis, [cnt], 0, (180, 180, 180), 1, cv2.LINE_AA)
            if holes:
                draw_holes_with_labels(circ_vis, holes)

            c_v1, c_v2, c_v3 = st.columns([1, 2, 1])
            with c_v2:
                st.image(cv_to_pil(circ_vis), use_container_width=True)

            st.markdown("---")

            with st.expander("Radiální profil (úhel → poloměr)"):
                profile = cm["profile"]
                if profile:
                    step = max(1, len(profile) // 72)
                    sampled = profile[::step]
                    r_fit = cm["r_fit_mm"]
                    rows_p = []
                    for ang, r in sampled:
                        dev_r = r - r_fit
                        bar_len = int(abs(dev_r) / cm["dev_mm"] * 20) if cm["dev_mm"] > 0 else 0
                        bar_len = min(bar_len, 20)
                        bar = ("+" if dev_r >= 0 else "-") * bar_len
                        rows_p.append(
                            f"| {ang:+.0f}° | {r:.3f} | {bar} | {dev_r:+.3f} mm |"
                        )
                    st.markdown("| Úhel | R [mm] | Odchylka | Δ [mm] |\n|---|---|---|---|\n" + "\n".join(rows_p))

            st.markdown("---")
            st.markdown("**Tolerance kruhu**")
            cfg = {}
            tc1, tc2, tc3 = st.columns(3)

            with tc1:
                st.markdown("**Kruhovitost**")
                cfg["circ_en"] = st.checkbox("Aktivní", value=True, key=f"circ_en_{obj_i}")
                cfg["circ_min"] = st.number_input("Min. [%]", 0.0, 100.0, 85.0, step=1.0,
                                                  key=f"circ_min_{obj_i}",
                                                  disabled=not cfg["circ_en"])
                if cfg["circ_en"]:
                    ok = circ >= cfg["circ_min"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Kruhovitost", circ, cfg["circ_min"], 100.0, "%")
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Kruhovitost", 0, 0, 100, "%", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with tc2:
                st.markdown("**Průměr (fitovaný)**")
                cfg["diam_en"] = st.checkbox("Aktivní", value=True, key=f"diam_en_{obj_i}_diam")
                cfg["diam_nom"] = st.number_input("Jmenovitý průměr [mm]", 0.0, 2000.0,
                                                  round(cm["r_fit_mm"] * 2, 2), step=0.1,
                                                  key=f"diam_nom_{obj_i}",
                                                  disabled=not cfg["diam_en"])
                cfg["diam_plus"] = st.number_input("Tol. + [mm]", 0.0, 100.0, 0.5, step=0.05,
                                                   key=f"diam_plus_{obj_i}",
                                                   disabled=not cfg["diam_en"])
                cfg["diam_minus"] = st.number_input("Tol. − [mm]", 0.0, 100.0, 0.5, step=0.05,
                                                    key=f"diam_minus_{obj_i}",
                                                    disabled=not cfg["diam_en"])
                if cfg["diam_en"]:
                    d_fit = cm["r_fit_mm"] * 2
                    lo = cfg["diam_nom"] - cfg["diam_minus"]
                    hi = cfg["diam_nom"] + cfg["diam_plus"]
                    ok = lo <= d_fit <= hi
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Průměr (fitovaný)", d_fit, lo, hi, "mm")
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Průměr", 0, 0, 0, "mm", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with tc3:
                st.markdown("**Radiální odchylka**")
                cfg["rad_en"] = st.checkbox("Aktivní", value=True, key=f"rad_en_{obj_i}_rad")
                _rad_default = round(cm["dev_mm"] * 1.5 + 0.1, 2)
                _rad_max_limit = max(100.0, _rad_default * 2)
                cfg["rad_max"] = st.number_input("Max. odchylka [mm]", 0.0, _rad_max_limit,
                                                 _rad_default, step=0.05,
                                                 key=f"rad_max_{obj_i}",
                                                 disabled=not cfg["rad_en"])
                if cfg["rad_en"]:
                    ok = cm["dev_mm"] <= cfg["rad_max"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row("Radiální odchylka", cm["dev_mm"], 0.0, cfg["rad_max"], "mm")
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Radiální odchylka", 0, 0, 0, "mm", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            # Výpis výsledků pro kruh
            st.markdown("---")
            st.markdown("**Výsledky tolerance hlavního tvaru**")
            for r in rows_html_main:
                st.markdown(r)

            # Otvory
            is_holes_nok, export_data_list = render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol,
                                                                  base_img, export_data_list)
            if is_holes_nok: is_nok = True

            st.markdown("---")
            st.markdown("**Výsledky tolerance celého objektu**")
            if is_nok:
                st.error(f"DÍL MIMO TOLERANCI | Alespoň jedna kontrola hlavního tvaru nebo děr selhala.")
            else:
                st.success(f"DÍL V TOLERANCI | Všechny provedené kontroly hlavního tvaru a děr prošly.")

            df_export = pd.DataFrame(export_data_list)
            csv_data = df_export.to_csv(index=False).encode('utf-8-sig')

            img_buf = io.BytesIO()
            cv_to_pil(circ_vis).save(img_buf, format="PNG")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(f"Stáhnout snímek (Objekt #{obj_i + 1})",
                                   data=img_buf.getvalue(),
                                   file_name=f"viziometer_obj{obj_i + 1}.png",
                                   mime="image/png", key=f"img_dl_{obj_i}", use_container_width=True)
            with dl_col2:
                st.download_button(f"Stáhnout tabulku s daty (CSV)",
                                   data=csv_data,
                                   file_name=f"viziometer_obj{obj_i + 1}_data.csv",
                                   mime="text/csv", key=f"csv_dl_{obj_i}", use_container_width=True)

        else:  # polygon mode
            is_nok = False
            highlights = {}
            highlight_conflicts = set()


            def assign_highlight(idx, color):
                if idx in highlights and highlights[idx] != color:
                    highlight_conflicts.add(idx)
                highlights[idx] = color


            PALETTE = [
                (0, 158, 230), (0, 114, 178), (0, 158, 115),
                (10, 194, 213), (0, 94, 213), (167, 121, 204), (0, 0, 0)
            ]

            export_data_list = []
            rows_html_main = []  # Ukládání výsledků pro polygon

            st.markdown("**Vizualizace nalezených hran**")
            st.markdown(
                f'<div class="info-box">Detekováno hran: {n_edges}. Čísla hran viz náhled.*',
                unsafe_allow_html=True
            )

            preview_highlights = {i: PALETTE[i % len(PALETTE)] for i in range(n_edges)}
            edge_preview = draw_edge_map(base_img, edges, preview_highlights, raw_cnt=cnt)
            cv2.rectangle(edge_preview, (x, y), (x + bw, y + bh), (150, 150, 150), 1)
            if holes:
                draw_holes_with_labels(edge_preview, holes)

            preview_legend = [
                (PALETTE[i % len(PALETTE)],
                 f"H{i + 1}  {edges[i]['length'] / px_per_mm:.1f} mm  {edges[i]['angle']:.1f} deg")
                for i in range(n_edges)
            ]
            edge_preview = draw_legend_pil(edge_preview, preview_legend)

            c_p1, c_p2, c_p3 = st.columns([1, 2, 1])
            with c_p2:
                st.image(cv_to_pil(edge_preview), use_container_width=True)

            st.markdown("**Seznam všech hran**")
            edges_df_data = []
            for i, e in enumerate(edges):
                length_mm = e['length'] / px_per_mm
                edges_df_data.append({
                    "Hrana": f"H{i + 1}",
                    "Délka [mm]": round(length_mm, 3),
                    "Úhel [°]": round(e['angle'], 1)
                })
                export_data_list.append({
                    "Veličina": f"Délka hrany H{i + 1}",
                    "Naměřeno": round(length_mm, 3),
                    "Minimum": None,
                    "Maximum": None,
                    "Jednotka": "mm",
                    "Výsledek": "Info"
                })

            st.dataframe(pd.DataFrame(edges_df_data), use_container_width=True)
            st.markdown("---")
            st.markdown("**Konfigurace tolerancí hlavního objektu**")

            nums = list(range(1, n_edges + 1))


            def edge_label(i):
                e = edges[i - 1]
                return f"H{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"


            cfg = {}

            with st.container():
                st.markdown("**Rovnoběžnost dvou hran hlavního tvaru**")
                pc = st.columns([1, 1, 1, 1])
                cfg["par_en"] = pc[0].checkbox("Aktivní", value=True, key=f"par_en_{obj_i}")
                cfg["par_a"] = pc[1].selectbox("Hrana A", nums, 0, key=f"par_a_{obj_i}",
                                               disabled=not cfg["par_en"],
                                               format_func=edge_label) - 1
                cfg["par_b"] = pc[2].selectbox("Hrana B", nums, min(1, n_edges - 1),
                                               key=f"par_b_{obj_i}",
                                               disabled=not cfg["par_en"],
                                               format_func=edge_label) - 1
                cfg["par_tol"] = pc[3].number_input("Max. odchylka [°]", 0.0, 90.0, 5.0,
                                                    step=0.5, key=f"par_tol_{obj_i}",
                                                    disabled=not cfg["par_en"])
                if cfg["par_en"]:
                    ia, ib = cfg["par_a"], cfg["par_b"]
                    dev = parallelism_deg(edges, ia, ib)
                    ok = dev <= cfg["par_tol"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Rovnoběžnost (odchylka) (H{ia + 1} | H{ib + 1})", dev, 0.0, cfg["par_tol"],
                                           "°")
                    assign_highlight(ia, HCOLORS["par_a"])
                    assign_highlight(ib, HCOLORS["par_b"])
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Rovnoběžnost (odchylka)", 0, 0, 0, "°", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Kolmost dvou hran hlavního tvaru**")
                qc = st.columns([1, 1, 1, 1])
                cfg["perp_en"] = qc[0].checkbox("Aktivní", value=True, key=f"perp_en_{obj_i}")
                cfg["perp_a"] = qc[1].selectbox("Hrana A", nums, 0, key=f"perp_a_{obj_i}",
                                                disabled=not cfg["perp_en"],
                                                format_func=edge_label) - 1
                cfg["perp_b"] = qc[2].selectbox("Hrana B", nums, min(1, n_edges - 1),
                                                key=f"perp_b_{obj_i}",
                                                disabled=not cfg["perp_en"],
                                                format_func=edge_label) - 1
                cfg["perp_tol"] = qc[3].number_input("Max. odch. od 90° [°]", 0.0, 45.0, 5.0,
                                                     step=0.5, key=f"perp_tol_{obj_i}",
                                                     disabled=not cfg["perp_en"])
                if cfg["perp_en"]:
                    ia, ib = cfg["perp_a"], cfg["perp_b"]
                    dev = perpendicularity_deg(edges, ia, ib)
                    ok = dev <= cfg["perp_tol"]
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Kolmost (odchylka)(H{ia + 1} | H{ib + 1})", dev, 0.0, cfg["perp_tol"], "°")
                    assign_highlight(ia, HCOLORS["perp_a"])
                    assign_highlight(ib, HCOLORS["perp_b"])
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Kolmost (odchylka)", 0, 0, 0, "°", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Libovolný úhel dvou hran hlavního tvaru**")
                ac = st.columns([1, 1, 1, 1, 1])
                cfg["ang_en"] = ac[0].checkbox("Aktivní", value=False, key=f"ang_en_{obj_i}")
                cfg["ang_a"] = ac[1].selectbox("Hrana A ", nums, 0, key=f"ang_a_{obj_i}",
                                               disabled=not cfg["ang_en"],
                                               format_func=edge_label) - 1
                cfg["ang_b"] = ac[2].selectbox("Hrana B ", nums, min(1, n_edges - 1),
                                               key=f"ang_b_{obj_i}",
                                               disabled=not cfg["ang_en"],
                                               format_func=edge_label) - 1
                cfg["ang_nom"] = ac[3].number_input("Jmenovitý úhel [°] (0-90)", 0.0, 90.0, 45.0,
                                                    step=0.5, key=f"ang_nom_{obj_i}",
                                                    disabled=not cfg["ang_en"])
                cfg["ang_tol"] = ac[4].number_input("Max. odchylka [°] ", 0.0, 45.0, 2.0,
                                                    step=0.5, key=f"ang_tol_{obj_i}",
                                                    disabled=not cfg["ang_en"])
                if cfg["ang_en"]:
                    ia, ib = cfg["ang_a"], cfg["ang_b"]
                    act_ang = angle_between_edges(edges, ia, ib)
                    lo = cfg["ang_nom"] - cfg["ang_tol"]
                    hi = cfg["ang_nom"] + cfg["ang_tol"]
                    ok = lo <= act_ang <= hi
                    if not ok: is_nok = True
                    r_md, r_data = tol_row(f"Úhel (H{ia + 1} | H{ib + 1})", act_ang, lo, hi, "°",
                                           extra=f" — jmenovitý **{cfg['ang_nom']}°**")
                    assign_highlight(ia, HCOLORS["ang_a"])
                    assign_highlight(ib, HCOLORS["ang_b"])
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Úhel (libovolný)", 0, 0, 0, "°", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Délka vybrané hrany hlavního tvaru**")
                lc = st.columns([1, 1, 1, 1, 1])
                cfg["len_en"] = lc[0].checkbox("Aktivní", value=False, key=f"len_en_{obj_i}")
                cfg["len_edge"] = lc[1].selectbox("Zkoumaná hrana", nums, 0, key=f"len_edge_{obj_i}",
                                                  disabled=not cfg["len_en"],
                                                  format_func=edge_label) - 1
                cfg["len_nom"] = lc[2].number_input("Jmenovitá [mm]", 0.0, 2000.0, 10.0,
                                                    step=0.5, key=f"len_nom_{obj_i}",
                                                    disabled=not cfg["len_en"])
                cfg["len_plus"] = lc[3].number_input("Tol. + [mm]", 0.0, 100.0, 1.0,
                                                     step=0.1, key=f"len_plus_{obj_i}",
                                                     disabled=not cfg["len_en"])
                cfg["len_minus"] = lc[4].number_input("Tol. − [mm]", 0.0, 100.0, 1.0,
                                                      step=0.1, key=f"len_minus_{obj_i}",
                                                      disabled=not cfg["len_en"])
                if cfg["len_en"]:
                    ie = cfg["len_edge"]
                    if ie < n_edges:
                        elen = edges[ie]["length"] / px_per_mm
                        lo = cfg["len_nom"] - cfg["len_minus"]
                        hi = cfg["len_nom"] + cfg["len_plus"]
                        ok = lo <= elen <= hi
                        if not ok: is_nok = True
                        r_md, r_data = tol_row(f"Tolerance délky H{ie + 1}", elen, lo, hi, "mm")
                        assign_highlight(ie, HCOLORS["len_e"])
                        rows_html_main.append(r_md)
                        export_data_list.append(r_data)
                    else:
                        r_md = "[NOK] Délka: hrana neexistuje"
                        rows_html_main.append(r_md)
                        r_data = {"Veličina": "Tolerance délky hrany", "Naměřeno": None,
                                  "Minimum": None, "Maximum": None,
                                  "Jednotka": "mm", "Výsledek": "Chyba (hrana chybí)"}
                        export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Tolerance délky hrany", 0, 0, 0, "mm", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            with st.container():
                st.markdown("**Rozměr objektu hlavního tvaru (šířka / výška)**")
                dc = st.columns([1, 1, 1, 1, 1, 1])
                cfg["dim_en"] = dc[0].checkbox("Aktivní", value=True, key=f"dim_en_{obj_i}")
                cfg["dim_axis"] = dc[1].selectbox("Osa", ["Šířka", "Výška", "Obě"],
                                                  key=f"dim_axis_{obj_i}",
                                                  disabled=not cfg["dim_en"])
                cfg["dim_wnom"] = dc[2].number_input("Jmenovitá šířka [mm]", 0.0, 2000.0,
                                                     round(rw_mm, 1), step=0.5,
                                                     key=f"dim_wnom_{obj_i}",
                                                     disabled=not cfg["dim_en"])
                cfg["dim_hnom"] = dc[3].number_input("Jmenovitá výška [mm]", 0.0, 2000.0,
                                                     round(rh_mm, 1), step=0.5,
                                                     key=f"dim_hnom_{obj_i}",
                                                     disabled=not cfg["dim_en"])
                cfg["dim_plus"] = dc[4].number_input("Tol. + [mm]", 0.0, 100.0, 1.0,
                                                     step=0.1, key=f"dim_plus_{obj_i}",
                                                     disabled=not cfg["dim_en"])
                cfg["dim_minus"] = dc[5].number_input("Tol. − [mm]", 0.0, 100.0, 1.0,
                                                      step=0.1, key=f"dim_minus_{obj_i}",
                                                      disabled=not cfg["dim_en"])
                if cfg["dim_en"]:
                    check_w = cfg["dim_axis"] in ("Šířka", "Obě")
                    check_h = cfg["dim_axis"] in ("Výška", "Obě")
                    if check_w:
                        lo = cfg["dim_wnom"] - cfg["dim_minus"]
                        hi = cfg["dim_wnom"] + cfg["dim_plus"]
                        ok = lo <= rw_mm <= hi
                        if not ok: is_nok = True
                        r_md, r_data = tol_row("Šířka objektu", rw_mm, lo, hi, "mm")
                        rows_html_main.append(r_md)
                        export_data_list.append(r_data)
                    if check_h:
                        lo = cfg["dim_hnom"] - cfg["dim_minus"]
                        hi = cfg["dim_hnom"] + cfg["dim_plus"]
                        ok = lo <= rh_mm <= hi
                        if not ok: is_nok = True
                        r_md, r_data = tol_row("Výška objektu", rh_mm, lo, hi, "mm")
                        rows_html_main.append(r_md)
                        export_data_list.append(r_data)
                else:
                    r_md, r_data = tol_row("Rozměry objektu", 0, 0, 0, "mm", skip=True)
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

            # Výpis výsledků hlavního tvaru najednou
            st.markdown("---")
            st.markdown("**Výsledky tolerance hlavního tvaru**")
            for r in rows_html_main:
                st.markdown(r)

            # Otvory
            is_holes_nok, export_data_list = render_hole_analysis(obj_i, holes, px_per_mm, mode_choice, angle_merge_tol,
                                                                  base_img, export_data_list)
            if is_holes_nok: is_nok = True

            st.markdown("---")
            st.markdown("**Vizualizace vybraných hran hlavního tvaru a děr**")
            annotated = draw_edge_map(base_img, edges, highlights, raw_cnt=cnt)
            if holes:
                draw_holes_with_labels(annotated, holes)

            legend_defs = []
            if cfg["par_en"]:
                ia, ib = cfg["par_a"], cfg["par_b"]
                legend_defs += [
                    (HCOLORS["par_a"],
                     f"H{ia + 1}  rovnobeznost A  ({edges[ia]['length'] / px_per_mm:.1f} mm, {edges[ia]['angle']:.1f} deg)"),
                    (HCOLORS["par_b"],
                     f"H{ib + 1}  rovnobeznost B  ({edges[ib]['length'] / px_per_mm:.1f} mm, {edges[ib]['angle']:.1f} deg)"),
                ]
            if cfg["perp_en"]:
                ia, ib = cfg["perp_a"], cfg["perp_b"]
                legend_defs += [
                    (HCOLORS["perp_a"],
                     f"H{ia + 1}  kolmost A  ({edges[ia]['length'] / px_per_mm:.1f} mm, {edges[ia]['angle']:.1f} deg)"),
                    (HCOLORS["perp_b"],
                     f"H{ib + 1}  kolmost B  ({edges[ib]['length'] / px_per_mm:.1f} mm, {edges[ib]['angle']:.1f} deg)"),
                ]
            if cfg["ang_en"]:
                ia, ib = cfg["ang_a"], cfg["ang_b"]
                legend_defs += [
                    (HCOLORS["ang_a"],
                     f"H{ia + 1}  uhel A  ({edges[ia]['length'] / px_per_mm:.1f} mm, {edges[ia]['angle']:.1f} deg)"),
                    (HCOLORS["ang_b"],
                     f"H{ib + 1}  uhel B  ({edges[ib]['length'] / px_per_mm:.1f} mm, {edges[ib]['angle']:.1f} deg)"),
                ]
            if cfg["len_en"] and cfg["len_edge"] < n_edges:
                ie = cfg["len_edge"]
                legend_defs.append(
                    (HCOLORS["len_e"],
                     f"H{ie + 1}  delka  ({edges[ie]['length'] / px_per_mm:.1f} mm, {edges[ie]['angle']:.1f} deg)")
                )
            annotated_legend = draw_legend_pil(annotated, legend_defs)

            c_a1, c_a2, c_a3 = st.columns([1, 2, 1])
            with c_a2:
                st.image(cv_to_pil(annotated_legend), use_container_width=True)

            st.markdown("---")
            st.markdown("**Výsledky tolerance celého objektu**")
            if is_nok:
                st.error(f"DÍL MIMO TOLERANCI | Alespoň jedna kontrola hlavního tvaru nebo děr selhala.")
            else:
                st.success(f"DÍL V TOLERANCI | Všechny provedené kontroly hlavního tvaru a děr prošly.")

            df_export = pd.DataFrame(export_data_list)
            csv_data = df_export.to_csv(index=False).encode('utf-8-sig')

            annotated_img = annotated_legend
            img_buf = io.BytesIO()
            cv_to_pil(annotated_img).save(img_buf, format="PNG")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(f"Stáhnout snímek (Objekt #{obj_i + 1})",
                                   data=img_buf.getvalue(),
                                   file_name=f"viziometer_obj{obj_i + 1}.png",
                                   mime="image/png", key=f"img_dl_{obj_i}", use_container_width=True)
            with dl_col2:
                st.download_button(f"Stáhnout tabulku s daty (CSV)",
                                   data=csv_data,
                                   file_name=f"viziometer_obj{obj_i + 1}_data.csv",
                                   mime="text/csv", key=f"csv_dl_{obj_i}", use_container_width=True)