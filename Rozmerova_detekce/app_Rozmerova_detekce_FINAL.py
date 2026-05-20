import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
import math
import os
import time

# Načtení loga VUT
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
logo = Image.open(logo_path)

# Nastavení stránky
st.set_page_config(
    page_title="Rozměrová detekce",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Kalibrace a měřítko

# Měřítko pro vykreslovéní
def get_drawing_scale(img_h, img_w, min_scale=0.1):
    diagonal = math.sqrt(img_h ** 2 + img_w ** 2)
    return max(min_scale, diagonal / 1500.0)

# Hledání rohů kalibrační šachovnice
def find_chessboard(gray, pattern):
    ret, corners = cv2.findChessboardCornersSB(
        gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        return corners, gray, 1.0
    ret, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    return (corners, gray, 1.0) if ret else (None, None, None)

# Vzdálenost mezi rohy šachovnice
def space_between_px(pts, rows, cols):
    grid = pts.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
    return (dx + dy) / 2.0

# Konkrétní rozměry v měřitku > příprava prvků do anotovaných obrazů
def draw_scale(img):
    s = get_drawing_scale(*img.shape[:2])
    return {
        "s": s, "thin_line": max(1, int(1 * s)), "line": max(1, int(2 * s)), "font": max(0.4, 0.7 * s),
        "circle": max(10, int(22 * s)), "pad": max(10, int(20 * s)),
    }

# Převod obrazu z formátu OpenCV do formátu PIL
def cv_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# Označneí rohů v šachovnici
def anotato_chessboard(img, pts):
    D = draw_scale(img)
    dot_r = max(3, int(6 * D["s"]))
    debug = img.copy()
    for cx, cy in pts:
        cv2.circle(debug, (int(cx), int(cy)), dot_r, (255, 0, 0), -1)
    return debug

# Hlavní funkce pro kalibraci
@st.cache_data(show_spinner=False)
def calibrate(img_bytes, rows, cols, square_mm):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    pattern = (cols, rows)

    corners, used_gray, scale = find_chessboard(gray, pattern)
    if corners is None:
        return None, "Rohy šachovnice nebyly nalezeny.", None

    corners = corners.astype(np.float32)
    pts = corners.reshape(-1, 2)
    ppm = space_between_px(pts, rows, cols) / square_mm
    debug = anotato_chessboard(img, pts)
    return ppm, debug, None, (w, h)

# Funkce pro zjednodušení nalezeného mnohoúhelníku
def poly_ver(pts, peri, angle_merge_tol=6.0):
    pts = list(pts)
    changed = True
    min_len_px = max(3.0, 0.015 * peri)
    iters = 0
    while changed and len(pts) > 3 and iters < 30:
        changed = False
        iters += 1
        n = len(pts)
        for i in range(n):
            if np.linalg.norm(pts[(i + 1) % n] - pts[i]) < min_len_px:
                pts.pop((i + 1) % n)
                changed = True
                break
        if changed:
            continue
        n = len(pts)
        for i in range(n):
            v1 = pts[i] - pts[(i - 1) % n]
            v2 = pts[(i + 1) % n] - pts[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                if math.degrees(math.acos(cos_theta)) < angle_merge_tol:
                    pts.pop(i)
                    changed = True
                    break
    return np.array(pts)

# Zajištění ostrých rohů > vezme hrany a polsedních 10 porocent nahradí přímkami aby se zajistili ostré rohy
def corners(cnt_pts, rough_pts, peri, angle_merge_tol):
    rough_pts = poly_ver(rough_pts, peri, angle_merge_tol)
    n_edges = len(rough_pts)
    if n_edges < 3:
        return rough_pts

    edge_points = [[] for _ in range(n_edges)]
    for pt in cnt_pts:
        min_dist, best_edge, best_t = float('inf'), -1, 0.0
        for i in range(n_edges):
            p1 = rough_pts[i]
            p2 = rough_pts[(i + 1) % n_edges]
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                continue
            t = np.dot(pt - p1, line_vec) / (line_len ** 2)
            t_clamped = max(0.0, min(1.0, t))
            dist = np.linalg.norm(pt - (p1 + t_clamped * line_vec))
            if dist < min_dist:
                min_dist, best_edge, best_t = dist, i, t_clamped
        if best_edge != -1:
            edge_points[best_edge].append((pt, best_t))

    fitted_lines = []
    for i in range(n_edges):
        pts_with_t = sorted(edge_points[i], key=lambda x: x[1])
        n_pts = len(pts_with_t)
        if n_pts >= 10:
            valid_pts = []
            for p in pts_with_t[int(0.20 * n_pts):int(0.80 * n_pts)]:
                valid_pts.append(p[0])
        elif n_pts >= 3:
            valid_pts = []
            for p in pts_with_t:
                valid_pts.append(p[0])
        else:
            valid_pts = []

        if len(valid_pts) >= 2:
            line = cv2.fitLine(np.array(valid_pts, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
            fitted_lines.append((float(line[0][0]), float(line[1][0]), float(line[2][0]), float(line[3][0])))
        else:
            p1 = rough_pts[i]
            p2 = rough_pts[(i + 1) % n_edges]
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            norm = math.hypot(vx, vy) or 1
            fitted_lines.append((vx / norm, vy / norm, float(p1[0]), float(p1[1])))

    sharp_pts = []
    for i in range(n_edges):
        vx1, vy1, x1, y1 = fitted_lines[i]
        vx2, vy2, x2, y2 = fitted_lines[(i + 1) % n_edges]
        denom = vx1 * vy2 - vy1 * vx2
        rough_p = rough_pts[(i + 1) % n_edges]
        if abs(denom) > 1e-6:
            t1 = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / denom
            ix, iy = x1 + t1 * vx1, y1 + t1 * vy1
            if math.hypot(ix - rough_p[0], iy - rough_p[1]) > 0.15 * peri:
                sharp_pts.append(rough_p)
            else:
                sharp_pts.append(np.array([ix, iy], dtype=np.float32))
        else:
            sharp_pts.append(rough_p)
    return np.array(sharp_pts, dtype=np.float32)

# Z obrysu mnohoúhelníku udělá jednotlivé hrany
def get_edges(cnt, angle_merge_tol=6.0):
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

    if raw_circ >= 0.75:
        approx = cv2.approxPolyDP(cnt, 0.005 * peri, True)
        for factor in [0.008, 0.012, 0.018, 0.025]:
            if len(approx) <= 40:
                break
            approx = cv2.approxPolyDP(cnt, factor * peri, True)
    else:
        best, prev_n = None, None
        for factor in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
            cand = cv2.approxPolyDP(cnt, factor * peri, True)
            n_cand = len(cand)
            if prev_n is not None and abs(n_cand - prev_n) <= 1 and n_cand >= 3:
                best = cand
                break
            best, prev_n = cand, n_cand
        approx = best if best is not None else cv2.approxPolyDP(cnt, 0.02 * peri, True)

    pts = approx.reshape(-1, 2).astype(np.float32)
    if raw_circ < 0.75:
        pts = corners(cnt.reshape(-1, 2).astype(np.float32), pts, peri, angle_merge_tol)

    n = len(pts)
    raw_edges = []
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length == 0:
            continue
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0
        raw_edges.append({"p1": p1, "p2": p2, "angle": angle, "length": length,
                          "mid": ((p1 + p2) / 2).astype(int)})
    return raw_edges, pts


# Vykreslí obrysy vnitřních děr
def draw_holes_with_labels(out, holes, angle_merge_tol):
    D = draw_scale(out)
    font, line_th, badge_r = D["font"], D["line"], D["circle"]
    th = max(1, int(2 * D["s"]))
    text_y_offset = int(7 * D["s"])

    for i, h_cnt in enumerate(holes):
        peri = cv2.arcLength(h_cnt, True)
        area = cv2.contourArea(h_cnt)
        raw_circ = (4 * math.pi * area) / (peri ** 2) if peri > 0 else 0

        if raw_circ < 0.75:
            _, sharp_pts = get_edges(h_cnt, angle_merge_tol)
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
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cx, cy = int(h_cnt[0][0][0]), int(h_cnt[0][0][1])

        label = f"#{i + 1}"
        cv2.circle(out, (cx, cy), badge_r, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), badge_r, (0, 0, 255), line_th, cv2.LINE_AA)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, th)[0][0]
        cv2.putText(out, label, (cx - tw // 2, cy + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), th, cv2.LINE_AA)


# Vykreslení mapu hran nalezeného objektu
def draw_edge_map(base_img, edges, highlights, raw_cnt=None, sharp_pts=None, badge_scale=1.0):
    out = base_img.copy()
    D = draw_scale(out)
    s = D["s"]
    th_dash = max(1, int(1.5 * s))
    th_highlight = max(1, int(3 * s))
    th_raw_cnt = max(1, int(1.5 * s))
    dash_len = gap_len = max(2, int(10 * s))
    font_edge = D["font"] * badge_scale
    font_edge_hi = max(0.4, 0.8 * s) * badge_scale
    th_edge_text = max(1, int(2 * s))

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
        dx, dy = (p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist
        pos, drawing = 0, True
        while pos < dist:
            seg_end = min(pos + (dash_len if drawing else gap_len), dist)
            if drawing:
                cv2.line(out, (int(p1[0] + pos * dx), int(p1[1] + pos * dy)),
                         (int(p1[0] + seg_end * dx), int(p1[1] + seg_end * dy)),
                         (150, 150, 150), th_dash, cv2.LINE_AA)
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

    badge_r_hi = max(6, int(24 * s * badge_scale))
    badge_r_norm = max(5, int(20 * s * badge_scale))
    text_y_offset = int(6 * s * badge_scale)

    for i, e in enumerate(edges):
        mid = tuple(e["mid"].astype(int))
        is_hi = i in highlights
        badge_color = highlights[i] if is_hi else (120, 120, 120)
        curr_badge_r = badge_r_hi if is_hi else badge_r_norm
        cv2.circle(out, mid, curr_badge_r, (255, 255, 255), -1)
        curr_badge_th = max(1, int(2 * s)) if is_hi else max(1, int(1 * s))
        cv2.circle(out, mid, curr_badge_r, badge_color, curr_badge_th, cv2.LINE_AA)
        label = str(i + 1)
        curr_font = font_edge_hi if is_hi else font_edge
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, curr_font, th_edge_text)[0][0]
        cv2.putText(out, label, (mid[0] - tw // 2, mid[1] + text_y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, curr_font, (0, 0, 0), th_edge_text, cv2.LINE_AA)
    return out


# Legenda co se dá do anotovaných obrazů
def draw_legend_cv(cv_img, legend_defs):
    if not legend_defs:
        return cv_img
    out = cv_img.copy()
    D = draw_scale(out)
    s = D["s"]
    font = D["font"]
    th = max(1, int(1.5 * s))
    pad = D["pad"]
    swatch = max(12, int(25 * s))
    gap = max(6, int(15 * s))
    line_spacing = max(6, int(15 * s))

    max_tw = max_th = 0
    for _, txt in legend_defs:
        (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font, th)
        max_tw, max_th = max(max_tw, tw), max(max_th, th_txt)

    row_h = max(swatch, max_th)
    panel_w = pad * 2 + swatch + gap + max_tw
    panel_h = pad * 2 + (len(legend_defs) * row_h) + ((len(legend_defs) - 1) * line_spacing)
    x0 = y0 = max(10, int(20 * s))

    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.85, out, 0.15, 0, out)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (100, 100, 100), max(1, int(1 * s)))

    curr_y = y0 + pad
    for bgr, txt in legend_defs:
        cv2.rectangle(out, (x0 + pad, curr_y), (x0 + pad + swatch, curr_y + swatch), bgr, -1)
        cv2.rectangle(out, (x0 + pad, curr_y), (x0 + pad + swatch, curr_y + swatch), (50, 50, 50), max(1, int(1 * s)))
        (tw, th_txt), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font, th)
        text_y = curr_y + (swatch + th_txt) // 2
        cv2.putText(out, txt, (x0 + pad + swatch + gap, text_y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), th,
                    cv2.LINE_AA)
        curr_y += row_h + line_spacing
    return out


# Geometrické funkce
# Rovnoběžnost
def parallelism_deg(edges, ia, ib):
    diff = abs(edges[ia]["angle"] - edges[ib]["angle"])
    if diff > 90.0:
        diff = 180.0 - diff
    return round(diff, 3)


# Kolmost
def perpendicularity_deg(edges, ia, ib):
    diff = abs(edges[ia]["angle"] - edges[ib]["angle"])
    diff = min(diff, 180.0 - diff)
    return round(abs(diff - 90.0), 3)


# Úhel mezi hranami
def angle_between_edges(edges, ia, ib):
    diff = abs(edges[ia]["angle"] - edges[ib]["angle"])
    return round(min(diff, 180.0 - diff), 3)


# Kruhovitost
def circularity_pct(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return round(min((4 * math.pi * area) / (peri ** 2), 1.0) * 100.0, 2)


# Nalezená kružnice se proloží kružnicí > radiální odchylka
def circle_metrics(cnt, px_per_mm):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
    b = pts[:, 0] ** 2 + pts[:, 1] ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    r_fit = math.sqrt(max(result[2] + cx ** 2 + cy ** 2, 0))
    radii = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    r_max, r_min = float(radii.max()), float(radii.min())
    return {
        "cx_px": float(cx), "cy_px": float(cy),
        "r_fit_mm": float(r_fit / px_per_mm),
        "r_max_mm": float(r_max / px_per_mm),
        "r_min_mm": float(r_min / px_per_mm),
        "dev_mm": float((r_max - r_min) / px_per_mm),
    }


# Pro zobrazení výsledků měření
def tol_row(label, measured, lo, hi, unit, extra=""):
    ok = lo <= measured <= hi
    icon = "[OK]" if ok else "[NOK]"

    md = "**" + icon + " " + label + "**: `" + str(round(measured, 3)) + " " + unit + "` ∈ `[" + str(
        round(lo, 2)) + " – " + str(round(hi, 2)) + " " + unit + "]` " + extra

    data = {
        "Veličina": label, "Naměřeno": round(measured, 3), "Minimum": round(lo, 3),
        "Maximum": round(hi, 3), "Jednotka": unit, "Výsledek": "OK" if ok else "NOK"
    }
    return md, data


# Nastavení pro tolerance
def render_tol_block(kind, edges, nums, edge_label, px_per_mm, key_prefix,
                     title, default_on, prefix_label, hi_a, hi_b, highlight_fn):
    cfg = {}
    n_edges = len(edges)
    idx_a, idx_b = 0, min(1, n_edges - 1)
    st.markdown(f"**{title}**")

    # Rovnoběžnost a kolmost
    if kind in ("par", "perp"):
        c = st.columns([1, 1, 1, 1])
        cfg["en"] = c[0].checkbox("Aktivní", value=default_on, key=f"{key_prefix}_en")
        cfg["a"] = c[1].selectbox("Hrana A", nums, idx_a, key=f"{key_prefix}_a",
                                  disabled=not cfg["en"], format_func=edge_label) - 1
        cfg["b"] = c[2].selectbox("Hrana B", nums, idx_b, key=f"{key_prefix}_b",
                                  disabled=not cfg["en"], format_func=edge_label) - 1

        lbl = "Max. odchylka [°]"
        cfg["tol"] = c[3].number_input(lbl, 0.0, 90.0, 2.0, step=0.5,
                                       key=f"{key_prefix}_tol", disabled=not cfg["en"])

        if not cfg["en"]:
            return False, None, None, cfg
        ia, ib = cfg["a"], cfg["b"]
        dev = parallelism_deg(edges, ia, ib) if kind == "par" else perpendicularity_deg(edges, ia, ib)
        ok = dev <= cfg["tol"]
        name = "Rovnoběžnost" if kind == "par" else "Kolmost"
        r_md, r_data = tol_row(f"{prefix_label}{name} ({hi_a}{ia + 1}|{hi_b}{ib + 1})", dev, 0.0, cfg["tol"], "°")
        if highlight_fn:
            highlight_fn(ia, "a")
            highlight_fn(ib, "b")
        return not ok, r_md, r_data, cfg

    # Vzájemný úhel
    if kind == "ang":
        c = st.columns([1, 1, 1, 1, 1])
        cfg["en"] = c[0].checkbox("Aktivní", value=default_on, key=f"{key_prefix}_en")
        cfg["a"] = c[1].selectbox("Hrana A", nums, idx_a, key=f"{key_prefix}_a",
                                  disabled=not cfg["en"], format_func=edge_label) - 1
        cfg["b"] = c[2].selectbox("Hrana B", nums, idx_b, key=f"{key_prefix}_b",
                                  disabled=not cfg["en"], format_func=edge_label) - 1
        cfg["nom"] = c[3].number_input("Jmen. [°]", 0.0, 90.0, 45.0, step=0.5,
                                       key=f"{key_prefix}_nom", disabled=not cfg["en"])
        cfg["tol"] = c[4].number_input("Max. odch. [°]", 0.0, 45.0, 2.0, step=0.5,
                                       key=f"{key_prefix}_tol", disabled=not cfg["en"])
        if not cfg["en"]:
            return False, None, None, cfg
        ia, ib = cfg["a"], cfg["b"]
        act = angle_between_edges(edges, ia, ib)
        lo, hi = cfg["nom"] - cfg["tol"], cfg["nom"] + cfg["tol"]
        ok = lo <= act <= hi
        r_md, r_data = tol_row(f"{prefix_label}Úhel ({hi_a}{ia + 1}|{hi_b}{ib + 1})", act, lo, hi, "°")
        if highlight_fn:
            highlight_fn(ia, "a")
            highlight_fn(ib, "b")
        return not ok, r_md, r_data, cfg

    # Délka
    if kind == "len":
        c = st.columns([1, 1, 1, 1, 1])
        cfg["en"] = c[0].checkbox("Aktivní", value=default_on, key=f"{key_prefix}_en")
        cfg["edge"] = c[1].selectbox("Vybrat hranu", nums, 0, key=f"{key_prefix}_e",
                                     disabled=not cfg["en"], format_func=edge_label) - 1
        cfg["nom"] = c[2].number_input("Jmenovitá [mm]", 0.0, 2000.0, 10.0, step=0.5,
                                       key=f"{key_prefix}_nom", disabled=not cfg["en"])
        cfg["plus"] = c[3].number_input("Tol. +", 0.0, 100.0, 1.0, step=0.1,
                                        key=f"{key_prefix}_p", disabled=not cfg["en"])
        cfg["minus"] = c[4].number_input("Tol. −", 0.0, 100.0, 1.0, step=0.1,
                                         key=f"{key_prefix}_m", disabled=not cfg["en"])
        if not cfg["en"]:
            return False, None, None, cfg
        ie = cfg["edge"]
        if ie >= n_edges:
            return False, None, None, cfg
        elen = edges[ie]["length"] / px_per_mm
        lo, hi = cfg["nom"] - cfg["minus"], cfg["nom"] + cfg["plus"]
        ok = lo <= elen <= hi
        r_md, r_data = tol_row(f"{prefix_label}Délka hrany {hi_a}{ie + 1}", elen, lo, hi, "mm")
        if highlight_fn:
            highlight_fn(ie, "len")
        return not ok, r_md, r_data, cfg

    return False, None, None, cfg


# Funkce pro nalezení objektů v obraze
@st.cache_data(show_spinner=False)
def detect_objects(img_bytes, _calib_size, canny_low, canny_high,
                   min_area, max_obj, merge, merge_dist, detect_holes, det_method,
                   invert_thresh, min_hole_area):
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    scale_ratio = 1.0

    if _calib_size is not None:
        h, w = img.shape[:2]
        cal_w, cal_h = _calib_size
        scale_ratio = ((w / cal_w) + (h / cal_h)) / 2.0

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_type = cv2.THRESH_BINARY_INV if invert_thresh else cv2.THRESH_BINARY

    if "Otsu" in det_method:
        _, processed = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
        if merge:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    else:
        processed = cv2.Canny(blurred, canny_low, canny_high)
        if merge:
            k = max(5, merge_dist)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            processed = cv2.dilate(processed, kernel, iterations=1)
            processed = cv2.erode(processed, kernel, iterations=1)

    retrieval_mode = cv2.RETR_TREE if detect_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(processed.copy(), retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)

    print(f"DEBUG: Nalezeno {len(contours)} kontur, zpracovávám...")

    parts_serialized = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        raw_parts = []
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue
            outer_area = cv2.contourArea(cnt)
            if outer_area > 0.98 * total_img_area or outer_area < min_area:
                continue
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
                            if cv2.contourArea(contours[gc_idx]) >= min_hole_area:
                                holes.append(contours[gc_idx])
                            gc_idx = hierarchy[gc_idx][0]
                    child_idx = hierarchy[child_idx][0]
            raw_parts.append({"outer": cnt, "holes": holes, "area": outer_area})

        raw_parts.sort(key=lambda x: x["area"], reverse=True)
        if max_obj > 0:
            raw_parts = raw_parts[:max_obj]
        for part in raw_parts:
            parts_serialized.append({
                "outer_bytes": part["outer"].tobytes(),
                "outer_shape": part["outer"].shape,
                "holes": [(h.tobytes(), h.shape) for h in part["holes"]]
            })

    map_th = draw_scale(img)["thin_line"]
    clean_map = np.zeros_like(img)
    for p in deserialize_parts(parts_serialized):
        cv2.drawContours(clean_map, [p["outer"]], -1, (255, 255, 255), map_th, cv2.LINE_AA)
        if p["holes"]:
            cv2.drawContours(clean_map, p["holes"], -1, (255, 255, 255), map_th, cv2.LINE_AA)

    edge_vis_bytes = cv2.imencode(".png", clean_map)[1].tobytes()
    img_bytes_out = cv2.imencode(".png", img)[1].tobytes()
    return img_bytes_out, edge_vis_bytes, parts_serialized, scale_ratio


# Převod předchozí funkce na arrays pro další výpočty
def deserialize_parts(parts_s):
    parts = []
    for p in parts_s:
        outer = np.frombuffer(p["outer_bytes"], dtype=np.int32).reshape(p["outer_shape"])
        holes = [np.frombuffer(hb, dtype=np.int32).reshape(hs) for hb, hs in p["holes"]]
        parts.append({"outer": outer, "holes": holes})
    return parts


# Sekce pro analýzu děr
def render_hole_analysis(obj_i, holes, px_per_mm, angle_merge_tol, base_img, export_data_list):
    if len(holes) == 0:
        return False, export_data_list

    is_holes_nok = False
    rows_html_holes = []
    st.markdown("---")
    st.markdown(f"**Analýza vnitřních otvorů (Nalezeno: {len(holes)})**")

    hole_mode = st.segmented_control(
        "Typ analýzy otvorů", options=["circle", "polygon"], default="circle",
        format_func=lambda x: "Kruhové (Průměr, Kruhovitost)" if x == "circle" else "Hranaté (Detailní geometrie)",
        key=f"hole_mode_{obj_i}"
    )

    if hole_mode == "circle":
        holes_data = []

        for h_idx, dira_cnt in enumerate(holes):
            h_cm = circle_metrics(dira_cnt, px_per_mm)
            holes_data.append({
                "Otvor": f"#{h_idx + 1}",
                "Průměr (fit) [mm]": round(h_cm["r_fit_mm"] * 2, 3),
                "Kruhovitost [%]": circularity_pct(dira_cnt),
                "Radiální odchylka [mm]": round(h_cm["dev_mm"], 3)
            })
        st.dataframe(pd.DataFrame(holes_data), use_container_width=True)

        st.markdown("**Tolerance kruhových otvorů**")
        hc1, hc2, hc3, hc4, hc5 = st.columns(5)
        en_c = hc1.checkbox("Průměr", key=f"h_en_{obj_i}_c")
        default_diam = float(holes_data[0]["Průměr (fit) [mm]"]) if holes_data else 10.0
        h_nom = hc2.number_input("Jmen. [mm]", 0.0, float(max(500.0, default_diam * 2.0)), default_diam,
                                 step=0.1, key=f"h_nom_{obj_i}_c", disabled=not en_c)
        h_tol = hc3.number_input("Tol. ± [mm]", 0.0, 50.0, 0.5, step=0.1, key=f"h_tol_{obj_i}_c", disabled=not en_c)
        en_circ = hc4.checkbox("Kruhovitost", key=f"h_en_{obj_i}_circ")
        h_min_circ = hc5.number_input("Min. [%]", 0.0, 100.0, 89.0, step=1.0,
                                      key=f"h_min_circ_{obj_i}", disabled=not en_circ)

        if en_c or en_circ:
            for h_idx, h_row in enumerate(holes_data):
                if en_c:
                    d_fit = h_row["Průměr (fit) [mm]"]
                    lo, hi = h_nom - h_tol, h_nom + h_tol
                    if not (lo <= d_fit <= hi):
                        is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{h_idx + 1} průměr", d_fit, lo, hi, "mm",
                                           extra=f" — naměřeno **{d_fit:.3f} mm**")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)
                if en_circ:
                    circ_val = h_row["Kruhovitost [%]"]
                    if not (circ_val >= h_min_circ):
                        is_holes_nok = True
                    r_md, r_data = tol_row(f"Otvor #{h_idx + 1} kruhovitost", circ_val, h_min_circ, 100.0, "%")
                    rows_html_holes.append(r_md)
                    export_data_list.append(r_data)
    else:
        holes_data = []
        for h_idx, hole_cnt in enumerate(holes):
            h_edges, _ = get_edges(hole_cnt, angle_merge_tol=angle_merge_tol)
            hx, hy, hw, hh = cv2.boundingRect(hole_cnt)
            holes_data.append({
                "Otvor": f"#{h_idx + 1}", "Počet hran": len(h_edges),
                "Šířka X [mm]": round(hw / px_per_mm, 3),
                "Výška Y [mm]": round(hh / px_per_mm, 3),
                "Obvod [mm]": round(cv2.arcLength(hole_cnt, True) / px_per_mm, 3)
            })
        st.dataframe(pd.DataFrame(holes_data), use_container_width=True)

        for h_row in holes_data:
            for klic, hodnota in h_row.items():
                if klic == "Otvor":
                    continue
                jednotka = "mm" if "[mm]" in klic else "%" if "[%]" in klic else ""
                export_data_list.append({
                    "Veličina": f"Díra {h_row['Otvor']}: {klic.split(' [')[0]}",
                    "Naměřeno": hodnota, "Minimum": None, "Maximum": None,
                    "Jednotka": jednotka, "Výsledek": "Info"
                })

        st.markdown("---")
        st.markdown("**Detailní geometrické tolerance konkrétního otvoru**")
        sel_h_idx = st.selectbox("Vyberte otvor pro detailní analýzu", range(len(holes)),
                                 format_func=lambda x: f"Otvor #{x + 1}", key=f"sel_h_{obj_i}")
        sel_h_cnt = holes[sel_h_idx]
        sel_h_edges, sel_sharp_pts = get_edges(sel_h_cnt, angle_merge_tol=angle_merge_tol)
        n_h_edges = len(sel_h_edges)
        st.markdown(f"*Detekováno {n_h_edges} hran u Otvoru #{sel_h_idx + 1}. Čísla hran viz náhled.*")

        if n_h_edges >= 2:
            hole_preview = draw_edge_map(base_img, sel_h_edges,
                                         {i: (0, 0, 255) for i in range(n_h_edges)},
                                         sharp_pts=sel_sharp_pts, badge_scale=0.4)
            _, hc_2, _ = st.columns([1, 1, 1])
            with hc_2:
                if hole_preview.size > 0:
                    st.image(cv_to_pil(hole_preview), caption=f"Detail hran Otvoru #{sel_h_idx + 1}",
                             use_container_width=True)

            h_nums = list(range(1, n_h_edges + 1))

            def h_edge_label(i):
                if i - 1 < 0 or i - 1 >= len(sel_h_edges):
                    return f"h{i}"
                e = sel_h_edges[i - 1]
                return f"h{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"

            prefix = f"Otvor #{sel_h_idx + 1} "
            blocks = [
                ("par", "Rovnoběžnost", False),
                ("perp", "Kolmost", False),
                ("ang", "Libovolný úhel", False),
                ("len", "Délka hrany otvoru", False),
            ]
            for kind, title, default_on in blocks:
                with st.container():
                    nok, r_md, r_data, _ = render_tol_block(
                        kind, sel_h_edges, h_nums, h_edge_label, px_per_mm,
                        key_prefix=f"h_{kind}_{obj_i}", title=title, default_on=default_on,
                        prefix_label=prefix, hi_a="h", hi_b="h", highlight_fn=None)
                    if nok:
                        is_holes_nok = True
                    if r_md:
                        rows_html_holes.append(r_md)
                        export_data_list.append(r_data)

    if rows_html_holes:
        st.markdown("---")
        st.markdown("**Výsledky tolerance otvorů**")
        for r in rows_html_holes:
            st.markdown(r)
    return is_holes_nok, export_data_list


# Uživatelské rozhraní
st.title("Rozměrová detekce")
st.markdown("Analýza rozměrů a geometrických tolerancí")

st.markdown("---")
st.markdown("---")

# Boční panel - sidebar
with st.sidebar:
    st.markdown("## Nastavení")
    st.markdown("---")
    # Kalibrace pomocí šachovnice
    with st.expander("Kalibrace", expanded=True):
        st.caption("Nastavení px/mm a šachovnice")
        calib_file = st.file_uploader("Snímek šachovnice", type=["jpg", "jpeg", "png", "bmp"], key="calib")
        c1, c2 = st.columns(2)
        with c1:
            cb_rows = st.number_input("Řádky rohů", 3, 40, 23)
        with c2:
            cb_cols = st.number_input("Sloupce rohů", 3, 50, 32)
        sq_mm = st.number_input("Čtverec [mm]", 0.1, 100.0, 5.0, step=0.1)
    # Ruční kalibrace
    with st.expander("Ruční px/mm", expanded=False):
        st.caption("Vlastní kalibrační hodnota")
        manual_ppm = st.number_input("Hodnota (0 = auto)", 0.0, 500.0, 0.0, step=0.1)
    # Detekce objektů
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

angle_merge_tol = 6
px_per_mm = None
calib_size = None

if calib_file:
    calib_file.seek(0)
    res = calibrate(calib_file.read(), cb_rows, cb_cols, sq_mm)
    px_per_mm_cal, debug_img, err, calib_size = res
    if err:
        st.error(err)
    else:
        px_per_mm = px_per_mm_cal
        st.success(f"Kalibrace úspěšná — {px_per_mm:.4f} px/mm")
        with st.expander("Kalibrační snímek", expanded=False):
            if debug_img is not None:
                st.image(cv_to_pil(debug_img), use_container_width=True)

if manual_ppm and manual_ppm > 0:
    px_per_mm = manual_ppm
    st.info(f"Ruční px/mm aktivní: {px_per_mm:.4f}")

if not px_per_mm:
    px_per_mm = 5.0
    if not calib_file:
        st.info("Nahrajte šachovnici nebo zadejte vlastní px/mm. Výchozí: 5.0 px/mm")

obj_file = st.file_uploader("Snímek měřeného objektu", type=["jpg", "jpeg", "png", "bmp"])

if not obj_file:
    st.info("Nahrajte snímek objektu ke zpracování.")
    st.stop()

obj_file.seek(0)
img_bytes = obj_file.read()
orig_pil = Image.open(io.BytesIO(img_bytes))

img_out_bytes, edge_vis_bytes, parts_s, scale_ratio = detect_objects(
    img_bytes, calib_size, canny_low, canny_high, min_area, max_obj,
    merge, merge_dist, detect_holes, det_method, invert_thresh, min_hole_area
)

if scale_ratio != 1.0 and not (manual_ppm and manual_ppm > 0):
    px_per_mm = px_per_mm * scale_ratio

base_img = cv2.imdecode(np.frombuffer(img_out_bytes, np.uint8), cv2.IMREAD_COLOR)
edge_img = cv2.imdecode(np.frombuffer(edge_vis_bytes, np.uint8), cv2.IMREAD_COLOR)
parts = deserialize_parts(parts_s)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Originální snímek**")
    st.image(orig_pil, use_container_width=True)
with col_b:
    st.markdown("**Segmentační mapa**")
    if edge_img is not None:
        st.image(cv_to_pil(edge_img), use_container_width=True)

if not parts:
    st.warning("Žádné objekty nebyly nalezeny. Zkuste upravit segmentaci nebo min. plochu v postranním panelu.")
    st.stop()

st.markdown("---")
st.markdown(f"**Nalezeno:** {len(parts)} hlavních objektů")

HCOLORS = {
    "par_a": (0, 158, 230), "par_b": (0, 94, 213), "perp_a": (0, 158, 115), "perp_b": (167, 121, 204),
    "len_e": (0, 114, 178), "ang_a": (230, 158, 0), "ang_b": (213, 94, 0),
}

needs_rerun = False

# Smyčka pro kontrolu nastavených tolerancí
for obj_i, part in enumerate(parts):
    cnt = part["outer"]
    holes = part["holes"]
    if cnt is None or cnt.size < 6:
        continue

    test_plochy_obdelniku = cv2.boundingRect(cnt)[2] * cv2.boundingRect(cnt)[3]

    area_px = cv2.contourArea(cnt)
    x, y, bw, bh = cv2.boundingRect(cnt)
    if area_px == 0 or bw == 0 or bh == 0:
        continue
    if px_per_mm == 0:
        px_per_mm = 1.0

    (cx_r, cy_r), (rw_px, rh_px), angle = cv2.minAreaRect(cnt)
    rw_mm = max(rw_px, rh_px) / px_per_mm
    rh_mm = min(rw_px, rh_px) / px_per_mm
    edges, sharp_pts = get_edges(cnt, angle_merge_tol=angle_merge_tol)
    n_edges = len(edges)

    prev_nok = st.session_state.get(f"is_nok_{obj_i}", None)

    if prev_nok is None:
        status_badge = "Počítám..."
    elif prev_nok == True:
        status_badge = "🔴 NOK"
    elif prev_nok == False:
        status_badge = "🟢 OK"

    with st.expander(
            f"Objekt #{obj_i + 1}  —  {rw_mm:.1f} × {rh_mm:.1f} mm  |  Stav: {status_badge}",
            expanded=True):

        is_nok = False
        highlights = {}
        export_data_list, rows_html_main = [], []

        PALETTE = [(0, 158, 230), (0, 114, 178), (0, 158, 115), (10, 194, 213),
                   (0, 94, 213), (167, 121, 204), (0, 0, 0)]

        st.markdown("**Vizualizace nalezených hran**")
        preview_highlights = {i: PALETTE[i % len(PALETTE)] for i in range(n_edges)}
        edge_preview = draw_edge_map(base_img, edges, preview_highlights, sharp_pts=sharp_pts)
        th_rect = draw_scale(base_img)["thin_line"]
        cv2.rectangle(edge_preview, (x, y), (x + bw, y + bh), (150, 150, 150), th_rect)
        if holes:
            draw_holes_with_labels(edge_preview, holes, angle_merge_tol)

        preview_legend = [(PALETTE[i % len(PALETTE)],
                           f"H{i + 1}  {edges[i]['length'] / px_per_mm:.1f} mm  {edges[i]['angle']:.1f} deg")
                          for i in range(n_edges)]
        edge_preview_annotated = draw_legend_cv(edge_preview, preview_legend)

        _, c_p2, _ = st.columns([1, 1, 1])
        with c_p2:
            if edge_preview_annotated is not None:
                st.image(cv_to_pil(edge_preview_annotated), use_container_width=True)

        edges_df_data = []
        for i, e in enumerate(edges):
            length_mm = e['length'] / px_per_mm
            edges_df_data.append({
                "Hrana": f"H{i + 1}", "Délka [mm]": round(length_mm, 3),
                "Úhel [°]": round(e['angle'], 1),
            })
            export_data_list.append({
                "Veličina": f"Délka hrany H{i + 1}", "Naměřeno": round(length_mm, 3),
                "Minimum": None, "Maximum": None, "Jednotka": "mm", "Výsledek": "Info"
            })
        st.dataframe(pd.DataFrame(edges_df_data), use_container_width=True)

        st.markdown("---")
        st.markdown("**Konfigurace tolerancí hlavního objektu**")
        nums = list(range(1, n_edges + 1))


        def edge_label(i):
            if i - 1 < 0 or i - 1 >= len(edges):
                return f"H{i}"
            e = edges[i - 1]
            return f"H{i} ({e['length'] / px_per_mm:.1f} mm, {e['angle']:.1f}°)"


        def make_highlight(kind):
            def _hi(idx, slot):
                color_key = f"{kind}_{slot}" if f"{kind}_{slot}" in HCOLORS else kind
                if color_key in HCOLORS:
                    highlights[idx] = HCOLORS[color_key]

            return _hi


        main_blocks = [
            ("par", "Rovnoběžnost", True),
            ("perp", "Kolmost", True),
            ("ang", "Libovolný úhel", False),
            ("len", "Délka hrany", False),
        ]
        block_cfgs = {}
        for kind, title, default_on in main_blocks:
            with st.container():
                nok, r_md, r_data, cfg = render_tol_block(
                    kind, edges, nums, edge_label, px_per_mm,
                    key_prefix=f"{kind}_{obj_i}", title=title, default_on=default_on,
                    prefix_label="", hi_a="H", hi_b="H", highlight_fn=make_highlight(kind))
                block_cfgs[kind] = cfg
                if nok:
                    is_nok = True
                if r_md:
                    rows_html_main.append(r_md)
                    export_data_list.append(r_data)

        if rows_html_main:
            st.markdown("---")
            for r in rows_html_main:
                st.markdown(r)

        is_holes_nok, export_data_list = render_hole_analysis(
            obj_i, holes, px_per_mm, angle_merge_tol, base_img, export_data_list)
        if is_holes_nok:
            is_nok = True

        st.markdown("---")
        st.markdown("**Vizualizace vybraných hran hlavního tvaru a děr**")
        annotated = draw_edge_map(base_img, edges, highlights, sharp_pts=sharp_pts)
        if holes:
            draw_holes_with_labels(annotated, holes, angle_merge_tol)

        legend_defs = []
        for kind in ("par", "perp", "ang"):
            cfg = block_cfgs.get(kind, {})
            if cfg.get("en"):
                ia, ib = cfg["a"], cfg["b"]
                name = {"par": "rovnobeznost", "perp": "kolmost", "ang": "uhel"}[kind]
                legend_defs += [(HCOLORS[f"{kind}_a"], f"H{ia + 1} {name}"),
                                (HCOLORS[f"{kind}_b"], f"H{ib + 1} {name}")]
        cfg_len = block_cfgs.get("len", {})
        if cfg_len.get("en") and cfg_len["edge"] < n_edges:
            legend_defs.append((HCOLORS["len_e"], f"H{cfg_len['edge'] + 1} delka"))

        annotated_legend = draw_legend_cv(annotated, legend_defs)
        _, c_a2, _ = st.columns([1, 1, 1])
        with c_a2:
            if annotated_legend is not None:
                st.image(cv_to_pil(annotated_legend), use_container_width=True)

        st.markdown("---")
        if is_nok:
            st.error("MIMO TOLERANCI | Alespoň jedna kontrola byla mimo toleranci.")
        else:
            st.success("V TOLERANCI | Všechny provedené kontroly jsou v tolerancích.")

        df_export = pd.DataFrame(export_data_list)
        annotated_img = annotated_legend if annotated_legend is not None else annotated
        img_buf = io.BytesIO()
        cv_to_pil(annotated_img).save(img_buf, format="PNG")

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button("Stáhnout snímek", data=img_buf.getvalue(), file_name=f"obj{obj_i + 1}.png",
                               mime="image/png", use_container_width=True)
        with dl_col2:
            st.download_button("Stáhnout CSV", data=df_export.to_csv(index=False).encode('utf-8-sig'),
                               file_name=f"obj{obj_i + 1}_data.csv", mime="text/csv", use_container_width=True)

    if prev_nok != is_nok:
        st.session_state[f"is_nok_{obj_i}"] = is_nok
        needs_rerun = True

if needs_rerun:
    st.rerun()