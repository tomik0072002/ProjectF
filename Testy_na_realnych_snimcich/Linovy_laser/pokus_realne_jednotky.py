import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import io
import time

# ─────────────────────────────────────────────────────────────
#  Konfigurace stránky
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liniový laser – analýza skenu",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.info-box {
    background: rgba(128,128,128,0.1);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.8rem;
    margin: 8px 0;
}
.cal-box {
    background: rgba(60,120,220,0.08);
    border: 1px solid rgba(60,120,220,0.25);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.78rem;
    margin: 6px 0;
    color: #7ab3ff;
}
.warn-box {
    background: rgba(220,160,0,0.08);
    border: 1px solid rgba(220,160,0,0.35);
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 0.78rem;
    margin: 6px 0;
    color: #ffcc55;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Kalibrace – výpočet mm/pixel
# ─────────────────────────────────────────────────────────────

def compute_calibration(cal: dict) -> dict:
    """
    Vypočítá kalibrační koeficienty ze zadaných fyzických parametrů.

    Geometrie: kamera snímá kolmo shora, laserový paprsek dopadá na objekt
    pod triangulačním úhlem α od vertikály.

    ── OPRAVA: Výběr správné osy senzoru podle orientace laseru ──────────
      laser_axis == 0  → laser je HORIZONTÁLNÍ → střed laseru se pohybuje
                         ve směru Y snímku → používáme mm_per_px_sensor_y
      laser_axis == 1  → laser je VERTIKÁLNÍ   → střed laseru se pohybuje
                         ve směru X snímku → používáme mm_per_px_sensor_x

    ── Osa X (šířka skenované plochy – příčná osa senzoru) ──────────────
      FOV_x       = sensor_width_mm / focal_length_mm × working_distance_mm
      mm_per_px_x = FOV_x / sensor_res_x

    ── Osa Y (HLOUBKA – triangulační přepočet) ──────────────────────────
      Triangulační úhel:
          α = arctan(laser_offset_mm / working_distance_mm)

      Výchylka středu laseru na senzoru o Δ_px odpovídá výškovému rozdílu:
          Δh = Δ_mm / tan(α)   kde Δ_mm = Δ_px × mm_per_px_sensor_[xy]

      Kombinovaný koeficient (px výchylky → mm výšky objektu):
          depth_per_px = mm_per_px_sensor_[xy] / tan(α)

      DŮLEŽITÉ: Δ_px = (naměřená pozice laseru) − (referenční pozice laseru
      na rovném povrchu). Tento odečet se provádí v px_to_mm_map(), ne zde.

    ── Osa Z (posuv objektu mezi snímky) ────────────────────────────────
      Zadán přímo jako mm/snímek.
    """
    focal_mm   = cal["focal_length_mm"]
    work_mm    = cal["working_distance_mm"]
    sensor_w   = cal["sensor_width_mm"]
    sensor_h   = cal["sensor_height_mm"]
    res_x      = cal["sensor_res_x"]
    res_y      = cal["sensor_res_y"]
    step_mm    = cal["step_mm_per_frame"]
    d          = cal["laser_offset_mm"]
    laser_axis = cal["laser_axis"]           # ← NOVÉ: 0 = horizontální laser, 1 = vertikální

    if focal_mm <= 0 or res_x <= 0 or res_y <= 0:
        return {"mm_per_px_x": 1.0, "depth_per_px": 1.0, "mm_per_px_z": step_mm,
                "fov_x_mm": 0.0, "fov_y_mm": 0.0,
                "alpha_deg": 0.0, "mm_per_px_sensor_laser": 1.0,
                "triangulation_valid": False, "valid": False}

    fov_x = sensor_w / focal_mm * work_mm
    fov_y = sensor_h / focal_mm * work_mm

    mm_per_px_sensor_x = fov_x / res_x      # přímé senzorové rozlišení v ose X [mm/px]
    mm_per_px_sensor_y = fov_y / res_y      # přímé senzorové rozlišení v ose Y [mm/px]
    mm_per_px_x        = mm_per_px_sensor_x  # příčné rozlišení (pro zobrazení osy senzoru)

    # ── OPRAVA č. 1: volba správné senzorové osy podle orientace laseru ──
    # Horizontální laser (axis=0): střed laseru se mění v ose Y snímku → sensor_y
    # Vertikální laser  (axis=1): střed laseru se mění v ose X snímku → sensor_x
    if laser_axis == 0:
        mm_per_px_sensor_laser = mm_per_px_sensor_y
    else:
        mm_per_px_sensor_laser = mm_per_px_sensor_x

    # Triangulační úhel a koeficient hloubky
    if d > 0:
        alpha_rad    = np.arctan(d / work_mm)
        alpha_deg    = float(np.degrees(alpha_rad))
        # depth_per_px: o kolik mm výšky objektu odpovídá 1 px výchylky laseru
        # Δh [mm] = Δ_sensor [mm] / tan(α) = Δ_px × mm_per_px_sensor_laser / tan(α)
        depth_per_px = mm_per_px_sensor_laser / np.tan(alpha_rad)
        triangulation_valid = True
    else:
        alpha_deg    = 0.0
        # Bez offsetu nelze triangulovat – vrátíme přímý senzorový přepočet
        # a nastavíme varování; výsledky v mm budou mít pouze lineární smysl.
        depth_per_px = mm_per_px_sensor_laser
        triangulation_valid = False

    return {
        "mm_per_px_x":             mm_per_px_x,
        "mm_per_px_sensor_x":      mm_per_px_sensor_x,
        "mm_per_px_sensor_y":      mm_per_px_sensor_y,
        "mm_per_px_sensor_laser":  mm_per_px_sensor_laser,  # osa, ve které se pohybuje laser
        "depth_per_px":            depth_per_px,             # hlavní koeficient: px výchylky → mm výšky
        "mm_per_px_z":             step_mm,
        "fov_x_mm":                round(fov_x, 2),
        "fov_y_mm":                round(fov_y, 2),
        "alpha_deg":               round(alpha_deg, 2),
        "laser_axis":              laser_axis,
        "triangulation_valid":     triangulation_valid,
        "valid": True,
    }


def px_to_mm_map(depth_map_px: np.ndarray, calib: dict, use_mm: bool,
                 laser_ref_px: float = 0.0) -> np.ndarray:
    """
    Převede depth mapu z absolutní pozice laseru [px] na výšku objektu [mm].

    ── OPRAVA č. 2: odečtení referenční pozice ──────────────────────────
      depth_map_px obsahuje ABSOLUTNÍ pozici středu laseru na senzoru [px].
      Pro výpočet výšky objektu potřebujeme RELATIVNÍ výchylku od referenční
      polohy (poloha laseru na rovném kalibračním povrchu):

          Δ_px = depth_map_px − laser_ref_px

      Pak platí:
          Δh [mm] = Δ_px × depth_per_px

      Záporná výchylka = objekt je výše než referenční povrch (laser se na
      senzoru posunul opačným směrem). Nuly (= žádný detekovaný bod)
      zůstanou nulami i po přepočtu.

    Parametry
    ----------
    depth_map_px  : 2D pole absolutních pozic laseru [px]; 0 = bez detekce
    calib         : kalibrační slovník z compute_calibration()
    use_mm        : True → přepočítat na mm; False → vrátit Δ_px
    laser_ref_px  : referenční poloha laseru na rovném povrchu [px]
                    (měřeno za stejných podmínek jako sken)
    """
    if not use_mm:
        # I v px režimu vrátíme relativní výchylku, aby byl nulový bod smysluplný
        result = depth_map_px.astype(np.float64) - laser_ref_px
        result[depth_map_px == 0] = 0.0
        return result

    # Relativní výchylka [px] → výška objektu [mm]
    delta_px = depth_map_px.astype(np.float64) - laser_ref_px
    result   = delta_px * calib["depth_per_px"]
    result[depth_map_px == 0] = 0.0
    return result


def axis_labels(use_mm: bool, calib: dict):
    """Vrátí popisky os a jednotku pro aktuální režim."""
    if use_mm:
        return {
            "sensor_ax": "Pozice na senzoru [mm]",
            "laser_ax":  "Výška objektu Δh [mm]",
            "frame_ax":  "Posuv objektu [mm]",
            "unit":      "mm",
        }
    return {
        "sensor_ax": "Pozice na senzoru [px]",
        "laser_ax":  "Výchylka laseru Δ [px]",   # upřesněno: jde o výchylku, ne absolutní pozici
        "frame_ax":  "Číslo snímku",
        "unit":      "px",
    }


# ─────────────────────────────────────────────────────────────
#  Zpracování obrazu
# ─────────────────────────────────────────────────────────────

def extract_laser_signal(img: np.ndarray, channel: str = 'GRAY') -> np.ndarray:
    if img.ndim == 2:
        img_f = img.astype(np.float32)
        mn, mx = img_f.min(), img_f.max()
        if mx > mn:
            return ((img_f - mn) / (mx - mn) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)
    if channel == 'RG':
        return cv2.subtract(img[:, :, 2], img[:, :, 1])
    elif channel == 'R':
        return img[:, :, 2].copy()
    elif channel == 'G':
        return img[:, :, 1].copy()
    elif channel == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Neznámý channel: '{channel}'")


def get_laser_center_subpixel(img_slice: np.ndarray, threshold: int, peak_window: int) -> float:
    vals = img_slice.astype(float)
    max_idx = int(np.argmax(vals))
    if vals[max_idx] < threshold:
        return np.nan
    s = max(0, max_idx - peak_window)
    e = min(len(vals), max_idx + peak_window + 1)
    w_vals = vals[s:e]
    w_idxs = np.arange(s, e, dtype=float)
    total = w_vals.sum()
    return float(np.dot(w_idxs, w_vals) / total) if total > 0 else np.nan


def get_laser_profile(signal_2d, threshold, peak_window, axis):
    if axis == 0:
        return np.array([
            get_laser_center_subpixel(signal_2d[y, :], threshold, peak_window)
            for y in range(signal_2d.shape[0])
        ])
    return np.array([
        get_laser_center_subpixel(signal_2d[:, x], threshold, peak_window)
        for x in range(signal_2d.shape[1])
    ])


def process_laser_scan(snimky_3d, params, progress_bar=None, status_text=None):
    n = len(snimky_3d)
    profiles = []

    for i, img in enumerate(snimky_3d):
        h, w = img.shape[:2]
        ct, cb = params['crop_t'], params['crop_b']
        cl, cr = params['crop_l'], params['crop_r']
        img_cropped = img[ct:h - cb, cl:w - cr]

        if params['alpha'] != 1.0 or params['beta'] != 0:
            img_cropped = cv2.convertScaleAbs(img_cropped, alpha=params['alpha'], beta=params['beta'])

        signal = extract_laser_signal(img_cropped, params['channel'])

        if params['median_k'] > 1:
            k = params['median_k'] | 1
            signal = cv2.medianBlur(signal, k)
        if params['gauss_sigma'] > 0:
            signal = cv2.GaussianBlur(signal, (0, 0), params['gauss_sigma'])

        if params['morph_k'] > 1:
            k_size = params['morph_k'] | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            if params['morph_op'] == 'Otevření (odstraní šum)':
                signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
            elif params['morph_op'] == 'Uzavření (spojí čáru)':
                signal = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, kernel)
            elif params['morph_op'] == 'Dilatace (ztloustnutí)':
                signal = cv2.dilate(signal, kernel, iterations=1)

        profile_cropped = get_laser_profile(signal, params['threshold'], params['peak_window'], params['laser_axis'])

        if params['laser_axis'] == 0:
            full_len = h
            offset_pos = ct
            offset_val = cl
        else:
            full_len = w
            offset_pos = cl
            offset_val = ct

        profile = np.full(full_len, np.nan)
        corrected = np.where(np.isnan(profile_cropped), np.nan, profile_cropped + offset_val)
        end_pos = offset_pos + len(corrected)
        profile[offset_pos:end_pos] = corrected
        profiles.append(profile)

        if progress_bar and ((i + 1) % 10 == 0 or i == n - 1):
            progress_bar.progress((i + 1) / n)
            if status_text:
                status_text.text(f"Zpracovávám snímek {i + 1} / {n}...")

    depth_map = np.array(profiles).T
    depth_map = np.nan_to_num(depth_map, nan=0.0)

    if params['smooth_kernel'] > 1:
        k = params['smooth_kernel'] | 1
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), k)

    if params['outlier_sigma'] > 0:
        nonzero = depth_map[depth_map != 0]
        if nonzero.size > 0:
            med = np.median(nonzero)
            std = np.std(nonzero)
            mask = (depth_map != 0) & (np.abs(depth_map - med) > params['outlier_sigma'] * std)
            depth_map[mask] = 0

    nonzero = depth_map[depth_map != 0]
    stats = {
        "n_snimku": n,
        "shape": depth_map.shape,
        "validnich_bodu": int(nonzero.size),
        "pokryti_pct": round(100 * nonzero.size / depth_map.size, 1),
        "min": round(float(nonzero.min()), 4) if nonzero.size else 0,
        "max": round(float(nonzero.max()), 4) if nonzero.size else 0,
        "mean": round(float(nonzero.mean()), 4) if nonzero.size else 0,
        "std": round(float(nonzero.std()), 4) if nonzero.size else 0,
    }

    return depth_map, stats


# ─────────────────────────────────────────────────────────────
#  Vizualizace
# ─────────────────────────────────────────────────────────────

def build_axes(depth_map_disp: np.ndarray, use_mm: bool, calib: dict):
    """
    Vrátí osy X (snímky → mm nebo idx) a Y (senzor → mm nebo px)
    pro zobrazení depth mapy.
    """
    n_rows, n_cols = depth_map_disp.shape  # rows = senzor, cols = snímky

    if use_mm:
        x_axis = np.arange(n_cols) * calib["mm_per_px_z"]
        y_axis = np.arange(n_rows) * calib["mm_per_px_x"]
    else:
        x_axis = np.arange(n_cols)
        y_axis = np.arange(n_rows)

    return x_axis, y_axis


def create_figure(depth_map_px: np.ndarray, stats_px: dict, file_name: str,
                  colormap: str, use_mm: bool, calib: dict, laser_ref_px: float):

    dm_disp = px_to_mm_map(depth_map_px, calib, use_mm, laser_ref_px)
    lbl = axis_labels(use_mm, calib)
    x_axis, y_axis = build_axes(dm_disp, use_mm, calib)

    nonzero_disp = dm_disp[dm_disp != 0]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_alpha(0.0)
    fig.suptitle(f"Liniový laser – sken: {file_name}", fontsize=14, fontweight='bold', color='gray')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    # Panel 1 – Depth mapa
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('none')
    extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]
    im = ax1.imshow(dm_disp, cmap=colormap, interpolation='nearest',
                    aspect='auto', origin='lower', extent=extent)
    ax1.set_title("Depth mapa", fontsize=11, color='gray', pad=8)
    ax1.set_ylabel(lbl["sensor_ax"], fontsize=10, color='gray')
    ax1.tick_params(colors='gray')
    for spine in ax1.spines.values():
        spine.set_color('gray')
    cb = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
    cb.set_label(lbl["laser_ax"], rotation=90, labelpad=12, color='gray', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='gray')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='gray')

    nz = nonzero_disp
    s_min  = round(float(nz.min()),  4) if nz.size else 0
    s_max  = round(float(nz.max()),  4) if nz.size else 0
    s_mean = round(float(nz.mean()), 4) if nz.size else 0
    s_std  = round(float(nz.std()),  4) if nz.size else 0
    u = lbl["unit"]

    info = (f"Snímků: {stats_px['n_snimku']}   Pokrytí: {stats_px['pokryti_pct']} %   "
            f"Min: {s_min} {u}   Max: {s_max} {u}   "
            f"Průměr: {s_mean} {u}   Std: {s_std} {u}")
    ax1.set_xlabel(f"{lbl['frame_ax']}\n{info}", fontsize=9, color='gray')

    # Panel 2 – Průměrný profil
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('none')
    with_data = np.where(dm_disp != 0, dm_disp, np.nan)
    mean_profile = np.nanmean(with_data, axis=1)
    ax2.plot(mean_profile, y_axis, color='#ff6b6b', linewidth=1.2)
    ax2.fill_betweenx(y_axis, mean_profile, alpha=0.15, color='#ff6b6b')
    ax2.set_title("Průměrný profil laseru", fontsize=11, color='gray', pad=8)
    ax2.set_xlabel(lbl["laser_ax"], fontsize=10, color='gray')
    ax2.set_ylabel(lbl["sensor_ax"], fontsize=10, color='gray')
    ax2.tick_params(colors='gray')
    ax2.grid(True, alpha=0.15, linestyle='--', color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # Panel 3 – Histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('none')
    if nz.size > 0:
        ax3.hist(nz.ravel(), bins=60, color='#cc3344', edgecolor='gray', linewidth=0.4, alpha=0.85)
        ax3.axvline(s_mean, color='#ffaa44', lw=1.5, ls='--', label=f"Průměr = {s_mean} {u}")
        ax3.axvline(s_mean - s_std, color='#44cc88', lw=1.0, ls=':',
                    label=f"± směr. odch. = {s_std} {u}")
        ax3.axvline(s_mean + s_std, color='#44cc88', lw=1.0, ls=':')
        ax3.legend(fontsize=9, framealpha=0.2)
    ax3.set_title("Histogram hodnot", fontsize=11, color='gray', pad=8)
    ax3.set_xlabel(f"Hodnota [{u}]", fontsize=10, color='gray')
    ax3.set_ylabel("Počet bodů", fontsize=10, color='gray')
    ax3.tick_params(colors='gray')
    ax3.grid(True, alpha=0.15, linestyle='--', color='gray')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    return fig


def create_3d_figure(depth_map_px: np.ndarray, colormap: str, downsample: int,
                     use_mm: bool, calib: dict, laser_ref_px: float,
                     z_clip_sigma: float = 3.0) -> go.Figure:
    """
    Parametry
    ----------
    z_clip_sigma : float
        Outliery nad/pod (medián ± z_clip_sigma × std) jsou nahrazeny NaN,
        aby neroztáhly rozsah osy Z. 0 = žádné oříznutí.
    """
    dm = depth_map_px[::downsample, ::downsample].copy()
    dm_disp = px_to_mm_map(dm, calib, use_mm, laser_ref_px)

    # Nuly → NaN (žádná detekce)
    dm_plot = np.where(dm_disp == 0, np.nan, dm_disp)

    # ── Oříznutí outlierů v ose Z ─────────────────────────────────────────
    # Svislé špičky (outliery) vznikají chybnou detekcí laseru a výrazně
    # zkreslují měřítko osy Z. Nahradíme je NaN, graf pak zobrazí skutečný
    # povrch objektu bez deformace měřítka.
    if z_clip_sigma > 0:
        valid = dm_plot[np.isfinite(dm_plot)]
        if valid.size > 0:
            med = np.median(valid)
            std = np.std(valid)
            lo  = med - z_clip_sigma * std
            hi  = med + z_clip_sigma * std
            dm_plot = np.where((dm_plot < lo) | (dm_plot > hi), np.nan, dm_plot)

    lbl = axis_labels(use_mm, calib)

    cmap_map = {'magma': 'Magma', 'viridis': 'Viridis', 'plasma': 'Plasma',
                'jet': 'Jet', 'inferno': 'Inferno', 'gray': 'Gray'}
    plotly_cmap = cmap_map.get(colormap, 'Magma')

    rows, cols = dm_plot.shape

    # ── Osy X a Y ─────────────────────────────────────────────────────────
    # go.Surface: matice dm_plot má tvar (rows, cols).
    #   rows = osa senzoru  → přiřadíme ose Y scény  → "Pozice na senzoru"
    #   cols = osa snímků   → přiřadíme ose X scény  → "Posuv objektu"
    # Dříve bylo x←cols, y←rows, ale popisky xaxis/yaxis byly prohozeny,
    # takže v grafu stál "Posuv objektu" podél senzorové osy a naopak.
    if use_mm:
        x_sensor = np.arange(rows) * downsample * calib["mm_per_px_x"]   # senzor → osa X
        y_frame  = np.arange(cols) * downsample * calib["mm_per_px_z"]   # snímky → osa Y
    else:
        x_sensor = np.arange(rows) * downsample
        y_frame  = np.arange(cols) * downsample

    # go.Surface očekává: x má délku cols (druhá dimenze), y má délku rows (první dimenze)
    # Proto transponujeme matici a přiřadíme osy správně:
    #   x = x_sensor (délka rows → po transponování cols)
    #   y = y_frame  (délka cols → po transponování rows)
    fig = go.Figure(data=[go.Surface(
        z=dm_plot.T,          # transponujeme: (rows,cols) → (cols,rows) → x=senzor, y=snímky
        x=x_sensor,           # osa X = pozice na senzoru
        y=y_frame,            # osa Y = posuv objektu (snímky)
        colorscale=plotly_cmap,
        colorbar=dict(title=dict(text=lbl["laser_ax"], side="right"), thickness=15),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="3D Depth mapa", font=dict(size=14), x=0.02),
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=dict(text=lbl["sensor_ax"])),   # X = senzor
            yaxis=dict(title=dict(text=lbl["frame_ax"])),    # Y = posuv
            zaxis=dict(title=dict(text=lbl["laser_ax"]), autorange='reversed'),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        height=650,
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Laserový sken")
    st.markdown("Analýza skenu")
    st.markdown("---")

    # ── Kalibrace ──────────────────────────────────────────────
    st.markdown("### Nastavení triangulace")
    with st.expander("Parametry kamery a optiky", expanded=True):
        st.caption(
            "Fyzické parametry optické soustavy. "
            "Používají se k přepočtu px → mm pro všechny osy."
        )

        use_mm = st.toggle("Zobrazit výsledky v mm", value=False,
                           help="Přepíná osy a statistiky mezi px a mm.")

        st.markdown("**Optika**")
        focal_length_mm = st.number_input(
            "Ohnisková vzdálenost [mm]", min_value=0.1, value=8.0, step=0.5,
            format="%.2f"
        )
        working_distance_mm = st.number_input(
            "Pracovní vzdálenost (kamera → objekt) [mm]", min_value=1.0, value=1000.0, step=5.0,
            format="%.1f"
        )

        st.markdown("**Senzor kamery**")
        col_sx, col_sy = st.columns(2)
        with col_sx:
            sensor_width_mm = st.number_input(
                "Šířka senzoru [mm]", min_value=0.1, value=5.02, step=0.1,
                format="%.3f"
            )
            sensor_res_x = st.number_input(
                "Rozlišení X [px]", min_value=1, value=1456, step=1
            )
        with col_sy:
            sensor_height_mm = st.number_input(
                "Výška senzoru [mm]", min_value=0.1, value=3.75, step=0.1,
                format="%.3f"
            )
            sensor_res_y = st.number_input(
                "Rozlišení Y [px]", min_value=1, value=1088, step=1
            )

        st.markdown("**Triangulační geometrie laseru**")
        laser_offset_mm = st.number_input(
            "Horizontální offset laseru od osy kamery [mm]", min_value=0.0, value=300.0, step=1.0,
            format="%.1f"
        )

        # ── NOVÉ: Referenční pozice laseru ─────────────────────────────
        st.markdown("**Referenční poloha laseru**")
        st.caption(
            "Poloha středu laseru na senzoru při skenování rovného "
            "referenčního povrchu ve stejné pracovní vzdálenosti [px]. "
            "Slouží jako nulová hladina pro výpočet výšky objektu. "
            "Pokud neznáte hodnotu, spusťte sken rovné plochy a odečtěte "
            "průměrnou hodnotu z depth mapy."
        )
        laser_ref_px = st.number_input(
            "Referenční pozice laseru [px]", min_value=0.0, value=0.0, step=1.0,
            format="%.2f",
            help=(
                "Absolutní px pozice laseru na rovném povrchu. "
                "Výsledná depth mapa = (naměřená pozice) − (tato hodnota). "
                "Při hodnotě 0 jsou výsledky absolutní pozice laseru, ne výška."
            )
        )
        if laser_ref_px == 0.0 and use_mm:
            st.markdown(
                '<div class="warn-box">⚠ Referenční pozice = 0. '
                'Výsledky v mm jsou absolutní poloha laseru, '
                'ne výška objektu. Zadejte referenční hodnotu pro správný přepočet.</div>',
                unsafe_allow_html=True
            )

        st.markdown("**Pohyb objektu**")
        step_mm_per_frame = st.number_input(
            "Posuv mezi snímky [mm/snímek]", min_value=0.0001, value=0.25, step=0.01,
            format="%.4f"
        )

    # ── Rozsah snímků ──────────────────────────────────────────
    with st.expander("Rozsah snímků", expanded=False):
        st.caption("Omezí zpracování pouze na vybraný rozsah snímků ze souboru.")
        frame_start = st.number_input("Od snímku", min_value=0, value=0, step=1,
                                      help="Index prvního snímku (0 = začátek).")
        frame_end = st.number_input("Do snímku", min_value=0, value=0, step=1,
                                    help="Index posledního snímku (včetně). 0 = vše.")

    # ── ROI ────────────────────────────────────────────────────
    with st.expander("Oříznutí obrazu (ROI)", expanded=False):
        st.caption("Omezí zpracování pouze na vybranou oblast snímku.")
        crop_t = st.number_input("Zdola v ose: Pozice na senzoru [px]", 0, step=10)
        crop_b = st.number_input("Shora v ose: Pozice na senzoru [px]", 0, step=10)
        crop_l = st.number_input("Zdola v ose: Poloha laseru [px]", 0, step=10)
        crop_r = st.number_input("Zhora v ose: Poloha laseru [px]", 0, step=10)

    # ── Filtry ─────────────────────────────────────────────────
    with st.expander("Korekce a Filtry (2D)", expanded=False):
        st.caption("Úpravy snímku před detekcí")
        alpha = st.slider("Kontrast (Alpha)", 0.5, 1.4, 1.0, 0.1)
        beta = st.slider("Jas (Beta)", -100, 100, 0, 5)
        st.markdown("---")
        median_k = st.slider("Medián filtr (Kernel)", 1, 15, 1, 2, help="1 = Vypnuto")
        gauss_sigma = st.slider("Gaussovo rozostření (Sigma)", 0.0, 5.0, 0.0, 0.5, help="0 = Vypnuto")
        st.markdown("---")
        morph_op = st.selectbox("Morfologická operace",
                                ['Žádná', 'Otevření (odstraní šum)', 'Uzavření (spojí čáru)',
                                 'Dilatace (ztloustnutí)'])
        morph_k = st.slider("Velikost morfologie (Kernel)", 1, 15, 3, 2,
                            help="Aktivní pouze pokud vyberete operaci") if morph_op != 'Žádná' else 1

    # ── Detekce ────────────────────────────────────────────────
    with st.expander("Detekce Laseru", expanded=False):
        st.caption("Parametry pro nalezení středu čáry")
        laser_axis = st.selectbox("Osa laseru", options=[0, 1],
                                  format_func=lambda x: "0 – Horizontálně" if x == 0 else "1 – Vertikálně")
        channel = st.selectbox("Kanál signálu", options=['GRAY', 'RG', 'R', 'G'])
        threshold = st.slider("Práh (Min. jas)", 0, 70, 10, 5)
        peak_window = st.slider("Prohledávací okno [px]", 1, 50, 10, 1,
                                help="Šířka okolí maxima pro výpočet těžiště")

    # ── Post-processing ────────────────────────────────────────
    with st.expander("Post-processing Depth Mapy", expanded=False):
        st.caption("Úpravy hotové 3D mapy povrchu")
        smooth_kernel = st.slider("Vyhlazení povrchu (Medián)", 1, 15, 1, 2, help="1 = Vypnuto")
        outlier_sigma = st.slider("Filtrace odchylek (Sigma)", 0.0, 6.0, 0.0, 0.5,
                                  help="0 = Vypnuto. Odstraní body mimo N×směrodatná odchylka")

    st.markdown("---")

    # ── Vizualizace ────────────────────────────────────────────
    st.markdown("**Vizualizace**")
    colormap = st.selectbox("Barvová mapa", ['magma', 'viridis', 'plasma', 'jet', 'inferno', 'gray'])
    downsample_3d = st.select_slider("Rozlišení 3D vizualizace", options=[1, 2, 4, 8], value=2,
                                     help="Vyšší hodnota = plynulejší 3D model, ale menší detail")
    z_clip_sigma = st.slider(
        "Oříznutí outlierů v ose Z (Sigma)", 0.0, 6.0, 3.0, 0.5,
        help=(
            "Odstraní bodové výstřelky (špatně detekované pozice laseru) z 3D grafu. "
            "Body mimo (medián ± N × std) jsou nahrazeny NaN. "
            "0 = žádné oříznutí."
        )
    )

    st.markdown("---")
    save_npy = st.checkbox("Uložit depth mapu ke stažení (.npy)", value=True)

    # ── Sestavení kalibrace (musí být po laser_axis) ────────────────────
    cal_input = dict(
        focal_length_mm=focal_length_mm,
        working_distance_mm=working_distance_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        sensor_res_x=sensor_res_x,
        sensor_res_y=sensor_res_y,
        laser_offset_mm=laser_offset_mm,
        step_mm_per_frame=step_mm_per_frame,
        laser_axis=laser_axis,               # ← předáváme orientaci laseru
    )
    calib = compute_calibration(cal_input)


# ─────────────────────────────────────────────────────────────
#  Hlavní panel
# ─────────────────────────────────────────────────────────────
st.title("Zpracování skenu")

input_col, btn_col = st.columns([4, 1])
with input_col:
    file_path = st.text_input(
        "Cesta k souboru (.npy)", value="./OUT/kamera_251.npy",
        label_visibility="collapsed", placeholder="Cesta k souboru (.npy)"
    )
with btn_col:
    run_btn = st.button("SPUSTIT ANALÝZU", use_container_width=True)

file_name = Path(file_path).stem.replace("kamera_", "") if file_path else "?"
st.markdown(f"**Soubor:** `{file_path}`")

# Stav session
for key in ('depth_map', 'stats', 'fig_bytes', 'npy_bytes', 'last_file'):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Spuštění analýzy ───────────────────────────────────────────
if run_btn:
    fp = Path(file_path)
    if not fp.exists():
        st.error(f"Soubor nenalezen: `{file_path}`")
    else:
        with st.spinner("Načítám a zpracovávám data..."):
            try:
                snimky = np.load(str(fp), allow_pickle=True)
                if snimky.dtype == object:
                    if snimky.ndim == 0:
                        inner = snimky.item()
                        snimky = inner if isinstance(inner, np.ndarray) else np.array(inner)
                    else:
                        snimky = np.stack(snimky)
            except Exception as e:
                st.error(f"Chyba při načítání souboru: {e}")
                st.stop()

        st.markdown(
            f'<div class="info-box">'
            f'Načteno {len(snimky)} snímků &nbsp;|&nbsp; '
            f'Rozlišení: {snimky.shape} &nbsp;|&nbsp; '
            f'Dtype: {snimky.dtype}'
            f'</div>',
            unsafe_allow_html=True
        )

        fs = int(frame_start)
        fe = int(frame_end) + 1 if int(frame_end) > 0 else len(snimky)
        fe = min(fe, len(snimky))
        if fs >= fe:
            st.error(f"Neplatný rozsah snímků: od {fs} do {fe - 1}.")
            st.stop()
        snimky = snimky[fs:fe]
        if fs > 0 or int(frame_end) > 0:
            st.markdown(
                f'<div class="info-box">Použit rozsah: {fs} – {fe - 1} &nbsp;|&nbsp; '
                f'Zpracováno: {len(snimky)} snímků</div>',
                unsafe_allow_html=True
            )

        params = dict(
            crop_t=crop_t, crop_b=crop_b, crop_l=crop_l, crop_r=crop_r,
            alpha=alpha, beta=beta,
            median_k=median_k, gauss_sigma=gauss_sigma,
            morph_op=morph_op, morph_k=morph_k,
            laser_axis=laser_axis, channel=channel,
            threshold=threshold, peak_window=peak_window,
            outlier_sigma=outlier_sigma,
            smooth_kernel=smooth_kernel,
        )

        progress_bar = st.progress(0)
        status_text = st.empty()
        t0 = time.time()

        depth_map, stats = process_laser_scan(snimky, params, progress_bar, status_text)

        elapsed = time.time() - t0
        progress_bar.empty()
        status_text.empty()
        st.success(f"Analýza dokončena za {elapsed:.1f} s")

        st.session_state.depth_map = depth_map
        st.session_state.stats = stats
        st.session_state.last_file = file_name

        fig = create_figure(depth_map, stats, file_name, colormap, use_mm, calib, laser_ref_px)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        st.session_state.fig_bytes = buf.getvalue()

        npy_buf = io.BytesIO()
        np.save(npy_buf, depth_map)
        npy_buf.seek(0)
        st.session_state.npy_bytes = npy_buf.getvalue()

# ── Zobrazení výsledků ─────────────────────────────────────────
if st.session_state.depth_map is not None:
    stats        = st.session_state.stats
    depth_map_px = st.session_state.depth_map
    fname        = st.session_state.last_file

    # Přepočet statistik do aktuální jednotky (s referenční korekcí)
    nz_px = depth_map_px[depth_map_px != 0]
    if use_mm and calib["valid"]:
        scale  = calib["depth_per_px"]
        unit   = "mm"
        # Odečteme referenci před statistikami (stejná logika jako px_to_mm_map)
        delta  = nz_px - laser_ref_px
        s_min  = round(float(delta.min())  * scale, 4) if nz_px.size else 0
        s_max  = round(float(delta.max())  * scale, 4) if nz_px.size else 0
        s_mean = round(float(delta.mean()) * scale, 4) if nz_px.size else 0
        s_std  = round(float(delta.std())  * scale, 4) if nz_px.size else 0
    else:
        unit  = "px"
        # I v px režimu zobrazíme výchylku (odečtená reference)
        delta = nz_px - laser_ref_px
        s_min  = round(float(delta.min()),  4) if nz_px.size else 0
        s_max  = round(float(delta.max()),  4) if nz_px.size else 0
        s_mean = round(float(delta.mean()), 4) if nz_px.size else 0
        s_std  = round(float(delta.std()),  4) if nz_px.size else 0

    st.markdown("### Statistické hodnoty")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Snímků",   stats["n_snimku"])
    c2.metric("Pokrytí",  f"{stats['pokryti_pct']} %")
    c3.metric(f"Min [{unit}]",    s_min)
    c4.metric(f"Max [{unit}]",    s_max)
    c5.metric(f"Průměr [{unit}]", s_mean)
    c6.metric(f"Std [{unit}]",    s_std)

    # Kalibrační přehled
    if use_mm and calib["valid"]:
        triang_note = (
            f'α = {calib["alpha_deg"]:.2f} ° &nbsp;|&nbsp; '
            f'{calib["depth_per_px"]:.4f} mm/px výchylky &nbsp;|&nbsp; '
            f'Senzorová osa laseru: {"Y" if calib["laser_axis"] == 0 else "X"} '
            f'({calib["mm_per_px_sensor_laser"]:.4f} mm/px)'
            if calib["triangulation_valid"]
            else "⚠ Triangulace neaktivní (offset = 0)"
        )
        st.markdown(
            f'<div class="cal-box">'
            f'Kalibrace aktivní &nbsp;|&nbsp; '
            f'Senzor X: {calib["mm_per_px_x"]:.4f} mm/px &nbsp;|&nbsp; '
            f'Hloubka: {triang_note} &nbsp;|&nbsp; '
            f'Posuv Z: {calib["mm_per_px_z"]:.4f} mm/snímek &nbsp;|&nbsp; '
            f'Ref. laser: {laser_ref_px:.1f} px'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Depth mapa (2D)")

    fig_live = create_figure(depth_map_px, stats, fname, colormap, use_mm, calib, laser_ref_px)
    buf_live = io.BytesIO()
    fig_live.savefig(buf_live, format='png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig_live)
    buf_live.seek(0)
    st.image(buf_live.getvalue(), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3D Vizualizace")
    with st.spinner("Generuji 3D graf..."):
        fig3d = create_3d_figure(depth_map_px, colormap, downsample_3d, use_mm, calib, laser_ref_px,
                                 z_clip_sigma=z_clip_sigma)
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown(
        '<div class="info-box">Click & drag = rotace &nbsp;|&nbsp; '
        'Ctrl + drag = translace &nbsp;|&nbsp; Kolečko = zoom</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Export")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Stáhnout 2D vizualizaci (.png)",
            data=buf_live.getvalue(),
            file_name=f"sken_{fname}_depth_map.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        if save_npy and st.session_state.npy_bytes:
            st.download_button(
                label="Stáhnout depth mapu (.npy) – vždy v px (absolutní pozice)",
                data=st.session_state.npy_bytes,
                file_name=f"sken_{fname}_depth_map.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
else:
    st.info("Zadejte cestu k souboru výše a klikněte na **SPUSTIT ANALÝZU**.")