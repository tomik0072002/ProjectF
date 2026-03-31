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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Kalibrace – výpočet mm/pixel
# ─────────────────────────────────────────────────────────────

def compute_calibration(cal: dict) -> dict:
    """
    Vypočítá kalibrační koeficienty ze zadaných fyzických parametrů.

    Geometrie: kamera snímá kolmo shora, horizontální laserový paprsek
    dopadá na objekt pod triangulačním úhlem α od vertikály.

    ── Osa X (šířka skenované plochy – příčná osa senzoru) ──────────────
      FOV_x      = sensor_width_mm / focal_length_mm × working_distance_mm
      mm_per_px_x = FOV_x / sensor_res_x

    ── Osa Y (HLOUBKA – triangulační přepočet) ──────────────────────────
      Laser leží v rovině kamery a objektu. Horizontální vzdálenost laseru
      od optické osy kamery je `laser_offset_mm` (d), pracovní vzdálenost
      kamery je H.  Triangulační úhel:
          α = arctan(d / H)

      Výchylka středu laseru na senzoru o Δy_px odpovídá výškovému rozdílu:
          Δh = Δy_mm / tan(α)   kde  Δy_mm = Δy_px × mm_per_px_sensor_y

      Kombinovaný koeficient (px → výška v mm):
          depth_per_px = mm_per_px_sensor_y / tan(α)
                       = (FOV_y / res_y) / (d / H)
                       = (FOV_y × H) / (res_y × d)

      Pokud d = 0 (laser přímo nad objektem, žádná triangulace) → depth_per_px = ∞
      V tom případě hloubkový přepočet nelze provést a vrátíme varování.

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
    d          = cal["laser_offset_mm"]          # horizontální offset laseru od osy kamery

    if focal_mm <= 0 or res_x <= 0 or res_y <= 0:
        return {"mm_per_px_x": 1.0, "depth_per_px": 1.0, "mm_per_px_z": step_mm,
                "fov_x_mm": 0.0, "fov_y_mm": 0.0,
                "alpha_deg": 0.0, "mm_per_px_sensor_y": 1.0,
                "triangulation_valid": False, "valid": False}

    fov_x = sensor_w / focal_mm * work_mm
    fov_y = sensor_h / focal_mm * work_mm

    mm_per_px_x        = fov_x / res_x          # příčné rozlišení [mm/px]
    mm_per_px_sensor_y = fov_y / res_y          # přímé senzorové rozlišení v ose laseru [mm/px]

    # Triangulační úhel a koeficient hloubky
    if d > 0:
        alpha_rad    = np.arctan(d / work_mm)
        alpha_deg    = float(np.degrees(alpha_rad))
        depth_per_px = mm_per_px_sensor_y / np.tan(alpha_rad)   # [mm výšky / px výchylky]
        triangulation_valid = True
    else:
        alpha_deg    = 0.0
        depth_per_px = mm_per_px_sensor_y       # bez triangulace – jen lineární přepočet
        triangulation_valid = False

    return {
        "mm_per_px_x":        mm_per_px_x,
        "mm_per_px_sensor_y": mm_per_px_sensor_y,
        "depth_per_px":       depth_per_px,      # hlavní koeficient: px výchylky → mm výšky
        "mm_per_px_z":        step_mm,
        "fov_x_mm":           round(fov_x, 2),
        "fov_y_mm":           round(fov_y, 2),
        "alpha_deg":          round(alpha_deg, 2),
        "triangulation_valid": triangulation_valid,
        "valid": True,
    }


def px_to_mm_map(depth_map_px: np.ndarray, calib: dict, use_mm: bool) -> np.ndarray:
    """
    Převede depth mapu z px výchylky laseru na mm výšky objektu.

    Používá triangulační koeficient depth_per_px:
        Δh [mm] = Δy [px] × depth_per_px
    kde depth_per_px = (FOV_y / res_y) / tan(α) a α = arctan(d / H).
    Nuly (= žádný detekovaný bod) zůstanou nulami.
    """
    if not use_mm:
        return depth_map_px
    result = depth_map_px.astype(np.float64) * calib["depth_per_px"]
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
        "laser_ax":  "Výchylka laseru [px]",
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
        # Osa Z (sloupcová) = posuv objektu
        x_axis = np.arange(n_cols) * calib["mm_per_px_z"]
        # Osa senzoru (řádková) – pixely senzoru na mm
        y_axis = np.arange(n_rows) * calib["mm_per_px_x"]
    else:
        x_axis = np.arange(n_cols)
        y_axis = np.arange(n_rows)

    return x_axis, y_axis


def create_figure(depth_map_px: np.ndarray, stats_px: dict, file_name: str,
                  colormap: str, use_mm: bool, calib: dict):

    dm_disp = px_to_mm_map(depth_map_px, calib, use_mm)
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

    # Statistiky zobrazit vždy v aktuální jednotce
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
                     use_mm: bool, calib: dict) -> go.Figure:

    dm = depth_map_px[::downsample, ::downsample].copy()
    dm_disp = px_to_mm_map(dm, calib, use_mm)
    dm_plot = np.where(dm_disp == 0, np.nan, dm_disp)

    lbl = axis_labels(use_mm, calib)

    cmap_map = {'magma': 'Magma', 'viridis': 'Viridis', 'plasma': 'Plasma',
                'jet': 'Jet', 'inferno': 'Inferno', 'gray': 'Gray'}
    plotly_cmap = cmap_map.get(colormap, 'Magma')

    rows, cols = dm_plot.shape

    if use_mm:
        x = np.arange(cols) * downsample * calib["mm_per_px_z"]
        y = np.arange(rows) * downsample * calib["mm_per_px_x"]
    else:
        x = np.arange(cols) * downsample
        y = np.arange(rows) * downsample

    fig = go.Figure(data=[go.Surface(
        z=dm_plot, x=x, y=y, colorscale=plotly_cmap,
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
            xaxis=dict(title=dict(text=lbl["frame_ax"])),
            yaxis=dict(title=dict(text=lbl["sensor_ax"])),
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
    st.markdown("### ⚙️ Kalibrace hardware")
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
            format="%.2f",
            help="Ohnisková vzdálenost objektivu v mm."
        )
        working_distance_mm = st.number_input(
            "Pracovní vzdálenost (kamera → objekt) [mm]", min_value=1.0, value=300.0, step=5.0,
            format="%.1f",
            help="Vzdálenost středu objektivu od skenované plochy."
        )

        st.markdown("**Senzor kamery**")
        col_sx, col_sy = st.columns(2)
        with col_sx:
            sensor_width_mm = st.number_input(
                "Šířka senzoru [mm]", min_value=0.1, value=6.276, step=0.1,
                format="%.3f",
                help="Fyzická šířka CMOS/CCD čipu v mm."
            )
            sensor_res_x = st.number_input(
                "Rozlišení X [px]", min_value=1, value=2448, step=1,
                help="Počet pixelů na šířku snímku."
            )
        with col_sy:
            sensor_height_mm = st.number_input(
                "Výška senzoru [mm]", min_value=0.1, value=4.712, step=0.1,
                format="%.3f",
                help="Fyzická výška CMOS/CCD čipu v mm."
            )
            sensor_res_y = st.number_input(
                "Rozlišení Y [px]", min_value=1, value=2048, step=1,
                help="Počet pixelů na výšku snímku."
            )

        st.markdown("**Triangulační geometrie laseru**")
        laser_offset_mm = st.number_input(
            "Horizontální offset laseru od osy kamery [mm]", min_value=0.0, value=50.0, step=1.0,
            format="%.1f",
            help=(
                "Vzdálenost mezi optickou osou kamery a laserovým paprskem, "
                "měřeno v rovině skenování (horizontálně). "
                "Z tohoto offsetu a pracovní vzdálenosti se vypočítá triangulační úhel α = arctan(d / H). "
                "Hodnota 0 = laser přímo pod kamerou, triangulace není možná."
            )
        )

        st.markdown("**Pohyb objektu**")
        step_mm_per_frame = st.number_input(
            "Posuv mezi snímky [mm/snímek]", min_value=0.0001, value=0.1, step=0.01,
            format="%.4f",
            help="O kolik mm se objekt posune mezi dvěma po sobě jdoucími snímky."
        )

        # Sestavení dict kalibrace a výpočet
        cal_input = dict(
            focal_length_mm=focal_length_mm,
            working_distance_mm=working_distance_mm,
            sensor_width_mm=sensor_width_mm,
            sensor_height_mm=sensor_height_mm,
            sensor_res_x=sensor_res_x,
            sensor_res_y=sensor_res_y,
            laser_offset_mm=laser_offset_mm,
            step_mm_per_frame=step_mm_per_frame,
        )
        calib = compute_calibration(cal_input)

        # Zobrazení vypočtených hodnot
        if calib["valid"]:
            if calib["triangulation_valid"]:
                triang_line = (
                    f'Triangulační úhel α: <b>{calib["alpha_deg"]:.2f} °</b><br>'
                    f'Senzor Y: <b>{calib["mm_per_px_sensor_y"]:.4f} mm/px</b> (přímý přepočet senzoru)<br>'
                    f'<b>Hloubka: {calib["depth_per_px"]:.4f} mm/px výchylky</b> (po triangulaci)'
                )
            else:
                triang_line = (
                    f'<span style="color:#ffaa44">⚠ Horizontální offset = 0 – triangulace není aktivní. '
                    f'Používá se přímý senzorový přepočet: {calib["mm_per_px_sensor_y"]:.4f} mm/px.</span>'
                )
            st.markdown(
                f'<div class="cal-box">'
                f'<b>Vypočtená kalibrace:</b><br>'
                f'FOV šířka: <b>{calib["fov_x_mm"]:.2f} mm</b> '
                f'→ <b>{calib["mm_per_px_x"]:.4f} mm/px</b> (osa X – senzor)<br>'
                f'{triang_line}<br>'
                f'Posuv: <b>{calib["mm_per_px_z"]:.4f} mm/snímek</b> (osa Z – pohyb)'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("Zadej platné parametry kamery.")

    st.markdown("---")

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

    st.markdown("---")
    save_npy = st.checkbox("Uložit depth mapu ke stažení (.npy)", value=True)


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

        fig = create_figure(depth_map, stats, file_name, colormap, use_mm, calib)
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
    stats    = st.session_state.stats
    depth_map_px = st.session_state.depth_map
    fname    = st.session_state.last_file

    # Přepočet statistik do aktuální jednotky
    nz_px   = depth_map_px[depth_map_px != 0]
    if use_mm and calib["valid"]:
        scale = calib["depth_per_px"]   # triangulační koeficient px → mm výšky
        unit  = "mm"
        s_min  = round(float(nz_px.min())  * scale, 4) if nz_px.size else 0
        s_max  = round(float(nz_px.max())  * scale, 4) if nz_px.size else 0
        s_mean = round(float(nz_px.mean()) * scale, 4) if nz_px.size else 0
        s_std  = round(float(nz_px.std())  * scale, 4) if nz_px.size else 0
    else:
        unit  = "px"
        s_min, s_max  = stats["min"],  stats["max"]
        s_mean, s_std = stats["mean"], stats["std"]

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
            f'α = {calib["alpha_deg"]:.2f} ° &nbsp;|&nbsp; {calib["depth_per_px"]:.4f} mm/px výchylky'
            if calib["triangulation_valid"]
            else "⚠ Triangulace neaktivní (offset = 0)"
        )
        st.markdown(
            f'<div class="cal-box">'
            f'Kalibrace aktivní &nbsp;|&nbsp; '
            f'Senzor X: {calib["mm_per_px_x"]:.4f} mm/px &nbsp;|&nbsp; '
            f'Hloubka: {triang_note} &nbsp;|&nbsp; '
            f'Posuv Z: {calib["mm_per_px_z"]:.4f} mm/snímek'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Depth mapa (2D)")

    # Živý překreslení grafu při přepnutí px/mm bez nutnosti re-analýzy
    fig_live = create_figure(depth_map_px, stats, fname, colormap, use_mm, calib)
    buf_live = io.BytesIO()
    fig_live.savefig(buf_live, format='png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig_live)
    buf_live.seek(0)
    st.image(buf_live.getvalue(), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3D Vizualizace")
    with st.spinner("Generuji 3D graf..."):
        fig3d = create_3d_figure(depth_map_px, colormap, downsample_3d, use_mm, calib)
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
                label="Stáhnout depth mapu (.npy) – vždy v px",
                data=st.session_state.npy_bytes,
                file_name=f"sken_{fname}_depth_map.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
else:
    st.info("Zadejte cestu k souboru výše a klikněte na **SPUSTIT ANALÝZU**.")