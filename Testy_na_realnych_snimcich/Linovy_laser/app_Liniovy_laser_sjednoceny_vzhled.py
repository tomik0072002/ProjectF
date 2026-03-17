import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import logging
from pathlib import Path
from typing import Tuple
import streamlit as st
import plotly.graph_objects as go
import io
import time

#  Konfigurace stranky
st.set_page_config(
    page_title="Liniovy laser - analyza skenu",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Vzhled stranky - Odlehčené CSS (pouze pro info box)
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


#  Funkce pro zpracovani obrazu

def extract_laser_signal(img: np.ndarray, channel: str = 'GRAY') -> np.ndarray:
    if img.ndim == 2:
        img_f = img.astype(np.float32)
        img_min, img_max = img_f.min(), img_f.max()
        if img_max > img_min:
            return ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    if channel == 'RG':
        return cv2.subtract(img[:, :, 2], img[:, :, 1])
    elif channel == 'R':
        return img[:, :, 2].copy()
    elif channel == 'G':
        return img[:, :, 1].copy()
    elif channel == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Nezname channel: '{channel}'")


def apply_noise_filter(signal, enable_median, median_kernel, enable_gaussian, gaussian_sigma):
    if enable_median:
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        signal = cv2.medianBlur(signal, k)
    if enable_gaussian:
        signal = cv2.GaussianBlur(signal, (0, 0), gaussian_sigma)
    return signal


def get_laser_center_subpixel(img_slice: np.ndarray, threshold: int, peak_window: int) -> float:
    vals = img_slice.astype(float)
    max_idx = int(np.argmax(vals))
    max_val = vals[max_idx]
    if max_val < threshold:
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
    else:
        return np.array([
            get_laser_center_subpixel(signal_2d[:, x], threshold, peak_window)
            for x in range(signal_2d.shape[1])
        ])


def smooth_depth_map(depth_map, kernel):
    if kernel <= 1:
        return depth_map
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(depth_map.astype(np.float32), k)


def apply_min_threshold(depth_map, min_val):
    if min_val <= 0:
        return depth_map
    result = depth_map.copy()
    result[result < min_val] = 0
    return result


def remove_outliers(depth_map, sigma):
    if sigma <= 0:
        return depth_map
    nonzero = depth_map[depth_map != 0]
    if nonzero.size == 0:
        return depth_map
    med = np.median(nonzero)
    std = np.std(nonzero)
    result = depth_map.copy()
    mask = (result != 0) & (np.abs(result - med) > sigma * std)
    result[mask] = 0
    return result


def process_laser_scan(snimky_3d, params, progress_bar=None, status_text=None):
    n = len(snimky_3d)
    profiles = []

    for i, img in enumerate(snimky_3d):
        signal = extract_laser_signal(img, params['channel'])
        signal = apply_noise_filter(
            signal,
            params['enable_median'], params['median_kernel'],
            params['enable_gaussian'], params['gaussian_sigma']
        )
        profile = get_laser_profile(signal, params['threshold'], params['peak_window'], params['laser_axis'])
        profiles.append(profile)

        if progress_bar and ((i + 1) % 10 == 0 or i == n - 1):
            progress_bar.progress((i + 1) / n)
            if status_text:
                status_text.text(f"Zpracovavam snimek {i + 1} / {n}...")

    depth_map = np.array(profiles).T
    depth_map = np.nan_to_num(depth_map, nan=0.0)

    depth_map = apply_min_threshold(depth_map, params['min_value'])
    depth_map = smooth_depth_map(depth_map, params['smooth_kernel'])
    if params['outlier_sigma'] > 0:
        depth_map = remove_outliers(depth_map, params['outlier_sigma'])

    nonzero = depth_map[depth_map != 0]
    stats = {
        "n_snimku": n,
        "shape": depth_map.shape,
        "validnich_bodu": int(nonzero.size),
        "pokryti_pct": round(100 * nonzero.size / depth_map.size, 1),
        "min": round(float(nonzero.min()), 2) if nonzero.size else 0,
        "max": round(float(nonzero.max()), 2) if nonzero.size else 0,
        "mean": round(float(nonzero.mean()), 2) if nonzero.size else 0,
        "std": round(float(nonzero.std()), 2) if nonzero.size else 0,
    }

    return depth_map, stats


def create_figure(depth_map, stats, file_name, colormap):
    nonzero = depth_map[depth_map != 0]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_alpha(0.0)  # Zprůhlednění pozadí
    fig.suptitle(f"Liniový laser - sken: {file_name}", fontsize=14, fontweight='bold', color='gray')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    # Panel 1: Depth mapa
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('none')
    im = ax1.imshow(depth_map, cmap=colormap, interpolation='nearest', aspect='auto', origin='lower')
    ax1.set_title("Depth mapa", fontsize=11, color='gray', pad=8)
    ax1.set_ylabel("Pozice na senzoru [px]", fontsize=10, color='gray')
    ax1.tick_params(colors='gray')
    for spine in ax1.spines.values():
        spine.set_color('gray')
    cb = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
    cb.set_label("Poloha stredu laseru [px]", rotation=90, labelpad=12, color='gray', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='gray')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='gray')

    info = (f"Snimku: {stats['n_snimku']}   Pokryti: {stats['pokryti_pct']} %   "
            f"Min: {stats['min']}   Max: {stats['max']}   "
            f"Prumer: {stats['mean']}   Std: {stats['std']}")
    ax1.set_xlabel(f"Cislo snimku\n{info}", fontsize=9, color='gray')

    # Panel 2: Prumerny profil
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('none')
    with_data = np.where(depth_map != 0, depth_map, np.nan)
    mean_profile = np.nanmean(with_data, axis=1)
    y_pos = np.arange(len(mean_profile))
    ax2.plot(mean_profile, y_pos, color='#ff6b6b', linewidth=1.2)
    ax2.fill_betweenx(y_pos, mean_profile, alpha=0.15, color='#ff6b6b')
    ax2.set_title("Prumerny profil laseru", fontsize=11, color='gray', pad=8)
    ax2.set_xlabel("Stred laseru [px]", fontsize=10, color='gray')
    ax2.set_ylabel("Pozice na senzoru [px]", fontsize=10, color='gray')
    ax2.tick_params(colors='gray')
    ax2.grid(True, alpha=0.15, linestyle='--', color='gray')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # Panel 3: Histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('none')
    if nonzero.size > 0:
        ax3.hist(nonzero.ravel(), bins=60, color='#cc3344', edgecolor='gray', linewidth=0.4, alpha=0.85)
        ax3.axvline(stats['mean'], color='#ffaa44', lw=1.5, ls='--', label=f"Prumer = {stats['mean']}")
        ax3.axvline(stats['mean'] - stats['std'], color='#44cc88', lw=1.0, ls=':',
                    label=f"+-smerodatna odchylka = {stats['std']}")
        ax3.axvline(stats['mean'] + stats['std'], color='#44cc88', lw=1.0, ls=':')
        ax3.legend(fontsize=9, framealpha=0.2)
    ax3.set_title("Histogram hodnot", fontsize=11, color='gray', pad=8)
    ax3.set_xlabel("Hodnota [px]", fontsize=10, color='gray')
    ax3.set_ylabel("Pocet bodu", fontsize=10, color='gray')
    ax3.tick_params(colors='gray')
    ax3.grid(True, alpha=0.15, linestyle='--', color='gray')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    plt.tight_layout()
    return fig


#  3D vizualizace

def create_3d_figure(depth_map: np.ndarray, colormap: str, downsample: int = 1) -> go.Figure:
    dm = depth_map[::downsample, ::downsample].copy()

    # Nulove hodnoty
    dm_plot = np.where(dm == 0, np.nan, dm)

    # Mapovani matplotlib colormap -> plotly
    cmap_map = {
        'magma': 'Magma', 'viridis': 'Viridis', 'plasma': 'Plasma',
        'jet': 'Jet', 'inferno': 'Inferno', 'gray': 'Gray',
    }
    plotly_cmap = cmap_map.get(colormap, 'Magma')

    rows, cols = dm_plot.shape
    x = np.arange(cols) * downsample  # cislo snimku
    y = np.arange(rows) * downsample  # pozice na senzoru

    fig = go.Figure(data=[go.Surface(
        z=dm_plot,
        x=x,
        y=y,
        colorscale=plotly_cmap,
        colorbar=dict(
            title=dict(text="Poloha laseru [px]", side="right"),
            thickness=15,
        ),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5,
        ),
    )])

    # Úprava pro průhledné pozadí
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text="3D Depth mapa",
            font=dict(size=14),
            x=0.02,
        ),
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=dict(text='Cislo snimku')),
            yaxis=dict(title=dict(text='Pozice na senzoru [px]')),
            zaxis=dict(title=dict(text='Hloubka [px]'), autorange='reversed'),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2)),
        ),
        height=650,
    )

    return fig


#  Sidebar

with st.sidebar:
    st.markdown("## Laserový sken")
    st.markdown("Analýza skenu")
    st.markdown("---")

    # Vstupni soubor
    st.markdown("**Vstupní data**")
    file_path = st.text_input(
        "Cesta k souboru (.npy)",
        value="./OUT/kamera_251.npy"
    )
    file_name = Path(file_path).stem.replace("kamera_", "") if file_path else "?"

    st.markdown("---")
    # Detekce laseru
    st.markdown("**Detekce laseru**")
    laser_axis = st.selectbox(
        "Osa laseru",
        options=[0, 1],
        format_func=lambda x: "0 - Horizontalne" if x == 0 else "1 - Vertikalne"
    )
    channel = st.selectbox(
        "Kanal signalu",
        options=['GRAY', 'RG', 'R', 'G']
    )
    threshold = st.slider("Threshold (min. jas)", 0, 200, 25, 5)
    peak_window = st.slider("Peak window [px]", 1, 50, 10, 1)

    st.markdown("---")
    # Filtrovani sumu
    st.markdown("**Filtrování šumu**")
    enable_median = st.checkbox("Medianovy filtr", value=True)
    median_kernel = st.slider("Kernel medianu", 3, 11, 3, 2,
                              disabled=not enable_median)
    enable_gaussian = st.checkbox("Gaussovsky filtr", value=False)
    gaussian_sigma = st.slider("Sigma gaussu", 0.5, 5.0, 1.0, 0.5,
                               disabled=not enable_gaussian)

    st.markdown("---")
    # Post-processing
    st.markdown("**Post-processing**")
    smooth_kernel = st.select_slider(
        "Velikost kernelu pro vyhlazeni",
        options=[1, 3, 5, 7, 9],
        value=1
    )
    min_value = st.slider("Min. hodnota (vynulovani)", 0, 100, 0, 1,
                          help="Pozor: jde o souradnici polohy laseru, ne intenzitu!")
    outlier_sigma = st.slider("Outlier sigma (0 = vypnuto)", 0.0, 6.0, 0.0, 0.5,
                              help="Body dale nez N*smerodatna odchylka od medianu jsou odstraneny")

    st.markdown("---")
    # Vizualizace
    st.markdown("**Vizualizace**")
    colormap = st.selectbox("Barvova mapa", ['magma', 'viridis', 'plasma', 'jet', 'inferno', 'gray'])

    downsample_3d = st.select_slider(
        "Rozliseni 3D (downsample)",
        options=[1, 2, 4, 8],
        value=2,
        help="Vyssi hodnota = rychlejsi vykreslovani, nizsi detail"
    )

    st.markdown("---")
    run_btn = st.button("SPUSTIT ANALÝZU", use_container_width=True)
    save_npy = st.checkbox("Ulozit depth mapu (.npy)", value=True)

#  Hlavni panel
st.title("Zpracování skenu")
st.markdown(f"**Soubor:** `{file_path}`")

# Stav session
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
    st.session_state.stats = None
    st.session_state.fig_bytes = None
    st.session_state.npy_bytes = None
    st.session_state.last_file = None

# Spusteni analyzy
if run_btn:
    fp = Path(file_path)
    if not fp.exists():
        st.error(f"Soubor nenalezen: `{file_path}`")
    else:
        with st.spinner("Nacitam data..."):
            try:
                snimky = np.load(str(fp), allow_pickle=True)

                if snimky.dtype == object:
                    if snimky.ndim == 0:
                        inner = snimky.item()
                        snimky = inner if isinstance(inner, np.ndarray) else np.array(inner)
                    else:
                        snimky = np.stack(snimky)

            except Exception as e:
                st.error(f"Chyba pri nacitani souboru: {e}")
                st.stop()

        st.markdown(
            f'<div class="info-box">'
            f'Nacteno {len(snimky)} snimku &nbsp;|&nbsp; '
            f'shape: {snimky.shape} &nbsp;|&nbsp; '
            f'dtype: {snimky.dtype}'
            f'</div>',
            unsafe_allow_html=True
        )

        params = dict(
            laser_axis=laser_axis, channel=channel,
            threshold=threshold, peak_window=peak_window,
            enable_median=enable_median, median_kernel=median_kernel,
            enable_gaussian=enable_gaussian, gaussian_sigma=gaussian_sigma,
            smooth_kernel=smooth_kernel, min_value=min_value,
            outlier_sigma=outlier_sigma,
        )

        progress_bar = st.progress(0)
        status_text = st.empty()
        t0 = time.time()

        depth_map, stats = process_laser_scan(snimky, params, progress_bar, status_text)

        elapsed = time.time() - t0
        progress_bar.empty()
        status_text.empty()

        st.success(f"Hotovo za {elapsed:.1f} s")

        st.session_state.depth_map = depth_map
        st.session_state.stats = stats
        st.session_state.last_file = file_name

        fig = create_figure(depth_map, stats, file_name, colormap)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        st.session_state.fig_bytes = buf.getvalue()

        npy_buf = io.BytesIO()
        np.save(npy_buf, depth_map)
        npy_buf.seek(0)
        st.session_state.npy_bytes = npy_buf.getvalue()

# Zobrazeni vysledku
if st.session_state.depth_map is not None:
    stats = st.session_state.stats
    depth_map = st.session_state.depth_map
    fname = st.session_state.last_file

    st.markdown("### Výsledky")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Snimku", stats['n_snimku'])
    c2.metric("Pokryti", f"{stats['pokryti_pct']} %")
    c3.metric("Min [px]", stats['min'])
    c4.metric("Max [px]", stats['max'])
    c5.metric("Prumer [px]", stats['mean'])
    c6.metric("Std [px]", stats['std'])

    st.markdown("---")

    # 2D depth mapa
    st.markdown("### Depth mapa (2D)")
    if st.session_state.fig_bytes:
        st.image(st.session_state.fig_bytes, use_container_width=True)

    st.markdown("---")

    # 3D vizualizace
    st.markdown("### 3D Vizualizace")
    with st.spinner("Generuji 3D graf..."):
        fig3d = create_3d_figure(depth_map, colormap, downsample=downsample_3d)
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown(
        '<div class="info-box">Click & drag s myší = rotace | Click & Ctrl = translační pohyb | Kolečko myši = zoom</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    # Export
    st.markdown("### Export")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Stáhnout graf (.png)",
            data=st.session_state.fig_bytes,
            file_name=f"sken_{fname}_depth_map.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        if save_npy and st.session_state.npy_bytes:
            st.download_button(
                label="Stáhnout depth mapu (.npy)",
                data=st.session_state.npy_bytes,
                file_name=f"sken_{fname}_depth_map.npy",
                mime="application/octet-stream",
                use_container_width=True
            )

else:
    st.info("Zadejte cestu k souboru v postranním panelu a klikněte na **SPUSTIT ANALÝZU**.")