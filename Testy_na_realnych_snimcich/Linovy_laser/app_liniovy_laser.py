import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import logging
from pathlib import Path
from typing import Tuple
import streamlit as st
import io
import time

#  Konfigurace stránky

st.set_page_config(
    page_title="Liniový laser – analýza skenu",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Vzhled stránky

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0d0d0f;
    color: #e8e8e8;
}

section[data-testid="stSidebar"] {
    background: #111115;
    border-right: 1px solid #2a2a35;
}

section[data-testid="stSidebar"] * {
    color: #c8c8d0 !important;
}

h1, h2, h3 {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.5px;
}

.main-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ff4444;
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1;
}

.sub-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #555566;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.stat-card {
    background: #16161c;
    border: 1px solid #2a2a35;
    border-left: 3px solid #ff4444;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 4px 0;
    font-family: 'JetBrains Mono', monospace;
}

.stat-label {
    font-size: 0.65rem;
    color: #555566;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ff6b6b;
}

.info-box {
    background: #111115;
    border: 1px solid #2a2a35;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #888899;
    margin: 8px 0;
}

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #ff4444;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 1.5rem 0 0.5rem 0;
    border-bottom: 1px solid #2a2a35;
    padding-bottom: 4px;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select {
    background: #16161c !important;
    border-color: #2a2a35 !important;
    color: #e8e8e8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

div.stButton > button {
    background: #ff4444;
    color: white;
    border: none;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.5rem 2rem;
    width: 100%;
    transition: background 0.2s;
}

div.stButton > button:hover {
    background: #cc2222;
    color: white;
}

div[data-testid="stDownloadButton"] button {
    background: #1a2a1a;
    color: #66ff88;
    border: 1px solid #2a4a2a;
    border-radius: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    width: 100%;
}

.stProgress > div > div {
    background-color: #ff4444 !important;
}

[data-testid="stMetric"] {
    background: #16161c;
    border: 1px solid #2a2a35;
    border-radius: 4px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


#  Funkce pro zpracování obrazu

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
        raise ValueError(f"Neznámý channel: '{channel}'")


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
                status_text.text(f"Zpracovávám snímek {i + 1} / {n}...")

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
        "min": round(float(nonzero.min()),  2) if nonzero.size else 0,
        "max": round(float(nonzero.max()),  2) if nonzero.size else 0,
        "mean": round(float(nonzero.mean()), 2) if nonzero.size else 0,
        "std": round(float(nonzero.std()),  2) if nonzero.size else 0,
    }

    return depth_map, stats


def create_figure(depth_map, stats, file_name, colormap):
    nonzero = depth_map[depth_map != 0]

    fig = plt.figure(figsize=(18, 10), facecolor='#0d0d0f')
    fig.suptitle(f"Liniový laser – sken: {file_name}", fontsize=14,
                 fontweight='bold', color='#e8e8e8', fontfamily='monospace')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)

    ax_style = dict(facecolor='#111115')

    # Panel 1: Depth mapa
    ax1 = fig.add_subplot(gs[0, :], **ax_style)
    im = ax1.imshow(depth_map, cmap=colormap, interpolation='nearest',
                    aspect='auto', origin='lower')
    ax1.set_title("Depth mapa", fontsize=11, color='#ccccdd', pad=8)
    ax1.set_ylabel("Pozice na senzoru [px]", fontsize=10, color='#888899')
    ax1.tick_params(colors='#555566')
    for spine in ax1.spines.values():
        spine.set_color('#2a2a35')
    cb = plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01)
    cb.set_label("Poloha středu laseru [px]", rotation=90, labelpad=12,
                 color='#888899', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#555566')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#888899')

    info = (f"Snímků: {stats['n_snimku']}   Pokrytí: {stats['pokryti_pct']} %   "
            f"Min: {stats['min']}   Max: {stats['max']}   "
            f"Průměr: {stats['mean']}   Std: {stats['std']}")
    ax1.set_xlabel(f"Číslo snímku\n{info}", fontsize=9, color='#555566')

    # Panel 2: Průměrný profil
    ax2 = fig.add_subplot(gs[1, 0], **ax_style)
    with_data = np.where(depth_map != 0, depth_map, np.nan)
    mean_profile = np.nanmean(with_data, axis=1)
    y_pos = np.arange(len(mean_profile))
    ax2.plot(mean_profile, y_pos, color='#ff6b6b', linewidth=1.2)
    ax2.fill_betweenx(y_pos, mean_profile, alpha=0.15, color='#ff6b6b')
    ax2.set_title("Průměrný profil laseru", fontsize=11, color='#ccccdd', pad=8)
    ax2.set_xlabel("Střed laseru [px]", fontsize=10, color='#888899')
    ax2.set_ylabel("Pozice na senzoru [px]", fontsize=10, color='#888899')
    ax2.tick_params(colors='#555566')
    ax2.grid(True, alpha=0.15, linestyle='--', color='#444455')
    for spine in ax2.spines.values():
        spine.set_color('#2a2a35')

    # Panel 3: Histogram
    ax3 = fig.add_subplot(gs[1, 1], **ax_style)
    if nonzero.size > 0:
        ax3.hist(nonzero.ravel(), bins=60, color='#cc3344',
                 edgecolor='#0d0d0f', linewidth=0.4, alpha=0.85)
        ax3.axvline(stats['mean'], color='#ffaa44', lw=1.5, ls='--',
                    label=f"Průměr = {stats['mean']}")
        ax3.axvline(stats['mean'] - stats['std'], color='#44cc88',
                    lw=1.0, ls=':', label=f"±směrodatná odchylka = {stats['std']}")
        ax3.axvline(stats['mean'] + stats['std'], color='#44cc88', lw=1.0, ls=':')
        ax3.legend(fontsize=9, facecolor='#16161c', edgecolor='#2a2a35',
                   labelcolor='#ccccdd')
    ax3.set_title("Histogram hodnot", fontsize=11, color='#ccccdd', pad=8)
    ax3.set_xlabel("Hodnota [px]", fontsize=10, color='#888899')
    ax3.set_ylabel("Počet bodů", fontsize=10, color='#888899')
    ax3.tick_params(colors='#555566')
    ax3.grid(True, alpha=0.15, linestyle='--', color='#444455')
    for spine in ax3.spines.values():
        spine.set_color('#2a2a35')

    plt.tight_layout()
    return fig


#  Sidebar

with st.sidebar:
    st.markdown('<div class="main-header">Laserový<br>sken</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">analýza skenu</div>', unsafe_allow_html=True)

    # Vstupní soubor
    st.markdown('<div class="section-label">Vstupní data</div>', unsafe_allow_html=True)
    file_path = st.text_input(
        "Cesta k souboru (.npy)",
        value="./OUT/kamera_251.npy"
    )
    file_name = Path(file_path).stem.replace("kamera_", "") if file_path else "?"

    # Detekce laseru
    st.markdown('<div class="section-label">Detekce laseru</div>', unsafe_allow_html=True)
    laser_axis = st.selectbox(
        "Osa laseru",
        options=[0, 1],
        format_func=lambda x: "0 – Horizontálně (po řádcích)" if x == 0 else "1 – Vertikálně (po sloupcích)"
    )
    channel = st.selectbox(
        "Kanál signálu",
        options=['GRAY', 'RG', 'R', 'G']
    )
    threshold = st.slider("Threshold (min. jas)", 0, 200, 25, 5)
    peak_window = st.slider("Peak window [px]", 1, 50, 10, 1)

    # Filtrování šumu
    st.markdown('<div class="section-label">Filtrování šumu</div>', unsafe_allow_html=True)
    enable_median = st.checkbox("Mediánový filtr", value=True)
    median_kernel = st.slider("Kernel mediánu", 3, 11, 3, 2,
                              disabled=not enable_median)
    enable_gaussian = st.checkbox("Gaussovský filtr", value=False)
    gaussian_sigma = st.slider("Sigma gaussu", 0.5, 5.0, 1.0, 0.5,
                               disabled=not enable_gaussian)

    # Post-processing
    st.markdown('<div class="section-label">Post-processing</div>', unsafe_allow_html=True)
    smooth_kernel = st.select_slider(
        "Velikost kernelu pro vyhlazení",
        options=[1, 3, 5, 7, 9],
        value=1
    )
    min_value = st.slider("Min. hodnota (vynulování)", 0, 100, 0, 1,
                          help="Pozor: jde o souřadnici polohy laseru, ne intenzitu!")
    outlier_sigma = st.slider("Outlier sigma (0 = vypnuto)", 0.0, 6.0, 0.0, 0.5,
                              help="Body dále než N×směrodatná odchylka od mediánu jsou odstraněny")

    # Vizualizace
    st.markdown('<div class="section-label">Vizualizace</div>', unsafe_allow_html=True)
    colormap = st.selectbox("Barvová mapa", ['magma', 'viridis', 'plasma', 'jet', 'inferno', 'gray'])

    st.markdown("---")
    run_btn = st.button("SPUSTIT ANALÝZU")
    save_npy = st.checkbox("Uložit depth mapu (.npy)", value=True)



#  Hlavní panel

st.markdown(
    f'<div style="font-family:JetBrains Mono,monospace; font-size:1.1rem; '
    f'color:#FFFFFF; margin-bottom:1rem;">soubor: '
    f'<span style="color:#FFFFFF">{file_path}</span></div>',
    unsafe_allow_html=True
)

# Stav session
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
    st.session_state.stats = None
    st.session_state.fig_bytes = None
    st.session_state.npy_bytes = None
    st.session_state.last_file = None

# Spuštění analýzy
if run_btn:
    fp = Path(file_path)
    if not fp.exists():
        st.error(f"Soubor nenalezen: `{file_path}`")
    else:
        # ── Čisté načítání ze souboru (jako v původním funkčním skriptu)
        with st.spinner("Načítám data..."):
            try:
                snimky = np.load(str(fp), allow_pickle=True)

                # Pokud to načetlo objektové pole, zkusíme ho rozbalit
                if snimky.dtype == object:
                    if snimky.ndim == 0:
                        inner = snimky.item()
                        snimky = inner if isinstance(inner, np.ndarray) else np.array(inner)
                    else:
                        snimky = np.stack(snimky)

            except Exception as e:
                st.error(f"Chyba při načítání souboru: {e}\n\nUjistěte se, že soubor na disku není zkrácený nebo poškozený.")
                st.stop()

        # Diagnostika – zobrazení info o načtených datech
        st.markdown(
            f'<div class="info-box">'
            f'Načteno {len(snimky)} snímků &nbsp;|&nbsp; '
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

        # Uložení do session state
        st.session_state.depth_map = depth_map
        st.session_state.stats = stats
        st.session_state.last_file = file_name

        # Vygenerování figury
        fig = create_figure(depth_map, stats, file_name, colormap)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0d0f')
        plt.close(fig)
        buf.seek(0)
        st.session_state.fig_bytes = buf.getvalue()

        # NPY bytes pro stažení
        npy_buf = io.BytesIO()
        np.save(npy_buf, depth_map)
        npy_buf.seek(0)
        st.session_state.npy_bytes = npy_buf.getvalue()

# Zobrazení výsledků
if st.session_state.depth_map is not None:
    stats = st.session_state.stats
    depth_map = st.session_state.depth_map
    fname = st.session_state.last_file

    # Statistiky – metriky
    st.markdown('<div class="section-label">Výsledky</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Snímků", stats['n_snimku'])
    with c2:
        st.metric("Pokrytí", f"{stats['pokryti_pct']} %")
    with c3:
        st.metric("Min [px]", stats['min'])
    with c4:
        st.metric("Max [px]", stats['max'])
    with c5:
        st.metric("Průměr [px]", stats['mean'])
    with c6:
        st.metric("Std [px]", stats['std'])

    st.markdown('<div class="section-label">Depth mapa</div>', unsafe_allow_html=True)

    if st.session_state.fig_bytes:

        st.image(st.session_state.fig_bytes, use_container_width=True)

    # Export
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Stáhnout graf (.png)",
            data=st.session_state.fig_bytes,
            file_name=f"sken_{fname}_depth_map.png",
            mime="image/png"
        )
    with dl_col2:
        if save_npy and st.session_state.npy_bytes:
            st.download_button(
                label="Stáhnout depth mapu (.npy)",
                data=st.session_state.npy_bytes,
                file_name=f"sken_{fname}_depth_map.npy",
                mime="application/octet-stream"
            )

else:
    st.markdown("""
    <div style="
        margin-top: 4rem;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        color: #FFFFFF;
    ">
        <div style="font-size: 1.2rem; margin-top: 1rem; color: #FFFFFF;">
            Zadejte cestu k souboru
        </div>
        <div style="font-size: 0.8rem; margin-top: 0.5rem; color: #E0E0E0;">
            Poté klikněte na SPUSTIT ANALÝZU v sidebaru
        </div>
    </div>
    """, unsafe_allow_html=True)