import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import io
import time
from PIL import Image

# Načtení loga VUT
script_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(script_dir, "vut_brno_00.jpg")
logo = Image.open(logo_path)

# Nastavení stránky
st.set_page_config(
    page_title="Liniovy laser - analyza skenu",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Funkce pro zpracování obrazu

def take_laser_signal(img: np.ndarray, channel: str = 'GRAY') -> np.ndarray: # Funkce pro extrakci laserového signálu
    if img.ndim == 2:
        img_f = img.astype(np.float32)
        img_min, img_max = img_f.min(), img_f.max()
        if img_max > img_min:
            return ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        return np.zeros_like(img, dtype=np.uint8)

    # Kanály laseru
    if channel == 'RG':
        return cv2.subtract(img[:, :, 2], img[:, :, 1])
    elif channel == 'R':
        return img[:, :, 2].copy()
    elif channel == 'G':
        return img[:, :, 1].copy()
    elif channel == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Fukce pro získání středu laserové čáry
def laser_center(img_slice: np.ndarray, threshold: int, peak_window: int) -> float:
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

# Získání profilu laseru
def laser_profile(signal_2d, threshold, peak_window, axis):
    if axis == 0:
        return np.array([
            laser_center(signal_2d[y, :], threshold, peak_window)
            for y in range(signal_2d.shape[0])
        ])
    else:
        return np.array([
            laser_center(signal_2d[:, x], threshold, peak_window)
            for x in range(signal_2d.shape[1])
        ])

# Upravení skenu
def edit_laser_scan(snimky_3d, params, progress_bar=None, status_text=None):
    n = len(snimky_3d)
    profiles = []

    for i, img in enumerate(snimky_3d):

        # Oříznutí
        h, w = img.shape[:2]
        ct, cb = params['crop_t'], params['crop_b']
        cl, cr = params['crop_l'], params['crop_r']

        img_cropped = img[ct:h - cb, cl:w - cr]

        # Jas a kontrast
        if params['alpha'] != 1.0 or params['beta'] != 0:
            img_cropped = cv2.convertScaleAbs(img_cropped, alpha=params['alpha'], beta=params['beta'])

        # Extrakce kanálu
        signal = take_laser_signal(img_cropped, params['channel'])

        # Rozostření obrazu
        if params['median_k'] > 1:
            k = params['median_k'] | 1
            signal = cv2.medianBlur(signal, k)
        if params['gauss_sigma'] > 0:
            signal = cv2.GaussianBlur(signal, (0, 0), params['gauss_sigma'])

        # Morfologie
        if params['morph_k'] > 1:
            k_size = params['morph_k'] | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
            if params['morph_op'] == 'Otevření':
                signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel)
            elif params['morph_op'] == 'Uzavření':
                signal = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, kernel)
            elif params['morph_op'] == 'Dilatace':
                signal = cv2.dilate(signal, kernel, iterations=1)

        # Extrakce profilu
        profile_cropped = laser_profile(signal, params['threshold'], params['peak_window'], params['laser_axis'])

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

    #  2D mapy
    depth_map = np.array(profiles).T
    depth_map = np.nan_to_num(depth_map, nan=0.0)

    # Vyhlazení 2D mapy
    if params['smooth_kernel'] > 1:
        k = params['smooth_kernel'] | 1
        depth_map = cv2.medianBlur(depth_map.astype(np.float32), k)

    # Odstranění odlehlých bodů
    if params['outlier_sigma'] > 0:
        nonzero = depth_map[depth_map != 0]
        if nonzero.size > 0:
            med = np.median(nonzero)
            std = np.std(nonzero)
            mask = (depth_map != 0) & (np.abs(depth_map - med) > params['outlier_sigma'] * std)
            depth_map[mask] = 0

    # Výpočet statistik
    nonzero = depth_map[depth_map != 0]
    stats = {
        "n_snimku": n,
        "shape": depth_map.shape,
        "pokryti_pct": round(100 * nonzero.size / depth_map.size, 1),

    }

    return depth_map, stats

# Vytváření 2D vizualizace
def create_figure(depth_map, stats, file_name, colormap):
    nonzero = depth_map[depth_map != 0]
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_alpha(0.0)
    fig.suptitle(f"Liniový laser - sken: {file_name}", fontsize=14, fontweight='bold', color='gray')

    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.60, height_ratios=[2, 1])

    # Hloubková mapa
    ax1 = fig.add_subplot(gs[0, 1:5])
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

    ax1.set_xlabel("Cislo snimku", fontsize=10, color='gray')

    return fig

# Vytvoření 3D vizualizace
def create_3D_figure(depth_map: np.ndarray, colormap: str, downsample: int = 1) -> go.Figure:
    dm = depth_map[::downsample, ::downsample].copy()
    dm_plot = np.where(dm == 0, np.nan, dm)

    plotly_cmap = colormap.capitalize()

    rows, cols = dm_plot.shape
    x = np.arange(cols) * downsample
    y = np.arange(rows) * downsample

    fig = go.Figure(data=[go.Surface(
        z=dm_plot, x=x, y=y, colorscale=plotly_cmap,
        colorbar=dict(title=dict(text="Poloha laseru [px]", side="right"), thickness=15),
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="3D Depth mapa", font=dict(size=14), x=0.02),
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

# Boční panel - sidebar

with st.sidebar:
    st.markdown("## Laserový sken")
    st.markdown("Analýza skenu")
    st.markdown("---")

    with st.expander("Rozsah snímků", expanded=False):
        st.caption(
            "Omezí zpracování pouze na vybraný rozsah snímků ze souboru."
        )
        frame_start = st.number_input(
            "Od snímku", min_value=0, value=0, step=1,
        )
        frame_end = st.number_input(
            "Do snímku", min_value=0, value=0, step=1,
        )

    with st.expander("Oříznutí obrazu (ROI)", expanded=False):
        st.caption(
            "Odřízne okraje každého snímku před zpracováním. "
            "Hodnoty jsou v pixelech od příslušného okraje snímku."
        )

        st.markdown("**Osa: Pozice na senzoru**")
        crop_t = st.number_input(
            "Od dolního okraje [px]", min_value=0, value=0, step=10, key="crop_t"
        )
        crop_b = st.number_input(
            "Od horního okraje [px]", min_value=0, value=0, step=10, key="crop_b"
        )

        st.markdown("**Osa: Hloubka")
        crop_l = st.number_input(
            "Od dolního okraje [px]", min_value=0, value=0, step=10, key="crop_l"
        )
        crop_r = st.number_input(
            "Od horního okraje [px]", min_value=0, value=0, step=10, key="crop_r"
        )

    with st.expander("Korekce a filtry", expanded=False):
        st.caption("Úpravy samotného snímku před detekcí")
        alpha = st.slider("Kontrast (Alpha)", 0.5, 1.4, 1.0, 0.1)
        beta = st.slider("Jas (Beta)", -100, 100, 0, 5)
        st.markdown("---")
        median_k = st.slider("Medián filtr (Kernel)", 1, 15, 1, 2)
        gauss_sigma = st.slider("Gaussovo rozostření (Sigma)", 0.0, 5.0, 0.0, 0.5)
        st.markdown("---")
        morph_op = st.selectbox("Morfologická operace", ['Žádná', 'Otevření', 'Uzavření', 'Dilatace'])
        morph_k = st.slider("Velikost morfologie (Kernel)", 1, 15, 3, 2) if morph_op != 'Žádná' else 1

    with st.expander("Detekce laseru", expanded=False):
        st.caption("Parametry pro nalezení středu čáry")
        laser_axis = st.selectbox("Osa laseru", options=[0, 1], format_func=lambda x: "0 - Horizontálně" if x == 0 else "1 - Vertikálně")
        channel = st.selectbox("Kanál signálu", options=['GRAY', 'RG', 'R', 'G'])
        threshold = st.slider("Práh (Min. jas)", 0, 70, 10, 5)
        peak_window = st.slider("Prohledávací okno [px]", 1, 50, 10, 1, help="Velikost okolí pro výpočet středu laseru")

    with st.expander("Post-processing", expanded=False):
        smooth_kernel = st.slider("Vyhlazení povrchu (Medián)", 1, 15, 1, 2)
        outlier_sigma = st.slider("Filtrace odchylek (Sigma)", 0.0, 6.0, 0.0, 0.5,
                                  help="Odstraní body mimo N*směrodatná odchylka")

    st.markdown("---")

    # Vizualizace
    st.markdown("**Vizualizace**")
    colormap = st.selectbox("Barevná mapa", ['magma', 'viridis', 'plasma', 'jet', 'inferno', 'gray'])
    downsample_3d = st.select_slider("Rozlišení 3D vizualizace", options=[1, 2, 4, 8], value=2, help="Vyšší hodnota = menší detaily")

    st.markdown("---")
    save_npy = st.checkbox("Uložit depth mapu ke stažení (.npy)", value=True)


#  Hlavni panel
st.title("Zpracování skenu")

# Vstupní soubor
input_col, btn_col = st.columns([4, 1])
with input_col:
    file_path = st.text_input("Cesta k souboru (.npy)", value="./OUT/kamera_251.npy", label_visibility="collapsed", placeholder="Cesta k souboru (.npy)")
with btn_col:
    run_btn = st.button("SPUSTIT ANALÝZU", use_container_width=True)

file_name = Path(file_path).stem.replace("kamera_", "") if file_path else "?"
st.markdown(f"**Soubor:** `{file_path}`")

# Stav
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

        # Ořez rozsahu snímků
        fs = int(frame_start)
        fe = int(frame_end) + 1 if int(frame_end) > 0 else len(snimky)
        fe = min(fe, len(snimky))
        if fs >= fe:
            st.error(
                f"Neplatný rozsah snímků: od {fs} do {fe - 1}. "
            )
            st.stop()

        if len(snimky) > 0:
            h_img, w_img = snimky[0].shape[:2]
            roi_errors = []

            roi_h = h_img - crop_t - crop_b
            roi_w = w_img - crop_l - crop_r
            min_roi_px = 10

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

        depth_map, stats = edit_laser_scan(snimky, params, progress_bar, status_text)

        elapsed = time.time() - t0
        progress_bar.empty()
        status_text.empty()

        st.success(f"Analýza dokončena za {elapsed:.1f} s")

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

# Zobrazení výsledků
if st.session_state.depth_map is not None:
    stats = st.session_state.stats
    depth_map = st.session_state.depth_map
    fname = st.session_state.last_file

    st.markdown("#### Statistické hodnoty")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Počet snímků", stats['n_snimku'])
    c2.metric("Pokrytí", f"{stats['pokryti_pct']} %")

    st.markdown("---")

    st.markdown("### Depth mapa (2D)")
    if st.session_state.fig_bytes:
        st.image(st.session_state.fig_bytes, use_container_width=True)

    st.markdown("---")

    st.markdown("### 3D Vizualizace")
    with st.spinner("Generuji 3D graf..."):
        fig3d = create_3D_figure(depth_map, colormap, downsample=downsample_3d)
    st.plotly_chart(fig3d, use_container_width=True)
    st.markdown(
        '<div class="info-box">'
        'Click &amp; drag s myší = rotace &nbsp;|&nbsp; '
        'Click &amp; Ctrl = translační pohyb &nbsp;|&nbsp; '
        'Kolečko myši = zoom'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Export")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Stáhnout 2D vizualizaci a grafy (.png)",
            data=st.session_state.fig_bytes,
            file_name=f"sken_{fname}_depth_map.png",
            mime="image/png",
            use_container_width=True
        )
    with dl_col2:
        if save_npy and st.session_state.npy_bytes:
            st.download_button(
                label="Stáhnout 3D depth mapu (.npy)",
                data=st.session_state.npy_bytes,
                file_name=f"sken_{fname}_depth_map.npy",
                mime="application/octet-stream",
                use_container_width=True
            )
else:
    st.info("Zadejte cestu k souboru výše a klikněte na **SPUSTIT ANALÝZU**.")