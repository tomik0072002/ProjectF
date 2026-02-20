"""
╔══════════════════════════════════════════════════════════════╗
║         LINIOVÝ LASER – zpracování skenu                     ║
║  Nastavení: upravte proměnné v sekci KONFIGURACE             ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import logging
from pathlib import Path
from typing import Tuple

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE – upravte dle potřeby
# ─────────────────────────────────────────────────────────────

OUT_ADR    = './OUT/'
FILE_NAME  = '251'

# Detekce laseru
LASER_AXIS    = 0      # 0 = laser svítí horizontálně (detekce po řádcích)
                       # 1 = laser svítí vertikálně (detekce po sloupcích)
THRESHOLD     = 50     # minimální jas pro platný bod laseru – pro uint16 data normalizovaná
PEAK_WINDOW   = 10     # pološířka okna pro výpočet těžiště okolo maxima [px]
LASER_CHANNEL = 'GRAY' # kanál pro extrakci signálu:
                       #   'RG'   = červený mínus zelený (doporučeno pro červený laser)
                       #   'R'    = jen červený kanál
                       #   'G'    = jen zelený kanál
                       #   'GRAY' = převod na šedotón

# Filtrování šumu
ENABLE_MEDIAN   = True   # mediánový filtr před detekcí (doporučeno zapnout)
MEDIAN_KERNEL   = 3      # velikost kernelu mediánu (lichá čísla: 3, 5, 7...)
ENABLE_GAUSSIAN = False  # gaussovský filtr (alternativa nebo doplněk k mediánu)
GAUSSIAN_SIGMA  = 1.0    # sigma gaussovského filtru
OUTLIER_SIGMA   = 0.0    # odstranění outlierů z depth mapy: body mimo N×std
                         # od mediánu jsou vynulovány. 0 = vypnuto.
                         # Doporučené hodnoty: 2.5–4.0 pokud je výsledek zašuměný.

# Post-processing depth mapy
SMOOTH_KERNEL   = 5      # mediánový filtr na výslednou depth mapu (1 = vypnuto, 3 nebo 5 doporučeno)
MIN_VALUE       = 30     # hodnoty pod tímto prahem jsou vynulovány (odstraní nízké falešné detekce)
                         # např. MIN_VALUE = 20 odstraní šum v oblasti 0-20px

# Vizualizace
COLORMAP  = 'magma'   # barvová mapa ('magma', 'viridis', 'plasma', 'jet', 'gray')
SAVE_PNG  = True      # uložit výsledný graf jako PNG
SAVE_NPY  = True      # uložit depth mapu jako .npy pro další zpracování
DPI       = 300       # rozlišení výstupního PNG

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  EXTRAKCE LASEROVÉHO SIGNÁLU Z SNÍMKU
# ─────────────────────────────────────────────────────────────

def extract_laser_signal(img: np.ndarray, channel: str = 'RG') -> np.ndarray:
    """
    Extrahuje laserový signál z barevného nebo šedotónového snímku.

    'RG'   – červený kanál mínus zelený: potlačí bílé/neutrální světlo,
             zvýrazní červený laser vůči okolnímu osvětlení
    'R'    – pouze červený kanál (vhodné pro silný červený laser)
    'G'    – pouze zelený kanál (pro zelený laser)
    'GRAY' – průměr všech kanálů (pro bílý laser nebo šedotónový vstup)

    Poznámka: uint16 snímky jsou automaticky normalizovány na uint8 (0–255)
    aby byly kompatibilní s cv2 filtry a prahem.
    """
    if img.ndim == 2:
        # Šedotónový snímek – normalizujeme na 0–255 (uint8)
        # Nutné pro cv2.medianBlur a konzistentní threshold
        img_f = img.astype(np.float32)
        img_min, img_max = img_f.min(), img_f.max()
        if img_max > img_min:
            normalized = ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(img, dtype=np.uint8)
        return normalized

    if channel == 'RG':
        return cv2.subtract(img[:, :, 2], img[:, :, 1])
    elif channel == 'R':
        return img[:, :, 2].copy()
    elif channel == 'G':
        return img[:, :, 1].copy()
    elif channel == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Neznámý channel: '{channel}'. Použij 'RG', 'R', 'G' nebo 'GRAY'.")


def apply_noise_filter(signal: np.ndarray,
                       enable_median: bool,
                       median_kernel: int,
                       enable_gaussian: bool,
                       gaussian_sigma: float) -> np.ndarray:
    """Aplikuje zvolené filtry šumu na 2D laserový signál."""
    if enable_median:
        k = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        signal = cv2.medianBlur(signal, k)
    if enable_gaussian:
        signal = cv2.GaussianBlur(signal, (0, 0), gaussian_sigma)
    return signal


# ─────────────────────────────────────────────────────────────
#  DETEKCE STŘEDU LASERU – TĚŽIŠTĚ (CENTER OF MASS)
# ─────────────────────────────────────────────────────────────

def get_laser_center_subpixel(img_slice: np.ndarray,
                               threshold: int = 25,
                               peak_window: int = 10) -> float:
    """
    Detekuje střed laseru v jednom řádku nebo sloupci pomocí těžiště.

    Algoritmus:
      1. Najde pozici maxima v řezu
      2. Pokud je maximum pod prahem, vrátí NaN (neplatný bod)
      3. Vyřízne okno ±peak_window okolo maxima
      4. Vypočítá vážené těžiště → sub-pixelová přesnost
    """
    vals    = img_slice.astype(float)
    max_idx = int(np.argmax(vals))
    max_val = vals[max_idx]

    if max_val < threshold:
        return np.nan

    s = max(0,         max_idx - peak_window)
    e = min(len(vals), max_idx + peak_window + 1)

    w_vals = vals[s:e]
    w_idxs = np.arange(s, e, dtype=float)
    total  = w_vals.sum()

    return float(np.dot(w_idxs, w_vals) / total) if total > 0 else np.nan


def get_laser_profile(signal_2d: np.ndarray,
                      threshold: int = 25,
                      peak_window: int = 10,
                      axis: int = 0) -> np.ndarray:
    """
    Detekuje profil laseru pro celý snímek.
    Volá get_laser_center_subpixel pro každý řádek (axis=0)
    nebo sloupec (axis=1) zvlášť.
    """
    if axis == 0:
        h = signal_2d.shape[0]
        return np.array([
            get_laser_center_subpixel(signal_2d[y, :], threshold, peak_window)
            for y in range(h)
        ])
    else:
        w = signal_2d.shape[1]
        return np.array([
            get_laser_center_subpixel(signal_2d[:, x], threshold, peak_window)
            for x in range(w)
        ])


# ─────────────────────────────────────────────────────────────
#  POST-PROCESSING DEPTH MAPY
# ─────────────────────────────────────────────────────────────

def smooth_depth_map(depth_map: np.ndarray, kernel: int) -> np.ndarray:
    """
    Aplikuje 2D mediánový filtr na výslednou depth mapu.
    Vyhlazuje lokální šum a odlehlé body mezi sousedními snímky.
    kernel = 1 → bez filtru, 3 nebo 5 = doporučeno.
    """
    if kernel <= 1:
        return depth_map
    k = kernel if kernel % 2 == 1 else kernel + 1
    import cv2 as _cv2
    # medianBlur vyžaduje uint8 nebo float32
    dm32 = depth_map.astype(np.float32)
    return _cv2.medianBlur(dm32, k)


def apply_min_threshold(depth_map: np.ndarray, min_val: float) -> np.ndarray:
    """
    Vynuluje všechny hodnoty pod min_val.
    Odstraní falešné detekce v oblasti nízkého signálu (šum pozadí).
    """
    if min_val <= 0:
        return depth_map
    result = depth_map.copy()
    result[result < min_val] = 0
    removed = int(np.sum((depth_map > 0) & (result == 0)))
    if removed > 0:
        log.info(f"  Min threshold ({min_val}): odstraněno {removed} bodů")
    return result


def remove_outliers(depth_map: np.ndarray, sigma: float) -> np.ndarray:
    """
    Odstraní statisticky odlehlé hodnoty z depth mapy.
    Body dále než sigma×std od mediánu jsou vynulovány.
    Použijte až po ověření že základní detekce funguje správně.
    """
    if sigma <= 0:
        return depth_map

    nonzero = depth_map[depth_map != 0]
    if nonzero.size == 0:
        return depth_map

    med    = np.median(nonzero)
    std    = np.std(nonzero)
    result = depth_map.copy()
    mask   = (result != 0) & (np.abs(result - med) > sigma * std)
    result[mask] = 0
    log.info(f"  Odstraněno outlierů: {mask.sum()} bodů  "
             f"(>{sigma:.1f}σ od mediánu {med:.1f})")
    return result


# ─────────────────────────────────────────────────────────────
#  HLAVNÍ PIPELINE ZPRACOVÁNÍ
# ─────────────────────────────────────────────────────────────

def process_laser_scan(snimky_3d: np.ndarray,
                       laser_axis: int       = LASER_AXIS,
                       threshold: int        = THRESHOLD,
                       peak_window: int      = PEAK_WINDOW,
                       channel: str          = LASER_CHANNEL,
                       enable_median: bool   = ENABLE_MEDIAN,
                       median_kernel: int    = MEDIAN_KERNEL,
                       enable_gaussian: bool = ENABLE_GAUSSIAN,
                       gaussian_sigma: float = GAUSSIAN_SIGMA,
                       outlier_sigma: float  = OUTLIER_SIGMA,
                       ) -> Tuple[np.ndarray, dict]:
    """
    Zpracuje celou sadu snímků a vrátí (depth_map, statistiky).

    Pipeline pro každý snímek:
      1. Extrakce laserového kanálu
      2. Filtrování šumu
      3. Detekce středu laseru řádek po řádku (těžiště)

    Výsledná depth_map má tvar (sensor_pos, číslo_snímku).
    """
    n = len(snimky_3d)
    log.info(f"Zpracovávám {n} snímků  "
             f"[osa: {'H' if laser_axis == 0 else 'V'}, kanál: {channel}, "
             f"threshold: {threshold}, peak_window: ±{peak_window}px]")

    profiles = []

    for i, img in enumerate(snimky_3d):
        signal  = extract_laser_signal(img, channel)
        signal  = apply_noise_filter(signal, enable_median, median_kernel,
                                     enable_gaussian, gaussian_sigma)
        profile = get_laser_profile(signal, threshold, peak_window, laser_axis)
        profiles.append(profile)

        if (i + 1) % 50 == 0 or i == n - 1:
            log.info(f"  Zpracováno {i + 1} / {n}")

    depth_map = np.array(profiles).T               # shape: (sensor_pos, snimek)
    depth_map = np.nan_to_num(depth_map, nan=0.0)  # NaN → 0

    # Post-processing
    depth_map = apply_min_threshold(depth_map, MIN_VALUE)
    depth_map = smooth_depth_map(depth_map, SMOOTH_KERNEL)
    if outlier_sigma > 0:
        depth_map = remove_outliers(depth_map, outlier_sigma)

    # Statistiky
    nonzero = depth_map[depth_map != 0]
    stats = {
        "n_snimku":       n,
        "shape":          depth_map.shape,
        "validnich_bodu": int(nonzero.size),
        "pokryti_pct":    round(100 * nonzero.size / depth_map.size, 1),
        "min":            round(float(nonzero.min()),  2) if nonzero.size else 0,
        "max":            round(float(nonzero.max()),  2) if nonzero.size else 0,
        "mean":           round(float(nonzero.mean()), 2) if nonzero.size else 0,
        "std":            round(float(nonzero.std()),  2) if nonzero.size else 0,
    }

    return depth_map, stats


# ─────────────────────────────────────────────────────────────
#  VIZUALIZACE
# ─────────────────────────────────────────────────────────────

def plot_results(depth_map: np.ndarray,
                 stats: dict,
                 file_name: str,
                 output_dir: Path,
                 colormap: str  = COLORMAP,
                 save_png: bool = SAVE_PNG,
                 dpi: int       = DPI):
    """
    Trojpanelový výstupní graf:
      1. Depth mapa – celý sken
      2. Průměrný profil laseru přes všechny snímky
      3. Histogram hodnot depth mapy
    """
    nonzero = depth_map[depth_map != 0]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Liniový laser – sken: {file_name}", fontsize=15, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── Panel 1: Depth mapa ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    im  = ax1.imshow(depth_map, cmap=colormap, interpolation='nearest',
                     aspect='auto', origin='lower')
    ax1.set_title("Depth mapa (celý sken)", fontsize=12)
    ax1.set_ylabel("Pozice na senzoru [px]", fontsize=11)
    cb = plt.colorbar(im, ax=ax1, fraction=0.02, pad=0.01)
    cb.set_label("Poloha středu laseru [px]", rotation=90, labelpad=12)

    info = (f"Snímků: {stats['n_snimku']}   "
            f"Pokrytí: {stats['pokryti_pct']} %   "
            f"Min: {stats['min']}   Max: {stats['max']}   "
            f"Průměr: {stats['mean']}   Std: {stats['std']}")
    ax1.set_xlabel(f"Číslo snímku\n{info}", fontsize=9)

    # ── Panel 2: Průměrný profil ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    with_data    = np.where(depth_map != 0, depth_map, np.nan)
    mean_profile = np.nanmean(with_data, axis=1)
    y_pos        = np.arange(len(mean_profile))

    ax2.plot(mean_profile, y_pos, color='#4fc3f7', linewidth=1.2)
    ax2.fill_betweenx(y_pos, mean_profile, alpha=0.15, color='#4fc3f7')
    ax2.set_title("Průměrný profil laseru", fontsize=12)
    ax2.set_xlabel("Střed laseru [px]",      fontsize=10)
    ax2.set_ylabel("Pozice na senzoru [px]", fontsize=10)
    ax2.grid(True, alpha=0.25, linestyle='--')

    # ── Panel 3: Histogram ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if nonzero.size > 0:
        ax3.hist(nonzero.ravel(), bins=60, color='#ab47bc',
                 edgecolor='#111', linewidth=0.3)
        ax3.axvline(stats['mean'], color='#ff7043', lw=1.5, ls='--',
                    label=f"Průměr = {stats['mean']}")
        ax3.axvline(stats['mean'] - stats['std'], color='#66bb6a',
                    lw=1.0, ls=':', label=f"±std = {stats['std']}")
        ax3.axvline(stats['mean'] + stats['std'], color='#66bb6a', lw=1.0, ls=':')
        ax3.legend(fontsize=9)
    ax3.set_title("Histogram hodnot depth mapy", fontsize=12)
    ax3.set_xlabel("Hodnota [px]", fontsize=10)
    ax3.set_ylabel("Počet bodů",   fontsize=10)
    ax3.grid(True, alpha=0.25, linestyle='--')

    plt.tight_layout()

    if save_png:
        out_path = output_dir / f"sken_{file_name}_vysledek.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        log.info(f"Graf uložen: {out_path}")

    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  SPUŠTĚNÍ
# ─────────────────────────────────────────────────────────────

def main():
    input_file = Path(OUT_ADR) / f'kamera_{FILE_NAME}.npy'
    output_dir = Path(OUT_ADR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        log.error(f"Soubor nenalezen: {input_file}")
        return

    log.info(f"Načítám: {input_file}")
    snimky = np.load(input_file)
    log.info(f"Načteno {len(snimky)} snímků  |  "
             f"shape: {snimky.shape}  |  dtype: {snimky.dtype}")

    # ── Diagnostika prvního snímku ────────────────────────────
    img0 = snimky[0]
    log.info(f"Diagnostika snímku 0:")
    log.info(f"  shape:        {img0.shape}")
    log.info(f"  dtype:        {img0.dtype}")
    log.info(f"  min/max:      {img0.min()} / {img0.max()}")
    if img0.ndim == 3:
        log.info(f"  kanál 0 max:  {img0[:,:,0].max()}  (BGR→Blue / RGB→Red)")
        log.info(f"  kanál 1 max:  {img0[:,:,1].max()}  (BGR→Green / RGB→Green)")
        log.info(f"  kanál 2 max:  {img0[:,:,2].max()}  (BGR→Red  / RGB→Blue)")
        # Test obou variant extrakce signálu
        import cv2 as _cv2
        rg_bgr = _cv2.subtract(img0[:,:,2], img0[:,:,1])  # BGR: R-G
        rg_rgb = _cv2.subtract(img0[:,:,0], img0[:,:,1])  # RGB: R-G
        log.info(f"  signál RG (BGR interpretace) max: {rg_bgr.max()}")
        log.info(f"  signál RG (RGB interpretace) max: {rg_rgb.max()}")
        log.info(f"  → Použij LASER_CHANNEL='RG' pro BGR nebo uprav index pokud jsou data RGB")
    # ─────────────────────────────────────────────────────────

    depth_map, stats = process_laser_scan(snimky)

    log.info("── Statistiky výsledku ──────────────────")
    for k, v in stats.items():
        log.info(f"  {k:<20} {v}")
    log.info("─────────────────────────────────────────")

    if SAVE_NPY:
        npy_path = output_dir / f"sken_{FILE_NAME}_depth_map.npy"
        np.save(npy_path, depth_map)
        log.info(f"Depth mapa uložena: {npy_path}")

    plot_results(depth_map, stats, FILE_NAME, output_dir)


if __name__ == "__main__":
    main()