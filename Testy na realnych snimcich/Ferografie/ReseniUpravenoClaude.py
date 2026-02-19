"""
╔══════════════════════════════════════════════════════════════╗
║           ANALÝZA ČÁSTIC - particle_analysis.py              ║
║  Nastavení skriptu: upravte proměnné v sekci KONFIGURACE     ║
╚══════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import logging
from typing import Dict, Optional, Any

# ─────────────────────────────────────────────────────────────
#  KONFIGURACE – upravte dle potřeby
# ─────────────────────────────────────────────────────────────

INPUT_FOLDER   = '.'          # složka se vstupními snímky
OUTPUT_FOLDER  = 'vysledky_Cloude'   # složka pro výsledky

# Kalibrace – přepočet pixelů na reálné jednotky
PX_PER_MM      = 10.0         # kolik pixelů odpovídá 1 mm (zadejte svou hodnotu)
UNIT_LABEL     = 'mm'         # zobrazovaná jednotka ('mm' nebo 'µm')

# Detekce – Canny prahy (0 = automatický odhad z obrazu)
CANNY_T1       = 0            # dolní práh (0 = auto)
CANNY_T2       = 0            # horní práh (0 = auto)
CANNY_SIGMA    = 0.33         # citlivost automatického odhadu (menší = přísnější)

# Filtrování částic
MIN_AREA_MM2   = 0.005        # minimální plocha v mm² (částice menší budou ignorovány)

# Export
EXPORT_CSV     = True         # uložit CSV
EXPORT_EXCEL   = True         # uložit Excel (.xlsx)

# Vizualizace – kolik řádků zobrazit v tabulce na obrázku
TABLE_ROWS     = 15

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
#  FUNKCE
# ─────────────────────────────────────────────────────────────

def auto_canny_thresholds(blurred: np.ndarray, sigma: float = 0.33):
    """Automatický odhad Canny prahů z mediánu jasu obrazu."""
    median = np.median(blurred)
    t1 = int(max(0,   (1.0 - sigma) * median))
    t2 = int(min(255, (1.0 + sigma) * median))
    # Zajistíme minimální rozdíl, aby Canny vůbec našel hrany
    if t2 - t1 < 20:
        t1 = max(0, median - 30)
        t2 = min(255, median + 30)
    return int(t1), int(t2)


def get_mask(img: np.ndarray,
             canny_t1: int = 0,
             canny_t2: int = 0,
             sigma: float = 0.33) -> np.ndarray:
    """
    Vytvoří binární masku objektů v obraze kombinací:
      1) detekce hran (Canny) + dilatace + výplň kontur
      2) Otsu prahování pro tmavé oblasti
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Prahy – automatické nebo ruční
    if canny_t1 == 0 and canny_t2 == 0:
        t1, t2 = auto_canny_thresholds(blurred, sigma)
        log.debug(f"  Auto Canny prahy: t1={t1}, t2={t2}")
    else:
        t1, t2 = canny_t1, canny_t2

    canny = cv2.Canny(blurred, t1, t2)

    # Propojení hran a vytvoření masky kontur
    kernel_connect  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    canny_connected = cv2.dilate(canny, kernel_connect, iterations=2)

    contours_c, _ = cv2.findContours(
        canny_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask_metal = np.zeros_like(gray)
    cv2.drawContours(mask_metal, contours_c, -1, 255, thickness=cv2.FILLED)

    # Otsu pro tmavé oblasti
    _, mask_dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    combined     = cv2.bitwise_or(mask_metal, mask_dark)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean   = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean, iterations=2)

    return mask_clean


def px_to_unit(px: float, px_per_mm: float) -> float:
    """Převod pixelů na zvolené jednotky (mm nebo µm)."""
    return px / px_per_mm


def analyze_shape(contour: np.ndarray,
                  min_area_px: float,
                  px_per_mm: float) -> Optional[Dict[str, Any]]:
    """
    Změří tvar kontury a vrátí slovník s výsledky.
    Vrátí None, pokud je plocha menší než min_area_px.
    """
    area_px = cv2.contourArea(contour)
    if area_px < min_area_px:
        return None

    rect              = cv2.minAreaRect(contour)
    center, (w, h), _ = rect
    length_px         = max(w, h)
    width_px          = min(w, h)
    aspect_ratio      = length_px / width_px if width_px > 0 else 0

    perimeter  = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

    # Konvexnost (convexity): poměr plochy kontury ku ploše konvexního obalu
    hull       = cv2.convexHull(contour)
    hull_area  = cv2.contourArea(hull)
    convexity  = area_px / hull_area if hull_area > 0 else 0

    # Ekvivalentní průměr (průměr kruhu se stejnou plochou)
    equiv_diam_px = np.sqrt(4 * area_px / np.pi)

    return {
        "area_px":       area_px,
        "area_mm2":      (area_px) / (px_per_mm ** 2),
        "length_mm":     px_to_unit(length_px, px_per_mm),
        "width_mm":      px_to_unit(width_px,  px_per_mm),
        "equiv_diam_mm": px_to_unit(equiv_diam_px, px_per_mm),
        "ar":            aspect_ratio,
        "circularity":   circularity,
        "convexity":     convexity,
        "rect":          rect,
        "center":        center,
    }


def build_dataframe(data_list: list, unit: str) -> pd.DataFrame:
    """Sestaví DataFrame z výsledků analýzy s přejmenovanými sloupci."""
    u = unit
    rename = {
        "ID":             "ID",
        f"Plocha ({u}²)": f"Plocha ({u}²)",
        f"Délka ({u})":   f"Délka ({u})",
        f"Šířka ({u})":   f"Šířka ({u})",
        f"Ekv. průměr ({u})": f"Ekv. průměr ({u})",
        "Poměr stran":    "Poměr stran",
        "Kruhovitost":    "Kruhovitost",
        "Konvexnost":     "Konvexnost",
    }
    return pd.DataFrame(data_list).rename(columns=rename)


def save_annotated_figure(img: np.ndarray,
                          annotated: np.ndarray,
                          df: pd.DataFrame,
                          filename: str,
                          out_path: str,
                          unit: str):
    """Uloží trojpanelový obrázek: anotovaný snímek | tabulka | histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(f"Analýza částic – {filename}", fontsize=13, fontweight='bold')

    # Panel 1 – anotovaný snímek
    axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Detekované částice")
    axes[0].axis('off')
    red_patch   = mpatches.Patch(color='red',   label='Kontura')
    green_patch = mpatches.Patch(color='green', label='Min. opsaný obdélník')
    axes[0].legend(handles=[red_patch, green_patch],
                   loc='lower right', fontsize=8, framealpha=0.7)

    # Panel 2 – tabulka (prvních TABLE_ROWS řádků)
    axes[1].axis('off')
    if not df.empty:
        vis_df = df.head(TABLE_ROWS)
        tbl = axes[1].table(
            cellText=vis_df.values,
            colLabels=vis_df.columns,
            cellLoc='center',
            loc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)
        axes[1].set_title(
            f"Tabulka (zobrazeno {len(vis_df)} z {len(df)} částic)",
            fontsize=10
        )
    else:
        axes[1].text(0.5, 0.5, "Žádné částice nenalezeny",
                     ha='center', va='center', fontsize=12)

    # Panel 3 – histogram ekvivalentních průměrů
    axes[2].axis('on')
    if not df.empty:
        col = f"Ekv. průměr ({unit})"
        if col in df.columns:
            axes[2].hist(df[col], bins=min(20, len(df)),
                         color='steelblue', edgecolor='white', linewidth=0.5)
            axes[2].set_xlabel(f"Ekvivalentní průměr ({unit})")
            axes[2].set_ylabel("Počet částic")
            axes[2].set_title("Distribuce velikostí")
            # Statistiky do titulku
            med = df[col].median()
            avg = df[col].mean()
            axes[2].axvline(avg, color='red',    linestyle='--', linewidth=1.2,
                            label=f'Průměr = {avg:.3f}')
            axes[2].axvline(med, color='orange', linestyle=':',  linewidth=1.2,
                            label=f'Medián = {med:.3f}')
            axes[2].legend(fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "Žádná data", ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_all_images(input_folder: str,
                       output_folder: str,
                       px_per_mm: float      = PX_PER_MM,
                       unit_label: str       = UNIT_LABEL,
                       canny_t1: int         = CANNY_T1,
                       canny_t2: int         = CANNY_T2,
                       canny_sigma: float    = CANNY_SIGMA,
                       min_area_mm2: float   = MIN_AREA_MM2,
                       export_csv: bool      = EXPORT_CSV,
                       export_excel: bool    = EXPORT_EXCEL):
    """
    Hlavní funkce – projde všechny PNG/JPG ve vstupní složce,
    analyzuje každý snímek a uloží výsledky.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(input_folder, "*.png")) +
        glob.glob(os.path.join(input_folder, "*.jpg")) +
        glob.glob(os.path.join(input_folder, "*.jpeg"))
    )

    if not files:
        log.warning(f"Ve složce '{input_folder}' nebyly nalezeny žádné snímky (PNG/JPG).")
        return

    log.info(f"Nalezeno {len(files)} snímků → zpracovávám...")

    # Minimální plocha v pixelech (převedeno z mm²)
    min_area_px = min_area_mm2 * (px_per_mm ** 2)

    # Souhrnný DataFrame ze všech snímků
    all_results = []

    u = unit_label  # zkratka pro sloupce

    for file_path in files:
        filename = os.path.basename(file_path)
        log.info(f"  → {filename}")

        img = cv2.imread(file_path)
        if img is None:
            log.warning(f"    ✗ Nepodařilo se načíst soubor: {file_path}")
            continue

        # Detekce masky a kontur
        mask     = get_mask(img, canny_t1, canny_t2, canny_sigma)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        annotated = img.copy()
        data_list = []

        for i, cnt in enumerate(contours):
            stats = analyze_shape(cnt, min_area_px, px_per_mm)
            if stats is None:
                continue

            particle_id = i + 1
            data_list.append({
                "ID":                      particle_id,
                f"Plocha ({u}²)":          round(stats["area_mm2"],      4),
                f"Délka ({u})":            round(stats["length_mm"],     3),
                f"Šířka ({u})":            round(stats["width_mm"],      3),
                f"Ekv. průměr ({u})":      round(stats["equiv_diam_mm"], 3),
                "Poměr stran":             round(stats["ar"],            2),
                "Kruhovitost":             round(stats["circularity"],   3),
                "Konvexnost":              round(stats["convexity"],     3),
            })

            # Kreslení do anotovaného snímku
            cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 1)
            box = np.int64(cv2.boxPoints(stats["rect"]))
            cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
            cx, cy = int(stats["center"][0]), int(stats["center"][1])
            cv2.putText(annotated, str(particle_id), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        df = pd.DataFrame(data_list)
        log.info(f"    Nalezeno {len(df)} částic")

        # Souhrnná statistika pro tento snímek
        if not df.empty:
            summary_row = {"Soubor": filename, "Počet částic": len(df)}
            for col in df.columns[1:]:
                summary_row[f"{col} – průměr"] = round(df[col].mean(), 4)
                summary_row[f"{col} – medián"] = round(df[col].median(), 4)
                summary_row[f"{col} – std"]    = round(df[col].std(), 4)
            all_results.append(summary_row)

        # Uložení anotovaného obrázku + tabulky + histogramu
        base_name = os.path.splitext(filename)[0]
        fig_path  = os.path.join(output_folder, f"{base_name}_analyza.png")
        save_annotated_figure(img, annotated, df, filename, fig_path, u)
        log.info(f"    Obrázek uložen: {fig_path}")

        # Export dat pro tento snímek
        if not df.empty:
            if export_csv:
                csv_path = os.path.join(output_folder, f"{base_name}_data.csv")
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                log.info(f"    CSV uloženo: {csv_path}")

            if export_excel:
                xlsx_path = os.path.join(output_folder, f"{base_name}_data.xlsx")
                with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Částice', index=False)
                    # Automatické šířky sloupců
                    ws = writer.sheets['Částice']
                    for col_cells in ws.columns:
                        max_len = max(len(str(c.value)) for c in col_cells if c.value)
                        ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4
                log.info(f"    Excel uložen: {xlsx_path}")

    # ── Souhrnný Excel ze všech snímků ─────────────────────────────────────
    if all_results:
        summary_df   = pd.DataFrame(all_results)
        summary_path = os.path.join(output_folder, "_SOUHRN_vsechny_snimky.xlsx")
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Souhrn', index=False)
            ws = writer.sheets['Souhrn']
            for col_cells in ws.columns:
                max_len = max(len(str(c.value)) for c in col_cells if c.value)
                ws.column_dimensions[col_cells[0].column_letter].width = max_len + 4
        log.info(f"\nSouhrnný Excel uložen: {summary_path}")

    log.info("\n✓ Hotovo! Výsledky jsou ve složce: " + output_folder)


# ─────────────────────────────────────────────────────────────
#  SPUŠTĚNÍ
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_all_images(
        input_folder  = INPUT_FOLDER,
        output_folder = OUTPUT_FOLDER,
    )