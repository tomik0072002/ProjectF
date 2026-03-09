"""
Coin Detection Pipeline
=======================
Generates 5 pipeline images from a coin photo:
  1. Noisy / raw input
  2. Preprocessing (grayscale + blur)
  3. Edge detection (Canny)
  4. Morphology (dilation + erosion)
  5. Extraction & classification (Hough circles + labels)

Usage:
    python coin_pipeline.py --input your_image.png --output_dir ./output
"""

import cv2
import numpy as np
import argparse
import os
from PIL import Image, ImageDraw, ImageFont


def save(img: np.ndarray, output_dir: str, step: int, name: str) -> str:
    filename = os.path.join(output_dir, f"step{step}_{name}.png")
    cv2.imwrite(filename, img)
    print(f"  Saved: {filename}")
    return filename


def step1_noisy(src: np.ndarray, output_dir: str) -> np.ndarray:
    """Step 1 – Sběr dat: add Gaussian noise to simulate raw/noisy input."""
    print("Step 1: Sběr dat (raw noisy input)")
    noisy = src.astype(np.float32) + np.random.normal(0, 35, src.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    save(noisy, output_dir, 1, "sbr_dat")
    return noisy


def step2_preprocess(src: np.ndarray, output_dir: str) -> np.ndarray:
    """Step 2 – Předzpracování: convert to grayscale and apply Gaussian blur."""
    print("Step 2: Předzpracování (grayscale + blur)")
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (9, 9), 2)
    smooth_bgr = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
    save(smooth_bgr, output_dir, 2, "predz")
    return smooth  # return single-channel for next steps


def step3_edges(gray_smooth: np.ndarray, output_dir: str) -> np.ndarray:
    """Step 3 – Segmentace a hrany: Canny edge detection."""
    print("Step 3: Segmentace a hrany (Canny)")
    edges = cv2.Canny(gray_smooth, 30, 100)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    save(edges_bgr, output_dir, 3, "hrany")
    return edges


def step4_morphology(edges: np.ndarray, output_dir: str) -> np.ndarray:
    """Step 4 – Morfologie: dilate then erode to produce a clean mask."""
    print("Step 4: Morfologie (dilation + erosion)")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    cleaned = cv2.erode(dilated, kernel, iterations=1)
    cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    save(cleaned_bgr, output_dir, 4, "morfologie")
    return cleaned


def step5_classify(src: np.ndarray, gray_smooth: np.ndarray, output_dir: str) -> np.ndarray:
    """Step 5 – Extrakce a klasifikace: Hough circles + colored labels."""
    print("Step 5: Extrakce a klasifikace (Hough circles)")

    result = src.copy()
    h, w = gray_smooth.shape
    # minDist = 25 % šířky obrazu, aby se nenašly duplicity
    min_dist   = int(w * 0.25)
    min_radius = int(w * 0.13)
    max_radius = int(w * 0.32)

    circles = cv2.HoughCircles(
        gray_smooth,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=120,
        param2=60,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    colors = [
        (0,   130, 255),  # oranžová
        (255,  80, 180),  # azurová
        (180,   0, 255),  # magenta
        (50,  220,  50),  # zelená
        (0,   220, 220),  # žlutá
        (255,  60,  60),  # modrá
    ]
    # Štítky seřazené zleva doprava, shora dolů
    labels = ["50 Kč", "20 Kč", "10 Kč", "5 Kč", "2 Kč", "1 Kč"]

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        print(f"  Detected {len(circles)} coin(s)")

        # Seřadit kruhy podle pozice: nejdřív řádek, pak sloupec.
        # Tolerance řádku = 80 % mediánového poloměru.
        median_r = int(np.median(circles[:, 2]))
        row_tol  = int(median_r * 0.8)
        sorted_circles = sorted(circles[:6], key=lambda c: (c[1] // row_tol, c[0]))

        # Převést na PIL pro Unicode text, pak zpět na OpenCV
        pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        # Načíst font — fallback na výchozí pokud soubor neexistuje
        # Hledáme tučný font na Linuxu i Windows
        font_candidates = [
            # Linux
            "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # Windows
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/verdanab.ttf",
        ]
        font_size = max(60, int(gray_smooth.shape[1] * 0.09))  # ~9 % šířky obrazu
        font = None
        for fp in font_candidates:
            try:
                font = ImageFont.truetype(fp, size=font_size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default(size=font_size)

        for i, (x, y, r) in enumerate(sorted_circles):
            col_bgr = colors[i % len(colors)]
            col_rgb = (col_bgr[2], col_bgr[1], col_bgr[0])  # BGR → RGB pro Pillow
            label   = labels[i] if i < len(labels) else f"#{i+1}"

            # Nakreslit kroužek přes OpenCV (pracuje in-place na result)
            cv2.circle(result, (x, y), r + 6, col_bgr, 6)

            # Aktualizovat PIL obraz po cv2.circle
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            draw    = ImageDraw.Draw(pil_img)

            # Vycentrovat text na střed mince
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x - tw // 2, y - th // 2), label, font=font, fill=col_rgb)

            # Zpět do OpenCV pro další iteraci
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        print("  No circles detected — try tuning Hough parameters.")

    save(result, output_dir, 5, "klasifikace")
    return result


def run_pipeline(input_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    src = cv2.imread(input_path)
    if src is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    print(f"Loaded: {input_path}  ({src.shape[1]}×{src.shape[0]} px)\n")

    step1_noisy(src, output_dir)
    gray_smooth = step2_preprocess(src, output_dir)
    edges = step3_edges(gray_smooth, output_dir)
    step4_morphology(edges, output_dir)
    step5_classify(src, gray_smooth, output_dir)

    print(f"\nDone! All images saved to: {output_dir}")


def find_image_in_script_dir() -> str:
    """Find the first image file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    for f in os.listdir(script_dir):
        if f.lower().endswith(extensions) and not f.startswith("step"):
            return os.path.join(script_dir, f)
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coin detection pipeline")
    parser.add_argument("--input",      default="", help="Input image path (auto-detected if not set)")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    args = parser.parse_args()

    input_path = args.input or find_image_in_script_dir()
    if not input_path:
        raise FileNotFoundError("No image found in the script directory. Use --input to specify the path.")

    run_pipeline(input_path, args.output_dir)