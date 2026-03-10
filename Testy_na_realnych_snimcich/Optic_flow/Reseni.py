import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


def select_image(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Obrazky", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Vsechny soubory", "*.*")]
    )
    root.destroy()
    return path


def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def fit_to_screen(image, margin=0.9):
    sw, sh = get_screen_size()
    max_w, max_h = int(sw * margin), int(sh * margin)
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def get_next_filename(out_dir):
    """Najde dalsi volne cislo vysledek1, vysledek2 atd."""
    i = 1
    while True:
        path = os.path.join(out_dir, f"vysledek{i}.png")
        if not os.path.exists(path):
            return path
        i += 1


def draw_label(panel, label):
    """Popisek přímo na obrázek — bílý text s černým stínem, velikost 2."""
    cv2.putText(panel, label, (12, 72), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(panel, label, (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)


def optical_flow_car():
    print("Vyber prvni snimek...")
    path1 = select_image("Vyber prvni snimek")
    if not path1:
        return

    print("Vyber druhy snimek...")
    path2 = select_image("Vyber druhy snimek")
    if not path2:
        return

    frame1 = cv2.imread(path1)
    frame2 = cv2.imread(path2)

    if frame1 is None or frame2 is None:
        print("Chyba pri nacitani obrazku.")
        return

    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=6, winsize=25,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    med = np.median(magnitude)
    std = np.std(magnitude)
    motion_threshold = med + 2.0 * std
    motion_mask = magnitude > motion_threshold

    moving_magnitudes = magnitude[motion_mask]
    typical_motion = np.percentile(moving_magnitudes, 50) if len(moving_magnitudes) > 0 else 1.0
    arrow_scale = 40.0 / max(typical_motion, 1.0)

    # Panel 1
    p1 = frame2.copy()
    draw_label(p1, "Povodni snimek")

    # Panel 2: Flow mapa
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    p2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    draw_label(p2, "Flow mapa")

    # Panel 3: Pohybujici se oblasti
    p3 = frame2.copy()
    highlight = np.zeros_like(frame2)
    highlight[motion_mask] = (0, 0, 255)
    p3 = cv2.addWeighted(p3, 0.65, highlight, 0.35, 0)
    draw_label(p3, "Pohybujici se oblasti")

    # Panel 4: Šipky pohybu
    p4 = frame2.copy()
    h, w = frame2.shape[:2]
    step = 25
    for y in range(0, h, step):
        for x in range(0, w, step):
            if motion_mask[y, x]:
                fx, fy = flow[y, x]
                end = (int(x + fx * arrow_scale), int(y + fy * arrow_scale))
                speed_norm = min(magnitude[y, x] / (motion_threshold * 2), 1.0)
                if speed_norm < 0.5:
                    t = speed_norm * 2
                    color = (int(255 * (1 - t)), 255, int(255 * t))
                else:
                    t = (speed_norm - 0.5) * 2
                    color = (0, int(255 * (1 - t)), 255)
                cv2.arrowedLine(p4, (x, y), end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)
    draw_label(p4, "Sipky pohybu")


    target_h = min(p.shape[0] for p in [p1, p2, p3, p4])
    target_w = min(p.shape[1] for p in [p1, p2, p3, p4])
    panels = [cv2.resize(p, (target_w, target_h)) for p in [p1, p2, p3, p4]]
    combined = np.hstack(panels)
    final = fit_to_screen(combined, margin=0.95)

    # Uložení
    out_dir = "vysledky"
    os.makedirs(out_dir, exist_ok=True)
    out_path = get_next_filename(out_dir)
    cv2.imwrite(out_path, final)
    print(f"Vysledek ulozen: {out_path}")

    cv2.imshow("Optical Flow", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    optical_flow_car()