import cv2
import numpy as np

# Cílem je generovat snímek, který simuluje laser se dvěma "schody"

def generovani_snimku(filename, width=800, height=600):

    image = np.zeros((height, width, 3), dtype=np.float32)

    base_level_y = height - 100
    line_thickness_sigma = 4.0

    # První "schod"
    step1_height_px = 60.0
    step1_start_x = 100
    step1_end_x = 350

    # Druhý "schod"
    step2_height_px = 130.0
    step2_start_x = 450
    step2_end_x = 600


    true_profile_y = np.full(width, base_level_y, dtype=np.float32)

    true_profile_y[step1_start_x:step1_end_x] -= step1_height_px
    true_profile_y[step2_start_x:step2_end_x] -= step2_height_px

    _, yy = np.meshgrid(np.arange(width), np.arange(height))

    distances = yy - true_profile_y

    # Gaussova funkce pro intenzitu
    intensity = np.exp(- (distances ** 2) / (2 * line_thickness_sigma ** 2))

    image[:, :, 2] = intensity * 255.0

    # Šum

    # 1. Multiplikativní "Speckle" šum (typický pro lasery)
    speckle_noise = np.random.normal(1.0, 0.2, (height, width))
    image[:, :, 2] *= speckle_noise

    # 2. Gaussovský šum
    sensor_noise = np.random.normal(0, 10.0, (height, width))
    image[:, :, 2] += sensor_noise

    # Ořezání hodnot na platný rozsah 0-255 (8bitový)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Uložení
    cv2.imwrite(filename, image)
    print(f"Vygenerován obrázek '{filename}'")
    print(f"Skutečná výška schodu 1: {step1_height_px} pixelů")
    print(f"Skutečná výška schodu 2: {step2_height_px} pixelů")


generovani_snimku("synteticke_dva_schody.png")