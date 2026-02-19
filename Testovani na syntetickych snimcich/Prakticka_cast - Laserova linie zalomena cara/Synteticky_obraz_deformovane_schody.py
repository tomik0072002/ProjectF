import cv2
import numpy as np

# Zase generování tří "schodů" akorát tentokrát to nejsou přímky

def generate_deformed_surface(length, wave_amp=2.0, wave_freq=0.05, roughness_sigma=0.5):

    x = np.arange(length, dtype=np.float32)
    # Sinusová vlna + náhodný šum
    wave = wave_amp * np.sin(x * wave_freq)
    roughness = np.random.normal(0, roughness_sigma, length).astype(np.float32)
    return wave + roughness


def generate_all_deformed_image(filename, width=800, height=600):
    # Inicializace
    image = np.zeros((height, width, 3), dtype=np.float32)

    # Parametry
    base_level_y = height - 100
    line_thickness_sigma = 4.0

    # Parametry schodů
    step1_h = 60                # výška schodu 1
    step1_range = (100, 350)    # šířka schodu 1

    step2_h = 130
    step2_range = (450, 600)

    step3_h = 200.0
    step3_range = (650, 750)

    # Vytvoření deformované základny
    base_deformation = generate_deformed_surface(width, wave_amp=4.0, wave_freq=0.015, roughness_sigma=0.5)

    true_profile_y = base_level_y + base_deformation

    # Vložení prvního schodu
    s1, e1 = step1_range
    len1 = e1 - s1

    # Deformace schodu
    step1_deformation = generate_deformed_surface(len1, wave_amp=2.0, wave_freq=0.05, roughness_sigma=0.8)
    true_profile_y[s1:e1] = (base_level_y - step1_h) + step1_deformation

    # Vložení druhého schodu
    s2, e2 = step2_range
    len2 = e2 - s2

    # Deformace
    step2_deformation = generate_deformed_surface(len2, wave_amp=3.0, wave_freq=0.08, roughness_sigma=0.8)
    true_profile_y[s2:e2] = (base_level_y - step2_h) + step2_deformation

    # ložení třetího schodu
    s3, e3 = step3_range
    len3 = e3 - s3

    # Deformace
    step3_deformation = generate_deformed_surface(len3, wave_amp=3.0, wave_freq=0.08, roughness_sigma=0.8)
    true_profile_y[s3:e3] = (base_level_y - step3_h) + step3_deformation

    # Kreslení
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    distances = yy - true_profile_y
    intensity = np.exp(- (distances ** 2) / (2 * line_thickness_sigma ** 2))
    image[:, :, 2] = intensity * 255.0

    # Šum
    speckle_noise = np.random.normal(1.0, 0.2, (height, width))
    image[:, :, 2] *= speckle_noise
    sensor_noise = np.random.normal(0, 10.0, (height, width))
    image[:, :, 2] += sensor_noise

    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, image)
    print(f"Vygenerován obrázek '{filename}'")


# Spuštění
generate_all_deformed_image("deformovane_tri_schody.png")