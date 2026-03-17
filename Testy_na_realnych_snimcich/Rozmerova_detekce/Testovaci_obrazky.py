import cv2
import numpy as np


def vytvor_sachovnici(sq_size=50, rows=7, cols=10):
    """
    Vytvoří kalibrační šachovnici.
    Pro 7x10 čtverců to je 6x9 vnitřních rohů.
    """
    width = cols * sq_size
    height = rows * sq_size
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                cv2.rectangle(image, (c * sq_size, r * sq_size),
                              ((c + 1) * sq_size, (r + 1) * sq_size),
                              (0, 0, 0), -1)

    bordered = cv2.copyMakeBorder(image, sq_size, sq_size, sq_size, sq_size,
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered


def vytvor_samostatne_objekty():
    """
    Vytvoří 4 samostatné obrázky s jedním objektem uprostřed.
    Plátno bude vždy 600x600 px.
    """
    velikost = (600, 600, 3)
    barva_pozadi = 255
    stred = (300, 300)

    # 1. Dokonalý kruh (poloměr 120px)
    img1 = np.full(velikost, barva_pozadi, dtype=np.uint8)
    cv2.circle(img1, stred, 120, (10, 10, 10), -1)
    cv2.imwrite("test_1_kruh.png", img1)

    # 2. Dokonalý obdélník rovný (400x150px)
    img2 = np.full(velikost, barva_pozadi, dtype=np.uint8)
    # Vypočítáme rohy, aby byl přesně uprostřed
    cv2.rectangle(img2, (100, 225), (500, 375), (30, 30, 30), -1)
    cv2.imwrite("test_2_obdelnik_rovny.png", img2)

    # 3. Natočený obdélník (-30 stupňů)
    img3 = np.full(velikost, barva_pozadi, dtype=np.uint8)
    rect = (stred, (400, 150), -30.0)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.fillPoly(img3, [box], (50, 50, 50))
    cv2.imwrite("test_3_obdelnik_natoceny.png", img3)

    # 4. Elipsa
    img4 = np.full(velikost, barva_pozadi, dtype=np.uint8)
    cv2.ellipse(img4, stred, (160, 80), 45, 0, 360, (20, 20, 20), -1)
    cv2.imwrite("test_4_elipsa.png", img4)


if __name__ == "__main__":
    # 1. Kalibrace
    cb_img = vytvor_sachovnici(sq_size=50, rows=7, cols=10)
    cv2.imwrite("kalibrace_6x9.png", cb_img)
    print("Vytvořeno: kalibrace_6x9.png (Šachovnice s 6x9 vnitřními rohy)")

    # 2. Samostatné testovací objekty
    vytvor_samostatne_objekty()
    print("Vytvořeny 4 samostatné obrázky objektů:")
    print(" - test_1_kruh.png")
    print(" - test_2_obdelnik_rovny.png")
    print(" - test_3_obdelnik_natoceny.png")
    print(" - test_4_elipsa.png")