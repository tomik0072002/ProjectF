import cv2
import numpy as np
import urllib.request
import os

# Zkoušel jsem vytvořit syntetický snímek, ale to nefungovalo
# Takže jsem tady zkusil stahování obrázků (tohle jsou přímo dva obrázky z OpenCV stránek na Optic Flow)

# Stažení snímků
def download_file(url, filename):

    if not os.path.exists(filename):
        print(f"Stahuji {filename} z internetu...")
        try:
            urllib.request.urlretrieve(url, filename)
            print("Staženo.")
        except Exception as e:
            print(f"Chyba při stahování: {e}")
            return False
    return True


def optical_flow_test():
    # Definice URL a stažení dat
    base_url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
    file1 = "basketball1.png"
    file2 = "basketball2.png"

    if not download_file(base_url + file1, file1) or not download_file(base_url + file2, file2):
        return

    # Načtení
    frame1 = cv2.imread(file1)
    frame2 = cv2.imread(file2)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Hledání bodů (Shi-Tomasi)
    feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    if p0 is None:
        print("Nenalezeny žádné body.")
        return

    # Výpočet Optical Flow (Lucas-Kanade)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Vizualizace se šipkami
    mask = np.zeros_like(frame1)
    output_img = frame2.copy()

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()  # Nová pozice
        c, d = old.ravel()  # Stará pozice

        # Filtrujeme velmi malé pohyby (šum)
        dist = np.sqrt((a - c) ** 2 + (b - d) ** 2)
        if dist > 1.5:
            start_pt = (int(c), int(d))
            end_pt = (int(a), int(b))

            mask = cv2.arrowedLine(mask, start_pt, end_pt, (0, 255, 0), 2, tipLength=0.25)


    img_final = cv2.add(output_img, mask)


    cv2.imshow('Optical Flow (Arrows)', img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    optical_flow_test()