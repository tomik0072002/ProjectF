import cv2
import numpy as np
import glob
import os

# Našel jsem online dataset (složka Online obrazky) se syntetickými obrázky nebo uměle vytvořenými obrazy pro Optic Flow

# Načtení snímků
def run_dataset_sequence_looped(folder_path):
    search_path = os.path.join(folder_path, "*.png")
    files = glob.glob(search_path)

    if not files:
        search_path = os.path.join(folder_path, "*.jpg")
        files = glob.glob(search_path)

    files = sorted(files)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    if len(images) < 2:
        print(f"Chyba: Ve složce '{folder_path}' nebylo nalezeno dost obrázků.")
        return

    print(f"Načteno {len(images)} snímků. Spouštím nekonečnou smyčku (ukončení 'q')...")

    # Parametry pro Shi-Tomasi (hledání bodů)
    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)

    # Parametry pro Lucas-Kanade (Optic Flow)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Hlavní nekonečná smyčka pro opakování videa
    while True:

        frame1 = cv2.imread(images[0])
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

        mask = np.zeros_like(frame1)

        # Smyčka přes jednotlivé snímky v sekvenci
        for i in range(1, len(images)):
            frame2 = cv2.imread(images[i])
            if frame2 is None:
                break

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Pokud nejsou žádné body ke sledování, najdou se nové
            if p0 is None or len(p0) < 5:
                p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

            # Výpočet Optic Flow
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

                # Výběr dobrých bodů
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    # Kreslení
                    for j, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()
                        c, d = old.ravel()

                        # Vykreslení čáry pohybu
                        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                        # Vykreslení aktuálního bod
                        frame2 = cv2.circle(frame2, (int(a), int(b)), 4, (0, 0, 255), -1)

                    img_show = cv2.add(frame2, mask)

                    # Aktualizace pro další krok
                    gray1 = gray2.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                else:
                    img_show = frame2
            else:
                img_show = frame2

            cv2.imshow('Optical Flow Player (Loop)', img_show)

            # Čekání 30ms.
            k = cv2.waitKey(30) & 0xff

            # Pokud uživatel stiskne 'q', ukončíme funkci úplně (vyskočíme z obou smyček)
            if k == ord('q'):
                cv2.destroyAllWindows()
                print("Ukončeno uživatelem.")
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    moje_slozka = r"C:\Users\dejme\PycharmProjects\DP_project\Testing\Prakticka_cast - Optic flow\Vice obrazku"

    if os.path.exists(moje_slozka):
        run_dataset_sequence_looped(moje_slozka)
    else:
        print(f"Cesta '{moje_slozka}' neexistuje. Upravte proměnnou 'moje_slozka' v kódu!")