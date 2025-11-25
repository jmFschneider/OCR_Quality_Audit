import cv2
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'test_scans/029-R.jpg'
OUTPUT_FOLDER = 'scans_pipeline_final'


def remove_lines(image_rgb):
    """ Supprime les lignes structurelles horizontales et verticales """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Noyaux ajustés (passés de 40 à 60 pour potentiellement mieux gérer les résidus)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))

    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    mask_lignes = cv2.addWeighted(detect_horizontal, 1, detect_vertical, 1, 0.0)
    mask_lignes = cv2.dilate(mask_lignes, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    resultat = gray.copy()
    # On met du blanc (255) à l'emplacement des lignes
    resultat[mask_lignes > 0] = 255

    return resultat


def normalisation_division(image_gray_cleaned):
    """ Applique la technique d'aplanissement pour supprimer le fond coloré """

    # Estimation du fond (Background Estimation)
    fond = cv2.GaussianBlur(image_gray_cleaned, (51, 51), 0)

    # Division (Normalisation)
    # On utilise np.divide pour éviter une erreur de division par zéro si le fond est noir
    resultat = np.divide(image_gray_cleaned, fond, out=np.zeros_like(image_gray_cleaned, dtype=np.float32),
                         where=fond != 0, dtype=np.float32)
    resultat = cv2.normalize(resultat, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return resultat


def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    original = cv2.imread(INPUT_FILE)
    if original is None:
        print("Erreur : Fichier introuvable.")
        return

    # 1. Isolation : Enlever les lignes
    image_sans_lignes = remove_lines(original)

    # 2. Normalisation : Supprimer le fond coloré
    image_normalisee = normalisation_division(image_sans_lignes)

    # 3. Finalisation : Binarisation avec débruitage léger
    # Le Denoising NLM est parfait pour les zones de manuscrit (supprime le bruit/grain de l'image)
    denoised = cv2.fastNlMeansDenoising(image_normalisee, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Seuil d'Otsu pour la conversion finale N&B
    blur = cv2.GaussianBlur(denoised, (5, 5), 0)
    _, finale = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sauvegardes
    cv2.imwrite(f"{OUTPUT_FOLDER}/0_original.jpg", original)
    cv2.imwrite(f"{OUTPUT_FOLDER}/1_sans_lignes.jpg", image_sans_lignes)
    cv2.imwrite(f"{OUTPUT_FOLDER}/2_normalisee.jpg", image_normalisee)
    cv2.imwrite(f"{OUTPUT_FOLDER}/3_finale_nettoyee.jpg", finale)

    print("Pipeline de correction terminé pour 029-R.jpg.")
    print("Veuillez comparer l'image '0_original.jpg' avec '3_finale_nettoyee.jpg'.")


if __name__ == "__main__":
    main()