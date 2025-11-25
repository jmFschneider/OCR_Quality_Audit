import cv2
import numpy as np

INPUT_FILE = 'test_scans/029-R.jpg'
OUTPUT_DEBUG = 'debug_isolation'


def remove_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation inversée (Texte blanc sur fond noir)
    # On utilise bitwise_not pour que les traits soient des valeurs "hautes"
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 2. Détection des lignes Horizontales
    # On cherche des structures qui sont très larges mais très fines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # 3. Détection des lignes Verticales
    # On cherche des structures qui sont très hautes mais très fines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # 4. Combinaison des grilles (Lignes H + Lignes V)
    mask_lignes = cv2.addWeighted(detect_horizontal, 1, detect_vertical, 1, 0.0)

    # 5. Dilatation légère des lignes pour être sûr de bien les effacer
    mask_lignes = cv2.dilate(mask_lignes, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # 6. Soustraction : On enlève les lignes de l'image originale
    # On utilise inpaint pour "reboucher" les trous laissés par les lignes (optionnel mais propre)
    # Ou plus simplement : on met du blanc là où il y avait des lignes
    resultat = gray.copy()
    # Là où le mask_lignes est blanc (>0), on met du blanc (255) dans l'image finale
    resultat[mask_lignes > 0] = 255

    return resultat, mask_lignes


def get_sharpness_score(image_gray):
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def main():
    if not os.path.exists(OUTPUT_DEBUG):
        os.makedirs(OUTPUT_DEBUG)

    img = cv2.imread(INPUT_FILE)
    if img is None:
        print("Image introuvable")
        return

    # Nettoyage des lignes
    image_sans_lignes, masque = remove_lines(img)

    # Calcul des scores
    score_original = get_sharpness_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    score_texte_seul = get_sharpness_score(image_sans_lignes)

    print(f"--- RÉSULTATS POUR {INPUT_FILE} ---")
    print(f"Score Netteté GLOBALE (avec tableau) : {score_original:.2f}")
    print(f"Score Netteté CONTENU (sans tableau) : {score_texte_seul:.2f}")

    # Sauvegarde pour vérification visuelle
    cv2.imwrite(f"{OUTPUT_DEBUG}/etape1_masque_lignes.jpg", masque)
    cv2.imwrite(f"{OUTPUT_DEBUG}/etape2_sans_lignes.jpg", image_sans_lignes)

    print("\nAllez voir l'image 'etape2_sans_lignes.jpg'.")
    print("Est-ce qu'il ne reste bien que le texte (imprimé + manuscrit) ?")


import os

if __name__ == "__main__":
    main()