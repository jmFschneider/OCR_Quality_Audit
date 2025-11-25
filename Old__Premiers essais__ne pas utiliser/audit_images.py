import cv2
import numpy as np
import pytesseract
import os
import pandas as pd
import time

# --- CONFIGURATION ---
# Chemin vers vos images (relatif ou absolu)
INPUT_FOLDER = 'test_scans'
# Fichier de rapport
OUTPUT_REPORT = 'rapport_qualite_ocr.csv'


# Si vous êtes sous Windows, décommentez et ajustez la ligne suivante si Tesseract n'est pas dans le PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_sharpness_score(image_gray):
    """
    Calcule la variance du Laplacien.
    Plus le score est haut, plus l'image est nette (bords francs).
    Seuil critique souvent autour de 100.
    """
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def get_contrast_score(image_gray):
    """
    Calcule l'écart-type des pixels (RMS Contrast).
    Plus c'est haut, plus l'histogramme est étalé (bon contraste).
    Si c'est bas (< 40), l'image est "grise" et plate.
    """
    return image_gray.std()


def get_tesseract_confidence(image_gray):
    """
    Demande à Tesseract de lire l'image brute et renvoie la confiance moyenne.
    C'est notre "Canari dans la mine".
    """
    try:
        data = pytesseract.image_to_data(image_gray, output_type=pytesseract.Output.DICT)
        # On ne garde que les confiances valides (> -1) et sur des mots non vides
        confidences = [int(x) for i, x in enumerate(data['conf']) if int(x) != -1 and data['text'][i].strip() != '']

        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    except Exception as e:
        print(f"Erreur Tesseract : {e}")
        return 0.0


def main():
    results = []
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

    print(f"🔍 Démarrage de l'audit sur {len(files)} fichiers...")
    print("-" * 50)

    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)

        # Chargement image
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ Impossible de lire : {filename}")
            continue

        # Conversion en niveaux de gris pour l'analyse
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Calcul Netteté
        sharpness = get_sharpness_score(gray)

        # 2. Calcul Contraste
        contrast = get_contrast_score(gray)

        # 3. Calcul Confiance Préliminaire (peut être lent sur des gros dossiers)
        # On redimensionne légèrement pour accélérer Tesseract si les images sont énormes
        h, w = gray.shape
        if w > 2000:
            scale_percent = 50
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            gray_resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
            ocr_conf = get_tesseract_confidence(gray_resized)
        else:
            ocr_conf = get_tesseract_confidence(gray)

        # Diagnostic rapide pour la console
        status = "✅ OK"
        if sharpness < 100 or ocr_conf < 40:
            status = "⚠️ ATTENTION"

        print(
            f"[{status}] {filename} -> Netteté: {sharpness:.1f} | Contraste: {contrast:.1f} | Confiance OCR: {ocr_conf:.1f}%")

        results.append({
            'Fichier': filename,
            'Nettete_Laplacien': round(sharpness, 2),
            'Contraste_RMS': round(contrast, 2),
            'Confiance_Tesseract': round(ocr_conf, 2),
            'Status_Global': 'Revue Requise' if (sharpness < 100 or ocr_conf < 50) else 'Bon'
        })

    # Création du DataFrame et export CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_REPORT, index=False, sep=';')  # Point-virgule pour ouverture facile dans Excel FR

    print("-" * 50)
    print(f"📄 Rapport généré : {os.path.abspath(OUTPUT_REPORT)}")


if __name__ == "__main__":
    main()