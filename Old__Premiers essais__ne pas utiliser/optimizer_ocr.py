import cv2
import numpy as np
import pytesseract
import optuna
import os
from glob import glob

# --- CONFIGURATION ---
INPUT_FOLDER = 'scans_input_test'  # Mettez ici vos 10-20 images représentatives


# Si Windows, configurez Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def traitement_parametrique(image, denoise_h, block_size, c_const):
    """
    Cette fonction applique le traitement, mais ses réglages sont variables.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Denoising (Variable : force h)
    # Plus h est grand, plus on floute le bruit, mais on perd des détails
    if denoise_h > 0:
        clean = cv2.fastNlMeansDenoising(gray, None, h=denoise_h, templateWindowSize=7, searchWindowSize=21)
    else:
        clean = gray

    # 2. Binarisation Adaptative (Variables : Block Size, C)
    # C'est souvent mieux que Otsu pour le manuscrit car ça garde de la texture
    # block_size doit être impair
    thresh = cv2.adaptiveThreshold(
        clean,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_const
    )

    return thresh


def evaluer_score_tesseract(image):
    """ Retourne la confiance moyenne (0-100) """
    try:
        # psm 6 = Assume a single uniform block of text (souvent bon pour des bouts de fiches)
        # psm 3 = Fully automatic (par défaut) -> À tester selon vos fiches
        data = pytesseract.image_to_data(image, config='--oem 1 --psm 3', output_type=pytesseract.Output.DICT)

        # On ne garde que les mots avec une confiance valide
        confs = [int(x) for x in data['conf'] if int(x) != -1]

        if not confs: return 0
        return sum(confs) / len(confs)
    except:
        return 0


def objective(trial):
    """
    C'est le Cerveau. Optuna va appeler cette fonction des centaines de fois
    avec des paramètres différents (trial) pour trouver le meilleur score.
    """

    # 1. Définition de l'espace de recherche (L'IA choisit des valeurs ici)
    denoise_h = trial.suggest_float('denoise_h', 2.0, 20.0)  # Force de nettoyage
    block_size_int = trial.suggest_int('block_size_base', 3, 25)  # Taille zone seuillage
    # Astuce : block_size doit être impair, donc on manipule :
    block_size = (block_size_int * 2) + 1

    c_const = trial.suggest_float('c_const', 2.0, 25.0)  # Sensibilité du seuillage

    # 2. Boucle sur les images de test
    scores = []
    fichiers = glob(os.path.join(INPUT_FOLDER, '*.*'))

    # On limite à 10 images pour que l'optimisation soit rapide
    for f in fichiers[:10]:
        img = cv2.imread(f)
        if img is None: continue

        # Application du traitement avec les paramètres choisis par l'IA
        processed_img = traitement_parametrique(img, denoise_h, block_size, c_const)

        # Évaluation
        score = evaluer_score_tesseract(processed_img)
        scores.append(score)

    # 3. On retourne la moyenne (c'est ce que l'algo veut MAXIMISER)
    mean_score = sum(scores) / len(scores) if scores else 0
    return mean_score


def main():
    print("Démarrage de l'optimisation intelligente...")

    # Création de l'étude
    study = optuna.create_study(direction='maximize')

    # Lancement de la recherche (n_trials = nombre d'essais)
    # 50 essais suffisent souvent à converger vers une solution excellente
    study.optimize(objective, n_trials=50)

    print("-" * 50)
    print("🏆 MEILLEURS PARAMÈTRES TROUVÉS :")
    best = study.best_params

    # Recalcul du block_size réel (impair)
    real_block_size = (best['block_size_base'] * 2) + 1

    print(f"  - Force Denoising (h) : {best['denoise_h']:.2f}")
    print(f"  - Taille Bloc (BlockSize) : {real_block_size}")
    print(f"  - Constante C : {best['c_const']:.2f}")
    print(f"  - Score Moyen Tesseract : {study.best_value:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()