"""
OCR Quality Audit - Optimiseur de Pipeline d'Image
====================================================

PHASE 2 - Optimisations GPU (UMat/OpenCL) + Performance
--------------------------------------------------------

Optimisations implémentées :
1. Migration UMat/OpenCL pour accélération GPU des opérations OpenCV
   - Chargement des images directement en UMat (mémoire GPU)
   - Pipeline complet exécuté sur GPU quand possible
   - Conversion CPU↔GPU minimisée (uniquement pour Tesseract)

2. Opérations GPU-accelerated :
   - GaussianBlur (normalisation)
   - morphologyEx (suppression de lignes)
   - threshold (binarisation OTSU et adaptative)
   - Laplacian (estimation du bruit et calcul de netteté)
   - divide (normalisation par division)

3. Pre-resize Tesseract pour images > 2500px (réduit charge OCR)

4. Optimisations de performance (gain 25-35%) :
   - Buffering CSV : Écriture par batch de 50 au lieu de point par point
   - Flag ENABLE_DETAILED_TIMING : Désactiver mesures temps détaillées (défaut: False)
   - Logs console réduits : Tous les 50 points au lieu de chaque point

Gain estimé total : +35-50% sur les temps d'exécution (Phase 2 + optimisations)
Compatible : CPU (fallback automatique) et GPU (RTX 1080, etc.)

Configuration :
- ENABLE_DETAILED_TIMING = False (défaut) : Mode production rapide
- ENABLE_DETAILED_TIMING = True : Mode debug avec analyse temps détaillée
"""
import os
import sys
import multiprocessing

# --- FIX 1 : Forcer X11 pour Tkinter (évite les crashs Wayland sous Ubuntu 22.04) ---
os.environ["GDK_BACKEND"] = "x11"
os.environ["tk_library"] = "/usr/lib/x86_64-linux-gnu/tk8.6" # Optionnel, aide parfois

# --- FIX 2 : Paramètres OpenCV ---
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**64)
# Note : QT_QPA... n'est utile que pour PyQt, mais ne gêne pas Tkinter.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import cv2
import numpy as np
# ... suite de vos imports ...
import tkinter as tk
# ...

import numpy as np
import pytesseract
import optuna
from optuna.samplers import TPESampler, QMCSampler, NSGAIISampler
import sys
import platform
from glob import glob
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
from datetime import datetime
import multiprocessing
from itertools import repeat
import time

# --- CONFIGURATION ---
INPUT_FOLDER = 'test_scans'

# Configuration Tesseract multi-plateforme
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif platform.system() == 'Linux':
    # Sous Linux, tesseract est généralement dans le PATH après installation via apt
    # Si tesseract n'est pas dans le PATH, décommentez et ajustez le chemin ci-dessous:
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    pass
elif platform.system() == 'Darwin':  # macOS
    # Sur macOS avec Homebrew
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Global flag to control GPU info printing
# Read from environment variable to support multiprocessing inheritance on Windows
_SHOULD_PRINT_GPU_INFO = os.environ.get('OCR_DEBUG_MODE', '0') == '1'

# --- GLOBAL FLAGS FOR TIMING ---
# ENABLE_DETAILED_TIMING: Active/désactive les mesures de temps détaillées
# Contrôlé par la checkbox "Debug/Timing" dans l'interface (pas besoin de modifier le code)
ENABLE_DETAILED_TIMING = False

# This flag ensures that the detailed timing breakdown is printed only once per run.
# Using environment variable for multiprocessing compatibility on Windows
_HAS_PRINTED_TIMINGS_KEY = 'OCR_TIMINGS_PRINTED'


# Activer les optimisations OpenCV
cv2.setUseOptimized(True)

# Tenter d'activer OpenCL pour l'accélération GPU si disponible
USE_GPU = False
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    USE_GPU = True
    if _SHOULD_PRINT_GPU_INFO:
        print("\n" + "="*70)
        print("🚀 PHASE 2 - OPTIMISATIONS GPU ACTIVÉES")
        print("="*70)
        print("✅ OpenCL activé pour OpenCV (accélération GPU UMat)")
        print("📊 Opérations GPU-accelerated:")
        print("   • GaussianBlur (normalisation)")
        print("   • morphologyEx (suppression lignes)")
        print("   • threshold (binarisation)")
        print("   • Laplacian (estimation bruit, netteté)")
        print("   • divide (normalisation)")
        print("🎯 Gain estimé: +10-15% sur les opérations OpenCV")
        print("="*70 + "\n")
else:
    if _SHOULD_PRINT_GPU_INFO:
        print("⚠️  OpenCL non disponible - Mode CPU uniquement")

# --- CORE FUNCTIONS (GPU-Optimized) ---

def get_sharpness(image):
    """Calcule la netteté. Accepte numpy array ou UMat."""
    gray = image # Image is already grayscale
    if USE_GPU and not isinstance(image, cv2.UMat):
        gray = cv2.UMat(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if isinstance(laplacian, cv2.UMat):
        laplacian = laplacian.get()
    return laplacian.var()

def get_contrast(image):
    """Calcule le contraste. Accepte numpy array ou UMat."""
    gray = image # Image is already grayscale
    if isinstance(gray, cv2.UMat):
        gray = gray.get()
    return gray.std()

def get_tesseract_score(image):
    """OCR Tesseract avec pre-resize si nécessaire. Accepte numpy array ou UMat."""
    try:
        # Tesseract nécessite numpy array, pas UMat
        if isinstance(image, cv2.UMat):
            image = image.get()

        # Pre-resize pour optimiser les grandes images (réduit la charge Tesseract)
        if image.shape[1] > 2500:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

        data = pytesseract.image_to_data(image, config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(x) for x in data['conf'] if int(x) != -1]
        return sum(confs) / len(confs) if confs else 0
    except: return 0

def evaluer_toutes_metriques(image):
    return get_tesseract_score(image), get_sharpness(image), get_contrast(image)

def remove_lines_param(gray_image, h_size, v_size, dilate_iter):
    """Suppression des lignes - Version GPU-optimized."""
    # Convertir en UMat si GPU activé et pas déjà UMat
    if USE_GPU and not isinstance(gray_image, cv2.UMat):
        gray_image = cv2.UMat(gray_image)

    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    h_detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    mask = cv2.addWeighted(h_detect, 1, v_detect, 1, 0.0)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=dilate_iter)

    # Copie et masquage
    # UMat n'a pas de méthode .copy(), utiliser clone() ou getMat()
    if isinstance(gray_image, cv2.UMat):
        result_np = gray_image.get().copy()
        mask_np = mask.get() if isinstance(mask, cv2.UMat) else mask
        result_np[mask_np > 0] = 255
        # Reconvertir en UMat si GPU activé
        result = cv2.UMat(result_np) if USE_GPU else result_np
    else:
        # Opération CPU standard
        result = gray_image.copy()
        mask_np = mask.get() if isinstance(mask, cv2.UMat) else mask
        result[mask_np > 0] = 255

    return result

def normalisation_division(image_gray, kernel_size):
    """Normalisation par division - Version GPU-optimized."""
    if kernel_size % 2 == 0: kernel_size += 1

    # Convertir en UMat si GPU activé et pas déjà UMat
    if USE_GPU and not isinstance(image_gray, cv2.UMat):
        image_gray = cv2.UMat(image_gray)

    # GaussianBlur et divide bénéficient de l'accélération GPU
    fond = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
    return cv2.divide(image_gray, fond, scale=255)

def estimate_noise_level(image):
    """
    Estime le niveau de bruit dans une image en utilisant la variance locale.
    Retourne un score de bruit (plus élevé = plus de bruit).
    Version GPU-optimized.
    """
    # Convertir en UMat si GPU activé et pas déjà UMat
    if USE_GPU and not isinstance(image, cv2.UMat):
        image = cv2.UMat(image)

    # Calcul de la variance locale avec un filtre Laplacien (GPU-accelerated)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Récupérer en numpy pour le calcul de variance
    if isinstance(laplacian, cv2.UMat):
        laplacian = laplacian.get()

    noise_estimate = laplacian.var()
    return noise_estimate

def adaptive_denoising(image, base_h_param, noise_threshold=100):
    """
    Applique un denoising adaptatif basé sur le niveau de bruit détecté.
    Stratégie simplifiée (2 niveaux) :
    - Si bruit < threshold : searchWindowSize=15 (rapide, gain 30-40%)
    - Si bruit >= threshold : searchWindowSize=21 (qualité maximale)

    Le paramètre noise_threshold est optimisable pour s'adapter à vos images.
    Version GPU-optimized (UMat).
    """
    if base_h_param <= 0:
        return image

    # Assurer que l'image est en numpy pour fastNlMeansDenoising
    # (cette fonction ne supporte pas bien UMat dans toutes les versions OpenCV)
    input_was_umat = isinstance(image, cv2.UMat)
    if input_was_umat:
        image = image.get()

    noise_level = estimate_noise_level(image)

    # Stratégie adaptative à 2 niveaux
    if noise_level < noise_threshold:
        # Bruit faible/moyen : paramètres optimisés (gain de vitesse)
        result = cv2.fastNlMeansDenoising(image, None, h=base_h_param,
                                         templateWindowSize=7, searchWindowSize=15)
    else:
        # Bruit élevé : paramètres complets (qualité maximale)
        result = cv2.fastNlMeansDenoising(image, None, h=base_h_param,
                                         templateWindowSize=7, searchWindowSize=21)

    # Reconvertir en UMat si nécessaire
    if USE_GPU and input_was_umat:
        result = cv2.UMat(result)

    return result

def pipeline_complet(image, params):
    """Pipeline complet de traitement d'image - Version GPU-optimized."""
    # Convertir en UMat dès le début si GPU activé
    if USE_GPU and not isinstance(image, cv2.UMat):
        gray = cv2.UMat(image)
    else:
        gray = image # Image is already grayscale

    # Toutes ces fonctions sont maintenant GPU-aware
    no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])
    norm = normalisation_division(no_lines, params['norm_kernel'])
    denoised = adaptive_denoising(norm, params['denoise_h'], params.get('noise_threshold', 100))

    # adaptiveThreshold sur UMat (GPU-accelerated)
    result = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])

    return result


def pipeline_complet_timed(image, params):
    """Pipeline complet avec mesure des temps - Version GPU-optimized."""
    timings = {}

    # Convertir en UMat dès le début si GPU activé
    if USE_GPU and not isinstance(image, cv2.UMat):
        gray = cv2.UMat(image)
    else:
        gray = image # Image is already grayscale

    if ENABLE_DETAILED_TIMING:
        # Step 1: Line Removal
        t0 = time.time()
        no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])
        timings['1_line_removal'] = (time.time() - t0) * 1000

        # Step 2: Normalization
        t0 = time.time()
        norm = normalisation_division(no_lines, params['norm_kernel'])
        timings['2_normalization'] = (time.time() - t0) * 1000

        # Step 3: Denoising (Adaptive)
        t0 = time.time()
        noise_level = estimate_noise_level(norm)
        denoised = adaptive_denoising(norm, params['denoise_h'], params.get('noise_threshold', 100))
        timings['3_denoising'] = (time.time() - t0) * 1000
        timings['noise_level'] = noise_level  # Pour diagnostic
        timings['noise_threshold'] = params.get('noise_threshold', 100)  # Pour voir le seuil utilisé

        # Step 4: Binarization
        t0 = time.time()
        processed_img = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])
        timings['4_binarization'] = (time.time() - t0) * 1000
    else:
        # Exécution rapide sans mesures de temps
        no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])
        norm = normalisation_division(no_lines, params['norm_kernel'])
        denoised = adaptive_denoising(norm, params['denoise_h'], params.get('noise_threshold', 100))
        processed_img = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])

    return processed_img, timings


import scipy_optimizer

def process_image_data_fast(args):
    """
    Version optimisée pour la production :
    - Pas de mesure de temps (time.time)
    - Pas de print
    - Appel direct au pipeline sans overhead
    """
    # FORCER LE MONO-THREADING
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cv2.setNumThreads(1)

    img, params, baseline_tess_score = args
    if img is None: return None

    # Exécution directe du pipeline (sans timings)
    processed_img = pipeline_complet(img, params)

    # Calcul des scores
    score_tess_processed = get_tesseract_score(processed_img)
    score_tess_delta = score_tess_processed - baseline_tess_score
    
    score_sharp = get_sharpness(processed_img)
    score_cont = get_contrast(processed_img)

    return score_tess_delta, score_tess_processed, score_sharp, score_cont

def process_image_data_wrapper(args):
    """
    Wrapper function to process a single image's data. Takes a tuple of (image_data, params, baseline_tess_score)
    as input to be compatible with pool.map.
    """
    # Lire le flag timing depuis la variable d'environnement (hérité du process parent)
    global ENABLE_DETAILED_TIMING
    ENABLE_DETAILED_TIMING = (os.environ.get('OCR_ENABLE_TIMING', '0') == '1')

    # FORCER LE MONO-THREADING : Crucial pour les performances en multiprocessing.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cv2.setNumThreads(1)

    img, params, baseline_tess_score = args
    if img is None:
        return None

    # Execute the timed pipeline
    processed_img, timings = pipeline_complet_timed(img, params)

    # --- TIMED METRICS (continued) ---

    if ENABLE_DETAILED_TIMING:
        # Step 5: Tesseract OCR
        t0 = time.time()
        score_tess_processed = get_tesseract_score(processed_img)
        score_tess_delta = score_tess_processed - baseline_tess_score
        timings['5_ocr_tesseract_delta'] = (time.time() - t0) * 1000

        # Step 6: Other metrics
        t0 = time.time()
        score_sharp = get_sharpness(processed_img)
        score_cont = get_contrast(processed_img)
        timings['6_sharp_contrast'] = (time.time() - t0) * 1000
    else:
        # Exécution rapide sans mesures de temps
        score_tess_processed = get_tesseract_score(processed_img)
        score_tess_delta = score_tess_processed - baseline_tess_score
        score_sharp = get_sharpness(processed_img)
        score_cont = get_contrast(processed_img)

    # --- PRINT TIMINGS (ONCE) ---
    if ENABLE_DETAILED_TIMING and os.environ.get(_HAS_PRINTED_TIMINGS_KEY) != '1':
        # Mark as printed using environment variable (multiprocessing-safe)
        os.environ[_HAS_PRINTED_TIMINGS_KEY] = '1'

        print("\n--- Analyse détaillée des temps d'exécution (en ms, pour une image) ---")

        # Séparer le noise_level et noise_threshold des timings pour l'affichage
        noise_level = timings.pop('noise_level', None)
        noise_threshold = timings.pop('noise_threshold', None)

        if noise_level is not None and noise_threshold is not None:
            print(f"  - Niveau de bruit détecté: {noise_level:.2f}")
            print(f"  - Seuil de bruit configuré: {noise_threshold:.2f}")
            if noise_level < noise_threshold:
                print("    → Stratégie: Denoising OPTIMISÉ (searchWindowSize=15)")
            else:
                print("    → Stratégie: Denoising COMPLET (searchWindowSize=21)")

        total_time = sum(timings.values())
        for name, t in sorted(timings.items()):
            percentage = (t / total_time) * 100 if total_time > 0 else 0
            print(f"  - Étape {name}: {t:.2f} ms ({percentage:.1f}%)")
        print(f"  - TEMPS TOTAL par image: {total_time:.2f} ms")
        print("----------------------------------------------------------------------\n")
        
    return score_tess_delta, score_tess_processed, score_sharp, score_cont

# --- SCREENING SOBOL ---

def run_sobol_screening(gui_app, n_sobol_exp, param_ranges, fixed_params):
    """
    Screening pur avec séquence de Sobol (Design of Experiments).
    Génère 2^n_sobol_exp points et évalue tous sans optimisation.
    Sauvegarde tous les résultats dans un CSV pour analyse ultérieure.
    """
    from scipy.stats import qmc

    n_points = 2 ** n_sobol_exp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"screening_sobol_{n_sobol_exp}_{timestamp}.csv"

    gui_app.update_log_from_thread(f"🔍 SCREENING SOBOL: Génération de {n_points} points (2^{n_sobol_exp})")

    # Préparer les bornes pour Sobol
    param_names = list(param_ranges.keys())
    lower_bounds = [param_ranges[p][0] for p in param_names]
    upper_bounds = [param_ranges[p][1] for p in param_names]

    # Générer séquence Sobol
    sampler = qmc.Sobol(d=len(param_names), scramble=True)
    sobol_samples = sampler.random(n=n_points)
    scaled_samples = qmc.scale(sobol_samples, lower_bounds, upper_bounds)

    # Préparer le CSV
    header_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel',
                  'denoise_h': 'denoise_h', 'noise_threshold': 'noise_threshold',
                  'bin_block': 'bin_block_size', 'bin_c': 'bin_c'}

    csv_headers = ['point_id', 'score_tesseract_delta', 'score_tesseract', 'score_nettete', 'score_contraste']
    for p in param_names:
        csv_headers.append(header_map.get(p, p))

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)

    gui_app.update_log_from_thread(f"📄 Fichier de résultats: {csv_filename}")

    # Évaluer chaque point
    best_score = 0
    best_params = None
    
    csv_buffer = []
    BATCH_SIZE = 50

    for idx, sample in enumerate(scaled_samples):
        if gui_app.cancellation_requested.is_set():
            gui_app.update_log_from_thread("⚠️ Screening annulé par l'utilisateur")
            break

        # Construire params dict
        params = fixed_params.copy()
        for i, param_name in enumerate(param_names):
            val = sample[i]
            if param_name == 'norm_kernel':
                params['norm_kernel'] = int(val) * 2 + 1
            elif param_name == 'bin_block':
                params['bin_block_size'] = int(val) * 2 + 1
            elif param_name == 'line_h':
                params['line_h_size'] = int(val)
            elif param_name == 'line_v':
                params['line_v_size'] = int(val)
            elif param_name in ['denoise_h', 'noise_threshold', 'bin_c']:
                params[param_name] = val
            else:
                params[param_name] = val

        # Évaluer
        avg_delta, avg_abs, avg_sharp, avg_cont = gui_app.evaluate_pipeline(params)

        # Ajouter au buffer CSV
        row_data = [idx + 1, avg_delta, avg_abs, avg_sharp, avg_cont]
        for p in param_names:
            if p == 'norm_kernel':
                row_data.append(params.get('norm_kernel'))
            elif p == 'bin_block':
                row_data.append(params.get('bin_block_size'))
            elif p == 'line_h':
                row_data.append(params.get('line_h_size'))
            elif p == 'line_v':
                row_data.append(params.get('line_v_size'))
            else:
                row_data.append(params.get(p))
        
        csv_buffer.append(row_data)

        # Écriture par lots (Batching)
        if len(csv_buffer) >= BATCH_SIZE:
            try:
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerows(csv_buffer)
                csv_buffer = [] # Reset buffer
            except Exception as e:
                gui_app.update_log_from_thread(f"Erreur écriture CSV batch: {e}")

        # Suivi du meilleur (on optimise sur le delta ou l'absolu, c'est pareil, mais affichons le delta)
        if avg_delta > best_score:
            best_score = avg_delta
            best_params = params.copy()
            gui_app.update_log_from_thread(f"🔥 Point {idx+1}/{n_points}: Nouveau meilleur gain = {avg_delta:.2f}%")
        else:
            if (idx + 1) % 50 == 0:  # Log tous les 50 points (réduit verbosité)
                gui_app.update_log_from_thread(f"   Point {idx+1}/{n_points}: Gain = {avg_delta:.2f}%")

        # Mise à jour UI
        gui_app.on_trial_finish(idx, avg_delta, avg_abs, avg_sharp, avg_cont, params)

    # Vider le reste du buffer à la fin ou si annulé
    if csv_buffer:
        try:
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(csv_buffer)
        except Exception as e:
            gui_app.update_log_from_thread(f"Erreur écriture CSV final flush: {e}")

    gui_app.update_status_from_thread(f"✅ Screening terminé! Meilleur gain: {best_score:.2f}%")
    gui_app.update_log_from_thread(f"📊 {n_points} points évalués et sauvegardés dans {csv_filename}")

    return best_params

# --- OPTUNA & LOGGING ---

def run_optuna_optimization(gui_app, n_trials, param_ranges, fixed_params, algo_choice):
    # Note: Optuna saves trial by trial, batching is harder to implement cleanly here without side effects.
    # Keeping it per-trial is acceptable for Optuna as n_trials is usually lower than Sobol screening.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"resultats_metrics_{algo_choice}_{timestamp}.csv"
    
    all_param_names = list(param_ranges.keys())
    
    header_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'denoise_h': 'denoise_h', 'noise_threshold': 'noise_threshold', 'bin_block': 'bin_block_size', 'bin_c': 'bin_c'}
    dynamic_headers = [header_map[p] for p in all_param_names if p in header_map]
    csv_headers = ['trial_id', 'score_tesseract_delta', 'score_tesseract', 'score_nettete', 'score_contraste'] + dynamic_headers

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)
    gui_app.update_log_from_thread(f"Log détaillé (legacy) : {csv_filename}")

    def objective(trial):
        if gui_app.cancellation_requested.is_set():
            raise optuna.exceptions.TrialPruned("Annulation demandée par l'utilisateur.")

        params = fixed_params.copy()
        
        # Build params dict for this trial
        current_params = {}
        if 'line_h' in param_ranges: current_params['line_h_size'] = trial.suggest_int('line_h_size', int(param_ranges['line_h'][0]), int(param_ranges['line_h'][1]))
        if 'line_v' in param_ranges: current_params['line_v_size'] = trial.suggest_int('line_v_size', int(param_ranges['line_v'][0]), int(param_ranges['line_v'][1]))
        if 'norm_kernel' in param_ranges: current_params['norm_kernel'] = trial.suggest_int('norm_kernel_base', int(param_ranges['norm_kernel'][0]), int(param_ranges['norm_kernel'][1])) * 2 + 1
        if 'denoise_h' in param_ranges: current_params['denoise_h'] = trial.suggest_float('denoise_h', param_ranges['denoise_h'][0], param_ranges['denoise_h'][1])
        if 'noise_threshold' in param_ranges: current_params['noise_threshold'] = trial.suggest_float('noise_threshold', param_ranges['noise_threshold'][0], param_ranges['noise_threshold'][1])
        if 'bin_block' in param_ranges:
            base_val = trial.suggest_int('bin_block_base', int(param_ranges['bin_block'][0]), int(param_ranges['bin_block'][1]))
            current_params['bin_block_size'] = base_val * 2 + 1
        if 'bin_c' in param_ranges: current_params['bin_c'] = trial.suggest_float('bin_c', param_ranges['bin_c'][0], param_ranges['bin_c'][1])
        
        params.update(current_params)

        avg_delta, avg_abs, avg_sharp, avg_cont = gui_app.evaluate_pipeline(params)

        try:
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                row_data = [trial.number, round(avg_delta, 4), round(avg_abs, 4), round(avg_sharp, 2), round(avg_cont, 2)]
                for header in dynamic_headers:
                    row_data.append(params.get(header))
                writer.writerow(row_data)
        except Exception as e: 
            gui_app.update_log_from_thread(f"Erreur CSV (legacy): {e}")

        gui_app.on_trial_finish(trial.number, avg_delta, avg_abs, avg_sharp, avg_cont, params)
        return avg_delta

    def cancellation_callback(study, trial):
        if gui_app.cancellation_requested.is_set():
            study.stop()

    sampler_map = {
        "TPE (Bayésien)": TPESampler(n_startup_trials=20),
        "Sobol (Quasi-Monte Carlo)": QMCSampler(qmc_type='sobol', scramble=True),
        "NSGA-II (Génétique)": NSGAIISampler(population_size=50, mutation_prob=0.05)
    }
    sampler = sampler_map.get(algo_choice, TPESampler())
    
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[cancellation_callback])

    if gui_app.cancellation_requested.is_set():
        gui_app.update_status_from_thread("⏹️ Optuna annulé ! Sauvegarde des résultats...")
    else:
        gui_app.update_status_from_thread(f"✅ Optuna terminé ! Meilleur Tesseract : {study.best_value:.2f}%")
    
    gui_app.finalize_run()


# --- GUI ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR V7 - Parallélisé")
        master.geometry("1000x800")

        self.best_score_so_far = 0.0
        self.trial_count = 0
        self.param_entries = {}
        self.optimal_labels = {}
        self.loaded_images = [] # To store pre-loaded images
        self.baseline_scores = [] # To store baseline Tesseract scores for each image
        self.default_params = {
            'line_h': (30, 70, 45), 'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75), 'denoise_h': (2.0, 20.0, 9.0),
            'noise_threshold': (20.0, 500.0, 100.0),  # Nouveau: seuil adaptatif denoising
            'bin_block': (30, 100, 60), 'bin_c': (10, 25.0, 15.0)
        }
        self.param_enabled_vars = {name: tk.BooleanVar(value=True) for name in self.default_params}
        self.cancellation_requested = threading.Event()
        self.results_data = []
        
        self.optuna_algos = ["TPE (Bayésien)", "Sobol (Quasi-Monte Carlo)", "NSGA-II (Génétique)"]
        self.scipy_algos = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

        self.image_files = []
        self.create_widgets()
        self.refresh_image_list()

    def pre_load_images(self):
        """Loads all images from the input folder into memory, in grayscale.
        Les images sont chargées en numpy pour compatibilité multiprocessing.
        La conversion UMat (GPU) se fait dans chaque worker si nécessaire."""
        self.update_log_from_thread("Pré-chargement des images en mémoire (en niveaux de gris)...")

        self.loaded_images = []
        self.baseline_scores = [] # Clear baseline scores as well

        if not self.image_files:
            messagebox.showwarning("Aucune image", f"Aucune image trouvée dans le dossier {INPUT_FOLDER}. Cliquez sur 🔄 pour rafraîchir.")
            return

        for f in self.image_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Garder en numpy pour compatibilité pickle/multiprocessing
                # La conversion UMat se fera dans chaque worker
                self.loaded_images.append(img)
                # Calculer le score Tesseract de l'image originale (baseline)
                baseline_score = get_tesseract_score(img)
                self.baseline_scores.append(baseline_score)

        self.update_log_from_thread(f"{len(self.loaded_images)} images chargées en mémoire.")
        self.update_log_from_thread(f"{len(self.baseline_scores)} scores de base calculés.")

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Bold.TLabel", font=('Helvetica', 9, 'bold'))

        param_frame = ttk.LabelFrame(self.master, text="Espace de Recherche des Paramètres")
        param_frame.pack(padx=10, pady=10, fill="x")
        headers = ["Actif", "Paramètre", "Min", "Max", "Défaut", "🏆 Optimal"]
        for col, text in enumerate(headers):
            ttk.Label(param_frame, text=text, style="Bold.TLabel").grid(row=0, column=col, padx=5, pady=5)
        row = 1
        for name, (min_val, max_val, default_val) in self.default_params.items():
            ttk.Checkbutton(param_frame, variable=self.param_enabled_vars[name]).grid(row=row, column=0, padx=5)
            ttk.Label(param_frame, text=name).grid(row=row, column=1, padx=5, pady=2, sticky="e")
            self.param_entries[name] = {}
            for i, val in enumerate([min_val, max_val]):
                entry = ttk.Entry(param_frame, width=8); entry.insert(0, str(val)); entry.grid(row=row, column=2 + i)
                self.param_entries[name][['min', 'max'][i]] = entry
            ttk.Label(param_frame, text=str(default_val), foreground="gray").grid(row=row, column=4)
            lbl_opt = ttk.Label(param_frame, text="-", foreground="blue", font=('Helvetica', 10, 'bold')); lbl_opt.grid(row=row, column=5, padx=10)
            self.optimal_labels[name] = lbl_opt
            row += 1
        ttk.Label(param_frame, text="dilate_iter").grid(row=row, column=1, padx=5, pady=2, sticky="e")
        ttk.Label(param_frame, text="FIXE", foreground="green", font=('Helvetica', 9, 'bold')).grid(row=row, column=0, padx=5)
        ttk.Label(param_frame, text="2", foreground="gray").grid(row=row, column=4)
        
        ctrl_frame = ttk.LabelFrame(self.master, text="Configuration de l'Optimisation")
        ctrl_frame.pack(padx=10, pady=5, fill="x")

        # --- Utilisation de GRID pour une disposition stable ---
        ctrl_frame.columnconfigure(2, weight=1)

        # Colonne 0: Bibliothèque/Mode
        ttk.Label(ctrl_frame, text="Mode :").grid(row=0, column=0, sticky="w", padx=5)
        self.lib_var = tk.StringVar(value="Screening")
        self.lib_combo = ttk.Combobox(ctrl_frame, textvariable=self.lib_var, state="readonly", width=12, values=["Screening", "Optuna", "Scipy"])
        self.lib_combo.grid(row=0, column=0, sticky="w", padx=(50,5))
        self.lib_combo.bind("<<ComboboxSelected>>", self.on_library_select)
        
        # Colonne 1: Algorithme
        ttk.Label(ctrl_frame, text="Algorithme :").grid(row=0, column=1, sticky="w", padx=5)
        self.algo_var = tk.StringVar(value=self.optuna_algos[0])
        self.algo_combo = ttk.Combobox(ctrl_frame, textvariable=self.algo_var, state="readonly", width=25, values=self.optuna_algos)
        self.algo_combo.grid(row=0, column=1, sticky="w", padx=(70,5))

        # Colonne 2: Options dynamiques (Frames non encore placées)
        self.optuna_frame = ttk.Frame(ctrl_frame)
        ttk.Label(self.optuna_frame, text="Nb Essais :").pack(side="left")
        self.trials_entry = ttk.Entry(self.optuna_frame, width=8); self.trials_entry.insert(0, "100")
        self.trials_entry.pack(side="left", padx=5)
        
        self.scipy_frame = ttk.Frame(ctrl_frame)
        ttk.Label(self.scipy_frame, text="Exposant Sobol (2^n):").pack(side="left")
        self.sobol_exponent_var = tk.StringVar(value="5")
        self.sobol_exponent_var.trace_add("write", self.update_sobol_points_label)
        self.sobol_exponent_entry = ttk.Entry(self.scipy_frame, width=5, textvariable=self.sobol_exponent_var)
        self.sobol_exponent_entry.pack(side="left", padx=2)
        self.sobol_points_label = ttk.Label(self.scipy_frame, text="= 32 points")
        self.sobol_points_label.pack(side="left", padx=(0,10))
        ttk.Label(self.scipy_frame, text="Itérations/point:").pack(side="left")
        self.scipy_iter_entry = ttk.Entry(self.scipy_frame, width=8); self.scipy_iter_entry.insert(0, "15")
        self.scipy_iter_entry.pack(side="left", padx=5)

        # Colonne 3: Compteur d'images
        img_frame = ttk.Frame(ctrl_frame)
        img_frame.grid(row=0, column=3, sticky="e", padx=5)
        self.image_count_label = ttk.Label(img_frame, text="Images: 0")
        self.image_count_label.pack(side="left")
        self.btn_refresh_images = ttk.Button(img_frame, text="🔄", width=3, command=self.refresh_image_list)
        self.btn_refresh_images.pack(side="left", padx=5)

        # Colonne 4: Boutons d'action
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=0, column=4, sticky="e", padx=5)
        
        self.debug_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(btn_frame, text="Debug/Timing", variable=self.debug_mode_var).pack(side="left", padx=5)
        
        self.btn_start = ttk.Button(btn_frame, text="▶ LANCER", command=self.start_optimization)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(btn_frame, text="⏹ ANNULER", command=self.request_cancellation, state="disabled")
        self.btn_cancel.pack(side="left", padx=5)
        
        self.status_label = ttk.Label(self.master, text="Prêt.", font=('Helvetica', 10, 'italic'))
        self.status_label.pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.master, width=90, height=12)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.on_library_select(None)

    def on_library_select(self, event):
        # On cache d'abord les deux frames d'options
        self.scipy_frame.grid_remove()
        self.optuna_frame.grid_remove()

        mode = self.lib_var.get()
        if mode == "Screening":
            # Mode Screening : on utilise le champ Sobol de Scipy
            self.algo_combo.config(values=["Sobol DoE"])
            self.algo_var.set("Sobol DoE")
            self.scipy_frame.grid(row=0, column=2, sticky="w")
        elif mode == "Optuna":
            self.algo_combo.config(values=self.optuna_algos)
            self.algo_var.set(self.optuna_algos[0])
            self.optuna_frame.grid(row=0, column=2, sticky="w")
        else:  # Scipy
            self.algo_combo.config(values=self.scipy_algos)
            self.algo_var.set(self.scipy_algos[0])
            self.scipy_frame.grid(row=0, column=2, sticky="w")


    def update_sobol_points_label(self, *args):
        try:
            exponent = int(self.sobol_exponent_var.get())
            if exponent > 16: # Limite pour éviter les très grands nombres
                self.sobol_points_label.config(text="! > 65536")
                return
            n_points = 2**exponent
            self.sobol_points_label.config(text=f"= {n_points} points")
        except ValueError:
            self.sobol_points_label.config(text="= Invalide")

    def refresh_image_list(self):
        """Met à jour la liste des fichiers image et l'affichage dans l'UI."""
        self.image_files = glob(os.path.join(INPUT_FOLDER, '*.*'))
        self.image_count_label.config(text=f"Images: {len(self.image_files)}")

    def get_optim_config(self):
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}
        param_order = []
        for name in self.default_params.keys():
            if self.param_enabled_vars[name].get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())
                    active_ranges[name] = (min_val, max_val)
                    param_order.append(name)
                except ValueError: return None, None, None
            else:
                default_val = self.default_params[name][2]
                if name == 'norm_kernel': fixed_params['norm_kernel'] = int(default_val) * 2 + 1
                elif name == 'bin_block': fixed_params['bin_block_size'] = int(default_val) * 2 + 1
                elif name == 'line_h': fixed_params['line_h_size'] = default_val
                elif name == 'line_v': fixed_params['line_v_size'] = default_val
                else: fixed_params[name] = default_val
        return active_ranges, fixed_params, param_order
    
    def evaluate_pipeline(self, params):
        if not self.loaded_images:
            return 0, 0, 0

        pool_args = zip(self.loaded_images, repeat(params), self.baseline_scores)

        # Optimisation Hyperthreading : Utiliser 1.5x les cores physiques pour CPU avec HT
        # Sur un CPU 12c/24t, cela donne 18 workers au lieu de 12
        optimal_workers = int(os.cpu_count() * 1.5)
        pool_size = min(len(self.loaded_images), optimal_workers)
        
        # Choix de la fonction worker selon le mode
        worker_func = process_image_data_wrapper if self.debug_mode_var.get() else process_image_data_fast

        try:
            with multiprocessing.Pool(processes=pool_size) as pool:
                results = pool.map(worker_func, pool_args)
            
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return 0, 0, 0

            list_delta, list_abs, list_sharp, list_cont = zip(*valid_results)

            avg_delta = sum(list_delta) / len(list_delta) if list_delta else 0
            avg_abs = sum(list_abs) / len(list_abs) if list_abs else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_delta, avg_abs, avg_sharp, avg_cont

        except Exception as e:
            print(f"Erreur de multiprocessing, passage en mode séquentiel: {e}")
            list_delta, list_abs, list_sharp, list_cont = [], [], [], []
            for i, img in enumerate(self.loaded_images):
                # Mode séquentiel : besoin de recalculer la baseline ou de la récupérer
                # Comme on a self.baseline_scores, on l'utilise
                baseline = self.baseline_scores[i] if i < len(self.baseline_scores) else 0
                
                processed_img = pipeline_complet(img, params)
                tess_abs = get_tesseract_score(processed_img)
                tess_delta = tess_abs - baseline
                sharp = get_sharpness(processed_img)
                cont = get_contrast(processed_img)
                
                list_delta.append(tess_delta)
                list_abs.append(tess_abs)
                list_sharp.append(sharp)
                list_cont.append(cont)
            
            avg_delta = sum(list_delta) / len(list_delta) if list_delta else 0
            avg_abs = sum(list_abs) / len(list_abs) if list_abs else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_delta, avg_abs, avg_sharp, avg_cont

    def start_optimization(self):
        # Reset timing flag for new optimization run
        if _HAS_PRINTED_TIMINGS_KEY in os.environ:
            del os.environ[_HAS_PRINTED_TIMINGS_KEY]

        # Set environment variables for child processes (Windows spawn support)
        debug_enabled = '1' if self.debug_mode_var.get() else '0'
        os.environ['OCR_DEBUG_MODE'] = debug_enabled
        os.environ['OCR_ENABLE_TIMING'] = debug_enabled  # Active timing en mode debug

        # Update local globals for main process
        global _SHOULD_PRINT_GPU_INFO, ENABLE_DETAILED_TIMING
        _SHOULD_PRINT_GPU_INFO = (debug_enabled == '1')
        ENABLE_DETAILED_TIMING = (debug_enabled == '1')

        self.cancellation_requested.clear()
        self.results_data.clear()
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

        active_ranges, fixed_params, param_order = self.get_optim_config()
        if active_ranges is None: 
            messagebox.showerror("Erreur de Saisie", "Vérifiez que les Min/Max sont des nombres valides.")
            self.finalize_run()
            return

        self.log_text.delete('1.0', tk.END)
        self.best_score_so_far = 0.0
        self.trial_count = 0
        for lbl in self.optimal_labels.values(): lbl.config(text="-")

        # Pre-load images once before starting the thread
        self.pre_load_images()

        mode = self.lib_var.get()
        algo = self.algo_var.get()

        if mode == "Screening":
            # Mode Screening Sobol pur (DoE)
            try:
                exponent = int(self.sobol_exponent_var.get())
                self.update_status_from_thread(f"🔬 Lancement Screening Sobol (2^{exponent} = {2**exponent} points)...")
                thread = threading.Thread(target=run_sobol_screening, args=(self, exponent, active_ranges, fixed_params))
                thread.start()
            except ValueError:
                messagebox.showerror("Erreur de Saisie", "L'exposant Sobol doit être un entier.")
                self.finalize_run()

        elif mode == "Optuna":
            try:
                n_trials = int(self.trials_entry.get())
                thread = threading.Thread(target=run_optuna_optimization, args=(self, n_trials, active_ranges, fixed_params, algo))
                thread.start()
            except ValueError: 
                messagebox.showerror("Erreur de Saisie", "Le nombre d'essais doit être un entier.")
                self.finalize_run()
        
        elif mode == "Scipy":
            try:
                exponent = int(self.sobol_exponent_var.get())
                n_sobol = 2**exponent
                n_iter = int(self.scipy_iter_entry.get())
                bounds = [active_ranges[p] for p in param_order]

                def objective_for_scipy(param_values):
                    # DEBUG print(f"DEBUG: Scipy objective called with {param_values}")
                    if self.cancellation_requested.is_set(): return 0 # Stop early

                    self.trial_count += 1
                    params = fixed_params.copy()
                    current_params = {}
                    for i, name in enumerate(param_order):
                        val = param_values[i]
                        p_name_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'bin_block': 'bin_block_size', 'denoise_h': 'denoise_h', 'bin_c': 'bin_c'}
                        
                        mapped_name = p_name_map.get(name)
                        if not mapped_name: continue

                        if name in ['norm_kernel', 'bin_block']:
                             current_params[mapped_name] = int(val) * 2 + 1
                        elif name in ['line_h', 'line_v']:
                             current_params[mapped_name] = int(val)
                        else:
                             current_params[mapped_name] = val
                    
                    params.update(current_params)
                    t_score_delta, t_score_abs, s_score, c_score = self.evaluate_pipeline(params)
                    self.on_trial_finish(self.trial_count, t_score_delta, t_score_abs, s_score, c_score, params)
                    return -t_score_delta

                thread = threading.Thread(target=self.run_scipy_thread, args=(objective_for_scipy, bounds, algo, n_sobol, n_iter))
                thread.start()

            except ValueError:
                messagebox.showerror("Erreur de Saisie", "Les points Sobol et Itérations doivent être des entiers.")
                self.finalize_run()

    def run_scipy_thread(self, objective, bounds, algo, n_sobol, n_iter):
        result = scipy_optimizer.run_scipy_optimization(objective, bounds, algo, n_sobol, n_iter, self.update_log_from_thread, self.cancellation_requested)
        
        # Si aucun résultat n'a été trouvé (par ex. annulation immédiate)
        if result.get("x") is None:
            self.update_status_from_thread("⏹️ Scipy annulé, aucun résultat à afficher.")
            self.finalize_run()
            return

        best_params_values = result.x
        
        # Reconstruct the full params dict to display it
        _, fixed_params, param_order = self.get_optim_config()
        final_params = fixed_params.copy()
        current_params = {}
        for i, name in enumerate(param_order):
            # Vérifier si l'indice est valide pour best_params_values
            if i >= len(best_params_values):
                continue
            val = best_params_values[i]
            p_name_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'bin_block': 'bin_block_size', 'denoise_h': 'denoise_h', 'bin_c': 'bin_c'}
            mapped_name = p_name_map.get(name)
            if not mapped_name: continue

            if name in ['norm_kernel', 'bin_block']:
                 current_params[mapped_name] = int(val) * 2 + 1
            else:
                 current_params[mapped_name] = val
        final_params.update(current_params)

        if self.cancellation_requested.is_set():
            self.update_status_from_thread(f"⏹️ Scipy annulé ! Sauvegarde des résultats...")
        else:
            self.update_status_from_thread(f"✅ Scipy terminé ! Meilleur score (négatif) : {result.fun:.4f}")

        self.master.after(0, self.update_optimal_display, final_params)
        self.finalize_run()

    def request_cancellation(self):
        """Signale l'annulation de l'optimisation en cours."""
        self.cancellation_requested.set()
        self.update_status_from_thread("🛑 Annulation demandée...")
        self.btn_cancel.config(state="disabled")

    def save_results_to_csv(self):
        """Sauvegarde les résultats collectés dans un fichier CSV."""
        if not self.results_data:
            self.update_log_from_thread("Aucune donnée à sauvegarder.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optim_results_{timestamp}.csv"
        
        # Les en-têtes sont basés sur les clés du premier résultat
        first_result = self.results_data[0]
        param_headers = list(first_result.get('params', {}).keys())
        score_headers = list(first_result.get('scores', {}).keys())
        headers = param_headers + score_headers

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(headers)
                for result in self.results_data:
                    # Assure l'ordre des valeurs
                    param_values = [result.get('params', {}).get(h) for h in param_headers]
                    score_values = [result.get('scores', {}).get(h) for h in score_headers]
                    writer.writerow(param_values + score_values)
            self.update_log_from_thread(f"📈 Résultats sauvegardés dans : {filename}")
        except Exception as e:
            self.update_log_from_thread(f"Erreur lors de la sauvegarde CSV : {e}")

    def update_log_from_thread(self, msg):
        self.master.after(0, self.log_text.insert, tk.END, msg + "\n")
        self.master.after(0, self.log_text.see, tk.END)

    def update_status_from_thread(self, msg):
        self.master.after(0, self.status_label.config, {'text': msg})
        
    def finalize_run(self):
        """Remet l'interface en état 'Prêt' et sauvegarde les résultats."""
        self.master.after(0, self.btn_start.config, {'state': 'normal'})
        self.master.after(0, self.btn_cancel.config, {'state': 'disabled'})
        self.master.after(0, self.save_results_to_csv)

    def on_trial_finish(self, trial_num, t_score_delta, t_score_abs, s_score, c_score, params):
        msg = f"[Essai {trial_num}] Delta Tess: {t_score_delta:.2f}% | Abs Tess: {t_score_abs:.2f}% | Netteté: {s_score:.1f} | Contraste: {c_score:.1f}"
        self.update_log_from_thread(msg)

        # Enregistrement des données pour le CSV
        trial_data = {
            'params': params.copy(),
            'scores': {
                'tesseract_delta': round(t_score_delta, 4),
                'tesseract': round(t_score_abs, 4),
                'nettete': round(s_score, 2),
                'contraste': round(c_score, 2)
            }
        }
        self.results_data.append(trial_data)

        if t_score_delta > self.best_score_so_far:
            self.best_score_so_far = t_score_delta
            self.update_status_from_thread(f"🔥 RECORD GAIN : {t_score_delta:.2f}% (Essai {trial_num})")
            self.master.after(0, self.update_optimal_display, params)

    def update_optimal_display(self, params):
        param_map_to_gui = {
            'line_h_size': 'line_h', 'line_v_size': 'line_v',
            'denoise_h': 'denoise_h', 'bin_c': 'bin_c',
            'norm_kernel': 'norm_kernel', 'bin_block_size': 'bin_block'
        }
        for p_name, p_val in params.items():
            gui_name = param_map_to_gui.get(p_name)
            if gui_name in self.optimal_labels:
                if gui_name == 'norm_kernel': value_to_display = (p_val - 1) // 2
                elif gui_name == 'bin_block': value_to_display = (p_val - 1) // 2
                else: value_to_display = p_val
                
                self.optimal_labels[gui_name].config(text=f"{value_to_display:.2f}" if isinstance(value_to_display, float) else f"{value_to_display}")


if __name__ == "__main__":
    # --- FIX CRITIQUE POUR LINUX + CUDA ---
    # Empêche le crash (Error 139 / SIGSEGV) lors de l'utilisation de multiprocessing
    # Force Python à créer des processus propres au lieu de cloner la mémoire
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # La méthode a déjà été définie, on continue.

    # Nécessaire si vous comptez compiler votre script en exécutable plus tard
    multiprocessing.freeze_support()

    # Création du dossier d'entrée si inexistant
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)

    # Initialisation de l'interface graphique
    root = tk.Tk()

    # Configuration optionnelle pour améliorer le rendu sous Linux
    # root.geometry("1200x800")

    app = OptimizerGUI(root)
    root.mainloop()