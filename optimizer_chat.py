"""
optimizer.py - Logique d'optimisation des paramètres
Gestion des algorithmes d'optimisation (Sobol, Optuna, SciPy)
"""

import multiprocessing
from itertools import repeat
import os
import pipeline


# ============================================================
# WORKERS MULTIPROCESSING (MODE CPU)
# ============================================================

def process_image_fast(args):
    """Worker pour traitement parallèle CPU."""
    # Forcer mono-threading pour multiprocessing
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    img, params, baseline_score = args
    if img is None:
        return None

    # Traitement
    processed_img = pipeline.pipeline_complet(img, params)

    # Métriques
    score_tess = pipeline.get_tesseract_score(processed_img)
    score_delta = score_tess - baseline_score
    score_sharp = pipeline.get_sharpness(processed_img)
    score_cont = pipeline.get_contrast(processed_img)

    return score_delta, score_tess, score_sharp, score_cont


# ============================================================
# ÉVALUATION DU PIPELINE
# ============================================================

def evaluate_pipeline(images, baseline_scores, params):
    """Évalue le pipeline sur un ensemble d'images.

    Args:
        images: Liste d'images (numpy arrays)
        baseline_scores: Liste des scores de base (OCR sur images originales)
        params: Dictionnaire de paramètres du pipeline

    Returns:
        (avg_delta, avg_abs, avg_sharp, avg_cont)
    """
    if not images:
        return 0, 0, 0, 0

    # STRATÉGIE ADAPTATIVE :
    # - Si CUDA : traitement séquentiel sur GPU
    # - Si CPU : multiprocessing parallèle

    if pipeline.USE_CUDA:
        # MODE GPU : Traitement séquentiel
        list_delta, list_abs, list_sharp, list_cont = [], [], [], []
        for i, img in enumerate(images):
            baseline = baseline_scores[i] if i < len(baseline_scores) else 0

            processed_img = pipeline.pipeline_complet(img, params)
            tess_abs = pipeline.get_tesseract_score(processed_img)
            tess_delta = tess_abs - baseline
            sharp = pipeline.get_sharpness(processed_img)
            cont = pipeline.get_contrast(processed_img)

            list_delta.append(tess_delta)
            list_abs.append(tess_abs)
            list_sharp.append(sharp)
            list_cont.append(cont)

    else:
        # MODE CPU : Multiprocessing
        pool_args = zip(images, repeat(params), baseline_scores)
        optimal_workers = int(os.cpu_count() * 1.5)
        pool_size = min(len(images), optimal_workers)

        try:
            with multiprocessing.Pool(processes=pool_size) as pool:
                results = pool.map(process_image_fast, pool_args)

            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return 0, 0, 0, 0

            list_delta, list_abs, list_sharp, list_cont = zip(*valid_results)

        except Exception as e:
            print(f"Erreur multiprocessing: {e}")
            # Fallback séquentiel
            list_delta, list_abs, list_sharp, list_cont = [], [], [], []
            for i, img in enumerate(images):
                baseline = baseline_scores[i] if i < len(baseline_scores) else 0
                processed_img = pipeline.pipeline_complet(img, params)
                tess_abs = pipeline.get_tesseract_score(processed_img)
                tess_delta = tess_abs - baseline
                sharp = pipeline.get_sharpness(processed_img)
                cont = pipeline.get_contrast(processed_img)

                list_delta.append(tess_delta)
                list_abs.append(tess_abs)
                list_sharp.append(sharp)
                list_cont.append(cont)

    # Moyennes
    avg_delta = sum(list_delta) / len(list_delta) if list_delta else 0
    avg_abs = sum(list_abs) / len(list_abs) if list_abs else 0
    avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
    avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0

    return avg_delta, avg_abs, avg_sharp, avg_cont


# ============================================================
# CALCUL DES SCORES BASELINE
# ============================================================

def calculate_baseline_scores(images):
    """Calcule les scores OCR des images originales.

    Args:
        images: Liste d'images (numpy arrays)

    Returns:
        Liste des scores baseline
    """
    baseline_scores = []
    for img in images:
        score = pipeline.get_tesseract_score(img)
        baseline_scores.append(score)
    return baseline_scores


# ============================================================
# UTILITAIRES PARAMÈTRES
# ============================================================

def build_params(line_h, line_v, norm_kernel_base, denoise_h, noise_threshold,
                bin_block_base, bin_c, dilate_iter=2):
    """Construit un dictionnaire de paramètres pour le pipeline.

    Args:
        line_h: Taille kernel horizontal pour suppression lignes
        line_v: Taille kernel vertical pour suppression lignes
        norm_kernel_base: Base pour norm_kernel (sera transformé en impair)
        denoise_h: Paramètre h pour denoising
        noise_threshold: Seuil pour denoising adaptatif
        bin_block_base: Base pour bin_block_size (sera transformé en impair)
        bin_c: Constante pour binarisation adaptative
        dilate_iter: Nombre d'itérations de dilatation

    Returns:
        Dictionnaire de paramètres
    """
    return {
        'line_h_size': int(line_h),
        'line_v_size': int(line_v),
        'dilate_iter': int(dilate_iter),
        'norm_kernel': int(norm_kernel_base) * 2 + 1,  # Toujours impair
        'denoise_h': float(denoise_h),
        'noise_threshold': float(noise_threshold),
        'bin_block_size': int(bin_block_base) * 2 + 1,  # Toujours impair
        'bin_c': float(bin_c)
    }


def params_to_tuple(params):
    """Convertit un dict de paramètres en tuple ordonné."""
    return (
        params['line_h_size'],
        params['line_v_size'],
        (params['norm_kernel'] - 1) // 2,  # Retour à la base
        params['denoise_h'],
        params['noise_threshold'],
        (params['bin_block_size'] - 1) // 2,  # Retour à la base
        params['bin_c'],
        params['dilate_iter']
    )
