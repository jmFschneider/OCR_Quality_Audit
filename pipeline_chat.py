"""
pipeline.py - Traitement d'images avec support CUDA
Basé sur sobol_test_pipeline.py (version stable qui fonctionne)
"""

import cv2
import numpy as np
import pytesseract

# ============================================================
# DÉTECTION CUDA
# ============================================================

USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

if USE_CUDA:
    cv2.cuda.setDevice(0)
    print(f"✅ GPU CUDA activé ({cv2.cuda.getCudaEnabledDeviceCount()} device(s))")
else:
    print("⚠️  Mode CPU uniquement")


# ============================================================
# UTILITAIRES GPU/CPU
# ============================================================

def ensure_gpu(image):
    """Upload image sur GPU si CUDA activé."""
    if not USE_CUDA:
        return image
    if isinstance(image, cv2.cuda_GpuMat):
        return image
    gpu = cv2.cuda_GpuMat()
    gpu.upload(image)
    return gpu


def ensure_cpu(image):
    """Download image du GPU si nécessaire."""
    if USE_CUDA and isinstance(image, cv2.cuda_GpuMat):
        return image.download()
    return image


# ============================================================
# FONCTIONS DE TRAITEMENT (VERSION STABLE)
# ============================================================

def remove_lines_param(img, h_size, v_size, dilate_iter):
    """Suppression des lignes - Version stable avec threshold CPU."""
    # 1) CPU image uint8 propre
    cpu_img = ensure_cpu(img)
    if cpu_img.ndim == 3:
        cpu_img = cv2.cvtColor(cpu_img, cv2.COLOR_BGR2GRAY)
    cpu_img = cpu_img.astype(np.uint8)

    # 2) Threshold OTSU CPU (stable, rapide)
    _, cpu_thresh = cv2.threshold(cpu_img, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- CPU only fallback ---
    if not USE_CUDA:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        h = cv2.morphologyEx(cpu_thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v = cv2.morphologyEx(cpu_thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        mask = cv2.addWeighted(h, 1, v, 1, 0.0)
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)), iterations=dilate_iter)
        result = cpu_img.copy()
        result[mask > 0] = 255
        return result

    # --- GPU steps ---
    gpu_src = ensure_gpu(cpu_img)
    gpu_thresh = ensure_gpu(cpu_thresh)

    # Morphologie H/V
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    morph_h = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, gpu_thresh.type(),
                                              h_kernel, iterations=2)
    morph_v = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, gpu_thresh.type(),
                                              v_kernel, iterations=2)

    gpu_h = morph_h.apply(gpu_thresh)
    gpu_v = morph_v.apply(gpu_thresh)

    # Fusion
    gpu_mask = cv2.cuda.addWeighted(gpu_h, 1.0, gpu_v, 1.0, 0.0)

    # Dilatation
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_dil = cv2.cuda.createMorphologyFilter(
        cv2.MORPH_DILATE, gpu_mask.type(), d_kernel,
        iterations=dilate_iter
    )
    gpu_mask = morph_dil.apply(gpu_mask)

    # Fusion finale avec l'image originale
    gpu_result = cv2.cuda.max(gpu_src, gpu_mask)
    return gpu_result


def normalisation_division(image_gray, kernel_size):
    """Normalisation par division."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    cpu_img = ensure_cpu(image_gray)

    # Limitation: GaussianBlur kernel <= 31 pour CUDA
    # Pour simplifier, toujours utiliser CPU (rapide et stable)
    fond = cv2.GaussianBlur(cpu_img, (kernel_size, kernel_size), 0)
    result = cv2.divide(cpu_img, fond, scale=255)

    return result


def adaptive_denoising(image, base_h_param, noise_threshold=100):
    """Denoising adaptatif basé sur le niveau de bruit."""
    if base_h_param <= 0:
        return image

    # Estimation du bruit (CPU)
    cpu_img = ensure_cpu(image)
    laplacian = cv2.Laplacian(cpu_img, cv2.CV_64F)
    noise_level = laplacian.var()

    # Denoising CPU (pas d'équivalent CUDA performant)
    if noise_level < noise_threshold:
        result = cv2.fastNlMeansDenoising(cpu_img, None, h=base_h_param,
                                         templateWindowSize=7, searchWindowSize=15)
    else:
        result = cv2.fastNlMeansDenoising(cpu_img, None, h=base_h_param,
                                         templateWindowSize=7, searchWindowSize=21)

    return result


def pipeline_complet(image, params):
    """Pipeline complet de traitement d'image.

    Args:
        image: Image en niveaux de gris (numpy array)
        params: Dict avec clés: line_h_size, line_v_size, dilate_iter,
                norm_kernel, denoise_h, noise_threshold, bin_block_size, bin_c

    Returns:
        Image traitée (numpy array CPU)
    """
    # 1. Suppression de lignes
    current_img = remove_lines_param(image, params['line_h_size'],
                                     params['line_v_size'], params['dilate_iter'])

    # 2. Normalisation par division
    current_img = ensure_cpu(current_img)  # S'assurer qu'on est sur CPU
    current_img = normalisation_division(current_img, params['norm_kernel'])

    # 3. Denoising adaptatif
    current_img = adaptive_denoising(current_img, params['denoise_h'],
                                    params.get('noise_threshold', 100))

    # 4. Binarisation adaptative (CPU uniquement)
    result = cv2.adaptiveThreshold(current_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])

    return result


# ============================================================
# MÉTRIQUES
# ============================================================

def get_sharpness(image):
    """Calcule la netteté (variance du Laplacien)."""
    cpu_img = ensure_cpu(image)
    laplacian = cv2.Laplacian(cpu_img, cv2.CV_64F)
    return float(laplacian.var())


def get_contrast(image):
    """Calcule le contraste (écart-type)."""
    cpu_img = ensure_cpu(image)
    return float(cpu_img.std())


def get_tesseract_score(image):
    """Score OCR Tesseract (confiance moyenne)."""
    try:
        cpu_img = ensure_cpu(image)

        # Pre-resize pour optimiser les grandes images
        if cpu_img.shape[1] > 2500:
            cpu_img = cv2.resize(cpu_img, None, fx=0.5, fy=0.5)

        data = pytesseract.image_to_data(cpu_img, config='--oem 1 --psm 6',
                                        output_type=pytesseract.Output.DICT)
        confs = [int(x) for x in data['conf'] if int(x) != -1]
        return sum(confs) / len(confs) if confs else 0
    except Exception:
        return 0


def evaluer_toutes_metriques(image):
    """Calcule toutes les métriques d'une image."""
    return (get_tesseract_score(image),
            get_sharpness(image),
            get_contrast(image))
