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

USE_CUDA = False
try:
    USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except AttributeError:
    USE_CUDA = False  # OpenCV sans support CUDA compilé

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

def _to_gray_uint8(img):
    """Utilitaire interne : retourne une image 2D uint8."""
    cpu_img = ensure_cpu(img)
    if cpu_img is None:
        return None
    if cpu_img.ndim == 3:
        cpu_img = cv2.cvtColor(cpu_img, cv2.COLOR_BGR2GRAY)
    if cpu_img.dtype != np.uint8:
        cpu_img = cv2.normalize(cpu_img, None, 0, 255, cv2.NORM_MINMAX)
        cpu_img = cpu_img.astype(np.uint8)
    return cpu_img


def remove_lines_param(img, h_size, v_size, dilate_iter):
    """Suppression des lignes - Version stable avec threshold CPU."""
    # 1) CPU image uint8 propre
    cpu_img = _to_gray_uint8(img)
    if cpu_img is None:
        return None

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
    """Normalisation par division (CPU)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3

    cpu_img = _to_gray_uint8(image_gray)
    if cpu_img is None:
        return None

    # Ici on reste en CPU pour éviter les limitations CUDA sur les gros kernels.
    fond = cv2.GaussianBlur(cpu_img, (kernel_size, kernel_size), 0)
    result = cv2.divide(cpu_img, fond, scale=255)

    # On force uint8 pour la suite du pipeline_chat
    if result.dtype != np.uint8:
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        result = result.astype(np.uint8)

    return result


def adaptive_denoising(image, base_h_param, noise_threshold=100):
    """Denoising adaptatif basé sur le niveau de bruit (CPU)."""
    if base_h_param <= 0:
        return _to_gray_uint8(image)

    cpu_img = _to_gray_uint8(image)
    if cpu_img is None:
        return None

    # Estimation du bruit (variance du Laplacien)
    laplacian = cv2.Laplacian(cpu_img, cv2.CV_64F)
    noise_level = laplacian.var()

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
        image: Image en niveaux de gris (numpy array ou GpuMat)
        params: Dict avec clés:
            line_h_size, line_v_size, dilate_iter,
            norm_kernel, denoise_h, noise_threshold,
            bin_block_size, bin_c

    Returns:
        Image traitée (numpy array CPU uint8)
    """
    # 1. Suppression de lignes
    current_img = remove_lines_param(
        image,
        params['line_h_size'],
        params['line_v_size'],
        params['dilate_iter']
    )

    # 2. Normalisation par division
    current_img = normalisation_division(current_img, params['norm_kernel'])

    # 3. Denoising adaptatif
    current_img = adaptive_denoising(
        current_img,
        params['denoise_h'],
        params.get('noise_threshold', 100)
    )

    # 4. Binarisation adaptative (CPU uniquement)
    cpu_img = _to_gray_uint8(current_img)
    if cpu_img is None:
        return None

    block_size = params['bin_block_size']
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    result = cv2.adaptiveThreshold(
        cpu_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        params['bin_c']
    )

    return result


def pipeline_complet_timed(image, params):
    """Pipeline complet avec mesure du temps de traitement.

    Args:
        image: Image en niveaux de gris (numpy array)
        params: Dict avec clés: line_h_size, line_v_size, dilate_iter,
                norm_kernel, denoise_h, noise_threshold, bin_block_size, bin_c

    Returns:
        Tuple (image_traitée, temps_ms)
    """
    import time
    t0 = time.time()

    # 1. Suppression de lignes
    current_img = remove_lines_param(image, params['line_h_size'],
                                     params['line_v_size'], params['dilate_iter'])

    # 2. Normalisation par division
    current_img = ensure_cpu(current_img)
    current_img = normalisation_division(current_img, params['norm_kernel'])

    # 3. Denoising adaptatif
    current_img = adaptive_denoising(current_img, params['denoise_h'],
                                    params.get('noise_threshold', 100))

    # 4. Binarisation adaptative
    result = cv2.adaptiveThreshold(current_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])

    temps_ms = (time.time() - t0) * 1000
    return result, temps_ms


# ============================================================
# MÉTRIQUES
# ============================================================

def get_sharpness(image):
    """Calcule la netteté (variance du Laplacien)."""
    cpu_img = _to_gray_uint8(image)
    if cpu_img is None:
        return 0.0
    laplacian = cv2.Laplacian(cpu_img, cv2.CV_64F)
    return float(laplacian.var())


def get_contrast(image):
    """Calcule le contraste (écart-type)."""
    cpu_img = _to_gray_uint8(image)
    if cpu_img is None:
        return 0.0
    return float(cpu_img.std())


def get_tesseract_score(image):
    """Score OCR Tesseract (confiance moyenne)."""
    try:
        cpu_img = _to_gray_uint8(image)
        if cpu_img is None:
            return 0.0

        # Pre-resize pour optimiser les grandes images
        if cpu_img.shape[1] > 2500:
            cpu_img = cv2.resize(cpu_img, None, fx=0.5, fy=0.5)

        data = pytesseract.image_to_data(
            cpu_img,
            config='--oem 1 --psm 6',
            output_type=pytesseract.Output.DICT
        )

        confs = []
        for c in data.get('conf', []):
            try:
                v = int(c)
                if v != -1:
                    confs.append(v)
            except (ValueError, TypeError):
                continue

        return sum(confs) / len(confs) if confs else 0.0
    except Exception:
        return 0.0

def evaluer_toutes_metriques(image):
    """Calcule toutes les métriques d'une image avec chronométrage."""
    import time

    # ---- TESSERACT ----
    t0 = time.time()
    tess = get_tesseract_score(image)
    t_tess = (time.time() - t0) * 1000

    # ---- SHARPNESS ----
    t0 = time.time()
    sharp = get_sharpness(image)
    t_sharp = (time.time() - t0) * 1000

    # ---- CONTRAST ----
    t0 = time.time()
    cont = get_contrast(image)
    t_cont = (time.time() - t0) * 1000

    return (
        tess, sharp, cont,
        t_tess, t_sharp, t_cont,
    )
