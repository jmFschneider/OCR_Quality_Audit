"""
pipeline.py - Traitement d'images avec support CUDA
Basé sur sobol_test_pipeline.py (version stable qui fonctionne)

Moteur OCR: Tesseract (excellent pour le français, stable avec CUDA)
"""

import cv2
import numpy as np
import pytesseract
import shutil
import os
import sys

# ============================================================
# CONFIGURATION TESSERACT
# ============================================================

# Vérification auto du path Tesseract
if not shutil.which("tesseract"):
    # Tesseract n'est pas dans le PATH
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"/usr/bin/tesseract",
        r"/usr/local/bin/tesseract"
    ]
    
    found_tesseract = False
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            print(f"[INFO] Tesseract trouve: {p}")
            found_tesseract = True
            break

    if not found_tesseract:
        print("[WARNING] Tesseract non trouve ! Assurez-vous qu'il est installe et dans le PATH.")

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
    print(f"[OK] GPU CUDA active ({cv2.cuda.getCudaEnabledDeviceCount()} device(s))")
    print("[INFO] Moteur OCR: Tesseract + OpenCV CUDA")
else:
    print("[WARNING] Mode CPU uniquement")
    print("[INFO] Moteur OCR: Tesseract")


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
    """Score OCR Tesseract (confiance moyenne).

    Returns:
        Score de confiance moyen (0-100)
    """
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
    except Exception as e:
        print(f"❌ Erreur Tesseract: {e}")
        return 0.0

def get_cnr_quality(image):
    """
    Calcule le Contrast-to-Noise Ratio (CNR) spécifique aux documents.
    Idéal pour évaluer la qualité pour des IA visuelles (Gemini, etc.)
    qui préfèrent un fond lisse et un bon contraste, sans binarisation stricte.
    
    Formula: (Mean_BG - Mean_FG) / Std_BG
    
    Returns:
        float: Score CNR (plus c'est haut, mieux c'est). 
               Typiquement: < 2 (Mauvais), 2-5 (Moyen), > 5 (Excellent)
    """
    try:
        cpu_img = _to_gray_uint8(image)
        if cpu_img is None:
            return 0.0

        # 1. Séparation grossière Texte/Fond via Otsu (juste pour l'analyse)
        # On utilise THRESH_BINARY_INV pour avoir Texte=Blanc(255), Fond=Noir(0) dans le masque
        thresh_val, mask_fg = cv2.threshold(cpu_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask_bg = cv2.bitwise_not(mask_fg)

        # Vérification sécurité (si image toute blanche ou toute noire)
        n_fg = cv2.countNonZero(mask_fg)
        n_bg = cv2.countNonZero(mask_bg)
        
        if n_fg == 0 or n_bg == 0:
            return 0.0

        # 2. Calcul des statistiques
        # MeanStdDev calcule moy et std pour les pixels où le masque != 0
        mean_bg, std_bg = cv2.meanStdDev(cpu_img, mask=mask_bg)
        mean_fg, _ = cv2.meanStdDev(cpu_img, mask=mask_fg)

        m_bg = mean_bg[0][0]
        s_bg = std_bg[0][0]
        m_fg = mean_fg[0][0]

        # 3. Calcul du score
        contrast = m_bg - m_fg
        
        # Si le contraste est inversé ou nul (texte plus clair que fond), c'est illisible
        if contrast <= 5:
            return 0.0
            
        # Pénalité bruit : on ajoute un epsilon pour éviter division par zéro
        # Si le fond est parfaitement uni (std=0), le score explose (ce qui est bien)
        noise = s_bg + 0.1 
        
        cnr = contrast / noise
        
        return float(cnr)
        
    except Exception as e:
        print(f"⚠️ Erreur calcul CNR: {e}")
        return 0.0


def evaluer_toutes_metriques(image):
    """Calcule toutes les métriques d'une image avec chronométrage."""
    import time

    # ---- TESSERACT ----
    t0 = time.time()
    tess = get_tesseract_score(image)
    t_tess = (time.time() - t0) * 1000

    # ---- CNR (Gemini Quality) ----
    # Remplaçons Sharpness/Contrast génériques par CNR et Sharpness
    t0 = time.time()
    cnr = get_cnr_quality(image)
    # On garde sharpness car utile pour détecter le flou excessif
    sharp = get_sharpness(image)
    t_metrics = (time.time() - t0) * 1000

    # On retourne un tuple compatible avec l'existant, mais on remplace 'Contrast' par 'CNR'
    # Structure: (tess, sharp, cnr, t_tess, t_metrics, 0)
    # Le dernier temps est mis à 0 pour garder la signature
    return (
        tess, sharp, cnr,
        t_tess, t_metrics, 0,
    )


def evaluer_toutes_metriques_batch(images, max_workers=None, verbose=False):
    """Calcule les métriques pour plusieurs images en parallèle.

    Utilise multiprocessing pour accélérer le traitement Tesseract OCR.
    Speedup typique: 2-3x sur CPU multi-core.

    Args:
        images: Liste d'images (numpy arrays)
        max_workers: Nombre de workers (None = auto-detect CPU count)
        verbose: Si True, affiche les messages de progression (défaut: False)

    Returns:
        Liste de tuples (tess, sharp, cont, t_tess, t_sharp, t_cont)
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(images))

    if verbose:
        print(f"🚀 Traitement parallèle: {len(images)} images avec {max_workers} workers")

    t0_total = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(evaluer_toutes_metriques, images))

    t_total = (time.time() - t0_total) * 1000

    if verbose:
        print(f"✅ Batch terminé en {t_total:.0f}ms ({t_total/len(images):.0f}ms/image)")

    return results


# ============================================================
# NOUVEAU PIPELINE: HAUTE FIDÉLITÉ (Blur + CLAHE)
# ============================================================

def pipeline_blur_clahe(image, params):
    """
    Pipeline 'Haute Fidélité' optimisé pour Gemini.
    Privilégie la conservation de la texture (niveaux de gris) vs binarisation.
    
    Args:
        image: Image source (numpy array uint8)
        params: Dict contenant:
            - inp_line_h (int): Largeur kernel ligne horizontale (ex: 40)
            - inp_line_v (int): Hauteur kernel ligne verticale (ex: 40)
            - denoise_h (float): Force du denoising NLMeans (ex: 12.0)
            - bg_dilate (int): Taille kernel dilatation fond (ex: 7)
            - bg_blur (int): Taille kernel median blur fond (ex: 21)
            - clahe_clip (float): Clip limit CLAHE (ex: 2.0)
            - clahe_tile (int): Taille grille CLAHE (ex: 8)
            
    Returns:
        Image traitée (numpy array uint8)
    """
    # 0. S'assurer qu'on est sur CPU (Inpainting et certains algos complexes)
    cpu_img = _to_gray_uint8(image)
    if cpu_img is None:
        return None

    # --- ÉTAPE A : Suppression des Lignes par "Inpainting" ---
    # Détection
    # Note: On utilise des valeurs fixes pour le threshold ici pour simplifier, 
    # ou on pourrait ajouter des params si nécessaire. OTSU est généralement bon.
    _, thresh = cv2.threshold(cpu_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Kernel Horizontal
    h_size = params.get('inp_line_h', 40)
    if h_size < 1: h_size = 1
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    remove_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Kernel Vertical
    v_size = params.get('inp_line_v', 40)
    if v_size < 1: v_size = 1
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    remove_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # Masque combiné + Dilatation
    mask_lines = cv2.add(remove_h, remove_v)
    mask_lines = cv2.dilate(mask_lines, np.ones((3,3), np.uint8), iterations=1)
    
    # Inpainting (Lent mais qualitatif)
    # Radius 3 est standard. TELEA est souvent plus rapide/stable que NS.
    img_no_lines = cv2.inpaint(cpu_img, mask_lines, 3, cv2.INPAINT_TELEA)
    
    # --- ÉTAPE B : Denoising (FastNLMeans) ---
    h_val = params.get('denoise_h', 12.0)
    # Template 7, Search 21 sont des standards
    img_denoised = cv2.fastNlMeansDenoising(img_no_lines, None, h=h_val, 
                                            templateWindowSize=7, searchWindowSize=21)
                                            
    # --- ÉTAPE C : Normalisation par Division (Preservation Grayscale) ---
    # 1. Estimation du fond
    dil_k = params.get('bg_dilate', 7)
    if dil_k % 2 == 0: dil_k += 1
    dilated_img = cv2.dilate(img_denoised, np.ones((dil_k, dil_k), np.uint8))
    
    blur_k = params.get('bg_blur', 21)
    if blur_k % 2 == 0: blur_k += 1
    bg_img = cv2.medianBlur(dilated_img, blur_k)
    
    # 2. Différence (Fond blanc - Différence)
    # Cela permet d'avoir le texte (sombre) sur fond blanc (255)
    diff_img = 255 - cv2.absdiff(img_denoised, bg_img)
    
    # 3. Normalisation (Étirement histogramme)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                             
    # --- ÉTAPE D : CLAHE Final ---
    clip = params.get('clahe_clip', 2.0)
    tile = params.get('clahe_tile', 8)
    if tile < 1: tile = 8
    
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    result = clahe.apply(norm_img)
    
    return result
