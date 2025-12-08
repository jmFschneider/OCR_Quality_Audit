#!/usr/bin/env python3
"""
Script de test pour identifier le goulot d'étranglement dans pipeline_blur_clahe
"""
import cv2
import time
import glob
import numpy as np

# Import du pipeline
import pipeline

def test_pipeline_blur_clahe():
    """Teste chaque étape du pipeline blur_clahe avec chronométrage détaillé."""

    # Charger une image de test
    images = glob.glob("test_scans/*.jpg")
    if not images:
        print("❌ Aucune image trouvée dans test_scans/")
        return

    img_path = images[0]
    print(f"📸 Test avec: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Impossible de charger l'image")
        return

    print(f"📏 Dimensions: {img.shape}")

    # Paramètres par défaut
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    print("\n🔍 CHRONOMÉTRAGE DÉTAILLÉ DU PIPELINE BLUR+CLAHE\n")
    print("="*60)

    # Copie pour ne pas modifier l'original
    cpu_img = img.copy()

    # ÉTAPE 0: Conversion grayscale (déjà fait)
    print("✅ Étape 0: Conversion grayscale - OK")

    # ÉTAPE A: Suppression des lignes par Inpainting
    print("\n📍 ÉTAPE A: Détection et suppression des lignes...")
    t0 = time.time()

    # A.1: Threshold
    t_sub = time.time()
    _, thresh = cv2.threshold(cpu_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    t_thresh = (time.time() - t_sub) * 1000
    print(f"  A.1 Threshold OTSU: {t_thresh:.0f}ms")

    # A.2: Détection lignes horizontales
    t_sub = time.time()
    h_size = params.get('inp_line_h', 40)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    remove_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    t_h = (time.time() - t_sub) * 1000
    print(f"  A.2 Lignes horizontales (kernel={h_size}): {t_h:.0f}ms")

    # A.3: Détection lignes verticales
    t_sub = time.time()
    v_size = params.get('inp_line_v', 40)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    remove_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    t_v = (time.time() - t_sub) * 1000
    print(f"  A.3 Lignes verticales (kernel={v_size}): {t_v:.0f}ms")

    # A.4: Fusion et dilatation du masque
    t_sub = time.time()
    mask_lines = cv2.add(remove_h, remove_v)
    mask_lines = cv2.dilate(mask_lines, np.ones((3,3), np.uint8), iterations=1)
    t_mask = (time.time() - t_sub) * 1000
    print(f"  A.4 Fusion masque: {t_mask:.0f}ms")

    # A.5: INPAINTING (GOULOT D'ÉTRANGLEMENT PROBABLE)
    print("\n  ⏳ A.5 INPAINTING (opération lente)...")
    t_sub = time.time()
    img_no_lines = cv2.inpaint(cpu_img, mask_lines, 3, cv2.INPAINT_TELEA)
    t_inpaint = (time.time() - t_sub) * 1000
    print(f"  ✅ A.5 Inpainting: {t_inpaint:.0f}ms ({'LENT!' if t_inpaint > 1000 else 'OK'})")

    t_etape_a = (time.time() - t0) * 1000
    print(f"\n📊 ÉTAPE A - TOTAL: {t_etape_a:.0f}ms")

    # ÉTAPE B: Denoising
    print("\n📍 ÉTAPE B: Denoising NLMeans...")
    t0 = time.time()
    h_val = params.get('denoise_h', 12.0)
    print(f"  ⏳ Denoising (h={h_val}, opération lente)...")
    img_denoised = cv2.fastNlMeansDenoising(img_no_lines, None, h=h_val,
                                            templateWindowSize=7, searchWindowSize=21)
    t_denoise = (time.time() - t0) * 1000
    print(f"  ✅ Denoising: {t_denoise:.0f}ms ({'LENT!' if t_denoise > 1000 else 'OK'})")

    # ÉTAPE C: Normalisation par division
    print("\n📍 ÉTAPE C: Normalisation par division...")
    t0 = time.time()

    dil_k = params.get('bg_dilate', 7)
    if dil_k % 2 == 0: dil_k += 1
    dilated_img = cv2.dilate(img_denoised, np.ones((dil_k, dil_k), np.uint8))

    blur_k = params.get('bg_blur', 21)
    if blur_k % 2 == 0: blur_k += 1
    bg_img = cv2.medianBlur(dilated_img, blur_k)

    diff_img = 255 - cv2.absdiff(img_denoised, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    t_norm = (time.time() - t0) * 1000
    print(f"  ✅ Normalisation: {t_norm:.0f}ms")

    # ÉTAPE D: CLAHE
    print("\n📍 ÉTAPE D: CLAHE...")
    t0 = time.time()
    clip = params.get('clahe_clip', 2.0)
    tile = params.get('clahe_tile', 8)
    if tile < 1: tile = 8

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    result = clahe.apply(norm_img)
    t_clahe = (time.time() - t0) * 1000
    print(f"  ✅ CLAHE: {t_clahe:.0f}ms")

    # RÉSUMÉ
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TEMPS:")
    print("="*60)
    print(f"  Étape A (Inpainting):      {t_etape_a:6.0f}ms")
    print(f"    - dont inpaint seul:     {t_inpaint:6.0f}ms ({t_inpaint/t_etape_a*100:.0f}%)")
    print(f"  Étape B (Denoising):       {t_denoise:6.0f}ms")
    print(f"  Étape C (Normalisation):   {t_norm:6.0f}ms")
    print(f"  Étape D (CLAHE):           {t_clahe:6.0f}ms")
    print(f"  {'─'*60}")
    total_time = t_etape_a + t_denoise + t_norm + t_clahe
    print(f"  TOTAL:                     {total_time:6.0f}ms ({total_time/1000:.2f}s)")
    print("="*60)

    # Identifier le goulot
    times = [
        ('Inpainting', t_inpaint),
        ('Denoising', t_denoise),
        ('Normalisation', t_norm),
        ('CLAHE', t_clahe)
    ]
    times.sort(key=lambda x: x[1], reverse=True)

    print("\n🎯 GOULOTS D'ÉTRANGLEMENT (par ordre):")
    for i, (name, t) in enumerate(times, 1):
        percent = (t / total_time) * 100
        print(f"  {i}. {name:20s}: {t:6.0f}ms ({percent:5.1f}%)")

    # Estimation pour 8 images
    print(f"\n⏱️  ESTIMATION POUR 8 IMAGES:")
    print(f"  Temps par image:  {total_time/1000:.2f}s")
    print(f"  Temps pour 8:     {total_time*8/1000:.2f}s ({total_time*8/1000/60:.1f} minutes)")

    # Estimation pour 4096 points Sobol avec 8 images
    print(f"\n⏱️  ESTIMATION POUR SCREENING SOBOL (4096 points, 8 images):")
    total_points_time = total_time * 8 * 4096 / 1000 / 60  # en minutes
    print(f"  Temps total:      {total_points_time:.0f} minutes ({total_points_time/60:.1f} heures)")

    # Avec multiprocessing (8 cores)
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    parallel_time = total_points_time / n_cores
    print(f"\n  Avec multiprocessing ({n_cores} cores):")
    print(f"  Temps parallèle:  {parallel_time:.0f} minutes ({parallel_time/60:.1f} heures)")

    print("\n" + "="*60)

    return result, total_time


if __name__ == "__main__":
    result, total_time = test_pipeline_blur_clahe()

    if total_time > 5000:  # Plus de 5 secondes par image
        print("\n⚠️  AVERTISSEMENT: Le pipeline est TRÈS LENT!")
        print("    Recommandations:")
        print("    1. Réduire la qualité du denoising (denoise_h)")
        print("    2. Réduire le rayon d'inpainting (actuellement 3)")
        print("    3. Réduire la résolution des images avant traitement")
        print("    4. Utiliser un pre-screening avec moins de points")
