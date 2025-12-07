#!/usr/bin/env python3
"""
Test du suivi des temps de traitement et OCR
"""

import os
import cv2
import numpy as np
from glob import glob
import pipeline
import optimizer

# Configuration
INPUT_FOLDER = "test_scans"

print("="*70)
print("TEST DU SUIVI DES TEMPS - Traitement d'image + OCR")
print("="*70)

# 1. Détection CUDA
print(f"\n1. MODE D'EXÉCUTION:")
if pipeline.USE_CUDA:
    print(f"   ✅ GPU CUDA activé")
else:
    print(f"   ⚠️  Mode CPU (multiprocessing)")

# 2. Chargement des images de test
print(f"\n2. CHARGEMENT DES IMAGES:")
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
print(f"   Images trouvées: {len(image_files)}")

if len(image_files) == 0:
    print("   ❌ Aucune image trouvée dans test_scans/")
    exit(1)

# Charger 2 images pour le test
test_images = []
for f in image_files[:2]:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        test_images.append(img)

print(f"   ✅ {len(test_images)} images chargées")

# 3. Calcul des scores baseline
print(f"\n3. CALCUL DES SCORES BASELINE:")
baseline_scores = optimizer.calculate_baseline_scores(test_images)
print(f"   Scores baseline: {[f'{s:.2f}' for s in baseline_scores]}")

# 4. Test du pipeline avec mesure des temps
print(f"\n4. TEST PIPELINE AVEC MESURE DES TEMPS:")
default_params = {
    'line_h_size': 45,
    'line_v_size': 50,
    'dilate_iter': 2,
    'norm_kernel': 151,  # 75*2+1
    'denoise_h': 9.0,
    'noise_threshold': 100.0,
    'bin_block_size': 121,  # 60*2+1
    'bin_c': 15.0
}

print(f"\n   Test avec evaluate_pipeline_timed (verbose=True):")
print(f"   " + "-"*60)
delta, abs_score, sharp, cont, temps_trait, temps_ocr = optimizer.evaluate_pipeline_timed(
    test_images, baseline_scores, default_params, verbose=True
)
print(f"   " + "-"*60)
print(f"\n   Résultats moyens:")
print(f"   - Delta Tesseract: {delta:.2f}%")
print(f"   - Score absolu: {abs_score:.2f}%")
print(f"   - Netteté: {sharp:.2f}")
print(f"   - Contraste: {cont:.2f}")
print(f"   - Temps traitement moyen: {temps_trait:.0f} ms")
print(f"   - Temps OCR moyen: {temps_ocr:.0f} ms")
print(f"   - TEMPS TOTAL moyen: {temps_trait + temps_ocr:.0f} ms")

# 5. Test Sobol avec affichage des temps
print(f"\n5. TEST SOBOL SCREENING AVEC TEMPS (4 points):")

param_ranges = {
    'line_h': (40, 50),
    'norm_kernel': (70, 80),
}

fixed_params = {
    'dilate_iter': 2,
    'line_v_size': 50,
    'denoise_h': 9.0,
    'noise_threshold': 100.0,
    'bin_block_size': 121,
    'bin_c': 15.0
}

print(f"\n   Lancement du screening avec verbose_timing=True...")
print(f"   " + "="*60)

best_params, csv_file = optimizer.run_sobol_screening(
    images=test_images,
    baseline_scores=baseline_scores,
    n_points=4,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    callback=None,
    cancellation_event=None,
    verbose_timing=True  # Affiche les temps de chaque image
)

print(f"   " + "="*60)

# 6. Résumé
print(f"\n6. RÉSUMÉ:")
print(f"   ✅ Mesure des temps implémentée avec succès")
print(f"   ✅ Temps de traitement d'image séparé du temps OCR")
print(f"   ✅ Affichage détaillé pour chaque image")
print(f"   ✅ Affichage des moyennes par point Sobol")
print(f"   📁 Résultats dans: {csv_file}")

print("\n" + "="*70)
print("TEST TERMINÉ")
print("="*70)
