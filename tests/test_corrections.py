#!/usr/bin/env python3
"""
Test des corrections après modifications de pipeline.py et optimizer.py
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
print("TEST DES CORRECTIONS - Pipeline + Optimizer")
print("="*70)

# 1. Test import et CUDA
print(f"\n1. TESTS D'IMPORT:")
print(f"   ✅ pipeline importé")
print(f"   ✅ optimizer importé")
print(f"   Mode CUDA: {pipeline.USE_CUDA}")

# 2. Chargement d'une image de test
print(f"\n2. CHARGEMENT IMAGE:")
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
if not image_files:
    print("   ❌ Aucune image trouvée")
    exit(1)

img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
if img is not None:
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    print(f"   ✅ Image chargée: {img.shape}")
else:
    print("   ❌ Échec chargement image")
    exit(1)

# 3. Test de la nouvelle fonction evaluer_toutes_metriques
print(f"\n3. TEST evaluer_toutes_metriques:")
try:
    result = pipeline.evaluer_toutes_metriques(img)
    print(f"   Retour: {len(result)} valeurs")
    tess, sharp, cont, t_tess, t_sharp, t_cont = result
    print(f"   ✅ Tesseract: {tess:.2f}% (temps: {t_tess:.0f}ms)")
    print(f"   ✅ Netteté: {sharp:.2f} (temps: {t_sharp:.0f}ms)")
    print(f"   ✅ Contraste: {cont:.2f} (temps: {t_cont:.0f}ms)")
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. Test du pipeline complet
print(f"\n4. TEST pipeline_complet:")
params = {
    'line_h_size': 45,
    'line_v_size': 50,
    'dilate_iter': 2,
    'norm_kernel': 151,
    'denoise_h': 9.0,
    'noise_threshold': 100.0,
    'bin_block_size': 121,
    'bin_c': 15.0
}

try:
    processed = pipeline.pipeline_complet(img, params)
    if processed is not None:
        print(f"   ✅ Pipeline OK: {processed.shape}, dtype={processed.dtype}")
    else:
        print(f"   ❌ Pipeline retourne None")
        exit(1)
except Exception as e:
    print(f"   ❌ Erreur pipeline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Test de evaluate_pipeline
print(f"\n5. TEST evaluate_pipeline:")
test_images = [img]
baseline_scores = [tess]  # On utilise le score de l'image originale

try:
    delta, abs_score, sharp_res, cont_res = optimizer.evaluate_pipeline(
        test_images, baseline_scores, params
    )
    print(f"   ✅ Delta: {delta:.2f}%")
    print(f"   ✅ Score absolu: {abs_score:.2f}%")
    print(f"   ✅ Netteté: {sharp_res:.2f}")
    print(f"   ✅ Contraste: {cont_res:.2f}")
except Exception as e:
    print(f"   ❌ Erreur evaluate_pipeline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test du screening Sobol (2 points seulement pour rapidité)
print(f"\n6. TEST run_sobol_screening (2 points):")
param_ranges = {
    'line_h': (40, 50),
}
fixed_params = {
    'dilate_iter': 2,
    'line_v_size': 50,
    'norm_kernel': 151,
    'denoise_h': 9.0,
    'noise_threshold': 100.0,
    'bin_block_size': 121,
    'bin_c': 15.0
}

try:
    best_params, csv_file = optimizer.run_sobol_screening(
        images=test_images,
        baseline_scores=baseline_scores,
        n_points=2,
        param_ranges=param_ranges,
        fixed_params=fixed_params,
        verbose_timing=False  # Paramètre déprécié mais gardé pour compatibilité
    )
    print(f"   ✅ Screening terminé")
    print(f"   ✅ Meilleurs params: {best_params}")
    print(f"   ✅ CSV généré: {csv_file}")
except Exception as e:
    print(f"   ❌ Erreur run_sobol_screening: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ TOUS LES TESTS RÉUSSIS")
print("="*70)
print("\nRÉSUMÉ DES MODIFICATIONS:")
print("  • pipeline.evaluer_toutes_metriques retourne maintenant 6 valeurs")
print("  • (tess, sharp, cont, t_tess, t_sharp, t_cont)")
print("  • Les temps sont affichés avec [PROFILE] en mode GPU")
print("  • evaluate_pipeline_timed a été supprimée")
print("  • process_image_timed a été supprimée")
print("  • run_sobol_screening utilise maintenant evaluate_pipeline")
print("="*70)
