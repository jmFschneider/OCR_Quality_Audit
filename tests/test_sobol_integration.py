#!/usr/bin/env python3
"""
Test d'intégration du screening Sobol
Vérifie que tout fonctionne avec la nouvelle architecture modulaire
"""

import os
import cv2
import numpy as np
from glob import glob
import pipeline
import optimizer

# Configuration
INPUT_FOLDER = "test_scans"
N_TEST_POINTS = 8  # Petit nombre pour test rapide (2^3)

print("="*70)
print("TEST D'INTÉGRATION SOBOL - Architecture modulaire")
print("="*70)

# 1. Vérification CUDA
print(f"\n1. DÉTECTION CUDA:")
print(f"   USE_CUDA: {pipeline.USE_CUDA}")
if pipeline.USE_CUDA:
    print(f"   ✅ GPU CUDA activé")
else:
    print(f"   ⚠️  Mode CPU uniquement")

# 2. Chargement des images de test
print(f"\n2. CHARGEMENT DES IMAGES:")
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
print(f"   Images trouvées: {len(image_files)}")

if len(image_files) == 0:
    print("   ❌ Aucune image trouvée dans test_scans/")
    exit(1)

# Charger seulement les 2 premières images pour le test
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

# 4. Test du pipeline avec paramètres par défaut
print(f"\n4. TEST PIPELINE:")
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

print(f"   Test avec paramètres par défaut...")
delta, abs_score, sharp, cont = optimizer.evaluate_pipeline(
    test_images, baseline_scores, default_params
)
print(f"   Delta Tesseract: {delta:.2f}%")
print(f"   Score absolu: {abs_score:.2f}%")
print(f"   Netteté: {sharp:.2f}")
print(f"   Contraste: {cont:.2f}")

# 5. Test du screening Sobol
print(f"\n5. TEST SCREENING SOBOL ({N_TEST_POINTS} points):")

# Définir les ranges de paramètres (réduits pour test rapide)
param_ranges = {
    'line_h': (40, 50),
    'norm_kernel': (70, 80),
    'denoise_h': (8.0, 10.0)
}

fixed_params = {
    'dilate_iter': 2,
    'line_v_size': 50,
    'noise_threshold': 100.0,
    'bin_block_size': 121,
    'bin_c': 15.0
}

print(f"   Paramètres actifs: {list(param_ranges.keys())}")
print(f"   Paramètres fixes: {list(fixed_params.keys())}")

# Callback pour afficher les progrès
def test_callback(point_idx, scores_dict, params_dict):
    print(f"   Point {point_idx+1}: Delta={scores_dict['tesseract_delta']:.2f}%")

# Lancer le screening
print(f"\n   Lancement du screening...")
best_params, csv_file = optimizer.run_sobol_screening(
    images=test_images,
    baseline_scores=baseline_scores,
    n_points=N_TEST_POINTS,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    callback=test_callback,
    cancellation_event=None
)

# 6. Résultats
print(f"\n6. RÉSULTATS:")
if best_params:
    print(f"   ✅ Meilleurs paramètres trouvés:")
    for key, val in best_params.items():
        print(f"      {key}: {val}")
    print(f"   📁 CSV généré: {csv_file}")

    # Vérifier que le fichier CSV existe
    if os.path.exists(csv_file):
        print(f"   ✅ Fichier CSV vérifié")
        # Lire les premières lignes
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            print(f"   CSV contient {len(lines)} lignes (header + {len(lines)-1} points)")
    else:
        print(f"   ❌ Fichier CSV introuvable!")
else:
    print(f"   ⚠️ Aucun meilleur paramètre (erreur?)")

print("\n" + "="*70)
print("TEST TERMINÉ")
print("="*70)
