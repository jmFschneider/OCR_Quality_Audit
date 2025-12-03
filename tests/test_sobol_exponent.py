#!/usr/bin/env python3
"""
Test du sélecteur d'exposant Sobol dans l'interface
"""

import os
import cv2
import numpy as np
from glob import glob
import optimizer
import pipeline

# Configuration
INPUT_FOLDER = "test_scans"

print("="*70)
print("TEST SÉLECTEUR EXPOSANT SOBOL")
print("="*70)

# Test du calcul 2^n
print("\n1. TEST DU CALCUL D'EXPOSANT:")
test_values = [3, 5, 7, 10, 12]
for exp in test_values:
    n_points = 2**exp
    print(f"   2^{exp:2d} = {n_points:6d} points")

# Test avec valeur limite
print(f"\n   Valeur limite:")
print(f"   2^16 = {2**16} points (max recommandé)")
print(f"   2^17 = {2**17} points (trop élevé)")

# 2. Chargement des images
print(f"\n2. CHARGEMENT DES IMAGES DE TEST:")
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
print(f"   Images trouvées: {len(image_files)}")

if len(image_files) == 0:
    print("   ❌ Aucune image trouvée")
    exit(1)

# Charger 2 images
test_images = []
for f in image_files[:2]:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        test_images.append(img)

print(f"   ✅ {len(test_images)} images chargées")

# 3. Calcul baseline
print(f"\n3. CALCUL DES SCORES BASELINE:")
baseline_scores = optimizer.calculate_baseline_scores(test_images)
print(f"   Scores: {[f'{s:.2f}' for s in baseline_scores]}")

# 4. Test Sobol avec différents exposants
print(f"\n4. TEST SCREENING SOBOL AVEC DIFFÉRENTS EXPOSANTS:")

# Test avec exposant 2 (4 points) - très rapide
exponent = 2
n_points = 2**exponent
print(f"\n   Test avec 2^{exponent} = {n_points} points:")

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

best_params, csv_file = optimizer.run_sobol_screening(
    images=test_images,
    baseline_scores=baseline_scores,
    n_points=n_points,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    callback=None,
    cancellation_event=None,
    verbose_timing=False
)

print(f"\n   ✅ Screening terminé")
print(f"   📁 Fichier: {csv_file}")

# 5. Estimation des temps pour différents exposants
print(f"\n5. ESTIMATION DES TEMPS (pour {len(test_images)} images):")
temps_par_point = 2  # secondes (estimation basée sur les tests précédents)

estimations = [
    (3, 8),      # Très rapide
    (5, 32),     # Rapide
    (7, 128),    # Moyen
    (8, 256),    # Long
    (10, 1024),  # Très long
]

for exp, pts in estimations:
    temps_s = pts * temps_par_point
    if temps_s < 60:
        temps_str = f"{temps_s:.0f}s"
    elif temps_s < 3600:
        temps_str = f"{temps_s/60:.1f}min"
    else:
        temps_str = f"{temps_s/3600:.1f}h"
    print(f"   2^{exp:2d} = {pts:5d} points → ~{temps_str:>8s}")

print("\n" + "="*70)
print("RECOMMANDATIONS:")
print("="*70)
print("  • Exploration rapide    : 2^5 = 32 points    (~1 min)")
print("  • Exploration standard  : 2^7 = 128 points   (~4 min)")
print("  • Exploration complète  : 2^8 = 256 points   (~8 min)")
print("  • Screening exhaustif   : 2^10 = 1024 points (~30 min)")
print("="*70)
