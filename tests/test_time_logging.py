#!/usr/bin/env python3
"""
Test du système de logging des temps
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
print("TEST DU SYSTÈME DE LOGGING DES TEMPS")
print("="*70)

# 1. Chargement des images
print(f"\n1. CHARGEMENT DES IMAGES:")
image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
if not image_files:
    print("   ❌ Aucune image trouvée")
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

# 2. Calcul baseline
print(f"\n2. CALCUL DES SCORES BASELINE:")
baseline_scores = optimizer.calculate_baseline_scores(test_images)
print(f"   Scores: {[f'{s:.2f}' for s in baseline_scores]}")

# 3. Test du screening avec logging
print(f"\n3. TEST SCREENING SOBOL AVEC LOGGING (4 points):")

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

print("\n   Lancement avec enable_time_logging=True...")
best_params, csv_file = optimizer.run_sobol_screening(
    images=test_images,
    baseline_scores=baseline_scores,
    n_points=4,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    enable_time_logging=True  # Active le logging
)

print(f"\n   ✅ Screening terminé")
print(f"   📁 CSV résultats: {csv_file}")

# 4. Vérifier que le fichier de timing a été créé
print(f"\n4. VÉRIFICATION DU FICHIER DE TIMING:")
timing_files = glob("timing_log_*.csv")
if timing_files:
    latest_timing = sorted(timing_files)[-1]
    print(f"   ✅ Fichier créé: {latest_timing}")

    # Compter les lignes
    with open(latest_timing, 'r') as f:
        lines = f.readlines()
        nb_lignes = len(lines) - 1  # -1 pour le header
        print(f"   ✅ {nb_lignes} mesures enregistrées")

        # Afficher les 3 premières lignes
        print(f"\n   Aperçu du fichier:")
        for i, line in enumerate(lines[:4]):  # Header + 3 lignes
            print(f"      {line.strip()}")
else:
    print(f"   ❌ Aucun fichier timing_log_*.csv trouvé")

# 5. Lancer l'analyseur
print(f"\n5. ANALYSE DES TEMPS:")
print("   " + "="*66)

import sys
sys.argv = ['analyser_temps.py', latest_timing]

try:
    exec(open('analyser_temps.py').read())
except Exception as e:
    print(f"   ⚠️ Erreur lors de l'analyse: {e}")

print("\n" + "="*70)
print("✅ TEST TERMINÉ")
print("="*70)
print("\nFichiers générés:")
print(f"  • {csv_file} (résultats Sobol)")
print(f"  • {latest_timing} (temps de traitement)")
print("\nPour analyser les temps:")
print(f"  python3 analyser_temps.py {latest_timing}")
print("="*70)
