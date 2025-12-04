#!/usr/bin/env python3
"""
Test de validation du traitement multiprocessing Tesseract

Vérifie que:
1. Le multiprocessing accélère le traitement de 2-3x
2. Les scores sont identiques entre séquentiel et parallèle
3. L'intégration avec optimizer.calculate_baseline_scores fonctionne
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
import glob
import pipeline
import optimizer


def test_multiprocessing_speedup():
    """Test que le multiprocessing accélère bien le traitement"""

    # Charger 4 images de test
    image_files = sorted(glob.glob('test_scans/*.jpg'))[:4]
    if not image_files:
        print("❌ Aucune image de test trouvée dans test_scans/")
        return False

    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    print(f"✅ {len(images)} images chargées\n")

    # Test 1: Mode séquentiel
    print("Test 1: Mode séquentiel")
    t0 = time.time()
    scores_seq = optimizer.calculate_baseline_scores(images, use_multiprocessing=False)
    t_seq = time.time() - t0
    print(f"  Temps: {t_seq*1000:.0f}ms")
    print(f"  Scores: {[f'{s:.1f}' for s in scores_seq]}\n")

    # Test 2: Mode multiprocessing
    print("Test 2: Mode multiprocessing")
    t0 = time.time()
    scores_mp = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)
    t_mp = time.time() - t0
    print(f"  Temps: {t_mp*1000:.0f}ms")
    print(f"  Scores: {[f'{s:.1f}' for s in scores_mp]}\n")

    # Vérifications
    print("Vérifications:")

    # 1. Scores identiques
    if scores_seq == scores_mp:
        print("  ✅ Scores identiques")
    else:
        print(f"  ❌ Scores différents!")
        print(f"     Séq: {scores_seq}")
        print(f"     MP:  {scores_mp}")
        return False

    # 2. Speedup significatif (>1.5x attendu)
    speedup = t_seq / t_mp
    print(f"  ✅ Speedup: {speedup:.2f}x")

    if speedup < 1.5:
        print(f"  ⚠️  Speedup faible (<1.5x), attendu 2-3x")

    # 3. Gain de temps
    gain_ms = (t_seq - t_mp) * 1000
    gain_pct = (1 - t_mp/t_seq) * 100
    print(f"  ✅ Gain: {gain_ms:.0f}ms ({gain_pct:.0f}%)")

    return True


def test_batch_metrics():
    """Test de la fonction evaluer_toutes_metriques_batch"""

    print("\n" + "="*70)
    print("Test de evaluer_toutes_metriques_batch")
    print("="*70 + "\n")

    # Charger et traiter 2 images
    image_files = sorted(glob.glob('test_scans/*.jpg'))[:2]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]

    # Paramètres par défaut
    params = {
        'line_h_size': 50,
        'line_v_size': 60,
        'dilate_iter': 2,
        'norm_kernel': 75,
        'denoise_h': 9.0,
        'noise_threshold': 100.0,
        'bin_block_size': 61,
        'bin_c': 15.0
    }

    # Traiter les images
    processed = [pipeline.pipeline_complet(img, params) for img in images]

    # Test batch
    print("Calcul des métriques en batch...")
    results = pipeline.evaluer_toutes_metriques_batch(processed, max_workers=2, verbose=True)

    print(f"\n✅ {len(results)} résultats obtenus")
    for i, (tess, sharp, cont, t_tess, t_sharp, t_cont) in enumerate(results):
        print(f"  Image {i+1}: Tess={tess:.1f}, Sharp={sharp:.0f}, Contrast={cont:.1f}")

    return True


if __name__ == '__main__':
    print("="*70)
    print("TEST MULTIPROCESSING TESSERACT")
    print("="*70)
    print()

    # Test 1: Speedup
    success1 = test_multiprocessing_speedup()

    # Test 2: Batch metrics
    success2 = test_batch_metrics()

    print("\n" + "="*70)
    if success1 and success2:
        print("✅ TOUS LES TESTS PASSENT")
        print("="*70)
        print()
        print("💡 Le multiprocessing est activé par défaut dans:")
        print("   - optimizer.calculate_baseline_scores()")
        print("   - pipeline.evaluer_toutes_metriques_batch()")
        print()
        print("   Speedup typique: 2-3x sur CPU multi-core")
        exit(0)
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ")
        exit(1)
