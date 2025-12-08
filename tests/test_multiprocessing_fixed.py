#!/usr/bin/env python3
"""
Test du multiprocessing AVEC spawn (correction)
"""
import cv2
import time
import glob
import multiprocessing
import os
import sys
import platform

# CORRECTION CRITIQUE: Configurer spawn AVANT tous les imports
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[✓] multiprocessing configuré en mode 'spawn'")
    except RuntimeError as e:
        print(f"[!] {e}")

# Imports APRÈS configuration multiprocessing
import pipeline
import optimizer

def test_evaluation_batch():
    """Test de l'évaluation batch avec spawn (devrait fonctionner)."""
    print("\n" + "="*60)
    print("TEST: Évaluation batch avec spawn")
    print("="*60)

    # Vérifier la méthode utilisée
    try:
        method = multiprocessing.get_start_method()
        print(f"📋 Méthode multiprocessing: {method}")
        if method != 'spawn':
            print("⚠️  ATTENTION: La méthode n'est pas 'spawn'!")
    except:
        print("📋 Méthode: non définie")

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")
    images = []
    for f in image_files[:8]:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    print(f"✅ {len(images)} images chargées")

    # Baseline
    print("\n⏳ Calcul baseline...")
    t0 = time.time()
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)
    t_baseline = time.time() - t0
    print(f"✅ Baseline: {t_baseline:.1f}s (moyenne: {sum(baseline_scores)/len(baseline_scores):.1f}%)")

    # Paramètres
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    # Test d'évaluation (c'est ici que ça bloquait avant)
    print(f"\n⏳ Évaluation pipeline blur_clahe sur {len(images)} images...")
    print("   (Si ça bloque ici, le problème persiste)")

    t0 = time.time()

    try:
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=1, pipeline_mode='blur_clahe'
        )
        t_eval = time.time() - t0

        print(f"\n✅ SUCCÈS! Évaluation terminée en {t_eval:.2f}s")
        print(f"   Delta: {avg_delta:+.2f}%")
        print(f"   Tesseract: {avg_abs:.2f}%")
        print(f"   CNR: {avg_cnr:.2f}")

        # Projection
        print(f"\n⏱️  PROJECTION pour 4096 points:")
        total_time_min = (t_eval * 4096) / 60
        parallel_time_min = total_time_min / multiprocessing.cpu_count()
        print(f"   Parallèle ({multiprocessing.cpu_count()} cores): {parallel_time_min:.0f} minutes")

        return True

    except Exception as e:
        print(f"\n❌ ÉCHEC: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_iterations():
    """Test de plusieurs itérations pour vérifier la stabilité."""
    print("\n" + "="*60)
    print("TEST: 5 itérations pour vérifier stabilité")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

    print(f"✅ {len(images)} images")

    # Baseline
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)

    # Params
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    print("\n⏳ Exécution de 5 itérations...")

    for i in range(5):
        t0 = time.time()
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=i+1, pipeline_mode='blur_clahe'
        )
        t_iter = time.time() - t0

        print(f"  Iter {i+1}: {t_iter:.2f}s | Delta: {avg_delta:+.2f}% | CNR: {avg_cnr:.2f}")

    print("\n✅ Toutes les itérations ont réussi!")
    return True


if __name__ == "__main__":
    print("\n🔧 TEST MULTIPROCESSING CORRIGÉ (avec spawn)")
    print("="*60)

    results = []

    # Test 1
    results.append(("Évaluation batch", test_evaluation_batch()))

    # Test 2
    results.append(("5 itérations", test_multiple_iterations()))

    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    if all(r[1] for r in results):
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n💡 Le problème de blocage est RÉSOLU avec spawn.")
        print("   Vous pouvez maintenant utiliser le moteur blur_clahe sans blocage.")
    else:
        print("\n❌ PROBLÈME PERSISTANT!")
