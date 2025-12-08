#!/usr/bin/env python3
"""
Test du pipeline blur_clahe avec traitement séquentiel (correction)
"""
import cv2
import glob
import time
import platform
import multiprocessing

# Configuration spawn (même si pas utilisé pour blur_clahe)
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import pipeline
import optimizer

def test_sequential():
    """Test du mode blur_clahe avec traitement séquentiel."""
    print("\n" + "="*60)
    print("TEST: Pipeline blur_clahe SÉQUENTIEL (correction)")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

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

    # Test blur_clahe (devrait maintenant fonctionner en séquentiel)
    print(f"\n⏳ Évaluation blur_clahe (mode séquentiel)...")
    t0 = time.time()

    try:
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=1, pipeline_mode='blur_clahe'
        )
        t_eval = time.time() - t0

        print(f"\n✅ SUCCÈS! Terminé en {t_eval:.2f}s ({t_eval/len(images):.2f}s/image)")
        print(f"\n📊 RÉSULTATS:")
        print(f"   Delta Tesseract: {avg_delta:+.2f}%")
        print(f"   Score Tesseract: {avg_abs:.2f}%")
        print(f"   Netteté:         {avg_sharp:.0f}")
        print(f"   CNR (Gemini):    {avg_cnr:.2f}")

        # Projections
        print(f"\n⏱️  PROJECTIONS:")
        time_per_point = t_eval  # temps pour 8 images

        for n_points in [256, 512, 1024, 2048, 4096]:
            total_time = time_per_point * n_points
            hours = total_time / 3600
            print(f"   {n_points:4d} points: {hours:6.1f}h ({total_time/60:6.0f} min)")

        print(f"\n💡 RECOMMANDATION:")
        print(f"   Commencez avec 256-512 points pour le screening blur_clahe")
        print(f"   (Temps estimé: {time_per_point*256/3600:.1f}h - {time_per_point*512/3600:.1f}h)")

        return True

    except Exception as e:
        print(f"\n❌ ÉCHEC: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_points():
    """Test de 3 points différents pour vérifier la stabilité."""
    print("\n" + "="*60)
    print("TEST: 3 points Sobol (vérification stabilité)")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

    print(f"✅ {len(images)} images")

    # Baseline
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)

    # Générer 3 points Sobol
    from scipy.stats import qmc

    param_ranges = {
        'inp_line_h': (20, 100),
        'inp_line_v': (20, 100),
        'denoise_h': (5.0, 20.0),
        'bg_dilate': (3, 15),
        'bg_blur': (11, 51),
        'clahe_clip': (1.0, 5.0),
        'clahe_tile': (4, 16)
    }

    param_names = list(param_ranges.keys())
    lower_bounds = [param_ranges[p][0] for p in param_names]
    upper_bounds = [param_ranges[p][1] for p in param_names]

    sampler = qmc.Sobol(d=len(param_names), scramble=True)
    sobol_samples = sampler.random(n=3)
    scaled_samples = qmc.scale(sobol_samples, lower_bounds, upper_bounds)

    print(f"\n⏳ Évaluation de 3 points...")

    for idx, sample in enumerate(scaled_samples):
        # Construire params
        params = {}
        for i, param_name in enumerate(param_names):
            val = sample[i]
            if param_name in ['bg_dilate', 'bg_blur']:
                params[param_name] = int(val) * 2 + 1
            elif param_name in ['inp_line_h', 'inp_line_v', 'clahe_tile']:
                params[param_name] = int(val)
            else:
                params[param_name] = val

        t0 = time.time()
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=idx+1, pipeline_mode='blur_clahe'
        )
        t_eval = time.time() - t0

        print(f"  Point {idx+1}: {t_eval:.2f}s | Delta: {avg_delta:+.2f}% | CNR: {avg_cnr:.2f}")

    print("\n✅ Tous les points ont été évalués avec succès!")
    return True


if __name__ == "__main__":
    print("\n🔧 TEST PIPELINE BLUR+CLAHE - MODE SÉQUENTIEL")
    print("="*60)

    results = []

    results.append(("Test séquentiel", test_sequential()))
    results.append(("3 points Sobol", test_multiple_points()))

    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    if all(r[1] for r in results):
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n💡 Le pipeline blur_clahe fonctionne maintenant en mode séquentiel.")
        print("   Vous pouvez l'utiliser pour l'optimisation.")
        print("\n⚠️  IMPORTANT:")
        print("   - Le traitement est séquentiel (pas de parallélisme)")
        print("   - Comptez ~2-3 secondes par point (8 images)")
        print("   - Utilisez moins de points (256-512) pour screening")
    else:
        print("\n❌ PROBLÈME PERSISTANT!")
