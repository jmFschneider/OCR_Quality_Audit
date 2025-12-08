#!/usr/bin/env python3
"""
Test du multiprocessing pour le pipeline blur_clahe
Identifie les problèmes de deadlock ou de démarrage
"""
import cv2
import time
import glob
import multiprocessing
import os
import sys

# Import des modules locaux
import pipeline
import optimizer

def test_multiprocessing_simple():
    """Test simple de multiprocessing avec une seule image."""
    print("\n" + "="*60)
    print("TEST 1: Multiprocessing simple (1 image)")
    print("="*60)

    # Charger une image
    images = glob.glob("test_scans/*.jpg")
    if not images:
        print("❌ Aucune image trouvée")
        return False

    img = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Impossible de charger l'image")
        return False

    print(f"✅ Image chargée: {img.shape}")

    # Paramètres de test
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    baseline_score = 50.0  # Score factice

    print("\n⏳ Lancement du worker multiprocessing...")
    t0 = time.time()

    try:
        # Test avec un seul worker
        args = (img, params, baseline_score, 'blur_clahe')

        with multiprocessing.Pool(processes=1) as pool:
            print("  Pool créé, envoi de la tâche...")
            result = pool.apply_async(optimizer.process_image_fast, (args,))
            print("  Attente du résultat...")
            output = result.get(timeout=30)  # Timeout de 30 secondes

        t_total = (time.time() - t0) * 1000
        print(f"\n✅ Worker terminé en {t_total:.0f}ms")
        print(f"   Résultat: {output}")
        return True

    except multiprocessing.TimeoutError:
        print("\n❌ TIMEOUT! Le worker ne répond pas après 30 secondes")
        print("   → Le processus est probablement bloqué dans le pipeline")
        return False
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiprocessing_batch():
    """Test avec plusieurs images (comme dans le vrai screening)."""
    print("\n" + "="*60)
    print("TEST 2: Multiprocessing batch (toutes les images)")
    print("="*60)

    # Charger toutes les images
    image_files = glob.glob("test_scans/*.jpg")
    if not image_files:
        print("❌ Aucune image trouvée")
        return False

    images = []
    for f in image_files[:8]:  # Limiter à 8 images
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    print(f"✅ {len(images)} images chargées")

    # Calculer baseline scores
    print("\n⏳ Calcul des scores baseline...")
    t0 = time.time()
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)
    t_baseline = time.time() - t0
    print(f"✅ Baseline calculés en {t_baseline:.1f}s")
    print(f"   Scores: {[f'{s:.1f}' for s in baseline_scores[:3]]}... (premiers 3)")

    # Paramètres de test
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    print(f"\n⏳ Évaluation du pipeline (blur_clahe) sur {len(images)} images...")
    t0 = time.time()

    try:
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=1, pipeline_mode='blur_clahe'
        )
        t_total = time.time() - t0

        print(f"\n✅ Évaluation terminée en {t_total:.2f}s")
        print(f"   Delta Tesseract: {avg_delta:+.2f}%")
        print(f"   Score absolu:     {avg_abs:.2f}%")
        print(f"   Netteté:          {avg_sharp:.0f}")
        print(f"   CNR:              {avg_cnr:.2f}")
        return True

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_first_sobol_point():
    """Simule l'évaluation du premier point Sobol (comme dans le vrai screening)."""
    print("\n" + "="*60)
    print("TEST 3: Premier point Sobol (simulation complète)")
    print("="*60)

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
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)
    print(f"✅ Baseline: {sum(baseline_scores)/len(baseline_scores):.1f}%")

    # Générer le premier point Sobol
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
    sobol_samples = sampler.random(n=1)
    scaled_samples = qmc.scale(sobol_samples, lower_bounds, upper_bounds)

    # Construire les paramètres
    sample = scaled_samples[0]
    params = {}
    for i, param_name in enumerate(param_names):
        val = sample[i]
        if param_name in ['bg_dilate', 'bg_blur']:
            params[param_name] = int(val) * 2 + 1
        elif param_name in ['inp_line_h', 'inp_line_v', 'clahe_tile']:
            params[param_name] = int(val)
        else:
            params[param_name] = val

    print(f"\n📊 Premier point Sobol:")
    for k, v in params.items():
        print(f"   {k}: {v}")

    print(f"\n⏳ Évaluation du point 1...")
    t0 = time.time()

    try:
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=1, pipeline_mode='blur_clahe'
        )
        t_total = time.time() - t0

        print(f"\n✅ Point 1 évalué en {t_total:.2f}s")
        print(f"   Delta: {avg_delta:+.2f}%")
        print(f"   Tesseract: {avg_abs:.2f}%")
        print(f"   CNR: {avg_cnr:.2f}")

        # Estimation pour 4096 points
        total_time_minutes = (t_total * 4096) / 60
        parallel_time_minutes = total_time_minutes / multiprocessing.cpu_count()

        print(f"\n⏱️  Estimation pour 4096 points:")
        print(f"   Séquentiel: {total_time_minutes:.0f} minutes ({total_time_minutes/60:.1f}h)")
        print(f"   Parallèle ({multiprocessing.cpu_count()} cores): {parallel_time_minutes:.0f} minutes ({parallel_time_minutes/60:.1f}h)")

        return True

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🔬 TESTS DE DIAGNOSTIC MULTIPROCESSING BLUR+CLAHE")
    print("="*60)

    # Vérifier le mode de démarrage
    try:
        method = multiprocessing.get_start_method()
        print(f"📋 Méthode multiprocessing: {method}")
    except:
        print("📋 Méthode multiprocessing: non définie (utilisera défaut)")

    print(f"📋 Nombre de CPUs: {multiprocessing.cpu_count()}")
    print(f"📋 Support CUDA: {'OUI' if pipeline.USE_CUDA else 'NON'}")

    # Exécuter les tests
    results = []

    print("\n" + "="*60)
    results.append(("Test 1 (worker simple)", test_multiprocessing_simple()))

    print("\n" + "="*60)
    results.append(("Test 2 (batch)", test_multiprocessing_batch()))

    print("\n" + "="*60)
    results.append(("Test 3 (premier point Sobol)", test_first_sobol_point()))

    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*60)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {name}")

    all_pass = all(r[1] for r in results)

    if all_pass:
        print("\n✅ TOUS LES TESTS RÉUSSIS!")
        print("\n💡 Le pipeline blur_clahe fonctionne correctement.")
        print("   Si vous observez un 'blocage', c'est probablement juste")
        print("   que le premier point prend du temps (normal).")
        print("\n   Suggestions:")
        print("   - Commencer avec moins de points (2^8 = 256)")
        print("   - Réduire denoise_h pour accélérer")
        print("   - Vérifier les logs dans l'interface GUI")
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ!")
        print("   Vérifiez les erreurs ci-dessus pour diagnostiquer.")
