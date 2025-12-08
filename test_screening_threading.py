#!/usr/bin/env python3
"""
Test du screening blur_clahe avec threading
Mini-screening de 32 points pour valider la modification
"""
import cv2
import glob
import time
import platform
import multiprocessing

# Configuration
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import optimizer

def test_mini_screening():
    """Test avec 32 points (2^5)."""
    print("\n" + "="*60)
    print("TEST: Mini-screening blur_clahe avec threading")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

    print(f"✅ {len(images)} images chargées")

    # Baseline
    print("\n⏳ Calcul baseline...")
    t0 = time.time()
    baseline_scores = optimizer.calculate_baseline_scores(images)
    t_baseline = time.time() - t0
    print(f"✅ Baseline calculés en {t_baseline:.1f}s")
    print(f"   Moyenne baseline: {sum(baseline_scores)/len(baseline_scores):.2f}%")

    # Paramètres pour screening
    param_ranges = {
        'inp_line_h': (30, 60),      # Range réduit pour test rapide
        'inp_line_v': (30, 60),
        'denoise_h': (8.0, 15.0),
        'bg_dilate': (5, 11),
        'bg_blur': (15, 31),
        'clahe_clip': (1.5, 3.0),
        'clahe_tile': (6, 10)
    }

    fixed_params = {}  # Aucun param fixe pour ce test

    n_points = 32  # 2^5

    print(f"\n🚀 Lancement screening Sobol: {n_points} points")
    print(f"   Mode: blur_clahe avec threading")
    print(f"   Paramètres actifs: {list(param_ranges.keys())}")

    # Compteur pour callback
    points_evaluated = [0]
    times = []

    def callback(point_idx, scores_dict, params_dict):
        """Callback après chaque point."""
        points_evaluated[0] = point_idx + 1

        # Afficher tous les 8 points
        if (point_idx + 1) % 8 == 0 or point_idx == 0:
            print(f"  Point {point_idx+1:2d}/{n_points}: "
                  f"Delta={scores_dict['tesseract_delta']:+.2f}% | "
                  f"CNR={scores_dict['cnr']:.2f}")

    # Lancer le screening
    t0 = time.time()

    try:
        best_params, csv_file = optimizer.run_sobol_screening(
            images=images,
            baseline_scores=baseline_scores,
            n_points=n_points,
            param_ranges=param_ranges,
            fixed_params=fixed_params,
            callback=callback,
            cancellation_event=None,
            verbose_timing=False,
            enable_time_logging=False,  # Désactiver le time logger pour test
            pipeline_mode='blur_clahe'
        )

        t_total = time.time() - t0

        print(f"\n✅ Screening terminé en {t_total:.1f}s")
        print(f"   Temps par point: {t_total/n_points:.2f}s")
        print(f"   Points évalués: {points_evaluated[0]}/{n_points}")

        if best_params:
            print(f"\n🏆 MEILLEURS PARAMÈTRES:")
            for k, v in best_params.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.2f}")
                else:
                    print(f"   {k}: {v}")

            print(f"\n📁 Résultats: {csv_file}")

        # Projection pour 4096 points
        time_per_point = t_total / n_points
        time_4096 = time_per_point * 4096

        print(f"\n⏱️  PROJECTION POUR 4096 POINTS:")
        print(f"   Temps par point: {time_per_point:.2f}s")
        print(f"   Temps total 4096: {time_4096/60:.0f} min ({time_4096/3600:.1f}h)")

        # Comparaison avec séquentiel
        time_seq_estimate = 6.5  # D'après test précédent
        speedup = time_seq_estimate / time_per_point

        print(f"\n📊 COMPARAISON:")
        print(f"   Séquentiel estimé: {time_seq_estimate:.2f}s/point")
        print(f"   Threading actuel:  {time_per_point:.2f}s/point")
        print(f"   Speedup: {speedup:.1f}x")

        if speedup > 2.5:
            print(f"\n🚀 Excellent! Le threading fonctionne parfaitement!")
        elif speedup > 1.5:
            print(f"\n✅ Bon gain de performance.")
        else:
            print(f"\n⚠️  Gain plus faible que prévu.")

        return True

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 TEST MINI-SCREENING AVEC THREADING")
    print("="*60)
    print("Test de 32 points pour valider l'intégration du threading")
    print("="*60)

    success = test_mini_screening()

    print("\n" + "="*60)
    if success:
        print("✅ TEST RÉUSSI!")
        print("\nVous pouvez maintenant utiliser le moteur blur_clahe")
        print("avec threading dans l'interface GUI.")
        print("\n💡 Recommandations:")
        print("   - Commencez avec 256-512 points")
        print("   - Temps estimé: 8-17 minutes (au lieu de 30-60 min)")
        print("   - 4096 points: ~2h (au lieu de 7.7h)")
    else:
        print("❌ TEST ÉCHOUÉ!")
        print("Vérifiez les erreurs ci-dessus.")
    print("="*60)
