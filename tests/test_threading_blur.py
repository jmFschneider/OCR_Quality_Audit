#!/usr/bin/env python3
"""
Test du threading pour accélérer le pipeline blur_clahe
"""
import cv2
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import platform
import multiprocessing

# Configuration spawn (au cas où)
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

import pipeline
import optimizer


def test_threading_vs_sequential():
    """Compare threading vs séquentiel."""
    print("\n" + "="*60)
    print("TEST: Threading vs Séquentiel pour blur_clahe")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

    print(f"✅ {len(images)} images chargées")

    # Baseline
    print("\n⏳ Calcul baseline...")
    baseline_scores = optimizer.calculate_baseline_scores(images)
    print(f"✅ Baseline: {sum(baseline_scores)/len(baseline_scores):.1f}%")

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

    # ========================================
    # Test 1 : SÉQUENTIEL
    # ========================================
    print("\n" + "-"*60)
    print("🐌 Test SÉQUENTIEL")
    print("-"*60)

    t0 = time.time()
    results_seq = []

    for i, img in enumerate(images):
        processed = pipeline.pipeline_blur_clahe(img, params)
        tess = pipeline.get_tesseract_score(processed)
        cnr = pipeline.get_cnr_quality(processed)
        sharp = pipeline.get_sharpness(processed)
        results_seq.append((tess, cnr, sharp))

    t_seq = time.time() - t0

    avg_tess_seq = sum(r[0] for r in results_seq) / len(results_seq)
    avg_cnr_seq = sum(r[1] for r in results_seq) / len(results_seq)

    print(f"✅ Temps: {t_seq:.2f}s ({t_seq/len(images):.2f}s/image)")
    print(f"   Tesseract moyen: {avg_tess_seq:.2f}%")
    print(f"   CNR moyen: {avg_cnr_seq:.2f}")

    # ========================================
    # Test 2 : THREADING (4 workers)
    # ========================================
    print("\n" + "-"*60)
    print("🚀 Test THREADING (4 workers)")
    print("-"*60)

    t0 = time.time()

    def process_one_image(img):
        """Worker thread pour une image."""
        processed = pipeline.pipeline_blur_clahe(img, params)
        tess = pipeline.get_tesseract_score(processed)
        cnr = pipeline.get_cnr_quality(processed)
        sharp = pipeline.get_sharpness(processed)
        return (tess, cnr, sharp)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results_thread_4 = list(executor.map(process_one_image, images))

    t_thread_4 = time.time() - t0

    avg_tess_t4 = sum(r[0] for r in results_thread_4) / len(results_thread_4)
    avg_cnr_t4 = sum(r[1] for r in results_thread_4) / len(results_thread_4)

    print(f"✅ Temps: {t_thread_4:.2f}s ({t_thread_4/len(images):.2f}s/image)")
    print(f"   Tesseract moyen: {avg_tess_t4:.2f}%")
    print(f"   CNR moyen: {avg_cnr_t4:.2f}")
    print(f"   Speedup: {t_seq/t_thread_4:.1f}x")

    # ========================================
    # Test 3 : THREADING (8 workers)
    # ========================================
    print("\n" + "-"*60)
    print("🚀 Test THREADING (8 workers)")
    print("-"*60)

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results_thread_8 = list(executor.map(process_one_image, images))

    t_thread_8 = time.time() - t0

    avg_tess_t8 = sum(r[0] for r in results_thread_8) / len(results_thread_8)
    avg_cnr_t8 = sum(r[1] for r in results_thread_8) / len(results_thread_8)

    print(f"✅ Temps: {t_thread_8:.2f}s ({t_thread_8/len(images):.2f}s/image)")
    print(f"   Tesseract moyen: {avg_tess_t8:.2f}%")
    print(f"   CNR moyen: {avg_cnr_t8:.2f}")
    print(f"   Speedup: {t_seq/t_thread_8:.1f}x")

    # ========================================
    # RÉSUMÉ
    # ========================================
    print("\n" + "="*60)
    print("📊 RÉSUMÉ COMPARATIF")
    print("="*60)

    print(f"\n{'Méthode':<20} {'Temps':>10} {'Speedup':>10} {'Tess':>10} {'CNR':>10}")
    print("-"*60)
    print(f"{'Séquentiel':<20} {t_seq:>9.2f}s {1.0:>9.1f}x {avg_tess_seq:>9.2f}% {avg_cnr_seq:>9.2f}")
    print(f"{'Threading (4)':<20} {t_thread_4:>9.2f}s {t_seq/t_thread_4:>9.1f}x {avg_tess_t4:>9.2f}% {avg_cnr_t4:>9.2f}")
    print(f"{'Threading (8)':<20} {t_thread_8:>9.2f}s {t_seq/t_thread_8:>9.1f}x {avg_tess_t8:>9.2f}% {avg_cnr_t8:>9.2f}")

    # Projections
    print("\n" + "="*60)
    print("⏱️  PROJECTIONS POUR 4096 POINTS")
    print("="*60)

    time_per_point_seq = t_seq
    time_per_point_t4 = t_thread_4
    time_per_point_t8 = t_thread_8

    n_points = 4096

    print(f"\n{'Méthode':<20} {'Temps total':>15} {'Heures':>10}")
    print("-"*60)
    print(f"{'Séquentiel':<20} {time_per_point_seq*n_points/60:>14.0f} min {time_per_point_seq*n_points/3600:>9.1f}h")
    print(f"{'Threading (4)':<20} {time_per_point_t4*n_points/60:>14.0f} min {time_per_point_t4*n_points/3600:>9.1f}h")
    print(f"{'Threading (8)':<20} {time_per_point_t8*n_points/60:>14.0f} min {time_per_point_t8*n_points/3600:>9.1f}h")

    # Vérifier cohérence
    print("\n" + "="*60)
    print("🔍 VÉRIFICATION COHÉRENCE")
    print("="*60)

    # Comparer premiers résultats
    diff_t4 = abs(results_seq[0][0] - results_thread_4[0][0])
    diff_t8 = abs(results_seq[0][0] - results_thread_8[0][0])

    print(f"\nDifférence Tesseract (1ère image):")
    print(f"  Séquentiel vs Thread(4): {diff_t4:.4f}%")
    print(f"  Séquentiel vs Thread(8): {diff_t8:.4f}%")

    if diff_t4 < 0.1 and diff_t8 < 0.1:
        print("\n✅ Résultats identiques (différence < 0.1%)")
    else:
        print("\n⚠️  Résultats légèrement différents (peut être normal)")

    # Recommandation
    print("\n" + "="*60)
    print("💡 RECOMMANDATION")
    print("="*60)

    best_speedup = max(t_seq/t_thread_4, t_seq/t_thread_8)
    best_method = "Threading (4)" if t_thread_4 < t_thread_8 else "Threading (8)"

    print(f"\n✨ Meilleure configuration: {best_method}")
    print(f"   Speedup: {best_speedup:.1f}x plus rapide que séquentiel")
    print(f"   Temps pour 4096 points: {min(time_per_point_t4, time_per_point_t8)*n_points/3600:.1f}h")

    if best_speedup > 3:
        print(f"\n🚀 Excellent! Le threading apporte un gain significatif!")
        print(f"   Vous pouvez modifier optimizer.py pour utiliser cette méthode.")
    elif best_speedup > 1.5:
        print(f"\n✅ Bon gain de performance avec le threading.")
    else:
        print(f"\n⚠️  Gain limité. Vérifiez si OpenCV/Tesseract relâchent bien le GIL.")


if __name__ == "__main__":
    print("\n🧪 TEST THREADING POUR BLUR+CLAHE")
    print("="*60)
    print("Ce test compare le traitement séquentiel vs threading")
    print("pour identifier le gain de performance potentiel.")
    print("="*60)

    test_threading_vs_sequential()

    print("\n" + "="*60)
    print("✅ Test terminé!")
    print("\nSi le speedup est > 3x, vous pouvez implémenter threading")
    print("en modifiant optimizer.py comme décrit dans:")
    print("  FUTURE_MULTIPROCESSING_SOLUTIONS.md")
    print("="*60)
