#!/usr/bin/env python3
"""
Test de fuite mémoire lors du screening blur_clahe
"""
import cv2
import glob
import psutil
import os
import platform
import multiprocessing

# Configurer spawn AVANT imports
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[DEBUG] multiprocessing.set_start_method('spawn') configuré")
    except RuntimeError as e:
        print(f"[DEBUG] {e}")

import pipeline
import optimizer

def get_memory_usage():
    """Retourne l'utilisation mémoire en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_leak():
    """Test de fuite mémoire sur plusieurs itérations."""
    print("\n" + "="*60)
    print("TEST DE FUITE MÉMOIRE - BLUR+CLAHE")
    print("="*60)

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = []
    for f in image_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    print(f"\n✅ {len(images)} images chargées")
    mem_start = get_memory_usage()
    print(f"📊 Mémoire initiale: {mem_start:.1f} MB")

    # Baseline
    print("\n⏳ Calcul baseline...")
    baseline_scores = optimizer.calculate_baseline_scores(images, use_multiprocessing=True)
    mem_after_baseline = get_memory_usage()
    print(f"✅ Baseline calculés")
    print(f"📊 Mémoire après baseline: {mem_after_baseline:.1f} MB (+{mem_after_baseline-mem_start:.1f} MB)")

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

    # Test sur 10 itérations pour détecter les fuites
    print(f"\n⏳ Test de 10 itérations du pipeline...")
    print("-"*60)

    for i in range(10):
        avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
            images, baseline_scores, params, point_id=i+1, pipeline_mode='blur_clahe'
        )

        mem_current = get_memory_usage()
        delta_mem = mem_current - mem_after_baseline

        print(f"Iter {i+1:2d}: Mem={mem_current:6.1f} MB (Δ={delta_mem:+6.1f} MB) | "
              f"Delta={avg_delta:+5.2f}% | CNR={avg_cnr:5.2f}")

        # Alarme si augmentation excessive
        if delta_mem > 1000:  # Plus de 1 GB d'augmentation
            print(f"\n⚠️  ALERTE: Fuite mémoire détectée! (+{delta_mem:.1f} MB)")
            break

    mem_final = get_memory_usage()
    total_increase = mem_final - mem_after_baseline

    print("-"*60)
    print(f"\n📊 RÉSUMÉ:")
    print(f"  Mémoire initiale:        {mem_start:.1f} MB")
    print(f"  Mémoire après baseline:  {mem_after_baseline:.1f} MB")
    print(f"  Mémoire finale:          {mem_final:.1f} MB")
    print(f"  Augmentation totale:     {total_increase:+.1f} MB")

    if total_increase > 500:
        print(f"\n⚠️  FUITE MÉMOIRE PROBABLE!")
        print(f"  L'augmentation de {total_increase:.1f} MB sur 10 itérations est anormale.")
        print(f"  Projection pour 4096 points: ~{total_increase*409.6:.0f} MB (~{total_increase*409.6/1024:.1f} GB)")
    elif total_increase > 100:
        print(f"\n⚠️  Légère augmentation mémoire détectée.")
        print(f"  Projection pour 4096 points: ~{total_increase*409.6:.0f} MB (~{total_increase*409.6/1024:.1f} GB)")
    else:
        print(f"\n✅ Pas de fuite mémoire détectée (variation normale)")

    print("="*60)


if __name__ == "__main__":
    # Vérifier psutil
    try:
        import psutil
    except ImportError:
        print("❌ Le module 'psutil' n'est pas installé.")
        print("   Installez-le avec: pip install psutil")
        exit(1)

    test_memory_leak()
