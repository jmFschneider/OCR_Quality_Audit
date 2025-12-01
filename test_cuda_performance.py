"""
Script de test de performance CUDA
===================================

Ce script teste les performances de l'accélération CUDA en comparant
le traitement d'images avec et sans GPU.

Usage:
    python test_cuda_performance.py
"""
import os
import sys
import cv2
import numpy as np
import time
from glob import glob

# Import des fonctions du programme principal
sys.path.insert(0, os.path.dirname(__file__))

# Configuration
INPUT_FOLDER = 'test_scans'
N_ITERATIONS = 5  # Nombre d'itérations pour moyenner les résultats

def test_cuda_available():
    """Vérifie si CUDA est disponible"""
    print("\n" + "="*70)
    print("TEST 1: Disponibilité CUDA")
    print("="*70)

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            print(f"✅ CUDA disponible: {count} GPU détecté(s)")
            cv2.cuda.setDevice(0)

            # Informations GPU
            device_info = cv2.cuda.getDevice()
            print(f"   GPU actif: #{device_info}")
            return True
        else:
            print("❌ Aucun GPU CUDA détecté")
            return False
    except AttributeError:
        print("❌ OpenCV compilé SANS support CUDA")
        return False

def test_image_loading():
    """Teste le chargement d'images"""
    print("\n" + "="*70)
    print("TEST 2: Chargement d'images")
    print("="*70)

    image_files = glob(os.path.join(INPUT_FOLDER, '*.*'))
    print(f"Images trouvées: {len(image_files)}")

    if not image_files:
        print("❌ Aucune image dans le dossier test_scans")
        return None

    # Charger la première image
    img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"✅ Image test chargée: {img.shape[1]}x{img.shape[0]} pixels")
        return img
    else:
        print("❌ Erreur lors du chargement de l'image")
        return None

def benchmark_cpu_vs_gpu(image):
    """Compare les performances CPU vs GPU"""
    print("\n" + "="*70)
    print("TEST 3: Benchmark CPU vs GPU")
    print("="*70)

    # Test paramètres
    kernel_size = 75

    # --- Test 1: GaussianBlur ---
    print("\n1. GaussianBlur (75x75):")

    # CPU
    times_cpu = []
    for _ in range(N_ITERATIONS):
        t0 = time.time()
        blurred_cpu = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        times_cpu.append((time.time() - t0) * 1000)
    avg_cpu = sum(times_cpu) / len(times_cpu)
    print(f"   CPU: {avg_cpu:.2f} ms (moyenne sur {N_ITERATIONS} runs)")

    # GPU
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gpu_float = gpu_img.convertTo(cv2.CV_32F)

        times_gpu = []
        for _ in range(N_ITERATIONS):
            t0 = time.time()
            gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F,
                                                            (kernel_size, kernel_size), 0)
            gpu_blur = gaussian_filter.apply(gpu_float)
            cv2.cuda.synchronize()  # Attendre la fin du traitement GPU
            times_gpu.append((time.time() - t0) * 1000)
        avg_gpu = sum(times_gpu) / len(times_gpu)
        print(f"   GPU: {avg_gpu:.2f} ms (moyenne sur {N_ITERATIONS} runs)")
        print(f"   🚀 Speedup: x{avg_cpu/avg_gpu:.2f}")
    except Exception as e:
        print(f"   ❌ Erreur GPU: {e}")

    # --- Test 2: Morphologie ---
    print("\n2. Morphologie (dilate 45x1):")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))

    # CPU
    times_cpu = []
    for _ in range(N_ITERATIONS):
        t0 = time.time()
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        morph_cpu = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        times_cpu.append((time.time() - t0) * 1000)
    avg_cpu = sum(times_cpu) / len(times_cpu)
    print(f"   CPU: {avg_cpu:.2f} ms")

    # GPU
    try:
        times_gpu = []
        for _ in range(N_ITERATIONS):
            t0 = time.time()
            _, gpu_thresh = cv2.cuda.threshold(gpu_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            morph_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, gpu_thresh.type(),
                                                          kernel, iterations=2)
            gpu_morph = morph_filter.apply(gpu_thresh)
            cv2.cuda.synchronize()
            times_gpu.append((time.time() - t0) * 1000)
        avg_gpu = sum(times_gpu) / len(times_gpu)
        print(f"   GPU: {avg_gpu:.2f} ms")
        print(f"   🚀 Speedup: x{avg_cpu/avg_gpu:.2f}")
    except Exception as e:
        print(f"   ❌ Erreur GPU: {e}")

    # --- Test 3: Laplacian + meanStdDev ---
    print("\n3. Laplacian + variance:")

    # CPU
    times_cpu = []
    for _ in range(N_ITERATIONS):
        t0 = time.time()
        lap = cv2.Laplacian(image, cv2.CV_64F)
        variance = lap.var()
        times_cpu.append((time.time() - t0) * 1000)
    avg_cpu = sum(times_cpu) / len(times_cpu)
    print(f"   CPU: {avg_cpu:.2f} ms")

    # GPU
    try:
        times_gpu = []
        for _ in range(N_ITERATIONS):
            t0 = time.time()
            lap_filter = cv2.cuda.createLaplacianFilter(gpu_img.type(), cv2.CV_64F, ksize=1)
            gpu_lap = lap_filter.apply(gpu_img)
            mean, std_dev = cv2.cuda.meanStdDev(gpu_lap)
            variance = std_dev[0][0] ** 2
            times_gpu.append((time.time() - t0) * 1000)
        avg_gpu = sum(times_gpu) / len(times_gpu)
        print(f"   GPU: {avg_gpu:.2f} ms")
        print(f"   🚀 Speedup: x{avg_cpu/avg_gpu:.2f}")
    except Exception as e:
        print(f"   ❌ Erreur GPU: {e}")

def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("SCRIPT DE TEST DE PERFORMANCE CUDA")
    print("="*70)

    # Test 1: CUDA disponible ?
    cuda_available = test_cuda_available()

    # Test 2: Charger une image
    image = test_image_loading()

    if image is None:
        print("\n❌ Impossible de continuer sans image de test")
        return

    # Test 3: Benchmark
    if cuda_available:
        benchmark_cpu_vs_gpu(image)
    else:
        print("\n⚠️  CUDA non disponible, benchmarks ignorés")

    print("\n" + "="*70)
    print("TESTS TERMINÉS")
    print("="*70)

if __name__ == "__main__":
    main()
