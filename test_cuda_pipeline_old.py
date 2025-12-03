import cv2
import numpy as np
import os

IMAGE_PATH = "test_scans/022-R.jpg"   # <-- Mets une image sûre ici

print("=== TEST CUDA PIPELINE MINIMAL ===")
print(cv2.getBuildInformation())


# --- Charger image CPU ---
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Image non trouvée : {IMAGE_PATH}")

print(f"Image chargée : shape={img.shape}, dtype={img.dtype}")

# S'assurer du bon type
if img.dtype != np.uint8:
    print("Normalisation CPU car dtype != uint8")
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = img.astype(np.uint8)

# --- Conversion GPU ---
gpu = cv2.cuda_GpuMat()
gpu.upload(img)

print("\n---- GPU UPLOAD ----")
print(f"gpu.type() = {gpu.type()} (attendu 0 pour CV_8U)")
print(f"gpu.depth() = {gpu.depth()}")
print(f"gpu.size() = {gpu.size()}")

# --- Test 1 : Threshold ---
print("\n---- TEST THRESHOLD ----")
try:
    _, gpu_thresh = cv2.cuda.threshold(gpu, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_cpu = gpu_thresh.download()
    print("✔ Threshold CUDA OK, dtype =", thresh_cpu.dtype)
except Exception as e:
    print("❌ ERREUR CUDA threshold :", e)
    exit()

# --- Test 2 : Morphology ---
print("\n---- TEST MORPHOLOGY ----")
try:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    morph = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, gpu_thresh.type(), kernel, iterations=1)
    gpu_morph = morph.apply(gpu_thresh)
    morph_cpu = gpu_morph.download()
    print("✔ Morphology CUDA OK, dtype =", morph_cpu.dtype)
except Exception as e:
    print("❌ ERREUR CUDA morphology :", e)
    exit()

# --- Test 3 : Normalisation division float32 ---
print("\n---- TEST NORMALISATION DIVISION ----")

try:
    # Conversion float32
    gpu_float = cv2.cuda_GpuMat()
    gpu_thresh.convertTo(gpu_float, cv2.CV_32F)

    # Gaussian blur
    gauss = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (41,41), 0)
    gpu_blur = gauss.apply(gpu_float)

    # Division
    gpu_div = cv2.cuda_GpuMat()
    cv2.cuda.divide(gpu_float, gpu_blur, gpu_div, scale=255.0)

    # Retour CPU fiable
    div_cpu = gpu_div.download().astype(np.uint8)

    print("✔ Normalisation CUDA OK, dtype =", div_cpu.dtype)

except Exception as e:
    print("❌ ERREUR CUDA normalisation :", e)
    exit()

# --- Test 4 : Laplacian (bruit) ---
print("\n---- TEST LAPLACIAN ----")
try:
    lapf = cv2.cuda.createLaplacianFilter(gpu.type(), cv2.CV_64F, ksize=1)
    gpu_lap = lapf.apply(gpu)
    lap_cpu = gpu_lap.download()
    print("✔ Laplacian CUDA OK, dtype =", lap_cpu.dtype)
except Exception as e:
    print("❌ ERREUR CUDA Laplacian :", e)
    exit()

print("\n=== ✔ TOUS LES TESTS CUDA SONT PASSÉS ===")
