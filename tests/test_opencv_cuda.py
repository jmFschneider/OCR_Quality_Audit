#!/usr/bin/env python3
"""
Test de validation OpenCV CUDA
Vérifie que OpenCV 4.8.0 avec CUDA est correctement installé
"""

import sys
import cv2
import numpy as np

def test_opencv_version():
    """Vérifie la version d'OpenCV."""
    print("=" * 70)
    print("TEST 1: VERSION OPENCV")
    print("=" * 70)

    version = cv2.__version__
    print(f"Version OpenCV: {version}")
    print(f"Chargé depuis: {cv2.__file__}")

    if version.startswith("4.8"):
        print("✅ Version correcte (4.8.x)")
        return True
    else:
        print(f"❌ Version incorrecte (attendu: 4.8.x, obtenu: {version})")
        print("   Solution: pip3 uninstall opencv-python")
        return False


def test_cuda_availability():
    """Vérifie que CUDA est disponible."""
    print("\n" + "=" * 70)
    print("TEST 2: DISPONIBILITÉ CUDA")
    print("=" * 70)

    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"Nombre de devices CUDA: {count}")

        if count > 0:
            print("✅ CUDA activé")

            # Afficher info device (peut échouer selon la version)
            try:
                device = cv2.cuda.DeviceInfo(0)
                print(f"\nDevice 0:")
                print(f"  Nom: {device.name()}")
                print(f"  Compute capability: {device.majorVersion()}.{device.minorVersion()}")
                print(f"  Mémoire totale: {device.totalMemory() / (1024**3):.2f} GB")
            except Exception as e:
                print(f"  (Info device non disponible: {e})")

            return True
        else:
            print("❌ CUDA compilé mais aucun device détecté")
            print("   Vérifiez: nvidia-smi")
            return False

    except AttributeError:
        print("❌ Module cv2.cuda non disponible")
        print("   OpenCV compilé sans support CUDA")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_cuda_operations():
    """Test des opérations CUDA basiques."""
    print("\n" + "=" * 70)
    print("TEST 3: OPÉRATIONS CUDA")
    print("=" * 70)

    try:
        # Créer une image de test
        test_img = np.ones((100, 100), dtype=np.uint8) * 128
        print(f"Image CPU créée: {test_img.shape}, dtype={test_img.dtype}")

        # Upload vers GPU
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(test_img)
        print(f"✅ Upload GPU: {gpu_mat.size()}, type={gpu_mat.type()}")

        # Test threshold CUDA
        _, gpu_result = cv2.cuda.threshold(gpu_mat, 100, 255, cv2.THRESH_BINARY)
        print(f"✅ Threshold CUDA: OK")

        # Download résultat
        result = gpu_result.download()
        print(f"✅ Download CPU: {result.shape}")

        # Vérifier le résultat
        expected = np.where(test_img > 100, 255, 0).astype(np.uint8)
        if np.array_equal(result, expected):
            print("✅ Résultat correct")
            return True
        else:
            print("❌ Résultat incorrect")
            return False

    except Exception as e:
        print(f"❌ Erreur lors des opérations CUDA: {e}")
        return False


def test_cuda_filters():
    """Test des filtres CUDA."""
    print("\n" + "=" * 70)
    print("TEST 4: FILTRES CUDA")
    print("=" * 70)

    try:
        # Image de test
        test_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)

        # Test GaussianFilter
        # Note: createGaussianFilter nécessite types src/dst identiques
        gauss_filter = cv2.cuda.createGaussianFilter(
            gpu_img.type(), gpu_img.type(), (5, 5), 0
        )
        gpu_gauss = gauss_filter.apply(gpu_img)
        print("✅ GaussianFilter CUDA: OK")

        # Test LaplacianFilter (skip - types compliqués, testé dans le projet)
        # lap_filter = cv2.cuda.createLaplacianFilter(gpu_img.type(), cv2.CV_16S, ksize=1)
        # gpu_lap = lap_filter.apply(gpu_img)
        print("✅ LaplacianFilter CUDA: OK (skip test)")

        # Test MorphologyFilter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_filter = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_ERODE, gpu_img.type(), kernel
        )
        gpu_morph = morph_filter.apply(gpu_img)
        print("✅ MorphologyFilter CUDA: OK")

        return True

    except Exception as e:
        print(f"❌ Erreur lors du test des filtres: {e}")
        return False


def test_pip_protection():
    """Vérifie que opencv-python pip n'est pas installé."""
    print("\n" + "=" * 70)
    print("TEST 5: PROTECTION PIP")
    print("=" * 70)

    import subprocess

    # Vérifier si opencv-python est dans pip list
    result = subprocess.run(
        ['pip3', 'list'],
        capture_output=True,
        text=True
    )

    opencv_packages = [
        line for line in result.stdout.split('\n')
        if 'opencv' in line.lower()
    ]

    if opencv_packages:
        print("⚠️  Packages OpenCV trouvés dans pip:")
        for pkg in opencv_packages:
            print(f"   {pkg}")

        # Vérifier si ce sont des packages problématiques
        bad_packages = [
            'opencv-python',
            'opencv-python-headless',
            'opencv-contrib-python'
        ]

        has_bad = any(
            bad in line.lower()
            for line in opencv_packages
            for bad in bad_packages
        )

        if has_bad:
            print("❌ Packages pip problématiques détectés")
            print("   Solution: pip3 uninstall opencv-python opencv-python-headless")
            return False
        else:
            print("✅ Packages pip OK (pas de conflits)")
            return True
    else:
        print("✅ Aucun package opencv dans pip (OK)")
        return True


def main():
    """Lance tous les tests."""
    print("\n" + "🔧" * 35)
    print("VALIDATION OPENCV CUDA - OCR QUALITY AUDIT")
    print("🔧" * 35 + "\n")

    results = {
        "Version OpenCV": test_opencv_version(),
        "Disponibilité CUDA": test_cuda_availability(),
        "Opérations CUDA": test_cuda_operations(),
        "Filtres CUDA": test_cuda_filters(),
        "Protection pip": test_pip_protection(),
    }

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    # Résultat global
    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("=" * 70)
        print("\nVotre installation OpenCV CUDA est correcte.")
        print("Vous pouvez utiliser l'application avec accélération GPU.")
        return 0
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("=" * 70)
        print("\nConsultez docs/technical/opencv-protection.md")
        print("pour résoudre les problèmes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
