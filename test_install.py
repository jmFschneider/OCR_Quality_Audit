#!/usr/bin/env python3
"""Script de vérification complète de l'installation Ubuntu"""

print("="*70)
print("TEST INSTALLATION - OCR Quality Audit Ubuntu")
print("="*70)

# 1. Python et système
import sys
import platform
print(f"\n1. Système")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Python: {sys.version.split()[0]}")

# 2. NumPy
try:
    import numpy as np
    print(f"\n2. NumPy")
    print(f"   ✓ Version: {np.__version__}")
    print(f"   ✓ Location: {np.__file__}")

    # Vérifier qu'il n'y a pas de .libs/ (conflit OpenBLAS)
    import os
    numpy_dir = os.path.dirname(np.__file__)
    libs_dir = os.path.join(numpy_dir, '.libs')
    if os.path.exists(libs_dir):
        print(f"   ⚠ WARNING: {libs_dir} exists (possible OpenBLAS conflict)")
    else:
        print(f"   ✓ No .libs/ directory (good)")
except ImportError as e:
    print(f"\n2. NumPy")
    print(f"   ✗ ERREUR: {e}")

# 3. OpenCV
try:
    import cv2
    print(f"\n3. OpenCV")
    print(f"   ✓ Version: {cv2.__version__}")
    print(f"   ✓ Location: {cv2.__file__}")

    # CUDA
    try:
        cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   ✓ CUDA devices: {cuda_count}")
        if cuda_count > 0:
            print(f"   ✓ CUDA ACTIVÉ !")
        else:
            print(f"   ⚠ CUDA non disponible")
    except:
        print(f"   ⚠ CUDA non disponible (OpenCV compilé sans CUDA)")

    # OpenCL
    print(f"   ✓ OpenCL available: {cv2.ocl.haveOpenCL()}")

except ImportError as e:
    print(f"\n3. OpenCV")
    print(f"   ✗ ERREUR: {e}")

# 4. Scipy
try:
    import scipy
    print(f"\n4. Scipy")
    print(f"   ✓ Version: {scipy.__version__}")
    print(f"   ✓ Location: {scipy.__file__}")

    # Vérifier qu'il n'y a pas de .libs/
    scipy_dir = os.path.dirname(scipy.__file__)
    libs_dir = os.path.join(scipy_dir, '.libs')
    if os.path.exists(libs_dir):
        print(f"   ⚠ WARNING: {libs_dir} exists (possible OpenBLAS conflict)")
    else:
        print(f"   ✓ No .libs/ directory (good)")
except ImportError as e:
    print(f"\n4. Scipy")
    print(f"   ✗ ERREUR: {e}")

# 5. Pandas et Matplotlib
try:
    import pandas as pd
    import matplotlib
    print(f"\n5. Autres packages scientifiques")
    print(f"   ✓ Pandas: {pd.__version__}")
    print(f"   ✓ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"\n5. Autres packages scientifiques")
    print(f"   ✗ ERREUR: {e}")

# 6. Tesseract
try:
    import pytesseract
    print(f"\n6. Tesseract OCR")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"   ✓ Version: {version}")
    except:
        print(f"   ⚠ Tesseract trouvé mais version inaccessible")

    # Langues
    import subprocess
    try:
        result = subprocess.run(['tesseract', '--list-langs'],
                              capture_output=True, text=True, timeout=5)
        langs = [l.strip() for l in result.stdout.split('\n')[1:] if l.strip()]
        print(f"   ✓ Langues: {', '.join(langs)}")
    except:
        print(f"   ⚠ Impossible de lister les langues")

except Exception as e:
    print(f"\n6. Tesseract OCR")
    print(f"   ✗ ERREUR: {e}")

# 7. Optuna
try:
    import optuna
    print(f"\n7. Optuna")
    print(f"   ✓ Version: {optuna.__version__}")
except ImportError as e:
    print(f"\n7. Optuna")
    print(f"   ✗ ERREUR: {e}")

# 8. Pillow
try:
    import PIL
    print(f"\n8. Pillow (PIL)")
    print(f"   ✓ Version: {PIL.__version__}")
except ImportError as e:
    print(f"\n8. Pillow (PIL)")
    print(f"   ✗ ERREUR: {e}")

# 9. Tkinter
try:
    import tkinter
    print(f"\n9. Tkinter (GUI)")
    print(f"   ✓ Tkinter disponible")
except ImportError as e:
    print(f"\n9. Tkinter (GUI)")
    print(f"   ✗ ERREUR: {e}")
    print(f"   → Installer: sudo apt install python3-tk")

# 10. Multiprocessing
print(f"\n10. Multiprocessing")
import multiprocessing
print(f"   ✓ CPU cores: {multiprocessing.cpu_count()}")
print(f"   ✓ Start method: {multiprocessing.get_start_method()}")

# 11. Test import complet (comme dans l'application)
print(f"\n11. Test import complet")
try:
    import cv2, numpy, scipy, pandas, matplotlib, pytesseract, optuna
    print(f"   ✓ Tous les imports réussis sans erreur")
except Exception as e:
    print(f"   ✗ ERREUR lors de l'import combiné: {e}")

print("\n" + "="*70)
print("✓ VÉRIFICATION TERMINÉE")
print("="*70)

# Résumé
print("\nRÉSUMÉ:")
print("  Si tous les tests affichent ✓, l'installation est complète.")
print("  Si vous voyez des ✗ ou ⚠, consultez le guide d'installation.")
print("\nPour lancer l'application:")
print("  cd ~/PycharmProjects/OCR_Quality_Audit")
print("  python3 gui_optimizer_v3_ultim.py")
