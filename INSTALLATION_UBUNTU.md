# Installation et Configuration - Ubuntu 22.04

Guide complet pour installer et exécuter **OCR Quality Audit** sur Ubuntu 22.04 avec support GPU (CUDA/OpenCL).

---

## Table des matières

1. [Prérequis matériels](#prérequis-matériels)
2. [Installation du système](#installation-du-système)
3. [Installation OpenCV avec CUDA](#installation-opencv-avec-cuda)
4. [Installation des dépendances Python](#installation-des-dépendances-python)
5. [Configuration de l'application](#configuration-de-lapplication)
6. [Vérification de l'installation](#vérification-de-linstallation)
7. [Lancement de l'application](#lancement-de-lapplication)
8. [Dépannage](#dépannage)

---

## Prérequis matériels

### Configuration recommandée
- **OS** : Ubuntu 22.04 LTS (ou 20.04 LTS)
- **GPU** : NVIDIA RTX série 10xx/20xx/30xx/40xx (testé sur RTX 1080)
- **RAM** : 16 GB minimum (32 GB recommandé)
- **CPU** : Processeur multi-cœurs (8+ cœurs recommandé)
- **Stockage** : 20 GB d'espace libre (pour OpenCV, CUDA, et dépendances)

### Vérification GPU NVIDIA
```bash
lspci | grep -i nvidia
```
Si aucune carte NVIDIA n'est détectée, ce guide ne s'applique pas (vous pouvez utiliser la version CPU uniquement).

---

## Installation du système

### 1. Mise à jour du système
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Installation des outils de base
```bash
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev
```

### 3. Installation des drivers NVIDIA
```bash
# Détection automatique et installation
sudo ubuntu-drivers autoinstall

# Redémarrer le système
sudo reboot
```

Après redémarrage, vérifier :
```bash
nvidia-smi
```
Vous devriez voir votre carte graphique et la version du driver.

---

## Installation OpenCV avec CUDA

### Option A : Script automatique (RECOMMANDÉ)

Le projet inclut un script de compilation automatique d'OpenCV avec support CUDA.

```bash
cd ~/OCR_Quality_Audit/Phase3_Migration_Ubuntu_CUDA
chmod +x build_opencv_cuda.sh
./build_opencv_cuda.sh
```

**Durée estimée** : 45-60 minutes (selon le CPU)

Le script va :
1. Vérifier la compatibilité Ubuntu (20.04/22.04)
2. Installer CUDA Toolkit 11.8 et cuDNN
3. Télécharger OpenCV 4.8.0 + opencv_contrib
4. Compiler avec optimisations CUDA pour votre GPU
5. Installer les bindings Python

### Option B : Installation manuelle

Suivez le guide détaillé dans `Phase3_Migration_Ubuntu_CUDA/PHASE3_OPENCV_CUDA_UBUNTU.md`

---

## Installation des dépendances Python

### 1. Création d'un environnement virtuel (recommandé)
```bash
cd ~/OCR_Quality_Audit
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installation des packages système pour Python
```bash
sudo apt install -y \
    python3-tk \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    libglib2.0-0
```

**Explication des packages :**
- `python3-tk` : Interface graphique Tkinter (pour la GUI)
- `tesseract-ocr` : Moteur OCR Tesseract
- `tesseract-ocr-fra` : Données linguistiques françaises pour Tesseract
- `libgl1-mesa-glx` : Support OpenGL pour OpenCV
- `libglib2.0-0` : Bibliothèque GLib (dépendance OpenCV)

### 3. Vérification de Tesseract
```bash
which tesseract
tesseract --version
tesseract --list-langs
```

Vous devriez voir `fra` (français) dans la liste des langues.

### 4. Installation des dépendances Python
```bash
# Activer l'environnement virtuel si ce n'est pas déjà fait
source .venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

**Note** : Si `requirements.txt` n'existe pas encore, voici les packages principaux :
```bash
pip install numpy scipy optuna pytesseract pillow matplotlib pandas
```

### 5. Vérification d'OpenCV avec CUDA
```bash
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import cv2; print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

**Résultat attendu :**
```
OpenCV version: 4.8.0
CUDA devices: 1
```

Si `CUDA devices: 0`, OpenCV n'a pas été compilé avec CUDA ou ne détecte pas votre GPU.

---

## Configuration de l'application

### 1. Vérification des chemins Tesseract

Le code détecte automatiquement Tesseract sur Linux. Vérifiez dans `gui_optimizer_v3_ultim.py` :

```python
# Configuration Tesseract multi-plateforme (lignes 57-67)
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
elif platform.system() == 'Linux':
    # Tesseract est généralement dans le PATH après installation via apt
    # Aucune configuration nécessaire si tesseract est dans /usr/bin/
    pass
```

### 2. Vérification du dossier d'images

Par défaut, l'application cherche les images dans `test_scans/` :

```bash
# Créer le dossier s'il n'existe pas
mkdir -p test_scans

# Copier vos images de test
cp /chemin/vers/vos/images/*.jpg test_scans/
```

### 3. Permissions OpenCL (optionnel, pour GPU)

Pour activer OpenCL sur GPU NVIDIA :

```bash
# Vérifier la présence d'OpenCL
clinfo

# Si clinfo n'est pas installé
sudo apt install -y clinfo ocl-icd-opencl-dev

# Vérifier à nouveau
clinfo | grep "Device Name"
```

Vous devriez voir votre GPU NVIDIA.

---

## Vérification de l'installation

### Script de test complet

Créez un fichier `test_installation.py` :

```python
#!/usr/bin/env python3
"""Script de vérification de l'installation Ubuntu"""

import sys
import platform

print("="*70)
print("VÉRIFICATION DE L'INSTALLATION - OCR Quality Audit")
print("="*70)

# 1. Système
print(f"\n1. Système d'exploitation")
print(f"   OS: {platform.system()} {platform.release()}")
print(f"   Version: {platform.version()}")

# 2. Python
print(f"\n2. Python")
print(f"   Version: {sys.version}")

# 3. OpenCV
try:
    import cv2
    print(f"\n3. OpenCV")
    print(f"   ✓ Version: {cv2.__version__}")
    print(f"   ✓ OpenCL disponible: {cv2.ocl.haveOpenCL()}")

    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
        print(f"   ✓ OpenCL activé: {cv2.ocl.useOpenCL()}")

    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   ✓ CUDA devices: {cuda_devices}")
        if cuda_devices > 0:
            print(f"   ✓ CUDA activé avec succès!")
    except:
        print(f"   ⚠ CUDA non disponible (OpenCV compilé sans CUDA)")

except ImportError as e:
    print(f"\n3. OpenCV")
    print(f"   ✗ ERREUR: {e}")

# 4. Tesseract
try:
    import pytesseract
    print(f"\n4. Tesseract OCR")
    version = pytesseract.get_tesseract_version()
    print(f"   ✓ Version: {version}")

    # Test langues
    import subprocess
    result = subprocess.run(['tesseract', '--list-langs'],
                          capture_output=True, text=True)
    langs = result.stdout.split('\n')[1:]  # Skip header
    print(f"   ✓ Langues disponibles: {', '.join([l for l in langs if l])}")

except Exception as e:
    print(f"\n4. Tesseract OCR")
    print(f"   ✗ ERREUR: {e}")

# 5. Autres dépendances
print(f"\n5. Autres dépendances Python")
packages = ['numpy', 'scipy', 'optuna', 'PIL', 'tkinter']
for pkg in packages:
    try:
        if pkg == 'PIL':
            import PIL
            print(f"   ✓ Pillow: {PIL.__version__}")
        elif pkg == 'tkinter':
            import tkinter
            print(f"   ✓ tkinter: disponible")
        else:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'OK')
            print(f"   ✓ {pkg}: {version}")
    except ImportError:
        print(f"   ✗ {pkg}: NON INSTALLÉ")

# 6. Multiprocessing
print(f"\n6. Multiprocessing")
import multiprocessing
print(f"   ✓ CPU cores: {multiprocessing.cpu_count()}")
print(f"   ✓ Start method: {multiprocessing.get_start_method()}")

print("\n" + "="*70)
print("VÉRIFICATION TERMINÉE")
print("="*70)
```

Exécutez le script :
```bash
chmod +x test_installation.py
python3 test_installation.py
```

---

## Lancement de l'application

### 1. Activation de l'environnement virtuel
```bash
cd ~/OCR_Quality_Audit
source .venv/bin/activate
```

### 2. Lancement de l'interface graphique
```bash
python3 gui_optimizer_v3_ultim.py
```

### 3. Vérification GPU dans l'application

Au lancement, vous devriez voir dans la console :

```
======================================================================
🚀 PHASE 2 - OPTIMISATIONS GPU ACTIVÉES
======================================================================
✓ OpenCL activé pour OpenCV (accélération GPU UMat)
📊 Opérations GPU-accelerated:
   • GaussianBlur (normalisation)
   • morphologyEx (suppression lignes)
   • threshold (binarisation)
   • Laplacian (estimation bruit, netteté)
   • divide (normalisation)
🎯 Gain estimé: +10-15% sur les opérations OpenCV
======================================================================
```

Si vous voyez : `⚠️ OpenCL non disponible - Mode CPU uniquement`, vérifiez votre installation OpenCL.

---

## Dépannage

### Problème : `ImportError: No module named 'cv2'`

**Solution :**
```bash
# Vérifier que OpenCV est installé
pip list | grep opencv

# Si absent, réinstaller
pip install opencv-python opencv-contrib-python
```

### Problème : `TclError: no display name and no $DISPLAY environment variable`

**Cause :** Exécution en SSH sans X11 forwarding

**Solution 1 - X11 Forwarding :**
```bash
ssh -X user@server
```

**Solution 2 - Mode headless (sans GUI) :**
Utilisez les scripts d'optimisation en ligne de commande au lieu de la GUI.

### Problème : Tesseract introuvable

**Solution :**
```bash
# Vérifier installation
which tesseract

# Si absent
sudo apt install -y tesseract-ocr tesseract-ocr-fra

# Vérifier à nouveau
tesseract --version
```

### Problème : OpenCV sans CUDA

**Symptôme :**
```python
cv2.cuda.getCudaEnabledDeviceCount()  # Retourne 0
```

**Solution :** Recompiler OpenCV avec CUDA en utilisant le script `build_opencv_cuda.sh`

### Problème : Performance GPU faible

**Vérifications :**
```bash
# 1. Vérifier que le GPU est utilisé
nvidia-smi

# 2. Activer le mode performance NVIDIA
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 250  # Limite de puissance (ajuster selon votre carte)

# 3. Vérifier OpenCL
clinfo
```

### Problème : `cv2.setNumThreads(1)` ne fait rien

**Explication :** Sur Linux, OpenCV peut utiliser différents backends (TBB, OpenMP, etc.)

**Solution :** Forcer les variables d'environnement avant import :
```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import cv2
cv2.setNumThreads(1)
```

Le code actuel fait déjà cela correctement.

---

## Optimisations avancées

### 1. Augmenter la priorité du processus
```bash
# Lancer avec nice (priorité élevée)
sudo nice -n -10 python3 gui_optimizer_v3_ultim.py
```

### 2. Désactiver le swap pendant l'exécution
```bash
# Voir utilisation swap
free -h

# Désactiver temporairement (si vous avez assez de RAM)
sudo swapoff -a

# Réactiver après
sudo swapon -a
```

### 3. Monitoring GPU en temps réel
```bash
# Terminal séparé
watch -n 1 nvidia-smi
```

---

## Résumé des commandes essentielles

```bash
# Installation complète (première fois)
sudo apt update && sudo apt upgrade -y
sudo ubuntu-drivers autoinstall
sudo reboot

cd ~/OCR_Quality_Audit/Phase3_Migration_Ubuntu_CUDA
./build_opencv_cuda.sh

sudo apt install -y python3-tk tesseract-ocr tesseract-ocr-fra libgl1-mesa-glx libglib2.0-0

cd ~/OCR_Quality_Audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Lancement quotidien
cd ~/OCR_Quality_Audit
source .venv/bin/activate
python3 gui_optimizer_v3_ultim.py
```

---

## Support et contact

Pour tout problème :
1. Vérifier les logs dans la console
2. Exécuter `test_installation.py` pour diagnostiquer
3. Consulter la documentation Phase 3 : `Phase3_Migration_Ubuntu_CUDA/PHASE3_OPENCV_CUDA_UBUNTU.md`

---

**Version :** 1.0
**Dernière mise à jour :** 2025-01-30
**Compatibilité testée :** Ubuntu 22.04 LTS + NVIDIA RTX 1080
