# Guide d'Installation Complet - OCR Quality Audit sur Ubuntu 22.04

**Version finale testée et validée - Décembre 2024**

Ce guide documente la procédure complète de migration et d'installation de l'application **OCR Quality Audit** de Windows vers Ubuntu 22.04 avec accélération GPU CUDA.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis matériels](#prérequis-matériels)
3. [Architecture finale](#architecture-finale)
4. [Installation complète](#installation-complète)
5. [Problèmes rencontrés et solutions](#problèmes-rencontrés-et-solutions)
6. [Utilisation](#utilisation)
7. [Vérifications](#vérifications)

---

## Vue d'ensemble

### Contexte du projet

**OCR Quality Audit** est une application d'analyse de qualité OCR développée initialement sur Windows avec OpenCL. L'objectif de cette migration était de :

- ✅ Passer de Windows à Ubuntu 22.04
- ✅ Utiliser CUDA au lieu d'OpenCL pour l'accélération GPU
- ✅ Éliminer les conflits de dépendances Python
- ✅ Obtenir un environnement stable et reproductible

### Résultat final

**Migration réussie** avec la configuration suivante :

```
OS: Ubuntu 22.04 LTS
GPU: NVIDIA GTX 1080 Ti (Pascal 6.1)
CUDA: 11.8
Python: 3.10.12 (système)
NumPy: 1.21.5 (package système Ubuntu)
OpenCV: 4.8.0 (compilé avec CUDA 11.8)
Autres packages: système Ubuntu (scipy, pandas, matplotlib)
```

**Points clés de la solution :**
- ❌ **Pas d'environnement virtuel** - Utilisation directe de Python système
- ✅ **Packages système Ubuntu** (apt) pour numpy, scipy, pandas, matplotlib
- ✅ **OpenCV compilé** avec NumPy 1.21.5 et CUDA 11.8
- ✅ **Aucun conflit OpenBLAS** - Toutes les bibliothèques partagent la même version système

---

## Prérequis matériels

### Configuration minimale

- **OS** : Ubuntu 22.04 LTS
- **GPU** : NVIDIA avec architecture Pascal ou supérieure (Compute Capability 6.1+)
  - Testé avec : GTX 1080 Ti
  - Compatible : GTX 1060/1070/1080, RTX 20xx/30xx/40xx
- **RAM** : 16 GB minimum
- **CPU** : Multi-cœurs (8+ recommandé pour la compilation)
- **Stockage** : 10 GB libres

### Vérification GPU

```bash
lspci | grep -i nvidia
nvidia-smi
```

Vous devriez voir votre carte graphique et le driver NVIDIA.

---

## Architecture finale

### Problème initial : Conflits OpenBLAS

Lors de l'utilisation de packages PyPI (pip), les bibliothèques comme NumPy et Scipy embarquent leurs propres versions d'OpenBLAS dans des répertoires `numpy.libs/` et `scipy.libs/`. Ces versions différentes causaient des **segmentation faults (erreur 139)** au chargement.

```
❌ Configuration initiale (échec)
~/.local/lib/python3.10/site-packages/
├── numpy/ (version 1.26.4)
│   └── .libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so
└── scipy/ (version 1.11.3)
    └── .libs/libopenblasp-r0-23e5df77.3.21.dev.so
        ↑
    Conflit ! Versions différentes d'OpenBLAS
```

### Solution : Architecture hybride

```
✅ Configuration finale (succès)
/usr/lib/x86_64-linux-gnu/
└── libopenblas.so.0 (version système unique)
    ↑
    Partagée par tous les packages

/usr/lib/python3/dist-packages/
├── numpy/ (1.21.5 - package système)
├── scipy/
├── pandas/
└── matplotlib/

/usr/local/lib/python3.10/dist-packages/
└── cv2/ (4.8.0 - compilé avec NumPy 1.21.5)

~/.local/lib/python3.10/site-packages/
├── pytesseract/ (pas de dépendance BLAS)
└── optuna/ (pas de dépendance BLAS)
```

**Avantages :**
- Une seule version d'OpenBLAS partagée
- Compatibilité garantie entre OpenCV et NumPy
- Stabilité maximale (packages testés ensemble par Ubuntu)

---

## Installation complète

### Étape 1 : Préparation du système

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Installation des outils de base
sudo apt install -y build-essential git wget curl vim
```

### Étape 2 : Installation drivers NVIDIA et CUDA

```bash
# Installation automatique des drivers NVIDIA
sudo ubuntu-drivers autoinstall
sudo reboot

# Après redémarrage, vérifier
nvidia-smi
```

**CUDA Toolkit 11.8** sera installé par le script de compilation OpenCV.

### Étape 3 : Installation des packages Python système

```bash
# Packages scientifiques (numpy, scipy, pandas, matplotlib)
sudo apt install -y python3-numpy python3-scipy python3-pandas python3-matplotlib

# Tesseract OCR
sudo apt install -y tesseract-ocr tesseract-ocr-fra

# Bibliothèques système requises
sudo apt install -y python3-tk libgl1-mesa-glx libglib2.0-0

# Pillow (PIL)
sudo apt install -y python3-pil
```

**Vérification :**
```bash
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
# Résultat attendu : NumPy: 1.21.5
```

### Étape 4 : Cloner le projet

```bash
cd ~/PycharmProjects
git clone https://github.com/jmFschneider/OCR_Quality_Audit
cd OCR_Quality_Audit
git checkout linux/ubuntu
```

### Étape 5 : Compilation OpenCV avec CUDA

#### 5.1 Nettoyer le répertoire de build (si compilation précédente)

```bash
cd ~/opencv_build/opencv/build
rm -rf *
```

#### 5.2 Lancer la compilation

```bash
cd ~/PycharmProjects/OCR_Quality_Audit
./Phase3_Migration_Ubuntu_CUDA/compile_opencv_numpy126.sh 2>&1 | tee opencv_build.log
```

**Durée estimée :** 10-15 minutes (make -j12 sur CPU moderne)

Le script va :
1. Vérifier NumPy 1.21.5 (version système)
2. Configurer CMake avec les flags CUDA appropriés
3. Compiler OpenCV avec support CUDA et NumPy 1.21.5
4. S'arrêter avant l'installation (nécessite sudo)

#### 5.3 Installation d'OpenCV

```bash
cd ~/opencv_build/opencv/build
sudo make install
```

#### 5.4 Vérification

```bash
python3 -c "import cv2; print('OpenCV:', cv2.__version__); print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
```

**Résultat attendu :**
```
OpenCV: 4.8.0
CUDA devices: 1
```

### Étape 6 : Installation des packages complémentaires

Ces packages n'ont pas de dépendances BLAS, donc peuvent être installés avec pip :

```bash
pip3 install --user pytesseract optuna
```

### Étape 7 : Vérification complète

```bash
python3 -c "import cv2, numpy, scipy, pandas, matplotlib, pytesseract, optuna; print('Tous les imports OK')"
```

Si cette commande s'exécute sans erreur, **l'installation est complète** !

---

## Problèmes rencontrés et solutions

### Problème 1 : Segmentation Fault (erreur 139)

**Symptôme :**
```bash
python3 gui_optimizer_v3_ultim.py
Segmentation fault (core dumped)  # Erreur 139
```

**Cause :** Conflit entre les bibliothèques OpenBLAS embarquées dans numpy.libs/ et scipy.libs/ des packages PyPI.

**Solution :** Utiliser les packages système Ubuntu au lieu de pip pour numpy, scipy, pandas, matplotlib.

```bash
# Désinstaller les packages pip
pip3 uninstall -y numpy scipy pandas matplotlib

# Installer les packages système
sudo apt install -y python3-numpy python3-scipy python3-pandas python3-matplotlib
```

### Problème 2 : OpenCV compilé avec mauvaise version NumPy

**Symptôme :** OpenCV compilé avec NumPy 1.26.4, mais système utilise 1.21.5.

**Cause :** Incompatibilité ABI entre les versions de NumPy.

**Solution :** Recompiler OpenCV avec la version NumPy système (1.21.5).

```bash
# 1. Nettoyer le build
cd ~/opencv_build/opencv/build
rm -rf *

# 2. Modifier le script de compilation pour accepter 1.21.5
# (déjà fait dans compile_opencv_numpy126.sh)

# 3. Recompiler
cd ~/PycharmProjects/OCR_Quality_Audit
./Phase3_Migration_Ubuntu_CUDA/compile_opencv_numpy126.sh
```

### Problème 3 : python3-pytesseract introuvable

**Symptôme :**
```bash
sudo apt install python3-pytesseract
E: Unable to locate package python3-pytesseract
```

**Cause :** pytesseract n'est pas disponible dans les dépôts Ubuntu.

**Solution :** Installer avec pip (pas de conflit car pas de dépendance BLAS).

```bash
pip3 install --user pytesseract
```

### Problème 4 : Environnement virtuel corrompu

**Symptôme :** Après installation de packages dans venv, segfault persistant.

**Solution :** Ne pas utiliser d'environnement virtuel. Utiliser directement Python système avec packages apt.

```bash
# Supprimer le venv
rm -rf .venv

# Utiliser python3 système directement
python3 gui_optimizer_v3_ultim.py
```

---

## Utilisation

### Lancement de l'application

```bash
cd ~/PycharmProjects/OCR_Quality_Audit
python3 gui_optimizer_v3_ultim.py
```

**Pas besoin d'activer un environnement virtuel !**

### Première utilisation

1. L'interface graphique s'ouvre
2. Cliquez sur "Charger dossier" pour sélectionner vos images
3. Configurez les paramètres d'optimisation
4. Lancez le traitement

### Monitoring GPU

Dans un terminal séparé :

```bash
watch -n 1 nvidia-smi
```

Vous devriez voir l'utilisation GPU pendant le traitement.

---

## Vérifications

### Vérification complète de l'installation

Créez un fichier `test_install.py` :

```python
#!/usr/bin/env python3
"""Test complet de l'installation Ubuntu"""

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
import numpy as np
print(f"\n2. NumPy")
print(f"   Version: {np.__version__}")
print(f"   Location: {np.__file__}")

# 3. OpenCV
import cv2
print(f"\n3. OpenCV")
print(f"   Version: {cv2.__version__}")
print(f"   Location: {cv2.__file__}")
cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"   CUDA devices: {cuda_count}")
if cuda_count > 0:
    print(f"   ✓ CUDA activé !")
else:
    print(f"   ✗ CUDA non disponible")

# 4. Autres packages
import scipy
import pandas as pd
import matplotlib
print(f"\n4. Autres packages scientifiques")
print(f"   Scipy: {scipy.__version__}")
print(f"   Pandas: {pd.__version__}")
print(f"   Matplotlib: {matplotlib.__version__}")

# 5. Tesseract
import pytesseract
print(f"\n5. Tesseract OCR")
try:
    version = pytesseract.get_tesseract_version()
    print(f"   Version: {version}")
except:
    print(f"   ✗ Tesseract non trouvé")

# 6. Optuna
import optuna
print(f"\n6. Optuna")
print(f"   Version: {optuna.__version__}")

print("\n" + "="*70)
print("✓ TOUS LES TESTS RÉUSSIS")
print("="*70)
```

Exécutez :
```bash
python3 test_install.py
```

### Vérification CUDA spécifique

```bash
python3 -c "
import cv2
print('OpenCV version:', cv2.__version__)
print('CUDA enabled:', cv2.cuda.getCudaEnabledDeviceCount() > 0)
print('OpenCL available:', cv2.ocl.haveOpenCL())
"
```

### Vérification absence de conflits OpenBLAS

```bash
# Vérifier qu'il n'y a pas de .libs/ dans les packages
find ~/.local/lib/python3.10/site-packages -name "*.libs" -type d
```

**Résultat attendu :** Aucun répertoire trouvé (ou seulement pour des packages non-BLAS comme packaging)

---

## Configuration système

### Variables d'environnement CUDA

Le script de compilation configure automatiquement CUDA. Vérifiez :

```bash
echo $PATH | grep cuda
# Devrait contenir /usr/local/cuda-11.8/bin

echo $LD_LIBRARY_PATH | grep cuda
# Devrait contenir /usr/local/cuda-11.8/lib64
```

Si absent, ajoutez à `~/.bashrc` :

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

Puis :
```bash
source ~/.bashrc
```

### Optimisations optionnelles

#### Performance GPU

```bash
# Mode performance persistant
sudo nvidia-smi -pm 1

# Limite de puissance (ajuster selon votre carte)
sudo nvidia-smi -pl 250  # Pour GTX 1080 Ti
```

#### Désactiver le swap temporairement (si assez de RAM)

```bash
# Avant traitement
sudo swapoff -a

# Après traitement
sudo swapon -a
```

---

## Résumé des commandes (Installation complète)

```bash
# ========================================
# INSTALLATION COMPLÈTE UBUNTU 22.04
# ========================================

# 1. Système de base
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl vim
sudo ubuntu-drivers autoinstall
sudo reboot

# 2. Packages Python système
sudo apt install -y python3-numpy python3-scipy python3-pandas python3-matplotlib
sudo apt install -y python3-pil python3-tk
sudo apt install -y tesseract-ocr tesseract-ocr-fra
sudo apt install -y libgl1-mesa-glx libglib2.0-0

# 3. Cloner le projet
cd ~/PycharmProjects
git clone https://github.com/jmFschneider/OCR_Quality_Audit
cd OCR_Quality_Audit
git checkout linux/ubuntu

# 4. Compiler OpenCV (10-15 min)
cd ~/opencv_build/opencv/build && rm -rf *
cd ~/PycharmProjects/OCR_Quality_Audit
./Phase3_Migration_Ubuntu_CUDA/compile_opencv_numpy126.sh
cd ~/opencv_build/opencv/build
sudo make install

# 5. Packages pip complémentaires
pip3 install --user pytesseract optuna

# 6. Vérification
python3 -c "import cv2, numpy, scipy, pandas, matplotlib, pytesseract, optuna; print('OK')"
python3 -c "import cv2; print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"

# 7. Lancer l'application
cd ~/PycharmProjects/OCR_Quality_Audit
python3 gui_optimizer_v3_ultim.py
```

---

## Fichiers importants du projet

```
OCR_Quality_Audit/
├── gui_optimizer_v3_ultim.py          # Application principale
├── requirements_ubuntu.txt            # Liste des dépendances
├── Phase3_Migration_Ubuntu_CUDA/
│   ├── compile_opencv_numpy126.sh     # Script de compilation OpenCV
│   ├── GUIDE_INSTALLATION_UBUNTU_FINAL.md  # Ce document
│   └── README.md                      # Vue d'ensemble Phase 3
└── test_scans/                        # Dossier images (à créer)
```

---

## Support et dépannage

### Logs

- Compilation OpenCV : `opencv_build.log` ou `opencv_build_numpy121.log`
- Application : Sortie console de `python3 gui_optimizer_v3_ultim.py`

### Commandes utiles

```bash
# Vérifier GPU
nvidia-smi
lspci | grep -i nvidia

# Vérifier CUDA
nvcc --version
which nvcc

# Vérifier Python et packages
python3 --version
pip3 list | grep -E "numpy|opencv|scipy|pandas"
dpkg -l | grep -E "python3-(numpy|scipy|pandas|matplotlib)"

# Vérifier Tesseract
which tesseract
tesseract --version
tesseract --list-langs
```

### En cas de problème

1. **Vérifier les versions :**
   ```bash
   python3 test_install.py
   ```

2. **Vérifier absence de conflits :**
   ```bash
   find ~/.local/lib -name "*.libs" -type d
   ```

3. **Recompiler OpenCV si nécessaire :**
   ```bash
   cd ~/opencv_build/opencv/build && rm -rf *
   cd ~/PycharmProjects/OCR_Quality_Audit
   ./Phase3_Migration_Ubuntu_CUDA/compile_opencv_numpy126.sh
   ```

4. **Nettoyer complètement et recommencer :**
   ```bash
   pip3 uninstall -y numpy scipy pandas matplotlib opencv-python opencv-contrib-python
   sudo apt install --reinstall python3-numpy python3-scipy python3-pandas python3-matplotlib
   ```

---

## Historique de la migration

### Décembre 2024 - Migration Windows → Ubuntu

**Défis rencontrés :**
1. ❌ Conflits OpenBLAS entre packages PyPI
2. ❌ Incompatibilités NumPy 1.26.4 vs 1.21.5
3. ❌ Environnements virtuels corrompus par des dépendances contradictoires

**Solution finale :**
1. ✅ Abandon des packages PyPI pour les bibliothèques scientifiques
2. ✅ Utilisation exclusive des packages système Ubuntu (apt)
3. ✅ Compilation OpenCV alignée sur NumPy système (1.21.5)
4. ✅ Architecture hybride : apt (BLAS) + pip (non-BLAS)

**Résultat :** Application stable, pas de segfault, CUDA fonctionnel.

---

## Informations de version

- **Guide version :** 1.0 Final
- **Date :** Décembre 2024
- **Testé sur :** Ubuntu 22.04 LTS
- **GPU testé :** NVIDIA GTX 1080 Ti (Pascal 6.1)
- **Configuration :** Python 3.10.12, NumPy 1.21.5, OpenCV 4.8.0, CUDA 11.8

---

## Prochaines étapes

1. ✅ Migration Ubuntu complète
2. ⏳ Benchmark performance CUDA vs OpenCL
3. ⏳ Migration PaddleOCR pour accélération OCR GPU
4. ⏳ Optimisation pipeline complet

Voir `Phase3_Migration_Ubuntu_CUDA/PHASE3B_PADDLEOCR.md` pour la suite.

---

**Auteur :** Documentation consolidée après migration réussie
**Contact :** Projet GitHub jmFschneider/OCR_Quality_Audit
**Licence :** Voir LICENSE dans le dépôt
