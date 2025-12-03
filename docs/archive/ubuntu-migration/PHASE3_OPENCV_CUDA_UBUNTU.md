# Phase 3 : Compilation OpenCV avec CUDA sous Ubuntu 20.04

**Date** : 2025-11-27
**Objectif** : Compiler OpenCV avec support CUDA pour accélérer le traitement d'images 300 DPI
**Plateforme** : Ubuntu 20.04 LTS + NVIDIA RTX 1080
**Gain estimé** : **×2.0-2.5** sur temps d'exécution total

---

## 🎯 Pourquoi cette Phase 3 ?

### **Contexte**
- **Phase 1** : Hyperthreading + denoising adaptatif → **+25%**
- **Phase 2** : UMat/OpenCL → **+33%** supplémentaire (total **×1.49**)
- **Problème** : Passage de **100 DPI → 300 DPI** = **×9 pixels** à traiter
- **Impact** : Temps screening passe de 8.6 min → **77 min** (impraticable !)

### **Solution : OpenCV-CUDA**
- **CUDA** est bien plus rapide qu'OpenCL sur NVIDIA
- Gain estimé **×2-2.5** supplémentaire
- **Résultat attendu** : Screening 300 DPI en **30-35 min** (acceptable)

---

## 📋 Prérequis

### **1. Vérifier votre configuration**

#### a) Vérifier Ubuntu
```bash
lsb_release -a
# Doit afficher Ubuntu 20.04 LTS
```

#### b) Vérifier la carte graphique NVIDIA
```bash
lspci | grep -i nvidia
# Doit afficher : NVIDIA Corporation GP104 [GeForce GTX 1080]
```

#### c) Vérifier les drivers NVIDIA
```bash
nvidia-smi
# Doit afficher la RTX 1080 et la version du driver
```

**Si `nvidia-smi` ne fonctionne pas**, installez les drivers :
```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

---

## 🚀 Installation - Méthode Rapide (Script Automatisé)

### **Option A : Utiliser le script fourni (RECOMMANDÉ)**

```bash
# 1. Rendre le script exécutable
chmod +x build_opencv_cuda.sh

# 2. Lancer la compilation (45-60 min)
./build_opencv_cuda.sh

# 3. Vérifier l'installation
python3 test_cuda.py
```

**Le script fait tout automatiquement** :
- Installation CUDA Toolkit
- Installation des dépendances
- Téléchargement OpenCV + opencv_contrib
- Compilation avec tous les flags CUDA
- Installation dans l'environnement Python

---

## 🛠️ Installation - Méthode Manuelle (Détaillée)

Si vous préférez comprendre chaque étape :

### **Étape 1 : Installer CUDA Toolkit (10-15 min)**

#### a) Télécharger CUDA Toolkit 11.8 (compatible RTX 1080)
```bash
# Ajouter le dépôt NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update

# Installer CUDA Toolkit
sudo apt install cuda-toolkit-11-8
```

#### b) Configurer les variables d'environnement
```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### c) Vérifier CUDA
```bash
nvcc --version
# Doit afficher : Cuda compilation tools, release 11.8
```

---

### **Étape 2 : Installer les dépendances (5 min)**

```bash
# Outils de build
sudo apt install -y build-essential cmake git pkg-config unzip

# Bibliothèques d'image
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev

# Bibliothèques vidéo (optionnel)
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev

# GTK pour interface (optionnel)
sudo apt install -y libgtk-3-dev

# Optimisations numériques
sudo apt install -y libatlas-base-dev gfortran

# Python
sudo apt install -y python3-dev python3-pip python3-venv
```

---

### **Étape 3 : Télécharger OpenCV sources (5 min)**

```bash
# Créer répertoire de travail
mkdir -p ~/opencv_build && cd ~/opencv_build

# Télécharger OpenCV 4.8.0 (version stable avec CUDA)
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.0.zip

# Extraire
unzip opencv.zip
unzip opencv_contrib.zip

# Renommer pour simplifier
mv opencv-4.8.0 opencv
mv opencv_contrib-4.8.0 opencv_contrib
```

---

### **Étape 4 : Configurer la compilation avec CMake (5 min)**

```bash
cd ~/opencv_build/opencv
mkdir build && cd build

# Configuration CMake avec CUDA activé
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=OFF \
      -D OPENCV_DNN_CUDA=OFF \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_ARCH_BIN=6.1 \
      -D CUDA_ARCH_PTX=6.1 \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_OPENGL=ON \
      -D WITH_OPENCL=ON \
      -D WITH_TBB=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
      -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
      ..
```

**Flags importants expliqués** :
- `WITH_CUDA=ON` : Active CUDA
- `CUDA_ARCH_BIN=6.1` : Architecture Pascal (GTX 1080)
- `CUDA_FAST_MATH=1` : Optimisations mathématiques
- `WITH_CUBLAS=1` : Bibliothèque BLAS CUDA (opérations matricielles)
- `WITH_TBB=ON` : Threading Building Blocks (multithreading)

#### **Vérifier la configuration**
À la fin de CMake, vérifiez :
```
--   NVIDIA CUDA:                   YES (ver 11.8, CUFFT CUBLAS FAST_MATH)
--     NVIDIA GPU arch:              61
--   Python 3:
--     Interpreter:                  /usr/bin/python3
```

---

### **Étape 5 : Compiler OpenCV (20-30 min)**

```bash
# Utiliser tous les cœurs disponibles (12 sur votre PC)
make -j12

# Si erreur de mémoire, réduire à -j8 ou -j6
```

**Attendez 20-30 minutes...**
Sur votre PC 12 cores, la compilation devrait prendre **~25 minutes**.

---

### **Étape 6 : Installer OpenCV**

```bash
sudo make install
sudo ldconfig
```

---

### **Étape 7 : Vérifier l'installation**

```bash
python3 -c "import cv2; print(cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"
```

**Sortie attendue** :
```
4.8.0
CUDA: 1
```

---

## 🧪 Tests de Validation

### **Test 1 : Script de validation Python**

```bash
python3 test_cuda.py
```

**Ce script teste** :
- Import cv2
- Version OpenCV
- Nombre de GPU CUDA détectés
- Opérations CUDA de base (upload, GaussianBlur, download)
- Benchmark CPU vs CUDA

---

### **Test 2 : Benchmark Simple**

```python
import cv2
import numpy as np
import time

# Créer une image test (3000x3000 comme 300 DPI)
img = np.random.randint(0, 255, (3000, 3000), dtype=np.uint8)

# Test CPU
start = time.time()
for _ in range(10):
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

# Test CUDA
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)
start = time.time()
for _ in range(10):
    gpu_blurred = cv2.cuda.createGaussianFilter(cv2.CV_8U, cv2.CV_8U, (21, 21), 0).apply(gpu_img)
cuda_time = time.time() - start
print(f"CUDA: {cuda_time:.3f}s")
print(f"Speedup: {cpu_time/cuda_time:.2f}x")
```

**Résultat attendu** : Speedup **×5-10** sur GaussianBlur

---

## 🔧 Adaptation du Code Python

### **Modifications à apporter dans `gui_optimizer_v3_ultim.py`**

#### **1. Détection CUDA au lieu d'OpenCL**

```python
# Remplacer la section OpenCL par CUDA
USE_CUDA = False
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    USE_CUDA = True
    print("\n" + "="*70)
    print("🚀 PHASE 3 - OPTIMISATIONS CUDA ACTIVÉES")
    print("="*70)
    print(f"✅ CUDA activé - {cv2.cuda.getCudaEnabledDeviceCount()} GPU détecté(s)")
    print("📊 Opérations GPU-accelerated (CUDA):")
    print("   • GaussianBlur (×5-10 plus rapide)")
    print("   • morphologyEx (×8-15 plus rapide)")
    print("   • threshold (×3-5 plus rapide)")
    print("   • Laplacian (×4-8 plus rapide)")
    print("🎯 Gain estimé: +50-80% sur les opérations OpenCV")
    print("="*70 + "\n")
else:
    print("⚠️  CUDA non disponible - Mode CPU/OpenCL uniquement")
```

#### **2. Créer des versions CUDA des fonctions**

**Exemple : GaussianBlur avec CUDA**

```python
# Version CUDA de normalisation_division
def normalisation_division_cuda(image_gray, kernel_size):
    """Normalisation par division - Version CUDA."""
    if kernel_size % 2 == 0: kernel_size += 1

    if USE_CUDA:
        # Upload vers GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image_gray)

        # GaussianBlur sur GPU
        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8U, cv2.CV_8U,
            (kernel_size, kernel_size), 0
        )
        gpu_fond = gaussian_filter.apply(gpu_img)

        # Divide sur GPU
        gpu_result = cv2.cuda.divide(gpu_img, gpu_fond, scale=255)

        # Download résultat
        return gpu_result.download()
    else:
        # Fallback CPU
        fond = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
        return cv2.divide(image_gray, fond, scale=255)
```

**Note** : Un guide complet de migration du code sera fourni après validation de la compilation.

---

## 📊 Gains Attendus

### **Opérations individuelles (300 DPI, 3000×3000 pixels)**

| Opération | Temps CPU | Temps CUDA | Speedup |
|-----------|-----------|------------|---------|
| **GaussianBlur (21×21)** | ~500 ms | **~50 ms** | **×10** |
| **morphologyEx** | ~800 ms | **~60 ms** | **×13** |
| **threshold** | ~50 ms | **~15 ms** | **×3.3** |
| **Laplacian** | ~200 ms | **~40 ms** | **×5** |

### **Pipeline complet (estimations)**

| Configuration | Temps/image | Screening 512 pts |
|---------------|-------------|-------------------|
| **Phase 2 (100 DPI)** | 1.87 s | 8.6 min |
| **Phase 2 (300 DPI)** | 16.8 s | 77 min ⚠️ |
| **Phase 3 (300 DPI)** | **6-8 s** ✅ | **30-35 min** ✅ |

**Gain Phase 3 sur Phase 2** : **×2.0-2.5**
**Temps redevient acceptable pour l'optimisation de paramètres !**

---

## 🐛 Dépannage

### **Problème : CUDA non détecté après compilation**

```bash
# Vérifier que les libs CUDA sont bien liées
ldd /usr/local/lib/python3.8/dist-packages/cv2/python-3.8/cv2.*.so | grep cuda

# Si vide, recompiler avec :
cmake -D WITH_CUDA=ON -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 ..
```

### **Problème : Erreur de mémoire pendant compilation**

```bash
# Réduire le parallélisme
make -j4  # au lieu de -j12
```

### **Problème : Version CUDA incompatible**

GTX 1080 = Architecture **Pascal (Compute Capability 6.1)**

```bash
# Vérifier dans CMake :
cmake .. | grep "NVIDIA GPU arch"
# Doit afficher : 61
```

### **Problème : Python ne trouve pas cv2**

```bash
# Vérifier l'installation
python3 -c "import sys; print('\n'.join(sys.path))"

# Créer un lien symbolique si nécessaire
sudo ln -s /usr/local/lib/python3.8/site-packages/cv2 /usr/lib/python3/dist-packages/cv2
```

---

## 📝 Checklist de Validation

Avant de modifier le code du projet, vérifier :

- [ ] `nvidia-smi` affiche la RTX 1080
- [ ] `nvcc --version` affiche CUDA 11.8
- [ ] `python3 -c "import cv2; print(cv2.__version__)"` affiche 4.8.0
- [ ] `python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"` affiche 1
- [ ] `python3 test_cuda.py` réussit tous les tests
- [ ] Benchmark GaussianBlur montre speedup ×5+

---

## 🚀 Prochaines Étapes

Une fois OpenCV-CUDA compilé et validé :

1. **Migrer le code** vers les fonctions `cv2.cuda.*`
2. **Tester sur 2-3 images** pour valider les résultats
3. **Mesurer les gains réels** sur images 300 DPI
4. **Optimiser les paramètres** avec screening sur 300 DPI
5. **(Optionnel) Compiler Tesseract avec CUDA** pour gain supplémentaire sur OCR

---

## 🔀 Stratégie Git pour la Migration CUDA

### **Objectif**
Adapter le code du projet pour Ubuntu/CUDA tout en préservant la version Windows/OpenCL fonctionnelle.

### **Approche recommandée : Feature Branch**

#### **Option 1 : Branche dédiée CUDA (RECOMMANDÉ pour démarrer)**

```bash
# Sur Ubuntu, après compilation OpenCV-CUDA réussie
git checkout main
git pull origin main
git checkout -b feature/cuda-migration

# Travailler sur cette branche
# - Adapter le code pour CUDA
# - Tester les performances
# - Valider les résultats

# Commiter progressivement
git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Migration GaussianBlur vers cv2.cuda"

git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Migration variance_laplacien vers cv2.cuda"

# Pousser la branche
git push origin feature/cuda-migration

# Une fois validé : Pull Request vers main
```

**Avantages** :
- ✅ Code Windows (OpenCL) reste fonctionnel sur `main`
- ✅ Vous pouvez tester CUDA sans casser l'existant
- ✅ Facile de comparer performances avant/après
- ✅ Possibilité de faire des allers-retours
- ✅ Historique propre avec Pull Request

---

#### **Option 2 : Compatibilité croisée Windows + Ubuntu (IDÉAL long terme)**

Rendre le code **compatible Windows ET Ubuntu** avec détection automatique :

```python
# Détection de la plateforme et capacités GPU
USE_CUDA = False
USE_OPENCL = False

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    USE_CUDA = True
    print("🚀 Mode CUDA (Ubuntu + NVIDIA)")
elif cv2.ocl.haveOpenCL():
    USE_OPENCL = True
    print("🚀 Mode OpenCL (Windows)")
else:
    print("⚠️ Mode CPU uniquement")

# Fonctions adaptatives
def apply_gaussian_blur(image, kernel_size):
    """GaussianBlur adaptatif selon plateforme."""
    if USE_CUDA:
        # Version CUDA pour Ubuntu
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(image)
        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8U, cv2.CV_8U,
            (kernel_size, kernel_size), 0
        )
        gpu_result = gaussian_filter.apply(gpu_img)
        return gpu_result.download()
    elif USE_OPENCL:
        # Version OpenCL pour Windows
        umat = cv2.UMat(image)
        result = cv2.GaussianBlur(umat, (kernel_size, kernel_size), 0)
        return result.get()
    else:
        # Fallback CPU
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
```

**Branche suggérée** : `feature/cross-platform-gpu`

**Avantages** :
- ✅ Un seul code source pour Windows et Ubuntu
- ✅ Détection automatique de la meilleure accélération disponible
- ✅ Fallback CPU si pas de GPU
- ✅ Maintenance simplifiée

---

### **Workflow Git détaillé recommandé**

#### **Phase 1 : Préparation (maintenant, sur Windows)**

```bash
# Créer la branche depuis main
git checkout main
git pull origin main
git checkout -b feature/cuda-migration
git push -u origin feature/cuda-migration
```

#### **Phase 2 : Migration (sur Ubuntu)**

```bash
# Sur Ubuntu, cloner et récupérer la branche
git clone https://github.com/jmFschneider/OCR_Quality_Audit
cd OCR_Quality_Audit
git checkout feature/cuda-migration

# Après compilation OpenCV-CUDA réussie
# Modifier le code progressivement

# Commits granulaires (recommandé)
git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Add CUDA detection and initialization"

git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Migrate normalisation_division to CUDA"

git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Migrate variance_laplacien to CUDA"

git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Migrate all preprocessing functions to CUDA"

# Pousser régulièrement
git push origin feature/cuda-migration
```

#### **Phase 3 : Validation et benchmarks**

```bash
# Ajouter les résultats de benchmarks
mkdir -p benchmarks
# ... créer fichier avec résultats ...
git add benchmarks/cuda_vs_opencl_300dpi.md
git commit -m "docs(cuda): Add performance benchmarks 300 DPI CUDA vs OpenCL"

# Commit final avec résultats
git add gui_optimizer_v3_ultim.py
git commit -m "feat(cuda): Complete CUDA migration - 2.5x speedup on 300 DPI images"
git push origin feature/cuda-migration
```

#### **Phase 4 : Merge via Pull Request**

Sur GitHub :
1. Créer Pull Request `feature/cuda-migration` → `main`
2. Description détaillée :
   - Résumé des modifications
   - Benchmarks avant/après
   - Instructions de test
3. Review du code
4. Merge vers `main`

---

### **Structure de branches suggérée**

```
main (stable, actuellement Windows/OpenCL)
  │
  ├── feature/cuda-migration (Ubuntu/CUDA uniquement)
  │   └── Objectif : migration rapide pour tester CUDA
  │
  └── feature/cross-platform-gpu (Windows + Ubuntu)
      └── Objectif : code universel (à créer après validation CUDA)
```

---

### **Convention de commits pour cette migration**

Format recommandé :
```
<type>(cuda): <description courte>

[corps optionnel avec détails]
```

**Types courants** :
- `feat(cuda):` - Nouvelles fonctionnalités CUDA
- `fix(cuda):` - Corrections de bugs
- `perf(cuda):` - Améliorations de performance
- `docs(cuda):` - Documentation
- `test(cuda):` - Ajout/modification de tests
- `refactor(cuda):` - Refactoring sans changement fonctionnel

**Exemples concrets** :
```bash
git commit -m "feat(cuda): Add cv2.cuda support detection and initialization"

git commit -m "feat(cuda): Implement CUDA-accelerated GaussianBlur in preprocessing"

git commit -m "perf(cuda): 2.5x speedup on 300 DPI images with full CUDA pipeline"

git commit -m "docs(cuda): Add CUDA vs OpenCL benchmark results"

git commit -m "fix(cuda): Handle GPU memory cleanup in error cases"

git commit -m "feat(cuda): Add cross-platform GPU detection (CUDA/OpenCL/CPU)"
```

---

### **Recommandation finale pour votre projet**

**Approche en 2 temps** :

1. **Court terme - Validation CUDA** :
   - Créer `feature/cuda-migration`
   - Migrer uniquement vers CUDA (pas de compatibilité OpenCL)
   - Valider les performances sur Ubuntu
   - Benchmarker vs Phase 2
   - **Durée estimée** : 2-3 jours de travail

2. **Moyen terme - Code universel** (si CUDA validé) :
   - Créer `feature/cross-platform-gpu`
   - Ajouter détection CUDA/OpenCL/CPU
   - Refactorer les fonctions pour supporter les 3 modes
   - **Avantage** : Un seul code pour Windows et Ubuntu
   - **Durée estimée** : 1-2 jours supplémentaires

**Ou bien** :
- Si performances CUDA excellentes → Ubuntu devient plateforme principale
- Merger `feature/cuda-migration` vers `main`
- Archiver la version Windows/OpenCL dans une branche `legacy/windows-opencl`

---

### **Checklist avant de démarrer la migration**

Avant de créer la branche `feature/cuda-migration` :

- [ ] OpenCV-CUDA compilé et validé sur Ubuntu
- [ ] `python3 test_cuda.py` réussit tous les tests
- [ ] Benchmark simple montre bien speedup CUDA vs CPU
- [ ] Environnement Python Ubuntu opérationnel
- [ ] Git configuré sur Ubuntu (`git config --global user.name/email`)
- [ ] Accès SSH à GitHub depuis Ubuntu (ou HTTPS avec token)

---

## 📚 Ressources

- **OpenCV CUDA Documentation** : https://docs.opencv.org/4.8.0/d1/d1a/group__cuda.html
- **CUDA Toolkit** : https://developer.nvidia.com/cuda-downloads
- **Compute Capability** : https://developer.nvidia.com/cuda-gpus (GTX 1080 = 6.1)
- **OpenCV GitHub** : https://github.com/opencv/opencv

---

## ⚠️ Notes Importantes

1. **Backup** : Faites une sauvegarde de votre environnement Python actuel avant compilation
2. **Temps** : Prévoyez 1h pour la compilation complète
3. **Espace disque** : ~5 GB nécessaires pour sources + build
4. **Double boot** : Si vous utilisez le même `/home`, l'environnement sera partagé Windows/Ubuntu

---

**Bon courage pour la compilation ! Une fois terminée, vous aurez des performances exceptionnelles sur vos images 300 DPI ! 🚀**
