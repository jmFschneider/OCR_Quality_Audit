# Phase 3 : Migration Ubuntu + Accélération CUDA

**Objectif** : Migrer le projet de Windows (OpenCL) vers Ubuntu (CUDA) pour accélérer le traitement d'images 300 DPI
**Gain attendu** : **×2.2** (Screening 77 min → 35 min)
**Plateforme cible** : Ubuntu 20.04 LTS + NVIDIA GTX 1080 + CUDA 11.8

---

## 📚 Documentation Disponible

### 🎯 [SYNTHESE_PROJET.md](./SYNTHESE_PROJET.md)
**Vue d'ensemble complète du projet**
- Évolution par phases (Baseline → Phase 3B)
- Problématique et solutions techniques
- Comparatif GPU et OCR engines
- Stratégie Git et workflow
- Prochaines étapes opérationnelles

👉 **Commencez par ce document pour comprendre le contexte global**

---

### 🔧 [PHASE3_OPENCV_CUDA_UBUNTU.md](./PHASE3_OPENCV_CUDA_UBUNTU.md)
**Phase 3A : Compilation OpenCV avec CUDA**
- Guide complet de compilation OpenCV 4.8.0 + CUDA 11.8
- Installation manuelle et script automatisé
- Migration du code Python vers `cv2.cuda.*`
- Stratégie Git pour la migration (feature branches)
- Benchmarks et gains attendus : **×2.0-2.5** sur preprocessing

📋 **Checklist** :
- [ ] Compiler OpenCV-CUDA (45-60 min)
- [ ] Vérifier `python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"` → 1
- [ ] Migrer les fonctions de preprocessing vers CUDA
- [ ] Benchmarker les gains réels

---

### 🚀 [PHASE3B_PADDLEOCR.md](./PHASE3B_PADDLEOCR.md)
**Phase 3B : Migration vers PaddleOCR GPU**
- Comparatif détaillé des OCR engines (PaddleOCR, EasyOCR, Tesseract, Chandra, DeepSeek)
- Installation PaddleOCR avec CUDA 11.8
- Script de benchmark PaddleOCR vs Tesseract
- Intégration dans le pipeline de traitement
- Gains attendus : **×1.9** sur OCR (32 min → 17 min)

📋 **Checklist** :
- [ ] Installer PaddlePaddle GPU (`pip3 install paddlepaddle-gpu`)
- [ ] Installer PaddleOCR (`pip3 install paddleocr`)
- [ ] Benchmarker PaddleOCR vs Tesseract
- [ ] Intégrer dans `gui_optimizer_v3_ultim.py`
- [ ] Valider scores de confiance

---

### ⚙️ [build_opencv_cuda.sh](./build_opencv_cuda.sh)
**Script automatisé de compilation OpenCV-CUDA**
- Installation CUDA Toolkit 11.8
- Compilation OpenCV 4.8.0 avec tous les flags CUDA
- Vérifications automatiques (NumPy, nvcc, CMake)
- Corrections critiques incluses :
  - ✅ Export PATH CUDA direct (pas de `source ~/.bashrc`)
  - ✅ Installation NumPy avant compilation
  - ✅ Validations post-CMake détaillées

**Usage** :
```bash
chmod +x build_opencv_cuda.sh
./build_opencv_cuda.sh
```

---

## 🎯 Gains Cumulés Attendus

| Phase | Optimisation | Temps Screening 512 images (300 DPI) | Gain |
|-------|-------------|---------------------------------------|------|
| **Phase 2** (Baseline) | OpenCL Windows | **77 min** | - |
| **Phase 3A** | OpenCV-CUDA Ubuntu | **~50 min** | ×1.54 |
| **Phase 3B** | + PaddleOCR GPU | **~35 min** | **×2.2** ✅ |

---

## 🔀 Workflow Git Recommandé

```bash
# 1. Créer branche feature
git checkout -b feature/cuda-migration

# 2. Phase 3A : OpenCV-CUDA
# ... compilation, migration code, tests ...
git commit -m "feat(cuda): Migrate preprocessing to OpenCV-CUDA"

# 3. Phase 3B : PaddleOCR
# ... installation, benchmark, intégration ...
git commit -m "feat(ocr): Migrate OCR to PaddleOCR GPU"

# 4. Pull Request vers main
git push origin feature/cuda-migration
# → Créer PR sur GitHub avec benchmarks
```

Voir détails : [PHASE3_OPENCV_CUDA_UBUNTU.md § Stratégie Git](./PHASE3_OPENCV_CUDA_UBUNTU.md#-stratégie-git-pour-la-migration-cuda)

---

## 🖥️ Alternatives GPU Évaluées

| GPU | VRAM | Prix (~) | Recommandation |
|-----|------|----------|----------------|
| **GTX 1080** (actuelle) | 8 GB | 0€ | ✅ **Optimal court terme** - Compatible PaddleOCR |
| **RTX 3060** | 12 GB | 300€ | ⭐ **Meilleur ratio €/perf** - Compatible Chandra OCR |
| **RTX 4060** | 8 GB | 350€ | ⚠️ Moins intéressant (même VRAM que GTX 1080) |
| **RTX 4070** | 12 GB | 600€ | ✅ **Optimal moyen terme** - Futureproof 2025-2030 |

**Recommandation** : Garder GTX 1080 pour Phase 3, upgrade RTX 3060/4070 seulement si nécessaire

---

## 📋 Ordre de Lecture Recommandé

1. **[SYNTHESE_PROJET.md](./SYNTHESE_PROJET.md)** - Vue d'ensemble (10 min)
2. **[PHASE3_OPENCV_CUDA_UBUNTU.md](./PHASE3_OPENCV_CUDA_UBUNTU.md)** - Compilation OpenCV-CUDA (30 min lecture + 60 min compilation)
3. **[PHASE3B_PADDLEOCR.md](./PHASE3B_PADDLEOCR.md)** - Migration PaddleOCR (30 min lecture + implémentation)
4. **[build_opencv_cuda.sh](./build_opencv_cuda.sh)** - Exécuter le script (45-60 min)

---

## ⚠️ Prérequis Techniques

Avant de commencer :
- [ ] Ubuntu 20.04 LTS installé (dual boot ou VM)
- [ ] NVIDIA GTX 1080 (ou supérieure) détectée
- [ ] Driver NVIDIA installé (`nvidia-smi` fonctionne)
- [ ] Python 3.8+ disponible
- [ ] Git configuré (`git config --global user.name/email`)
- [ ] ~5 GB d'espace disque libre (compilation OpenCV)

---

## 🚀 Quick Start

```bash
# 1. Sur Ubuntu, cloner le projet
git clone https://github.com/jmFschneider/OCR_Quality_Audit
cd OCR_Quality_Audit/Phase3_Migration_Ubuntu_CUDA

# 2. Lire la synthèse
cat SYNTHESE_PROJET.md

# 3. Compiler OpenCV-CUDA
chmod +x build_opencv_cuda.sh
./build_opencv_cuda.sh

# 4. Vérifier installation
python3 -c "import cv2; print(cv2.__version__); print('CUDA:', cv2.cuda.getCudaEnabledDeviceCount())"

# 5. Installer PaddleOCR
pip3 install paddlepaddle-gpu==2.6.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip3 install paddleocr

# 6. Lire les guides détaillés et migrer le code
```

---

**Dernière mise à jour** : 2025-11-28
**Statut** : Documentation complète ✅ | Implémentation en attente 📋
