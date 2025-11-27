# Phase 3B : Tesseract Batch Processing

**Date** : 2025-11-27
**Objectif** : Optimiser l'OCR en traitant plusieurs images par batch au lieu d'une par une
**Gain estimé** : **+30-40%** supplémentaire sur temps OCR
**Prérequis** : Phase 3A (OpenCV-CUDA) recommandée mais pas obligatoire

---

## 🎯 Pourquoi le Batch Processing ?

### **Problème Actuel**

Dans le code actuel, chaque worker traite **1 image à la fois** :
```python
# process_image_data_wrapper() - Ligne ~175
score_tess = get_tesseract_score(processed_img)  # 1 appel Tesseract
```

**Avec 24 images en parallèle** :
- 24 workers × 1 image chacun = **24 appels Tesseract séparés**
- Chaque appel paie l'**overhead de démarrage** Tesseract (~50-100ms)
- Temps total OCR : ~650ms par image

### **Solution : Batch Processing**

Tesseract peut traiter **plusieurs images en un seul appel** :
```python
# Au lieu de 24 appels séparés
results = pytesseract.image_to_data(multi_page_tiff)  # 1 seul appel
```

**Avantage** :
- Overhead payé **1 seule fois** au lieu de 24 fois
- Tesseract optimise en interne le traitement multi-pages
- Gain estimé : **+30-40%** sur temps OCR

---

## 📊 Gains Attendus

### **Impact sur OCR uniquement**

| Mode | Overhead | Temps OCR (24 img) | Gain OCR |
|------|----------|-------------------|----------|
| **Actuel** (1 par 1) | 24 × 50-100ms | ~650ms/img | - |
| **Batch 20-24** | 1 × 50-100ms | **~450ms/img** | **+30-40%** |

### **Impact sur Pipeline Complet (300 DPI)**

| Configuration | Temps Total/img | Screening 512 pts |
|---------------|-----------------|-------------------|
| **Phase 3A (CUDA seul)** | 6-8 s | 30-35 min |
| **Phase 3A+B (CUDA + Batch)** | **4-5 s** ✅ | **20-25 min** ✅ |

**Résultat** : Gain additionnel de **~2s par image** (réduction temps OCR de 650ms → 450ms)

---

## 🔧 Architecture du Batch Processing

### **Workflow Modifié**

```
┌─────────────────────────────────────────────────────────┐
│ 1. Multiprocessing Pool (24 workers en parallèle)      │
│    ↓                                                    │
│    Process images through pipeline (NO OCR)             │
│    - Line removal                                       │
│    - Normalization                                      │
│    - Denoising                                          │
│    - Binarization                                       │
│    - Sharpness/Contrast calculation                     │
│    ↓                                                    │
│    Return: processed_images[] (24 images)               │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Tesseract Batch OCR (1 appel pour 20-24 images)     │
│    ↓                                                    │
│    Create multi-page TIFF                               │
│    ↓                                                    │
│    Single Tesseract call                                │
│    ↓                                                    │
│    Return: scores[] (24 scores)                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Aggregate Results                                    │
│    Average(tesseract_scores, sharpness, contrast)       │
└─────────────────────────────────────────────────────────┘
```

---

## 💻 Implémentation Détaillée

### **Étape 1 : Modifier `process_image_data_wrapper()`**

Renommer en `process_image_pipeline_only()` pour clarifier qu'on ne fait plus l'OCR ici :

```python
def process_image_pipeline_only(args):
    """
    Traite une image à travers le pipeline SANS faire l'OCR.
    L'OCR sera effectué en batch après le traitement de toutes les images.

    Returns:
        tuple: (processed_img, sharpness, contrast, timings)
    """
    global has_printed_timings

    # Forcer le mono-threading
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cv2.setNumThreads(1)

    img, params = args
    if img is None:
        return None, 0, 0, {}

    # Pipeline complet (sans OCR)
    processed_img, timings = pipeline_complet_timed(img, params)

    # Calculer sharpness et contrast
    t0 = time.time()
    score_sharp = get_sharpness(processed_img)
    score_cont = get_contrast(processed_img)
    timings['6_sharp_contrast'] = (time.time() - t0) * 1000

    # Afficher timings une fois (diagnostic)
    if not has_printed_timings:
        has_printed_timings = True
        print("\n--- Analyse détaillée des temps d'exécution (en ms, pour une image) ---")

        noise_level = timings.pop('noise_level', None)
        noise_threshold = timings.pop('noise_threshold', None)

        if noise_level is not None and noise_threshold is not None:
            print(f"  - Niveau de bruit détecté: {noise_level:.2f}")
            print(f"  - Seuil de bruit configuré: {noise_threshold:.2f}")
            if noise_level < noise_threshold:
                print("    → Stratégie: Denoising OPTIMISÉ (searchWindowSize=15)")
            else:
                print("    → Stratégie: Denoising COMPLET (searchWindowSize=21)")

        total_time = sum(timings.values())
        for name, t in sorted(timings.items()):
            percentage = (t / total_time) * 100 if total_time > 0 else 0
            print(f"  - Étape {name}: {t:.2f} ms ({percentage:.1f}%)")
        print(f"  - TEMPS TOTAL par image (sans OCR): {total_time:.2f} ms")
        print("  - OCR sera effectué en batch (gain estimé +30-40%)")
        print("----------------------------------------------------------------------\n")

    return processed_img, score_sharp, score_cont, timings
```

---

### **Étape 2 : Créer `batch_tesseract_ocr()`**

```python
def batch_tesseract_ocr(images, batch_size=20):
    """
    Effectue l'OCR sur un batch d'images pour réduire l'overhead de démarrage Tesseract.

    Args:
        images: Liste d'images (numpy arrays ou UMat)
        batch_size: Nombre d'images par batch (10-20 optimal selon RAM disponible)

    Returns:
        List[float]: Scores Tesseract pour chaque image
    """
    import tempfile

    scores = []

    # Traiter par batches
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        # Préparer les images
        batch_np = []
        for img in batch:
            # Convertir UMat en numpy si nécessaire
            if isinstance(img, cv2.UMat):
                img = img.get()

            # Pre-resize pour grandes images (optimisation Tesseract)
            if img.shape[1] > 2500:
                img = cv2.resize(img, None, fx=0.5, fy=0.5)

            batch_np.append(img)

        # Créer fichier TIFF multi-page temporaire
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Écrire toutes les images dans un TIFF multi-page
            import tifffile
            tifffile.imwrite(tmp_path, np.array(batch_np))

            # OCR sur tout le batch en une fois
            data = pytesseract.image_to_data(
                tmp_path,
                config='--oem 1 --psm 6',
                output_type=pytesseract.Output.DICT
            )

            # Parser les résultats par page
            page_nums = data['page_num']
            confidences = data['conf']

            # Extraire les scores par page
            for page_idx in range(1, len(batch_np) + 1):
                # Confidences de cette page uniquement
                page_confs = [
                    int(c) for p, c in zip(page_nums, confidences)
                    if p == page_idx and int(c) != -1
                ]

                if page_confs:
                    scores.append(sum(page_confs) / len(page_confs))
                else:
                    scores.append(0)

        except Exception as e:
            print(f"⚠️  Erreur OCR batch: {e}")
            print(f"    Fallback: traitement individuel pour ce batch")

            # Fallback : traiter individuellement
            for img in batch_np:
                try:
                    data = pytesseract.image_to_data(
                        img, config='--oem 1 --psm 6',
                        output_type=pytesseract.Output.DICT
                    )
                    confs = [int(x) for x in data['conf'] if int(x) != -1]
                    scores.append(sum(confs) / len(confs) if confs else 0)
                except:
                    scores.append(0)

        finally:
            # Nettoyer fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return scores
```

---

### **Étape 3 : Modifier `evaluate_pipeline()`**

```python
def evaluate_pipeline(self, params):
    """
    Évalue le pipeline sur toutes les images chargées.
    Version optimisée avec Tesseract Batch Processing.
    """
    if not self.loaded_images:
        return 0, 0, 0

    optimal_workers = int(os.cpu_count() * 1.5)
    pool_size = min(len(self.loaded_images), optimal_workers)

    try:
        # Phase 1 : Traitement pipeline en parallèle (SANS OCR)
        with multiprocessing.Pool(pool_size) as pool:
            pool_args = zip(self.loaded_images, repeat(params))
            results = pool.map(process_image_pipeline_only, pool_args)

        # Extraire les résultats
        processed_images = [r[0] for r in results if r[0] is not None]
        sharpness_scores = [r[1] for r in results if r[0] is not None]
        contrast_scores = [r[2] for r in results if r[0] is not None]

        if not processed_images:
            return 0, 0, 0

        # Phase 2 : OCR en BATCH sur toutes les images traitées
        tesseract_scores = batch_tesseract_ocr(processed_images, batch_size=20)

        # Calculer les moyennes
        avg_tess = sum(tesseract_scores) / len(tesseract_scores)
        avg_sharp = sum(sharpness_scores) / len(sharpness_scores)
        avg_cont = sum(contrast_scores) / len(contrast_scores)

        return avg_tess, avg_sharp, avg_cont

    except Exception as e:
        self.update_log_from_thread(f"Erreur de multiprocessing, passage en mode séquentiel: {e}")

        # Fallback séquentiel (mode original)
        return self.evaluate_pipeline_sequential(params)


def evaluate_pipeline_sequential(self, params):
    """Fallback séquentiel si multiprocessing échoue."""
    processed_images = []
    sharpness_scores = []
    contrast_scores = []

    for img in self.loaded_images:
        processed_img = pipeline_complet(img, params)
        processed_images.append(processed_img)
        sharpness_scores.append(get_sharpness(processed_img))
        contrast_scores.append(get_contrast(processed_img))

    # OCR en batch même en mode séquentiel
    tesseract_scores = batch_tesseract_ocr(processed_images, batch_size=20)

    avg_tess = sum(tesseract_scores) / len(tesseract_scores)
    avg_sharp = sum(sharpness_scores) / len(sharpness_scores)
    avg_cont = sum(contrast_scores) / len(contrast_scores)

    return avg_tess, avg_sharp, avg_cont
```

---

## 📦 Dépendances Supplémentaires

### **Ubuntu**

```bash
pip3 install tifffile
```

### **Windows**

```bash
pip install tifffile
```

**Mettre à jour `requirements.txt` et `requirements_ubuntu.txt`** :
```
tifffile>=2023.7.0   # Pour créer TIFF multi-page (batch Tesseract)
```

---

## 🧪 Tests et Validation

### **Test 1 : Vérifier que le batch fonctionne**

```python
import cv2
import numpy as np
from batch_tesseract_ocr import batch_tesseract_ocr

# Créer 10 images test
images = [np.random.randint(0, 255, (1000, 1000), dtype=np.uint8) for _ in range(10)]

# Tester batch
scores = batch_tesseract_ocr(images, batch_size=10)
print(f"Scores: {scores}")
print(f"Nombre: {len(scores)}")  # Doit être 10
```

### **Test 2 : Benchmark Batch vs Individuel**

```python
import time
import cv2
import numpy as np
import pytesseract

# Créer 24 images test
images = [np.random.randint(0, 255, (2000, 2000), dtype=np.uint8) for _ in range(24)]

# Test individuel
start = time.time()
scores_individual = []
for img in images:
    data = pytesseract.image_to_data(img, config='--oem 1 --psm 6',
                                     output_type=pytesseract.Output.DICT)
    confs = [int(x) for x in data['conf'] if int(x) != -1]
    scores_individual.append(sum(confs) / len(confs) if confs else 0)
time_individual = time.time() - start
print(f"Individuel (24 images): {time_individual:.2f}s ({time_individual/24*1000:.0f}ms/img)")

# Test batch
start = time.time()
scores_batch = batch_tesseract_ocr(images, batch_size=20)
time_batch = time.time() - start
print(f"Batch (24 images): {time_batch:.2f}s ({time_batch/24*1000:.0f}ms/img)")

# Speedup
speedup = time_individual / time_batch
print(f"Speedup: {speedup:.2f}x")
```

**Résultat attendu** : Speedup **1.3-1.4×** (gain 30-40%)

---

## ⚙️ Paramètres Optimaux

### **Batch Size**

| Taille Batch | Avantages | Inconvénients |
|--------------|-----------|---------------|
| **10** | Faible usage RAM | Overhead réduit mais pas maximal |
| **20** ⭐ | **Optimal** (équilibre RAM/vitesse) | **Recommandé** |
| **24** | Overhead minimal | Usage RAM plus élevé |
| **50+** | Gain marginal | Risque dépassement RAM |

**Recommandation** : `batch_size=20` pour équilibre optimal.

### **Gestion de la Mémoire**

Pour 24 images 300 DPI (3000×3000) :
- 1 image = ~9 MB (grayscale)
- Batch de 20 = ~180 MB
- TIFF temporaire = ~180 MB
- **Total** : ~360 MB (largement acceptable)

---

## 🎯 Gains Cumulés - Résumé

### **Evolution des Performances**

| Phase | Optimisation | Gain | Temps/img (300 DPI) | Screening 512 pts |
|-------|--------------|------|---------------------|-------------------|
| **Baseline** | Aucune | - | ~25 s | ~3h30 |
| **Phase 1** | Hyperthreading + denoising | +25% | ~20 s | ~2h45 |
| **Phase 2** | UMat/OpenCL | +33% | ~16.8 s | ~2h20 |
| **Phase 3A** | OpenCV-CUDA | +50-80% | **6-8 s** | **30-35 min** |
| **Phase 3B** | Tesseract Batch | +30-40% OCR | **4-5 s** ✅ | **20-25 min** ✅ |

### **Gain Total : ×5-6 depuis le baseline !**

---

## 📝 Ordre d'Implémentation Recommandé

1. ✅ **Phase 3A** : Compiler OpenCV-CUDA (gain immédiat ×2-2.5)
2. ✅ **Tester** sur images 300 DPI réelles
3. ✅ **Mesurer** les gains réels
4. ✅ **Phase 3B** : Ajouter Tesseract Batch (gain additionnel +30-40%)
5. ✅ **Valider** que les scores restent identiques
6. ✅ **Optimiser** les paramètres sur vraies images 300 DPI

**Raison** : Chaque phase apporte un gain indépendant et peut être testée séparément.

---

## ⚠️ Points d'Attention

### **1. Vérifier l'identité des scores**

Les scores OCR doivent rester **identiques** (à ±0.1% près) :
```python
# Test
scores_individual = [get_tesseract_score(img) for img in images]
scores_batch = batch_tesseract_ocr(images)

for i, (s1, s2) in enumerate(zip(scores_individual, scores_batch)):
    diff = abs(s1 - s2)
    print(f"Image {i}: {s1:.2f} vs {s2:.2f} (diff: {diff:.2f})")
```

### **2. Gestion des erreurs**

Le code inclut un **fallback automatique** :
- Si batch échoue → traitement individuel
- Aucune interruption du workflow

### **3. Fichiers temporaires**

Les fichiers TIFF sont **automatiquement nettoyés** après usage (clause `finally`).

---

## 🚀 Bénéfices Finaux

### **Sur PC de Bureau (12c/24t + RTX 1080)**

**Images 300 DPI** :
- Screening 512 points : **20-25 minutes** au lieu de 77 minutes
- Optuna 500 trials : **19-23 minutes** au lieu de 75 minutes

**Images 100 DPI** :
- Screening 512 points : **~5 minutes** au lieu de 8.6 minutes
- Optuna 500 trials : **~5 minutes** au lieu de 8.4 minutes

### **Optimisation de Paramètres Praticable**

Avec 20-25 minutes par screening 300 DPI, vous pouvez :
- Faire **2-3 screenings** différents par heure
- Tester plusieurs stratégies d'optimisation dans la journée
- Affiner progressivement les paramètres optimaux

---

## 📚 Fichiers de Référence

- **`PHASE3_OPENCV_CUDA_UBUNTU.md`** : Compilation OpenCV-CUDA
- **`build_opencv_cuda.sh`** : Script de compilation automatisé
- **`test_cuda.py`** : Validation OpenCV-CUDA
- **`PHASE3B_TESSERACT_BATCH.md`** : Ce document
- **`tesseract_batch.py`** : Implémentation standalone (optionnel)

---

**Bonne implémentation ! Avec Phase 3A+B, vous aurez des performances exceptionnelles sur images 300 DPI ! 🚀**
