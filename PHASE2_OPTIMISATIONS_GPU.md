# Phase 2 : Optimisations GPU et Avancées

## 📋 Contexte

**Phase 1 (COMPLÉTÉE)** :
- ✅ Hyperthreading optimisé (×1.5 cores)
- ✅ Denoising adaptatif
- ✅ Pré-chargement images en mémoire
- ✅ **Gain : -25% du temps total**

**Phase 2 (PLANIFIÉE)** :
- 🎯 Utilisation GPU (OpenCV CUDA/OpenCL)
- 🎯 Batch processing Tesseract
- 🎯 Optimisations mémoire avancées
- 🎯 **Objectif : Gain additionnel de -30 à -40%**

---

## 🎯 Objectifs de la Phase 2

### **Réduction du temps par image :**
- **Actuel** : ~1.5s par trial (24 images en parallèle)
- **Cible Phase 2** : ~1.0s par trial
- **Gain cumulé Phases 1+2** : **-50 à -60%** du temps initial

### **Temps d'optimisation projetés :**

| Scénario | Temps Actuel | Temps Cible Phase 2 |
|----------|--------------|---------------------|
| Screening 512 pts | 12.8 min | **8.5 min** |
| Screening 1024 pts | 25.6 min | **17 min** |
| Optuna 500 trials | 12.5 min | **8.3 min** |

---

## 🚀 Optimisations Prévues

### **1. Migration vers UMat (OpenCL GPU)** ⭐ PRIORITÉ 1

#### **Principe :**
- Utiliser `cv2.UMat` au lieu de `np.ndarray` pour les images
- Les opérations OpenCV utilisent automatiquement le GPU si disponible
- Transparent pour le code (API identique)

#### **Implémentation :**

**Modifications dans `gui_optimizer_v3_ultim.py` :**

```python
# Ligne 326 - Pré-chargement des images
def pre_load_images(self):
    self.update_log_from_thread("Pré-chargement des images en mémoire (GPU)...")
    self.loaded_images = []

    for f in self.image_files:
        # Charger en UMat pour GPU
        img_cpu = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_cpu is not None:
            img_gpu = cv2.UMat(img_cpu)  # Transfert vers GPU
            self.loaded_images.append(img_gpu)

    self.update_log_from_thread(f"{len(self.loaded_images)} images chargées en GPU memory.")
```

**Avantages :**
- ✅ GaussianBlur accéléré (normalisation)
- ✅ Morphological operations accélérées (line removal)
- ✅ adaptiveThreshold accéléré (binarisation)
- ✅ **Aucun changement de code** dans le pipeline (API identique)

**Limitations :**
- ⚠️ `fastNlMeansDenoising` : Peut ne pas être accéléré selon la version OpenCV
- ⚠️ Tesseract ne supporte pas UMat → Conversion nécessaire avant OCR

#### **Code détaillé :**

```python
def pipeline_complet(image, params):
    # image est déjà un UMat (GPU)

    # Étape 1 : Line removal (GPU)
    no_lines = remove_lines_param(image, params['line_h_size'],
                                   params['line_v_size'], params['dilate_iter'])

    # Étape 2 : Normalisation (GPU)
    norm = normalisation_division(no_lines, params['norm_kernel'])

    # Étape 3 : Denoising (CPU ou GPU selon implémentation OpenCV)
    denoised = adaptive_denoising(norm, params['denoise_h'],
                                   params.get('noise_threshold', 100))

    # Étape 4 : Binarisation (GPU)
    binarized = cv2.adaptiveThreshold(denoised, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY,
                                      params['bin_block_size'], params['bin_c'])

    # Conversion UMat → numpy pour Tesseract
    return binarized.get()  # Transfert GPU → CPU
```

**Gain estimé :** **+15-20%** sur les étapes 1, 2, 4

---

### **2. Optimisation Tesseract** ⭐ PRIORITÉ 2

#### **Option A : Batch Processing (RECOMMANDÉ)**

**Principe :**
- Grouper les 24 images en un seul appel Tesseract
- Réduire l'overhead de démarrage de Tesseract

**Implémentation :**

```python
def process_images_batch(images_list, params):
    """
    Traite un batch d'images en une seule fois.
    Réduit l'overhead de Tesseract (démarrage, chargement modèle).
    """
    # Preprocessing de toutes les images
    processed_images = []
    for img in images_list:
        processed = pipeline_complet(img, params)
        processed_images.append(processed)

    # Créer un "pseudo-PDF" multi-pages en mémoire
    # ou appeler Tesseract une seule fois avec toutes les images

    # Méthode 1 : Fichier temporaire multi-pages
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
        # Sauvegarder toutes les images en TIFF multi-pages
        cv2.imwritemulti(tmp.name, processed_images)

        # Un seul appel Tesseract
        data = pytesseract.image_to_data(tmp.name, output_type=pytesseract.Output.DICT)

        # Parser les résultats par page
        scores = parse_multipage_results(data)

    return scores
```

**Gain estimé :** **+5-10%** (réduction overhead)

**Complexité :** Moyenne (gestion multi-pages)

---

#### **Option B : Réduction de résolution intelligente**

**Principe :**
- Redimensionner les images UNIQUEMENT pour Tesseract
- Garder résolution native pour preprocessing

**Implémentation :**

```python
def get_tesseract_score(image):
    """Version optimisée avec resize systématique."""
    # Tesseract est plus rapide sur images < 2000px de largeur
    h, w = image.shape[:2]

    if w > 2000:
        scale = 2000 / w
        resized = cv2.resize(image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    else:
        resized = image

    try:
        data = pytesseract.image_to_data(resized, config='--oem 1 --psm 6',
                                         output_type=pytesseract.Output.DICT)
        confs = [int(x) for x in data['conf'] if int(x) != -1]
        return sum(confs) / len(confs) if confs else 0
    except:
        return 0
```

**Gain estimé :** **+10-15%** si images > 2000px
**Impact OCR :** Minimal (<1-2% de dégradation)

---

#### **Option C : Tesseract GPU (CUDA)** 🔥 MAXIMUM

**Principe :**
- Compiler Tesseract avec support CUDA
- OCR accéléré par GPU

**Implémentation :**
```bash
# Installation (complexe)
# 1. Installer CUDA Toolkit 11.x
# 2. Compiler Tesseract from source avec flag CUDA
# 3. Compiler pytesseract compatible

# Configuration
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
export OMP_THREAD_LIMIT=1  # Important pour ne pas confliter avec multiprocessing
```

**Gain estimé :** **+60-70%** sur l'étape OCR (857ms → 260ms)
**Complexité :** **TRÈS ÉLEVÉE** (compilation, compatibilité)

**Recommandation :** **Éviter sauf besoin critique** (gain Phase 2A+2B suffit)

---

### **3. Cache Preprocessing (Optionnel)** ⭐ PRIORITÉ 3

#### **Principe :**
- Si les mêmes images sont utilisées, cacher les résultats du preprocessing
- Utile si vous optimisez UNIQUEMENT `bin_c` par exemple

**Implémentation :**

```python
class PreprocessingCache:
    def __init__(self):
        self.cache = {}  # {(img_id, params_hash): processed_img}

    def get_or_compute(self, img_id, params, compute_fn):
        # Hash des paramètres qui affectent le preprocessing
        cache_params = {
            'line_h_size': params['line_h_size'],
            'line_v_size': params['line_v_size'],
            'norm_kernel': params['norm_kernel'],
            'denoise_h': params['denoise_h'],
            'noise_threshold': params.get('noise_threshold', 100)
        }

        key = (img_id, hash(frozenset(cache_params.items())))

        if key not in self.cache:
            self.cache[key] = compute_fn()

        return self.cache[key]
```

**Cas d'usage :**
- Optimisation de `bin_c` et `bin_block_size` UNIQUEMENT
- Les étapes 1-3 (line removal, normalisation, denoising) sont cachées

**Gain estimé :** **+20-30%** si applicable
**Limitation :** Seulement si paramètres preprocessing fixés

---

### **4. Optimisations Mémoire** ⭐ PRIORITÉ 4

#### **A. Pool size dynamique selon RAM disponible**

```python
import psutil

def get_optimal_pool_size(n_images):
    # Mesurer RAM disponible
    available_ram = psutil.virtual_memory().available / (1024**3)  # GB

    # Estimer RAM par worker (image + processing)
    ram_per_worker = 0.5  # GB (à ajuster selon vos images)

    max_workers_by_ram = int(available_ram / ram_per_worker)
    max_workers_by_cpu = int(os.cpu_count() * 1.5)

    optimal = min(n_images, max_workers_by_ram, max_workers_by_cpu)

    print(f"Pool size optimal: {optimal} (RAM: {max_workers_by_ram}, CPU: {max_workers_by_cpu})")

    return optimal
```

**Gain :** Évite les swaps mémoire (stabilité)

---

#### **B. Libération mémoire explicite**

```python
import gc

def process_image_data_wrapper(args):
    # ... traitement ...

    result = (score_tess, score_sharp, score_cont)

    # Libération explicite
    del processed_img, timings
    gc.collect()

    return result
```

**Gain :** Réduit la pression mémoire (utile si beaucoup d'images)

---

## 📊 Récapitulatif des Gains Estimés

| Optimisation | Gain Estimé | Complexité | Recommandation |
|--------------|-------------|------------|----------------|
| **UMat (OpenCL)** | +15-20% | Faible | ⭐⭐⭐ OUI |
| **Batch Tesseract** | +5-10% | Moyenne | ⭐⭐ Si besoin |
| **Resize Tesseract** | +10-15% | Faible | ⭐⭐⭐ OUI |
| **Tesseract CUDA** | +60-70% (OCR) | Très élevée | ❌ NON (overkill) |
| **Cache Preprocessing** | +20-30% | Moyenne | ⭐ Cas spécifique |
| **Pool dynamique RAM** | Stabilité | Faible | ⭐⭐ OUI |

### **Combinaison Recommandée (Phase 2A) :**
✅ UMat (OpenCL)
✅ Resize Tesseract
✅ Pool dynamique RAM

**Gain cumulé attendu :** **-30%**
**Complexité :** Faible à moyenne
**Temps de dev :** 1-2 jours

---

## 🛠️ Plan d'Implémentation

### **Étape 1 : Tests Préliminaires**
1. Vérifier que OpenCL fonctionne :
   ```python
   import cv2
   print(f"OpenCL disponible : {cv2.ocl.haveOpenCL()}")
   print(f"Device : {cv2.ocl.Device.getDefault().name()}")
   ```

2. Benchmark UMat vs numpy :
   ```python
   import time
   img_cpu = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
   img_gpu = cv2.UMat(img_cpu)

   # Test GaussianBlur
   t0 = time.time()
   blurred_cpu = cv2.GaussianBlur(img_cpu, (51, 51), 0)
   print(f"CPU : {(time.time()-t0)*1000:.2f}ms")

   t0 = time.time()
   blurred_gpu = cv2.GaussianBlur(img_gpu, (51, 51), 0)
   result = blurred_gpu.get()  # Force GPU sync
   print(f"GPU : {(time.time()-t0)*1000:.2f}ms")
   ```

### **Étape 2 : Migration Progressive**
1. **Jour 1** : UMat pour pré-chargement uniquement (test)
2. **Jour 2** : Resize Tesseract systématique
3. **Jour 3** : Pool dynamique RAM
4. **Jour 4** : Tests et mesures de gains réels

### **Étape 3 : Validation**
- Comparer scores OCR avant/après (doivent être identiques ±1%)
- Mesurer temps par trial (gain attendu ~30%)
- Vérifier stabilité (pas de crashes mémoire)

---

## ⚠️ Points d'Attention

### **UMat et Multiprocessing :**
```python
# ATTENTION : UMat n'est PAS pickle-able
# Il faut convertir en numpy avant de passer au pool

def pre_load_images(self):
    # Charger en numpy (CPU)
    for f in self.image_files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        self.loaded_images.append(img)

    # Dans le worker, convertir en UMat
def process_image_data_wrapper(args):
    img_cpu, params = args
    img_gpu = cv2.UMat(img_cpu)  # Conversion dans chaque worker
    # ... traitement avec img_gpu ...
```

### **Tesseract et UMat :**
```python
# Tesseract ne supporte PAS UMat
# Conversion obligatoire avant OCR
processed_umat = pipeline_complet(img_gpu, params)
processed_numpy = processed_umat.get()  # UMat → numpy
score = get_tesseract_score(processed_numpy)
```

---

## 🎓 Ressources et Références

### **Documentation OpenCV :**
- UMat : https://docs.opencv.org/4.x/d7/d60/classcv_1_1UMat.html
- OpenCL : https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html

### **Benchmarks GPU vs CPU :**
- GaussianBlur : 2-3× plus rapide sur GPU
- morphologyEx : 1.5-2× plus rapide
- adaptiveThreshold : 1.5-2× plus rapide
- fastNlMeansDenoising : Peut être plus LENT sur GPU (selon version)

### **Hardware Compatible :**
- ✅ NVIDIA (CUDA + OpenCL)
- ✅ AMD (OpenCL)
- ✅ Intel integrated GPU (OpenCL)

---

## 📋 Checklist Phase 2

### **Avant de commencer :**
- [ ] Vérifier OpenCL disponible (`cv2.ocl.haveOpenCL()`)
- [ ] Benchmarker 1 opération GPU vs CPU
- [ ] Sauvegarder le code actuel (branch)

### **Implémentation :**
- [ ] Migrer pré-chargement vers UMat
- [ ] Ajouter conversion UMat dans workers
- [ ] Implémenter resize systématique Tesseract
- [ ] Ajouter pool size dynamique (RAM)

### **Tests :**
- [ ] Vérifier scores OCR identiques
- [ ] Mesurer gains réels sur 64 points
- [ ] Vérifier utilisation GPU (nvidia-smi ou radeontop)
- [ ] Tester avec 1024 points (stabilité)

### **Documentation :**
- [ ] Mettre à jour gemini.md
- [ ] Ajouter notes de performance
- [ ] Documenter les gains obtenus

---

## 🎯 Décision : Implémenter ou Non ?

### **Phase 2 est OPTIONNELLE si :**
- ✅ Phase 1 suffit (12-13 min pour 512 points acceptable)
- ✅ Pas besoin de screenings > 1024 points régulièrement
- ✅ Complexité ajoutée pas justifiée

### **Phase 2 est RECOMMANDÉE si :**
- 🎯 Besoin de screenings fréquents (plusieurs par jour)
- 🎯 Besoin de screenings larges (2^11 = 2048 points)
- 🎯 GPU disponible mais sous-utilisé
- 🎯 Recherche du temps minimal absolu

---

## 💡 Conclusion

**Phase 1 SEULE** apporte déjà **-25%** de gain → **EXCELLENT**

**Phase 2** peut ajouter **-30%** supplémentaires avec effort modéré

**Recommandation** :
1. **Tester Phase 1** sur PC de bureau avec screening 512 points
2. **Mesurer si le temps actuel est acceptable** pour votre usage
3. **Décider ensuite** si Phase 2 vaut l'investissement

**Mon avis** : Phase 1 suffit largement pour la plupart des cas d'usage ! 🎯

---

*Document créé le 2025-11-27 - Référence pour implémentation future*
