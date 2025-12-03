# Changelog - Phase 2 : Optimisations GPU (UMat/OpenCL)

**Date** : 2025-11-27
**Branche** : `opti/vitesse_execussion_phase2`
**Gain estimé** : +10-15% sur les temps d'exécution

---

## 🚀 Optimisations Implémentées

### 1. Migration UMat/OpenCL

**Principe** : Utiliser `cv2.UMat` au lieu de `np.ndarray` pour permettre l'exécution sur GPU via OpenCL.

**Modifications** :

#### a) Détection et activation GPU
```python
USE_GPU = False
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    USE_GPU = True
```

#### b) Chargement des images en UMat (ligne 522-528)
```python
for f in self.image_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        if USE_GPU:
            img = cv2.UMat(img)  # Conversion en UMat pour GPU
        self.loaded_images.append(img)
```

#### c) Fonctions GPU-aware
Toutes les fonctions principales du pipeline acceptent maintenant `UMat` ou `numpy array` :

- **`remove_lines_param()`** : Opérations morphologiques sur GPU
- **`normalisation_division()`** : GaussianBlur et divide sur GPU
- **`estimate_noise_level()`** : Laplacian sur GPU
- **`adaptive_denoising()`** : Gestion intelligente CPU/GPU (fastNlMeans en CPU, reconversion en UMat après)
- **`pipeline_complet()`** : Pipeline complet en UMat
- **`pipeline_complet_timed()`** : Idem avec mesures de temps
- **`get_sharpness()`** : Laplacian sur GPU
- **`get_contrast()`** : Compatible UMat
- **`get_tesseract_score()`** : Conversion UMat→numpy pour Tesseract uniquement

---

### 2. Opérations GPU-Accelerated

Les opérations suivantes bénéficient de l'accélération GPU lorsque disponible :

| Opération | Fonction OpenCV | Étape du pipeline |
|-----------|-----------------|-------------------|
| **GaussianBlur** | `cv2.GaussianBlur()` | Normalisation (fond estimé) |
| **Threshold** | `cv2.threshold()` | Binarisation OTSU (suppression lignes) |
| **MorphologyEx** | `cv2.morphologyEx()` | Détection lignes horizontales/verticales |
| **Dilate** | `cv2.dilate()` | Expansion du masque de lignes |
| **AddWeighted** | `cv2.addWeighted()` | Fusion masques H+V |
| **Divide** | `cv2.divide()` | Normalisation par division |
| **AdaptiveThreshold** | `cv2.adaptiveThreshold()` | Binarisation finale |
| **Laplacian** | `cv2.Laplacian()` | Estimation bruit + netteté |

---

### 3. Pre-resize Tesseract (déjà présent Phase 1)

Pour les images de largeur > 2500px, un resize 0.5× est appliqué avant l'OCR pour réduire la charge Tesseract.

```python
if image.shape[1] > 2500:
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
```

**Gain estimé** : +3-5% sur grandes images

---

## 📊 Stratégie d'Optimisation

### Flux GPU optimal

```
Chargement image (CPU)
        ↓
Conversion UMat (GPU)
        ↓
Pipeline complet (GPU)
  ├─ Suppression lignes
  ├─ Normalisation
  ├─ Denoising (CPU, puis UMat)
  └─ Binarisation
        ↓
Conversion numpy (CPU) uniquement pour Tesseract
        ↓
OCR Tesseract (CPU)
```

**Minimisation des transferts CPU↔GPU** : Les images restent en mémoire GPU tout le long du pipeline, sauf pour Tesseract.

---

## 🔧 Compatibilité

### Mode GPU (avec OpenCL)
- **Requis** : Carte graphique compatible OpenCL (NVIDIA, AMD, Intel)
- **Activation automatique** : Si `cv2.ocl.haveOpenCL()` retourne `True`
- **Message de confirmation** :
  ```
  🚀 PHASE 2 - OPTIMISATIONS GPU ACTIVÉES
  ✅ OpenCL activé pour OpenCV (accélération GPU UMat)
  ```

### Mode CPU (fallback)
- **Activation** : Si OpenCL non disponible
- **Comportement** : Le code fonctionne exactement comme avant (numpy arrays)
- **Message** : `⚠️ OpenCL non disponible - Mode CPU uniquement`
- **Performance** : Identique à Phase 1

---

## 🧪 Tests Recommandés

### Test 1 : Vérifier l'activation GPU

```bash
python gui_optimizer_v3_ultim.py
```

**Attendu** : Message de démarrage indiquant si GPU est activé ou non.

### Test 2 : Screening 64 points avec mesure de temps

```
Mode: Screening
Exposant Sobol: 6 (64 points)
Tous les paramètres cochés
LANCER
```

**Comparer** :
- Temps total Phase 1 (CPU) vs Phase 2 (GPU)
- Temps par étape dans le log détaillé

### Test 3 : Vérifier que les scores sont identiques

**Important** : Les optimisations GPU ne doivent PAS changer les résultats numériques (scores OCR).

- Lancer le même screening avec Phase 1 et Phase 2
- Comparer les CSV générés
- Les scores doivent être identiques (à ±0.1% près dû aux approximations flottantes)

---

## 📈 Gains Attendus

### Configuration testée
- **PC** : 12 cores / 24 threads + RTX 1080
- **Images** : 24 images de test

### Estimations

| Composant | Gain Phase 2 | Temps avant | Temps après |
|-----------|--------------|-------------|-------------|
| **Opérations OpenCV** | **+10-15%** | ~500ms/trial | ~425ms/trial |
| **Tesseract (pre-resize)** | +3-5% | ~800ms/trial | ~760ms/trial |
| **Total par trial** | **+8-12%** | ~1500ms | ~1320ms |

**Pour 512 points (screening)** :
- Phase 1 : ~12.8 min
- Phase 2 : ~11.3 min (**gain de ~1.5 min**)

**Pour 500 trials (Optuna)** :
- Phase 1 : ~12.5 min
- Phase 2 : ~11.0 min (**gain de ~1.5 min**)

---

## ⚠️ Limitations Connues et Solutions

1. **fastNlMeansDenoising** : Ne supporte pas UMat dans toutes les versions OpenCV
   - **Solution** : Conversion temporaire en numpy, puis reconversion en UMat

2. **Tesseract** : Nécessite numpy array (pas UMat)
   - **Solution** : Conversion UMat→numpy juste avant l'appel Tesseract

3. **Multiprocessing** : UMat ne peut pas être sérialisé (pickle) pour multiprocessing
   - **Solution** : Images chargées en numpy, conversion UMat dans chaque worker
   - **Impact** : Léger overhead de conversion, mais gain GPU reste positif

4. **UMat.copy()** : Méthode inexistante sur cv2.UMat
   - **Solution** : Utiliser `umat.get().copy()` puis reconvertir en UMat

5. **Overhead de conversion** : Sur de très petites images (<500×500), le gain peut être négligeable

---

## 🔍 Analyse de Performance

### Pour mesurer le gain réel :

1. **Lancer un screening identique** en Phase 1 et Phase 2
2. **Comparer les temps** :
   ```python
   # Dans le log détaillé (première image)
   - Étape 1_line_removal: XX ms
   - Étape 2_normalization: XX ms
   - Étape 4_binarization: XX ms
   - TEMPS TOTAL par image: XX ms
   ```
3. **Calculer le speedup** :
   ```
   Speedup = Temps_Phase1 / Temps_Phase2
   Gain_pourcentage = (1 - Temps_Phase2/Temps_Phase1) × 100%
   ```

---

## 📝 Fichiers Modifiés

- **`gui_optimizer_v3_ultim.py`** : Intégration complète UMat/OpenCL
  - Lignes 1-25 : Documentation header
  - Lignes 33-50 : Détection et activation OpenCL
  - Lignes 52-234 : Fonctions GPU-aware
  - Lignes 509-533 : Chargement images en UMat

---

## 🎯 Prochaines Étapes (Phase 3 potentielle)

Si un gain supplémentaire est nécessaire :

1. **Batch Tesseract** : Grouper les 24 images en 1 appel (+5-10%)
2. **Cache preprocessing** : Mémoriser étapes 1-2 pour images récurrentes (+20-30%)
3. **Tesseract CUDA** : Version GPU de Tesseract (+10-15%, si compilation possible)

**Gain cumulé Phase 3** : +30-40% supplémentaires

---

**Note** : Cette implémentation est **100% compatible** avec les PC sans GPU (fallback automatique sur CPU).
