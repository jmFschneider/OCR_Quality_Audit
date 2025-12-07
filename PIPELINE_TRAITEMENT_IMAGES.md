# Pipeline de Traitement d'Images - Documentation

## 📋 Vue d'ensemble

Ce document décrit le pipeline complet de traitement d'images utilisé pour optimiser la qualité OCR.

### Point d'entrée : `pipeline_complet(image, params)` (ligne 390)

---

## 🔄 Étapes du Pipeline

### ÉTAPE 0 : Chargement initial

- **Fonction** : `ensure_gpu(image)` (ligne 165)
- **Action** : Charge l'image en mémoire GPU (GpuMat) si CUDA est activé
- **Paramètres** : Aucun
- **Mode** : GPU si CUDA activé, sinon CPU

---

### ÉTAPE 1 : Suppression des lignes horizontales et verticales 📐

- **Fonction** : `remove_lines_param()` (ligne 250)
- **Action** :
  1. Binarisation OTSU (inverse)
  2. Détection lignes horizontales avec morphologie MORPH_OPEN
  3. Détection lignes verticales avec morphologie MORPH_OPEN
  4. Fusion des deux masques
  5. Dilatation du masque
  6. Remplacement des pixels de lignes par du blanc (255)

- **Paramètres contrôlant cette étape** :
  - **`line_h_size`** : Largeur du kernel horizontal (ex: 30-70)
  - **`line_v_size`** : Hauteur du kernel vertical (ex: 40-120)
  - **`dilate_iter`** : Nombre d'itérations de dilatation (fixe à 2)

- **Kernels utilisés** :
  - Horizontal : `(line_h_size, 1)`
  - Vertical : `(1, line_v_size)`
  - Dilatation : `(3, 3)` avec `dilate_iter` itérations

- **Mode** : GPU (morphologie CUDA) si activé, sinon CPU

---

### ÉTAPE 2 : Normalisation par division 🔆

- **Fonction** : `normalisation_division()` (ligne 302)
- **Action** :
  1. Conversion en float32
  2. Flou gaussien pour extraire le fond
  3. Division de l'image par le fond (× 255 pour garder la plage)
  4. Reconversion en uint8

- **Paramètres contrôlant cette étape** :
  - **`norm_kernel`** : Taille du kernel gaussien (doit être impair, ex: 81, 101, 151)

- **Mode** :
  - GPU (GaussianFilter CUDA) si `norm_kernel ≤ 31`
  - CPU (fallback) si `norm_kernel > 31` (limitation CUDA)

---

### ÉTAPE 3 : Débruitage adaptatif 🔇

- **Fonction** : `adaptive_denoising()` (ligne 355)
- **Action** :
  1. Estimation du niveau de bruit (variance Laplacien)
  2. Sélection stratégie denoising :
     - **Bruit < threshold** → `searchWindowSize=15` (rapide)
     - **Bruit ≥ threshold** → `searchWindowSize=21` (qualité max)
  3. Application du denoising non-local means

- **Paramètres contrôlant cette étape** :
  - **`denoise_h`** : Force du denoising (0 = désactivé, typiquement 2.0-20.0)
  - **`noise_threshold`** : Seuil de décision pour la stratégie (typiquement 20-500)

- **Sous-fonction** : `estimate_noise_level()` (ligne 342) - calcule variance du Laplacien

- **Mode** :
  - Estimation bruit : CPU (rapide)
  - Denoising : **Toujours CPU** (pas d'équivalent CUDA performant pour fastNlMeansDenoising)
  - Retour sur GPU après si CUDA activé

---

### ÉTAPE 4 : Binarisation adaptative ⚫⚪

- **Fonction** : `cv2.adaptiveThreshold()` (ligne 411)
- **Action** :
  1. Transfert CPU si nécessaire
  2. Binarisation adaptative avec seuil gaussien
  3. Résultat : image en noir et blanc pur (0 ou 255)

- **Paramètres contrôlant cette étape** :
  - **`bin_block_size`** : Taille du voisinage pour le seuil adaptatif (impair, ex: 61, 101, 201)
  - **`bin_c`** : Constante soustraite à la moyenne (ex: 10-25)

- **Mode** : **Toujours CPU** (algorithme adaptatif complexe sans équivalent CUDA)

---

## 🎯 Schéma du Flux de Traitement

```
IMAGE ORIGINALE (grayscale)
        ↓
[0] ensure_gpu() → Chargement GPU si CUDA
        ↓
[1] remove_lines_param(line_h_size, line_v_size, dilate_iter)
    → Suppression lignes horizontales/verticales
        ↓
[2] normalisation_division(norm_kernel)
    → Normalisation de l'éclairage
        ↓
[3] adaptive_denoising(denoise_h, noise_threshold)
    → Réduction du bruit adaptative
        ↓
[4] adaptiveThreshold(bin_block_size, bin_c)
    → Binarisation adaptative
        ↓
IMAGE FINALE BINAIRE (0 ou 255)
```

---

## 📊 Tableau Récapitulatif des Paramètres

| **Paramètre** | **Étape** | **Rôle** | **Plage typique** | **GPU/CPU** |
|---------------|-----------|----------|-------------------|-------------|
| `line_h_size` | 1 | Largeur kernel lignes horizontales | 30-70 | GPU (si CUDA) |
| `line_v_size` | 1 | Hauteur kernel lignes verticales | 40-120 | GPU (si CUDA) |
| `dilate_iter` | 1 | Itérations de dilatation masque | 2 (fixe) | GPU (si CUDA) |
| `norm_kernel` | 2 | Taille kernel gaussien normalisation | 81-201 (impair) | GPU si ≤31, sinon CPU |
| `denoise_h` | 3 | Force du débruitage | 0-20 (0=off) | CPU |
| `noise_threshold` | 3 | Seuil stratégie denoising | 20-500 | CPU |
| `bin_block_size` | 4 | Taille voisinage binarisation | 61-201 (impair) | CPU |
| `bin_c` | 4 | Constante seuil adaptatif | 10-25 | CPU |

---

## ⏱️ Impact des Paramètres sur le Temps de Calcul

### 1. Paramètres avec impact MAJEUR ⚡

#### **`denoise_h` (2.0 - 20.0)** - IMPACT TRÈS ÉLEVÉ
- **Si `denoise_h = 0`** : Le denoising est complètement ignoré → gain de temps maximal
- **Si `denoise_h > 0`** : Le temps dépend du niveau de bruit détecté
  - **Bruit < `noise_threshold`** : `searchWindowSize=15` (optimisé, gain 30-40%)
  - **Bruit ≥ `noise_threshold`** : `searchWindowSize=21` (qualité max, plus lent)
- **Nature** : Opération CPU uniquement (pas d'équivalent CUDA performant)

#### **`noise_threshold` (20.0 - 500.0)** - IMPACT ÉLEVÉ
- **Valeur basse** (ex: 20) : Force le mode `searchWindowSize=21` (lent) même pour images peu bruitées
- **Valeur haute** (ex: 500) : Active le mode rapide `searchWindowSize=15` pour la majorité des images
- **Nature** : Contrôle indirect du temps de denoising
- **Conclusion** : Valeurs élevées = exécution plus rapide

---

### 2. Paramètres avec impact MODÉRÉ ⚙️

#### **`norm_kernel` (40 - 100, impair)** - IMPACT MODÉRÉ
- **Limitation CUDA** : `kernel_size <= 31` → traitement GPU ultra-rapide
- **Si `kernel_size > 31`** : Fallback CPU (plus lent)
  - Valeur 40→81, 50→101, etc. déclenche le fallback CPU
- **Nature** : Opération normalement GPU (GaussianBlur CUDA) sauf si > 31

#### **`line_h_size` et `line_v_size` (30-70 et 40-120)** - IMPACT FAIBLE À MODÉRÉ
- **Impact** : Taille des kernels morphologiques pour détecter les lignes
- **Valeurs élevées** : Kernels plus grands → légèrement plus lent (mais reste sur GPU si CUDA activé)
- **Nature** : Opérations morphologiques sur GPU si CUDA activé

---

### 3. Paramètres avec impact FAIBLE 🔹

#### **`bin_block_size` (30 - 100, impair)** - IMPACT FAIBLE
- **Nature** : Opération CPU uniquement (`adaptiveThreshold` n'a pas d'équivalent CUDA)
- **Impact** : Valeurs élevées augmentent légèrement le temps, mais reste rapide

#### **`bin_c` (10 - 25.0)** - IMPACT NÉGLIGEABLE
- **Nature** : Simple constante de soustraction dans `adaptiveThreshold`
- **Impact** : Aucun impact sur le temps de calcul
- **Conclusion** : Paramètre qualitatif uniquement

#### **`dilate_iter` (fixe à 2)** - IMPACT FAIBLE
- **Nature** : Nombre d'itérations de dilatation (opération GPU si CUDA)
- **Impact** : Proportionnel au nombre d'itérations, mais fixé à 2

---

### 4. Facteurs système ayant un impact MAJEUR 🚀

#### **Mode CUDA (GPU) vs CPU** - GAIN x2 à x5
- **GPU activé** : Traitement séquentiel (le GPU parallélise en interne)
- **CPU uniquement** : Multiprocessing avec `1.5 × nb_cores_physiques` workers
- **Exemple** : CPU 12c/24t → 18 workers en parallèle
- **Impact** : Le mode GPU est généralement plus rapide pour le pipeline image, mais le multiprocessing CPU compense sur plusieurs images

#### **Mode Debug/Timing** - OVERHEAD ~5-10%
- **Activé** : Mesure temps détaillées avec `time.time()` à chaque étape
- **Désactivé** : Exécution directe sans overhead de mesure
- **Fonction** : `process_image_data_fast` (production) vs `process_image_data_wrapper` (debug)

#### **Nombre d'images** - IMPACT LINÉAIRE
- Plus d'images = temps proportionnellement plus long
- Le multiprocessing CPU permet de paralléliser efficacement

---

## ✅ Recommandations pour optimiser le temps de calcul

### Pour minimiser le temps d'exécution :
1. **`denoise_h = 0`** : Désactiver complètement le denoising (gain majeur)
2. **`noise_threshold` élevé** (ex: 500) : Forcer le mode rapide même avec denoising
3. **`norm_kernel ≤ 31`** : Rester sur GPU CUDA (éviter fallback CPU)
4. **Désactiver "Debug/Timing"** : Utiliser `process_image_data_fast`
5. **GPU CUDA activé** : x2-x5 plus rapide que CPU (si disponible)

### Pour le plan d'étude :
- **Paramètres critiques temps** : `denoise_h`, `noise_threshold`, `norm_kernel`
- **Paramètres secondaires** : `line_h_size`, `line_v_size`, `bin_block_size`
- **Paramètres sans impact** : `bin_c`

---

## ⚙️ Version avec Timing (Debug Mode)

Si `ENABLE_DETAILED_TIMING = True`, la fonction `pipeline_complet_timed()` (ligne 417) mesure le temps de chaque étape :

```python
timings = {
    '1_line_removal': temps_ms,
    '2_normalization': temps_ms,
    '3_denoising': temps_ms,
    '4_binarization': temps_ms,
    'noise_level': valeur_bruit,
    'noise_threshold': seuil_config
}
```

### Exemple d'affichage (au premier traitement) :

```
--- Analyse détaillée des temps d'exécution (en ms, pour une image) ---
  - Niveau de bruit détecté: 123.45
  - Seuil de bruit configuré: 100.00
    → Stratégie: Denoising COMPLET (searchWindowSize=21)
  - Étape 1_line_removal: 45.23 ms (15.2%)
  - Étape 2_normalization: 67.89 ms (22.8%)
  - Étape 3_denoising: 156.78 ms (52.7%)
  - Étape 4_binarization: 27.45 ms (9.3%)
  - TEMPS TOTAL par image: 297.35 ms
```

---

## 📝 Résumé

Le pipeline se décompose en **4 étapes séquentielles** contrôlées par **8 paramètres**, avec un mix d'opérations GPU (étapes 1-2) et CPU (étapes 3-4) selon les capacités matérielles.

**L'objectif final** est d'obtenir une image binaire optimale pour l'OCR en :
1. Supprimant les artefacts (lignes)
2. Normalisant l'éclairage
3. Réduisant le bruit
4. Binarisant avec un seuil adaptatif

Le temps de calcul est principalement impacté par le **denoising** (étape 3) et le **mode GPU/CPU** utilisé.
