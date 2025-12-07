# Screening Sobol - Architecture Modulaire

## 📋 Vue d'ensemble

Le screening Sobol a été intégré avec succès dans la nouvelle architecture modulaire (pipeline.py, optimizer.py, gui_main.py).

## 🎯 Fonctionnalités

### Architecture modulaire
- **pipeline.py** : Traitement d'images avec support CUDA
- **optimizer.py** : Logique d'optimisation incluant le screening Sobol
- **gui_main.py** : Interface graphique

### Screening Sobol (Design of Experiments)
- Génération de points avec séquence Sobol (scipy.stats.qmc)
- Évaluation exhaustive de tous les points
- Sauvegarde automatique des résultats en CSV
- Support GPU CUDA pour accélération

## 🚀 Utilisation

### 1. Via l'interface graphique (gui_main.py)

```bash
python3 gui_main.py
```

**Étapes :**
1. Cliquer sur "🔄 Rafraîchir" pour détecter les images dans `test_scans/`
2. Cliquer sur "📥 Charger en mémoire" pour précharger les images
3. Configurer les paramètres à optimiser (cocher/décocher)
4. Entrer le nombre de points Sobol (ex: 32)
5. Cliquer sur "▶️ Lancer Sobol"

**Résultats :**
- Logs en temps réel dans l'interface
- Fichier CSV généré : `screening_sobol_XXpts_YYYYMMDD_HHMMSS.csv`
- Meilleurs paramètres affichés à la fin

### 2. Via script Python

```python
import optimizer
import cv2
from glob import glob
import numpy as np

# Charger les images
images = []
for f in glob("test_scans/*.jpg"):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        images.append(img.astype(np.uint8))

# Calculer les scores baseline
baseline_scores = optimizer.calculate_baseline_scores(images)

# Définir les ranges de paramètres
param_ranges = {
    'line_h': (30, 70),
    'line_v': (40, 120),
    'norm_kernel': (40, 100),
    'denoise_h': (2.0, 20.0),
    'noise_threshold': (20.0, 500.0),
    'bin_block': (30, 100),
    'bin_c': (10, 25.0)
}

fixed_params = {'dilate_iter': 2}

# Lancer le screening
best_params, csv_file = optimizer.run_sobol_screening(
    images=images,
    baseline_scores=baseline_scores,
    n_points=32,  # 2^5
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    callback=None,
    cancellation_event=None
)

print(f"Meilleurs paramètres: {best_params}")
print(f"Résultats dans: {csv_file}")
```

### 3. Test d'intégration

```bash
python3 test_sobol_integration.py
```

Ce script teste automatiquement :
- Détection CUDA
- Chargement des images
- Calcul des scores baseline
- Évaluation du pipeline
- Screening Sobol (8 points)
- Génération du CSV

## 📊 Format du fichier CSV

Le fichier CSV généré contient :

| Colonne | Description |
|---------|-------------|
| point_id | Numéro du point évalué (1 à n_points) |
| score_tesseract_delta | Gain OCR par rapport à l'image originale (%) |
| score_tesseract | Score OCR absolu (%) |
| score_nettete | Netteté (variance du Laplacien) |
| score_contraste | Contraste (écart-type) |
| line_h_size | Taille kernel horizontal (lignes) |
| line_v_size | Taille kernel vertical (lignes) |
| norm_kernel | Taille kernel normalisation (impair) |
| denoise_h | Paramètre h du denoising |
| noise_threshold | Seuil adaptatif du denoising |
| bin_block_size | Taille bloc binarisation (impair) |
| bin_c | Constante binarisation adaptative |

**Exemple :**
```csv
point_id;score_tesseract_delta;score_tesseract;score_nettete;score_contraste;line_h_size;norm_kernel;denoise_h
1;6.33;47.95;15557.92;63.91;49;153;9.55
2;6.14;47.76;15522.02;63.46;41;141;8.31
3;7.27;48.90;15440.46;63.60;43;157;9.31
```

## ⚡ Performance

### Mode GPU (CUDA activé)
- Traitement séquentiel sur GPU
- Accélération des opérations morphologiques
- Gain estimé : x2 à x5 par rapport au CPU

### Mode CPU (fallback)
- Multiprocessing automatique
- Utilisation optimale des cores (1.5x cores physiques)
- Exemple : 18 workers sur CPU 12c/24t

## 🔧 Optimisations

### Écriture CSV par lots
- Buffering de 50 points avant écriture
- Réduit les I/O disque
- Gain de performance : ~30%

### Logs console réduits
- Affichage tous les 50 points (sauf nouveaux records)
- Réduit l'overhead d'affichage
- Améliore les performances en mode batch

### Pre-resize Tesseract
- Images > 2500px redimensionnées à 50%
- Réduit la charge OCR
- Pas d'impact sur la qualité des résultats

## 🎛️ Paramètres recommandés

### Pour exploration rapide
- n_points = 32 (2^5)
- 2-3 paramètres actifs
- Temps estimé : 5-10 min sur GPU

### Pour exploration complète
- n_points = 128 à 256 (2^7 à 2^8)
- 5-7 paramètres actifs
- Temps estimé : 30-60 min sur GPU

### Pour screening exhaustif
- n_points = 512 à 1024 (2^9 à 2^10)
- Tous les paramètres actifs
- Temps estimé : 2-4h sur GPU

## 🐛 Dépannage

### "Aucune image trouvée"
- Vérifier que les images sont bien dans `test_scans/`
- Formats supportés : .jpg, .jpeg, .png
- Cliquer sur "🔄 Rafraîchir"

### "Aucun paramètre actif"
- Cocher au moins un paramètre dans l'interface
- Vérifier que Min < Max pour chaque paramètre

### Erreur multiprocessing
- Vérifier que `multiprocessing.set_start_method('spawn')` est appelé
- Sous Windows : pas de problème
- Sous Linux : nécessaire avec CUDA

### CSV incomplet
- Vérifier l'espace disque disponible
- Le buffer est vidé automatiquement à la fin
- En cas d'annulation, les points déjà évalués sont sauvegardés

## 📝 Notes techniques

### Séquence de Sobol
- Génération avec `scipy.stats.qmc.Sobol`
- Scramble=True pour meilleure couverture
- Scaling aux bornes définies par l'utilisateur

### Paramètres impairs (norm_kernel, bin_block)
- Valeur interne = base * 2 + 1
- Ex: base=75 → norm_kernel=151
- Garantit des valeurs impaires (requis par OpenCV)

### Multiprocessing vs GPU
- GPU : traitement séquentiel (le GPU parallélise en interne)
- CPU : multiprocessing avec pool de workers
- Détection automatique du mode optimal

## 🔗 Fichiers connexes

- `pipeline.py` : Pipeline de traitement d'images (196 lignes)
- `optimizer.py` : Logique d'optimisation Sobol (346 lignes)
- `gui_main.py` : Interface graphique (270 lignes)
- `test_sobol_integration.py` : Script de test automatique
- `sobol_test_pipeline.py` : Version standalone pour tests

## ✅ Tests

Le test d'intégration vérifie :
1. Détection CUDA
2. Chargement d'images
3. Calcul des scores baseline
4. Évaluation du pipeline
5. Screening Sobol (8 points)
6. Génération du CSV

**Résultat attendu :**
```
✅ GPU CUDA activé
✅ 2 images chargées
✅ Scores baseline calculés
✅ Pipeline testé avec succès
✅ Screening terminé! Meilleur gain: 7.27%
✅ CSV généré et vérifié
```
