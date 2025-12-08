# OCR Quality Audit - Optimiseur de Pipeline d'Image

Outil d'optimisation de paramètres de prétraitement d'images pour améliorer la qualité de reconnaissance optique de caractères (OCR) avec Tesseract.

## 🚀 Caractéristiques principales

- **Package Python installable** : Utilisable comme bibliothèque dans d'autres projets
- **Architecture modulaire** : Code séparé en modules `pipeline.py`, `optimizer.py`, et `gui_main.py`
- **Accélération GPU CUDA** : Support natif NVIDIA (GTX 1080 Ti, RTX, etc.) pour le traitement d'images
- **Optimisation Sobol** : Screening quasi-Monte Carlo pour exploration efficace de l'espace des paramètres
- **Logging des temps** : Sauvegarde automatique des métriques de performance en CSV
- **Interface graphique** : GUI Tkinter intuitive avec sélecteur d'exposant Sobol (2^n)
- **Multi-plateforme** : Fonctionne sur Windows et Linux (Ubuntu) avec détection automatique CUDA

## 📊 Performance

### Mode GPU CUDA (GTX 1080 Ti)
- **Traitement d'image** : ~200ms par image (pipeline CUDA complet)
- **OCR Tesseract** : ~800ms par image (80% du temps total)
- **Gain vs CPU** : x2-3 sur le traitement d'images

### Estimations de temps
| Images | Points Sobol (2^n) | Temps estimé |
|--------|-------------------|--------------|
| 2      | 2^5 (32)          | ~1 min       |
| 24     | 2^7 (128)         | ~51 min      |
| 24     | 2^8 (256)         | ~1h42        |

## 🛠️ Installation

### Prérequis
- Python 3.8+
- CUDA Toolkit 11.x (optionnel, pour GPU)
- Tesseract OCR

### Installation du package

Le projet peut être installé comme package Python pour être utilisé dans d'autres projets :

```bash
# Cloner le repository
git clone https://github.com/jmFschneider/OCR_Quality_Audit.git
cd OCR_Quality_Audit

# Installer en mode éditable (développement)
pip install -e .

# OU installer avec les dépendances Windows
pip install -e ".[windows]"
```

### Installation pour utilisation standalone

#### Ubuntu/Linux
```bash
# 1. Cloner le repository
git clone https://github.com/jmFschneider/OCR_Quality_Audit.git
cd OCR_Quality_Audit

# 2. Installer les dépendances système
sudo apt install tesseract-ocr tesseract-ocr-fra python3-tk

# 3. Installer le package
pip install -e .

# 4. Pour support GPU (optionnel)
# Compiler OpenCV avec CUDA (voir docs/archive/ubuntu-migration/)
```

#### Windows
```bash
# 1. Installer Tesseract
# Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki

# 2. Cloner le repository
git clone https://github.com/jmFschneider/OCR_Quality_Audit.git
cd OCR_Quality_Audit

# 3. Installer le package avec dépendances Windows
pip install -e ".[windows]"
```

Note : Le chemin Tesseract est détecté automatiquement sur Windows et Linux.

## 🎯 Démarrage rapide

### Utilisation comme package Python

```python
# Importer les fonctions de traitement
from ocr_quality_audit import pipeline_complet, pipeline_blur_clahe
import cv2

# Charger et traiter une image
image = cv2.imread("scan.jpg", cv2.IMREAD_GRAYSCALE)

# Définir les paramètres
params = {
    'h_size': 50,
    'v_size': 2,
    'dilate_iter': 2,
    'kernel_size': 50,
    'base_h_param': 10,
    'block_size': 15,
    'C_constant': 10
}

# Traiter l'image
image_traitee = pipeline_complet(image, params)

# Évaluer les métriques
from ocr_quality_audit import evaluer_toutes_metriques
metriques = evaluer_toutes_metriques(image_traitee)
print(metriques)  # {'tesseract': 95.2, 'sharpness': 123.4, ...}
```

Voir **[USAGE_PACKAGE.md](USAGE_PACKAGE.md)** pour plus d'exemples d'utilisation.

### Utilisation de l'interface graphique

#### 1. Placer vos images
```bash
# Copier vos images JPG dans le dossier test_scans/
cp /path/to/images/*.jpg test_scans/
```

#### 2. Lancer l'interface graphique
```bash
python3 gui_main.py
```

#### 3. Utilisation de l'interface
1. Cliquer sur **"🔄 Rafraîchir"** pour détecter les images
2. Cliquer sur **"📥 Charger en mémoire"** pour précharger les images
3. Sélectionner les paramètres à optimiser (cocher/décocher)
4. Entrer l'exposant Sobol : **5** (pour 32 points, ~1-2 min)
5. Cliquer sur **"▶️ Lancer Sobol"**
6. Observer les résultats dans les logs

#### 4. Analyser les résultats
```bash
# Les résultats sont sauvegardés automatiquement :
# - screening_sobol_XXpts_YYYYMMDD_HHMMSS.csv (scores)
# - timing_log_YYYYMMDD_HHMMSS.csv (temps de traitement)

# Analyser les temps avec le script d'analyse
python3 tools/analyser_temps.py
```

## 📁 Structure du projet

```
OCR_Quality_Audit/
├── README.md                    # Ce fichier
├── USAGE_PACKAGE.md            # Guide d'utilisation du package
├── pyproject.toml              # Configuration du package Python
│
├── src/                        # Package Python installable
│   └── ocr_quality_audit/
│       ├── __init__.py         # API publique du package
│       ├── pipeline.py         # Pipeline de traitement d'images (CUDA)
│       ├── optimizer.py        # Algorithmes d'optimisation (Sobol, TimeLogger)
│       └── scipy_optimizer.py  # Optimisation scipy
│
├── gui_main.py                 # Interface graphique Tkinter (point d'entrée)
├── pipeline.py                 # Copie pour compatibilité (à la racine)
├── optimizer.py                # Copie pour compatibilité (à la racine)
│
├── tools/                      # Utilitaires
│   ├── analyser_temps.py       # Analyse des temps de traitement
│   └── tesseract_batch.py      # Traitement batch Tesseract
│
├── tests/                      # Scripts de test
│   ├── test_time_logging.py
│   ├── test_timing.py
│   ├── test_blur_clahe_timing.py
│   └── ...
│
├── docs/                       # Documentation complète
│   ├── user-guide/             # Guides utilisateur
│   ├── technical/              # Documentation technique
│   ├── changelogs/             # Historiques des modifications
│   └── archive/                # Documentation obsolète (référence)
│
└── test_scans/                 # Images à traiter (vos fichiers)
```

## 📚 Documentation

### Guide principal
- **[USAGE_PACKAGE.md](USAGE_PACKAGE.md)** : Guide complet d'utilisation du package Python

### Guides utilisateur
- **[Guide Sobol Screening](docs/user-guide/sobol-screening.md)** : Utilisation de l'optimisation Sobol
- **[Exposant Sobol (2^n)](docs/user-guide/sobol-exponent.md)** : Sélection du nombre de points
- **[Logging des temps](docs/user-guide/time-logging.md)** : Système de sauvegarde CSV des performances
- **[Mesure des temps](docs/user-guide/timing-measurement.md)** : Analyse détaillée des temps de traitement

### Documentation technique
- **[Résumé de modularisation](docs/technical/modularization-summary.md)** : Architecture et améliorations
- **[Corrections appliquées](docs/technical/CORRECTIONS_APPLIED.md)** : Historique des corrections

### Changelogs
- **[Exposant Sobol](docs/changelogs/sobol-exponent.md)** : Système 2^n
- **[Time Logging](docs/changelogs/time-logging.md)** : CSV logging
- **[Timing](docs/changelogs/timing.md)** : Mesure des temps

## 🧪 Tests

```bash
# Test du système de logging des temps
python3 tests/test_time_logging.py

# Test de mesure des temps
python3 tests/test_timing.py

# Test de l'exposant Sobol
python3 tests/test_sobol_exponent.py

# Test d'intégration complète
python3 tests/test_sobol_integration.py
```

## ⚙️ Configuration des paramètres

### Paramètres optimisables
| Paramètre | Rôle | Plage par défaut |
|-----------|------|------------------|
| `line_h_size` | Suppression lignes horizontales | 30-70 |
| `line_v_size` | Suppression lignes verticales | 40-120 |
| `norm_kernel` | Taille kernel normalisation | 40-100 |
| `denoise_h` | Force du denoising | 2.0-20.0 |
| `noise_threshold` | Seuil détection bruit | 20.0-500.0 |
| `bin_block_size` | Taille bloc binarisation | 30-100 |
| `bin_c` | Constante binarisation | 10-25 |

### Valeurs recommandées pour Sobol

**Pour exploration rapide (1-2 min):**
- Exposant : **5** → 32 points

**Pour exploration standard (4-5 min avec 2 images):**
- Exposant : **7** → 128 points

**Pour production (1-6h avec 24 images):**
- Exposant : **8-10** → 256-1024 points

## 🔧 Optimisations CUDA

Le pipeline utilise des opérations CUDA natives pour maximiser les performances GPU :

- `cv2.cuda.createGaussianFilter` (normalisation)
- `cv2.cuda.createMorphologyFilter` (suppression de lignes)
- `cv2.cuda.threshold` (binarisation)
- `cv2.cuda.createLaplacianFilter` (netteté, estimation du bruit)
- `cv2.cuda.divide` (normalisation par division)
- `cv2.cuda.meanStdDev` (métriques sans transfert CPU)

**Fallback automatique CPU** si CUDA non disponible.

## 📊 Système de logging

### Fichiers générés

**Scores et paramètres :**
```
screening_sobol_XXpts_YYYYMMDD_HHMMSS.csv
```
Colonnes : `point_id`, `score_tesseract_delta`, `score_tesseract`, `score_nettete`, `score_contraste`, paramètres testés

**Temps de traitement :**
```
timing_log_YYYYMMDD_HHMMSS.csv
```
Colonnes : `timestamp`, `point_id`, `image_id`, `temps_total_ms`, `temps_cuda_ms`, `temps_tesseract_ms`, `temps_sharpness_ms`, `temps_contrast_ms`, scores

### Analyse automatique
```bash
python3 tools/analyser_temps.py [fichier.csv]
```

Fournit :
- Statistiques globales (min, max, moyenne, médiane, écart-type)
- Répartition des temps en pourcentage
- Statistiques par point Sobol
- Statistiques par image
- Recommandations d'optimisation
- Estimations de temps pour différents volumes

## 🐛 Résolution de problèmes

### GPU CUDA non détecté
```bash
# Vérifier les devices CUDA
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

# Si 0 → OpenCV compilé sans CUDA
# Voir docs/archive/ubuntu-migration/ pour recompiler avec CUDA
```

### Erreur Tesseract
```python
# Vérifier l'installation
tesseract --version

# Vérifier le chemin dans gui_main.py (ligne ~25)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
```

### Images non chargées
```bash
# Vérifier le format (JPG uniquement)
ls test_scans/*.jpg

# Permissions de lecture
chmod +r test_scans/*.jpg
```

## 🤝 Contribution

Ce projet est en développement actif. Pour contribuer :
1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/amelioration`)
3. Commit les changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `docs/archive/old-md-files/LICENSE` pour plus de détails.

## 📧 Contact

Pour questions ou suggestions :
- GitHub Issues : https://github.com/jmFschneider/OCR_Quality_Audit/issues
- Email : [votre email]

## 🎓 Références

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV CUDA](https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html)
- [Sobol Sequences](https://en.wikipedia.org/wiki/Sobol_sequence)
- [Quasi-Monte Carlo](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html)

---

**Version** : 4.0 (Package Python installable + Architecture modulaire + CUDA)
**Dernière mise à jour** : 2025-12-08
