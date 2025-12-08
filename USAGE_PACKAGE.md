# Utilisation du Package OCR Quality Audit

Ce document explique comment utiliser `ocr_quality_audit` comme package Python installable dans d'autres projets.

## Installation

### Sur votre poste de développement (Windows)

```bash
cd /chemin/vers/OCR_Quality_Audit
pip install -e .
```

### Sur votre poste de calcul (Ubuntu avec CUDA)

```bash
cd /chemin/vers/OCR_Quality_Audit
pip install -e .
```

**Note**: L'installation en mode éditable (`-e`) permet de modifier le code source sans réinstaller le package.

## Utilisation dans un autre projet

### Exemple 1: Traiter une image avec un pipeline complet

```python
from ocr_quality_audit import pipeline_complet
import cv2

# Charger une image
image = cv2.imread("mon_image.png", cv2.IMREAD_GRAYSCALE)

# Définir les paramètres de traitement
params = {
    'h_size': 50,           # Taille du kernel horizontal pour suppression de lignes
    'v_size': 2,            # Taille du kernel vertical
    'dilate_iter': 2,       # Nombre d'itérations de dilatation
    'kernel_size': 50,      # Taille du kernel de normalisation
    'base_h_param': 10,     # Paramètre de débruitage
    'block_size': 15,       # Taille du bloc pour binarisation adaptative
    'C_constant': 10        # Constante pour binarisation adaptative
}

# Traiter l'image
image_traitee = pipeline_complet(image, params)

# Sauvegarder le résultat
cv2.imwrite("image_traitee.png", image_traitee)
```

### Exemple 2: Pipeline Blur+CLAHE (High Fidelity)

```python
from ocr_quality_audit import pipeline_blur_clahe
import cv2

# Charger une image
image = cv2.imread("mon_image.png", cv2.IMREAD_GRAYSCALE)

# Paramètres Blur+CLAHE
params = {
    'blur_ksize': 5,        # Taille du kernel de flou (impair)
    'clahe_clip': 2.0,      # Limite de contraste CLAHE
    'clahe_grid': 8         # Taille de la grille CLAHE
}

# Traiter l'image
image_traitee = pipeline_blur_clahe(image, params)
```

### Exemple 3: Évaluer les métriques d'une image

```python
from ocr_quality_audit import (
    get_sharpness,
    get_contrast,
    get_tesseract_score,
    evaluer_toutes_metriques
)
import cv2

# Charger une image
image = cv2.imread("mon_image.png", cv2.IMREAD_GRAYSCALE)

# Évaluer individuellement
sharpness = get_sharpness(image)
contrast = get_contrast(image)
ocr_score = get_tesseract_score(image)

print(f"Netteté: {sharpness}")
print(f"Contraste: {contrast}")
print(f"Score OCR: {ocr_score}")

# Ou évaluer toutes les métriques d'un coup
metriques = evaluer_toutes_metriques(image)
print(metriques)
# {'tesseract': 95.2, 'sharpness': 123.4, 'contrast': 78.9, 'cnr': 45.6}
```

### Exemple 4: Traitement batch avec accélération

```python
from ocr_quality_audit import evaluer_toutes_metriques_batch
import cv2

# Charger plusieurs images
images = [
    cv2.imread("image1.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("image2.png", cv2.IMREAD_GRAYSCALE),
    cv2.imread("image3.png", cv2.IMREAD_GRAYSCALE),
]

# Évaluer en parallèle (utilise tous les CPU cores)
resultats = evaluer_toutes_metriques_batch(images, verbose=True)

for i, metrics in enumerate(resultats):
    print(f"Image {i+1}: {metrics}")
```

### Exemple 5: Utilisation dans votre visualisateur d'espace

```python
# Dans votre projet de visualisation d'espace
from ocr_quality_audit import pipeline_complet, USE_CUDA
import cv2
import numpy as np

class ImageProcessor:
    """Processeur d'images pour la visualisation d'espace d'optimisation."""

    def __init__(self):
        self.use_cuda = USE_CUDA
        print(f"Processeur initialisé avec CUDA: {self.use_cuda}")

    def process_point(self, image_path, point_params):
        """
        Traite une image avec les paramètres d'un point de l'espace.

        Args:
            image_path: Chemin vers l'image
            point_params: Dict des paramètres du point (h_size, v_size, etc.)

        Returns:
            L'image traitée
        """
        # Charger l'image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Traiter avec les paramètres du point
        image_traitee = pipeline_complet(image, point_params)

        return image_traitee

    def visualize_processing_steps(self, image, params):
        """Visualise les étapes de traitement pour un point intéressant."""
        # Import des fonctions intermédiaires
        from ocr_quality_audit import (
            remove_lines_param,
            normalisation_division,
            adaptive_denoising
        )

        steps = []

        # Étape 1: Suppression de lignes
        step1 = remove_lines_param(
            image,
            params['h_size'],
            params['v_size'],
            params['dilate_iter']
        )
        steps.append(("Suppression lignes", step1))

        # Étape 2: Normalisation
        step2 = normalisation_division(step1, params['kernel_size'])
        steps.append(("Normalisation", step2))

        # Étape 3: Débruitage
        step3 = adaptive_denoising(step2, params['base_h_param'])
        steps.append(("Débruitage", step3))

        return steps


# Utilisation
processor = ImageProcessor()

# Traiter un point intéressant identifié par l'optimiseur
params_interessant = {
    'h_size': 45,
    'v_size': 3,
    'dilate_iter': 2,
    'kernel_size': 55,
    'base_h_param': 8,
    'block_size': 17,
    'C_constant': 12
}

image_traitee = processor.process_point("scan.png", params_interessant)

# Afficher les étapes de traitement
steps = processor.visualize_processing_steps(
    cv2.imread("scan.png", cv2.IMREAD_GRAYSCALE),
    params_interessant
)

for nom, image_step in steps:
    cv2.imshow(nom, image_step)
    cv2.waitKey(0)
```

## Fonctions disponibles

### Pipelines complets
- `pipeline_complet(image, params)` - Pipeline standard avec suppression lignes, normalisation, débruitage
- `pipeline_complet_timed(image, params)` - Même chose avec mesure de temps détaillée
- `pipeline_blur_clahe(image, params)` - Pipeline High Fidelity (Blur + CLAHE)

### Fonctions de traitement
- `remove_lines_param(img, h_size, v_size, dilate_iter)` - Suppression de lignes
- `normalisation_division(image_gray, kernel_size)` - Normalisation par division
- `adaptive_denoising(image, base_h_param, noise_threshold)` - Débruitage adaptatif

### Métriques
- `get_sharpness(image)` - Calcule la netteté (variance du Laplacien)
- `get_contrast(image)` - Calcule le contraste (écart-type)
- `get_tesseract_score(image)` - Score OCR Tesseract (0-100)
- `get_cnr_quality(image)` - CNR (Contrast-to-Noise Ratio)
- `evaluer_toutes_metriques(image)` - Toutes les métriques en un appel
- `evaluer_toutes_metriques_batch(images, max_workers, verbose)` - Batch avec parallélisation

### Utilitaires
- `ensure_gpu(image)` - Upload image sur GPU si CUDA disponible
- `ensure_cpu(image)` - Download image du GPU si nécessaire
- `USE_CUDA` - Booléen indiquant si CUDA est activé

## Configuration

### Variables d'environnement
Le package détecte automatiquement:
- Tesseract OCR (Windows et Linux)
- Support CUDA (si OpenCV compilé avec CUDA)

### Support multi-plateforme
Le package fonctionne sur:
- **Windows**: OpenCV standard (pip)
- **Ubuntu avec CUDA**: OpenCV compilé avec support CUDA
- **macOS**: OpenCV standard (pip)

Le même code fonctionne partout - le package détecte automatiquement les capacités GPU.

## Structure du projet

```
OCR_Quality_Audit/
├── src/
│   └── ocr_quality_audit/      # Package installable
│       ├── __init__.py          # Exports publics
│       ├── pipeline.py          # Fonctions de traitement
│       ├── optimizer.py         # Algorithmes d'optimisation
│       └── scipy_optimizer.py   # Optimisation scipy
├── gui_main.py                  # Point d'entrée GUI
├── tools/                       # Utilitaires (analyse, batch)
├── tests/                       # Tests unitaires
├── pyproject.toml              # Configuration du package
└── requirements.txt            # Dépendances
```

## Dépannage

### Import Error
Si vous obtenez une erreur d'import:
```bash
# Vérifiez que le package est installé
pip list | grep ocr-quality-audit

# Réinstallez si nécessaire
pip install -e /chemin/vers/OCR_Quality_Audit
```

### CUDA non détecté sur Ubuntu
```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())  # Doit être > 0
```

Si CUDA n'est pas détecté, recompilez OpenCV avec support CUDA.

### Tesseract non trouvé
Le package cherche automatiquement Tesseract dans les emplacements standards.
Si non trouvé, installez Tesseract OCR:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-fra`
