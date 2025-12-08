"""
OCR Quality Audit - Package pour l'optimisation du traitement d'images OCR

Ce package fournit des outils pour traiter et optimiser des images avant OCR,
avec support CUDA pour l'accélération GPU.

Modules principaux:
- pipeline: Fonctions de traitement d'images
- optimizer: Algorithmes d'optimisation des paramètres
- scipy_optimizer: Optimisation avec scipy

Usage typique:
    from ocr_quality_audit import pipeline_complet, pipeline_blur_clahe

    # Traiter une image avec des paramètres
    params = {
        'h_size': 50,
        'v_size': 2,
        'dilate_iter': 2,
        'kernel_size': 50,
        'base_h_param': 10,
        'block_size': 15,
        'C_constant': 10
    }
    image_traitee = pipeline_complet(image, params)
"""

__version__ = "0.1.0"

# Imports des fonctions principales de pipeline
from .pipeline import (
    # Pipelines complets
    pipeline_complet,
    pipeline_complet_timed,
    pipeline_blur_clahe,

    # Fonctions de traitement
    remove_lines_param,
    normalisation_division,
    adaptive_denoising,

    # Métriques
    get_sharpness,
    get_contrast,
    get_tesseract_score,
    get_cnr_quality,
    evaluer_toutes_metriques,
    evaluer_toutes_metriques_batch,

    # Utilitaires GPU/CPU
    ensure_gpu,
    ensure_cpu,

    # Configuration
    USE_CUDA,
)

# Imports depuis optimizer (si nécessaire pour usage externe)
from .optimizer import (
    TimeLogger,
)

__all__ = [
    # Version
    '__version__',

    # Pipelines
    'pipeline_complet',
    'pipeline_complet_timed',
    'pipeline_blur_clahe',

    # Traitement
    'remove_lines_param',
    'normalisation_division',
    'adaptive_denoising',

    # Métriques
    'get_sharpness',
    'get_contrast',
    'get_tesseract_score',
    'get_cnr_quality',
    'evaluer_toutes_metriques',
    'evaluer_toutes_metriques_batch',

    # Utilitaires
    'ensure_gpu',
    'ensure_cpu',
    'TimeLogger',

    # Configuration
    'USE_CUDA',
]
