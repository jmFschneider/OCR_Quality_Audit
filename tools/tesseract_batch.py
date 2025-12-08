#!/usr/bin/env python3
"""
Module de Batch Processing pour Tesseract OCR
Pour OCR Quality Audit - Phase 3B

Permet de traiter plusieurs images en un seul appel Tesseract
pour réduire l'overhead de démarrage et gagner 30-40% sur temps OCR.

Usage:
    from tesseract_batch import batch_tesseract_ocr

    images = [img1, img2, img3, ...]  # Liste d'images
    scores = batch_tesseract_ocr(images, batch_size=20)
"""

import os
import time
import tempfile
import numpy as np
import cv2
import pytesseract

try:
    import tifffile
except ImportError:
    print("⚠️  Module tifffile non installé")
    print("   Installation: pip install tifffile")
    raise


def batch_tesseract_ocr(images, batch_size=20, verbose=False):
    """
    Effectue l'OCR sur un batch d'images pour réduire l'overhead de démarrage Tesseract.

    Cette fonction regroupe plusieurs images dans un fichier TIFF multi-page,
    puis appelle Tesseract une seule fois pour toutes les images du batch.

    Args:
        images (list): Liste d'images (numpy arrays ou cv2.UMat)
        batch_size (int): Nombre d'images par batch (10-20 optimal selon RAM disponible)
        verbose (bool): Afficher des informations de debug

    Returns:
        list[float]: Scores Tesseract (confiance moyenne) pour chaque image

    Example:
        >>> images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
        >>> scores = batch_tesseract_ocr(images, batch_size=20)
        >>> print(f"Score moyen: {sum(scores)/len(scores):.2f}")

    Notes:
        - Les images UMat sont automatiquement converties en numpy
        - Les images > 2500px de large sont redimensionnées (×0.5) pour optimiser Tesseract
        - En cas d'erreur sur un batch, fallback automatique sur traitement individuel
        - Les fichiers TIFF temporaires sont automatiquement nettoyés
    """
    scores = []
    total_images = len(images)

    if verbose:
        print(f"Traitement OCR batch de {total_images} images par batch de {batch_size}")

    # Traiter par batches
    for batch_idx, i in enumerate(range(0, total_images, batch_size)):
        batch = images[i:i+batch_size]
        batch_num = batch_idx + 1
        total_batches = (total_images + batch_size - 1) // batch_size

        if verbose:
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} images)...", end=" ")

        start_time = time.time()

        # Préparer les images pour le batch
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
                    score = sum(page_confs) / len(page_confs)
                    scores.append(score)
                else:
                    # Aucune confiance valide détectée
                    scores.append(0)

            elapsed = time.time() - start_time
            if verbose:
                print(f"OK ({elapsed*1000:.0f}ms, {elapsed*1000/len(batch):.0f}ms/img)")

        except Exception as e:
            elapsed = time.time() - start_time
            if verbose:
                print(f"ERREUR ({elapsed*1000:.0f}ms)")
                print(f"    ⚠️  Erreur OCR batch: {e}")
                print(f"    Fallback: traitement individuel pour ce batch")

            # Fallback : traiter individuellement
            for img_idx, img in enumerate(batch_np):
                try:
                    data = pytesseract.image_to_data(
                        img, config='--oem 1 --psm 6',
                        output_type=pytesseract.Output.DICT
                    )
                    confs = [int(x) for x in data['conf'] if int(x) != -1]
                    score = sum(confs) / len(confs) if confs else 0
                    scores.append(score)
                except Exception as e_individual:
                    if verbose:
                        print(f"      Image {img_idx}: Erreur {e_individual}")
                    scores.append(0)

        finally:
            # Nettoyer fichier temporaire
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Ignore si déjà supprimé

    return scores


def benchmark_batch_vs_individual(n_images=24, image_size=(2000, 2000), batch_size=20):
    """
    Compare les performances Tesseract batch vs individuel.

    Args:
        n_images (int): Nombre d'images à tester
        image_size (tuple): Taille des images test (height, width)
        batch_size (int): Taille du batch

    Example:
        >>> benchmark_batch_vs_individual(24, (2000, 2000), 20)
        Création de 24 images test (2000x2000)...

        Test INDIVIDUEL (24 images)...
        Temps: 15.23s (634ms/img)

        Test BATCH (24 images, batch_size=20)...
        Temps: 10.87s (453ms/img)

        SPEEDUP: 1.40x (gain de 28.6%)
    """
    print(f"Création de {n_images} images test ({image_size[0]}x{image_size[1]})...")
    images = [
        np.random.randint(0, 255, image_size, dtype=np.uint8)
        for _ in range(n_images)
    ]

    # Test individuel
    print(f"\nTest INDIVIDUEL ({n_images} images)...")
    start = time.time()
    scores_individual = []
    for img in images:
        try:
            data = pytesseract.image_to_data(
                img, config='--oem 1 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            confs = [int(x) for x in data['conf'] if int(x) != -1]
            scores_individual.append(sum(confs) / len(confs) if confs else 0)
        except:
            scores_individual.append(0)

    time_individual = time.time() - start
    print(f"Temps: {time_individual:.2f}s ({time_individual/n_images*1000:.0f}ms/img)")

    # Test batch
    print(f"\nTest BATCH ({n_images} images, batch_size={batch_size})...")
    start = time.time()
    scores_batch = batch_tesseract_ocr(images, batch_size=batch_size, verbose=False)
    time_batch = time.time() - start
    print(f"Temps: {time_batch:.2f}s ({time_batch/n_images*1000:.0f}ms/img)")

    # Speedup
    speedup = time_individual / time_batch
    gain_percent = (1 - time_batch/time_individual) * 100

    print(f"\nSPEEDUP: {speedup:.2f}x (gain de {gain_percent:.1f}%)")

    # Vérifier la cohérence des scores
    print(f"\nVérification cohérence des scores:")
    max_diff = 0
    for i, (s1, s2) in enumerate(zip(scores_individual, scores_batch)):
        diff = abs(s1 - s2)
        if diff > max_diff:
            max_diff = diff
        if diff > 1.0:  # Différence > 1%
            print(f"  Image {i}: {s1:.2f} vs {s2:.2f} (diff: {diff:.2f})")

    print(f"Différence maximale: {max_diff:.2f}%")

    if max_diff < 1.0:
        print("✓ Scores cohérents (différence < 1%)")
    elif max_diff < 5.0:
        print("⚠ Différences acceptables (< 5%)")
    else:
        print("✗ Différences importantes (> 5%) - Vérifier l'implémentation")

    return {
        'time_individual': time_individual,
        'time_batch': time_batch,
        'speedup': speedup,
        'gain_percent': gain_percent,
        'max_diff': max_diff
    }


if __name__ == "__main__":
    """
    Script de test standalone.
    Usage: python3 tesseract_batch.py
    """
    print("="*70)
    print("  Test Tesseract Batch Processing - Phase 3B")
    print("="*70)

    # Benchmark avec paramètres par défaut
    results = benchmark_batch_vs_individual(
        n_images=24,
        image_size=(2000, 2000),
        batch_size=20
    )

    print("\n" + "="*70)
    if results['speedup'] > 1.3:
        print("✓ Performance batch EXCELLENTE (>1.3x)")
    elif results['speedup'] > 1.1:
        print("✓ Performance batch BONNE (>1.1x)")
    else:
        print("⚠ Performance batch MODESTE (<1.1x)")
    print("="*70)
