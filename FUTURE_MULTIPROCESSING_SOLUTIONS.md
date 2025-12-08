# Solutions pour accélérer le pipeline Blur+CLAHE avec parallélisme

## Problème actuel

Le pipeline blur_clahe utilise un **traitement séquentiel** pour éviter les deadlocks multiprocessing.
- **Performance actuelle** : ~6.5s par point (8 images)
- **Potentiel d'amélioration** : 10-15x plus rapide avec 24 cores

## 🎯 Solution 1 : Threading (Recommandé)

### Avantages
- ✅ Pas de problème fork/spawn
- ✅ Partage de mémoire (pas de copies)
- ✅ OpenCV et Tesseract relâchent le GIL
- ✅ Facile à implémenter

### Code à ajouter dans optimizer.py

```python
from concurrent.futures import ThreadPoolExecutor
import threading

def evaluate_pipeline_threaded(images, baseline_scores, params, point_id=0, pipeline_mode='blur_clahe'):
    """Version threadée pour blur_clahe."""

    def process_single_image(idx, img, baseline):
        """Worker thread pour une image."""
        if pipeline_mode == 'blur_clahe':
            processed_img = pipeline.pipeline_blur_clahe(img, params)
        else:
            processed_img = pipeline.pipeline_complet(img, params)

        tess_abs = pipeline.get_tesseract_score(processed_img)
        cnr_val = pipeline.get_cnr_quality(processed_img)
        sharp_val = pipeline.get_sharpness(processed_img)

        delta = tess_abs - baseline
        return delta, tess_abs, sharp_val, cnr_val

    # Utiliser ThreadPoolExecutor
    max_workers = min(len(images), 8)  # 8 threads simultanés max

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, (img, baseline) in enumerate(zip(images, baseline_scores)):
            future = executor.submit(process_single_image, idx, img, baseline)
            futures.append(future)

        # Attendre tous les résultats
        results = [f.result() for f in futures]

    # Calculer moyennes
    list_delta, list_abs, list_sharp, list_cont = zip(*results)

    return (
        sum(list_delta) / len(list_delta),
        sum(list_abs) / len(list_abs),
        sum(list_sharp) / len(list_sharp),
        sum(list_cont) / len(list_cont)
    )
```

### Modification à faire dans optimizer.py (ligne 228)

```python
# MODE CPU (multiprocessing ou séquentiel)
else:
    # Pour blur_clahe : threading au lieu de séquentiel
    if pipeline_mode == 'blur_clahe':
        # Utiliser la version threadée
        list_delta, list_abs, list_sharp, list_cont = [], [], [], []

        def process_single(idx, img, baseline):
            processed_img = pipeline.pipeline_blur_clahe(img, params)
            tess_abs = pipeline.get_tesseract_score(processed_img)
            cnr_val = pipeline.get_cnr_quality(processed_img)
            sharp_val = pipeline.get_sharpness(processed_img)
            return tess_abs - baseline, tess_abs, sharp_val, cnr_val

        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(len(images), 8)  # 8 threads simultanés

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single, i, img, baseline_scores[i])
                      for i, img in enumerate(images)]
            results = [f.result() for f in futures]

        list_delta, list_abs, list_sharp, list_cont = zip(*results)

    # Mode standard : multiprocessing comme avant
    else:
        # ... code existant multiprocessing ...
```

### Gain attendu
- **Séquentiel actuel** : 6.5s/point
- **Threading (8 workers)** : ~1.0s/point (**6.5x plus rapide**)
- **4096 points** : 7.7h → **1.1h** 🚀

---

## 🔧 Solution 2 : Multiprocessing avec subprocess wrapper

### Principe
Chaque worker lance un **subprocess Python** séparé pour éviter les conflits de mémoire.

### Code

```python
import subprocess
import json
import tempfile

def process_image_subprocess(img_path, params_json):
    """Worker qui lance un subprocess."""
    script = """
import cv2
import json
import sys
import pipeline

# Charger image
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
params = json.loads(sys.argv[2])

# Traiter
processed = pipeline.pipeline_blur_clahe(img, params)

# Métriques
tess = pipeline.get_tesseract_score(processed)
cnr = pipeline.get_cnr_quality(processed)
sharp = pipeline.get_sharpness(processed)

# Retourner résultats
print(json.dumps({'tess': tess, 'cnr': cnr, 'sharp': sharp}))
"""

    result = subprocess.run(
        ['python3', '-c', script, img_path, params_json],
        capture_output=True,
        text=True,
        timeout=30
    )

    return json.loads(result.stdout)
```

### Avantages
- ✅ Isolation complète (pas de conflits)
- ✅ Utilise tous les cores

### Inconvénients
- ❌ Overhead de lancement subprocess
- ❌ Plus complexe

---

## 🌟 Solution 3 : Optimisation du pipeline lui-même

### Pistes d'optimisation

1. **Réduire la résolution avant traitement**
   ```python
   # Dans pipeline_blur_clahe, ajouter au début:
   if cpu_img.shape[1] > 1500:  # Si largeur > 1500px
       scale = 1500 / cpu_img.shape[1]
       cpu_img = cv2.resize(cpu_img, None, fx=scale, fy=scale)
   ```
   **Gain** : 2-3x plus rapide

2. **Denoising moins agressif**
   ```python
   # Réduire searchWindowSize de 21 à 15
   img_denoised = cv2.fastNlMeansDenoising(img_no_lines, None, h=h_val,
                                           templateWindowSize=7,
                                           searchWindowSize=15)  # au lieu de 21
   ```
   **Gain** : 30-40% plus rapide

3. **Inpainting plus rapide**
   ```python
   # Utiliser INPAINT_NS au lieu de TELEA
   img_no_lines = cv2.inpaint(cpu_img, mask_lines, 3, cv2.INPAINT_NS)
   ```
   **Gain** : 15-20% plus rapide

### Combiné avec threading
- **Optimisations** : 2x plus rapide
- **Threading (8 workers)** : 6.5x plus rapide
- **TOTAL** : **13x plus rapide** → 4096 points en **35 minutes** ! 🎉

---

## 📊 Comparaison des solutions

| Solution | Complexité | Gain | Temps 4096pts |
|----------|------------|------|---------------|
| Séquentiel actuel | ✅ Simple | 1x | 7.7h |
| Threading | ⚠️ Moyen | 6.5x | 1.1h |
| Threading + Optim | ⚠️ Moyen | 13x | 35min |
| Multiprocessing subprocess | ❌ Complexe | 8-10x | 45-55min |

---

## 🎯 Plan d'action recommandé

### Phase 1 : Threading (ce soir ou demain)
1. Modifier `optimizer.py` ligne 228 (ajouter le code threading)
2. Tester avec `python3 test_sequential_blur.py`
3. Vérifier stabilité sur 10-20 points

**Effort** : 15 minutes
**Gain** : 6.5x plus rapide

### Phase 2 : Optimisations pipeline (optionnel)
1. Ajouter resize si largeur > 1500px
2. Réduire searchWindowSize à 15
3. Tester qualité OCR (vérifier pas de régression)

**Effort** : 30 minutes
**Gain supplémentaire** : 2x (total 13x)

### Phase 3 : Fine-tuning (si besoin)
1. Ajuster nombre de threads (8, 12, 16)
2. Profiler pour identifier autres goulots
3. Considérer GPU pour certaines opérations

---

## 🧪 Code de test pour threading

```python
# test_threading_blur.py
import cv2
import glob
import time
from concurrent.futures import ThreadPoolExecutor
import pipeline
import optimizer

def test_threading():
    """Test du threading pour blur_clahe."""

    # Charger images
    image_files = glob.glob("test_scans/*.jpg")[:8]
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    images = [img for img in images if img is not None]

    # Baseline
    baseline_scores = optimizer.calculate_baseline_scores(images)

    # Params
    params = {
        'inp_line_h': 40,
        'inp_line_v': 40,
        'denoise_h': 12.0,
        'bg_dilate': 7,
        'bg_blur': 21,
        'clahe_clip': 2.0,
        'clahe_tile': 8
    }

    # Test séquentiel
    print("🐌 Test SÉQUENTIEL...")
    t0 = time.time()
    results_seq = []
    for i, img in enumerate(images):
        processed = pipeline.pipeline_blur_clahe(img, params)
        tess = pipeline.get_tesseract_score(processed)
        results_seq.append(tess)
    t_seq = time.time() - t0
    print(f"   Temps: {t_seq:.2f}s")

    # Test threading
    print("\n🚀 Test THREADING (8 workers)...")
    t0 = time.time()

    def process_one(img):
        processed = pipeline.pipeline_blur_clahe(img, params)
        return pipeline.get_tesseract_score(processed)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results_thread = list(executor.map(process_one, images))

    t_thread = time.time() - t0
    print(f"   Temps: {t_thread:.2f}s")

    # Comparaison
    print(f"\n📊 RÉSULTATS:")
    print(f"   Séquentiel:  {t_seq:.2f}s")
    print(f"   Threading:   {t_thread:.2f}s")
    print(f"   Speedup:     {t_seq/t_thread:.1f}x")
    print(f"\n   Résultats identiques: {results_seq == results_thread}")

if __name__ == "__main__":
    test_threading()
```

---

## ⚠️ Points d'attention

1. **Thread safety** : OpenCV et Tesseract sont généralement thread-safe, mais testez bien
2. **Nombre de threads** : Ne pas dépasser 8-12 pour éviter la contention
3. **Mémoire** : Chaque thread garde une image en traitement (~2 MB × 8 = 16 MB)
4. **Stabilité** : Tester sur plusieurs runs avant de lancer un gros screening

---

## 💡 Conclusion

La solution **Threading** est le meilleur compromis :
- ✅ Simple à implémenter (15 min)
- ✅ Gain significatif (6.5x)
- ✅ Pas de problème de mémoire
- ✅ Stable

Avec threading + optimisations : **4096 points en 35 minutes** au lieu de 7.7h ! 🚀
