# Solution : Problème de blocage avec le moteur Haute Fidélité (Blur+CLAHE)

## 🔍 Problème identifié

Lorsque vous lanciez l'application avec le moteur Haute Fidélité (Blur+CLAHE) en mode screening, **le système se bloquait avant que le premier point ne soit calculé**.

### Cause racine

Le **multiprocessing causait un deadlock** avec le pipeline blur_clahe :
- Le multiprocessing par défaut utilise `fork` sur Linux
- `fork` n'est pas compatible avec OpenCV, Tesseract et CUDA
- Même avec `spawn`, le multiprocessing créait des conflits car chaque worker chargeait OpenCV/Tesseract

## ✅ Solution implémentée

### Modification 1 : gui_main.py (lignes 18-27)
Configuration de `spawn` **AVANT** d'importer les modules pipeline/optimizer :

```python
# CRITIQUE: Configurer multiprocessing AVANT d'importer pipeline/optimizer
# pour éviter les deadlocks avec le mode blur_clahe
# ATTENTION: Ceci doit être au niveau module, pas dans main()
if platform.system() != 'Windows':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[DEBUG] multiprocessing.set_start_method('spawn') configuré")
    except RuntimeError as e:
        # Déjà défini, c'est OK
        print(f"[DEBUG] spawn déjà configuré ou impossible: {e}")
```

### Modification 2 : optimizer.py (lignes 229-275)
**Désactivation du multiprocessing pour blur_clahe** (traitement séquentiel) :

```python
# CORRECTION CRITIQUE: Le multiprocessing cause des deadlocks avec blur_clahe
# à cause de conflits OpenCV/Tesseract. On force le traitement séquentiel.
use_multiprocessing = (pipeline_mode != 'blur_clahe')

if use_multiprocessing:
    # Multiprocessing pour mode standard
    ...
else:
    # Traitement séquentiel pour blur_clahe
    if pipeline_mode == 'blur_clahe':
        print(f"[optimizer] Mode blur_clahe: traitement séquentiel ({len(images)} images)")
    ...
```

## 📊 Performances

Avec la correction, le pipeline blur_clahe fonctionne **sans blocage** :

- **Temps par image** : ~0.85 secondes
- **Temps par point** (8 images) : ~6.5 secondes

### Projections de temps pour différents nombres de points :

| Points | Temps estimé |
|--------|--------------|
| 256    | 0.5 heures   |
| 512    | 1.0 heure    |
| 1024   | 1.9 heures   |
| 2048   | 3.9 heures   |
| 4096   | 7.7 heures   |

## 💡 Recommandations

1. **Commencez avec 256-512 points** pour le screening blur_clahe
   - Temps raisonnable : 30 min à 1 heure
   - Suffisant pour explorer l'espace de paramètres

2. **Utilisez des valeurs initiales optimisées** :
   - `inp_line_h`: 20-100
   - `inp_line_v`: 20-100
   - `denoise_h`: 5.0-20.0
   - `bg_dilate`: 3-15
   - `bg_blur`: 11-51
   - `clahe_clip`: 1.0-5.0
   - `clahe_tile`: 4-16

3. **Pour réduire l'utilisation mémoire** :
   - Ne chargez pas trop d'images simultanément
   - Le mode séquentiel évite la multiplication des processus

## 🧪 Tests disponibles

Trois scripts de test ont été créés pour diagnostiquer et valider la solution :

1. **test_blur_clahe_timing.py** : Chronométrage détaillé de chaque étape du pipeline
2. **test_multiprocessing_blur.py** : Démonstration du blocage avec multiprocessing
3. **test_sequential_blur.py** : Validation de la correction avec traitement séquentiel

Pour exécuter les tests :
```bash
python3 test_sequential_blur.py
```

## ⚠️ Limitations connues

- **Pas de parallélisme** : Le mode blur_clahe traite les images séquentiellement
- **Plus lent** que le mode standard avec multiprocessing
- **Mémoire** : Peut atteindre plusieurs GB avec beaucoup d'images (comportement normal)

## 🎯 Utilisation dans l'interface GUI

1. Sélectionnez **"Haute Fidélité (Blur+CLAHE)"** dans le menu déroulant "Moteur"
2. Configurez vos paramètres (tous actifs par défaut)
3. Choisissez **Mode: Screening**
4. Réglez l'**Exposant Sobol** sur **8** (= 256 points) pour commencer
5. Cliquez sur **"▶️ Lancer"**

Le système va maintenant :
- Afficher "[optimizer] Mode blur_clahe: traitement séquentiel (8 images)"
- Traiter chaque point sans blocage
- Afficher la progression dans les logs

## 📈 Résultats attendus

Le pipeline blur_clahe donne généralement :
- **Delta Tesseract** : +1% à +5% (amélioration modérée)
- **CNR (Gemini Quality)** : 7-12 (bon pour IA visuelles)
- **Netteté** : 1500-2000 (préservation texture)

**Avantage principal** : Préservation des niveaux de gris (meilleur pour Gemini que la binarisation stricte).

---

## 📝 Notes techniques

- La méthode `cv2.inpaint()` représente ~31% du temps de traitement
- Le `fastNlMeansDenoising()` représente ~56% du temps
- Ces opérations ne peuvent pas être accélérées avec GPU facilement
- Le traitement séquentiel est le compromis optimal stabilité/performance
