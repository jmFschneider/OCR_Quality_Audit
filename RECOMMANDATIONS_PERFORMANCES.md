# Recommandations pour Analyse des Performances

## 📊 Contexte

**PC de Bureau** : 12 cores / 24 threads logiques + RTX 1080
**Résultat observé** : 24 images traitées en ~1.5s (temps total par trial)
**Comparaison PC Portable** : 2 images en ~3.5s

---

## 🔬 Tests à Effectuer pour Analyse Détaillée

### **Test 1 : Screening de Référence (64 points)**

**Objectif** : Mesurer précisément les performances et identifier la stratégie de denoising appliquée

**Procédure** :
```
1. Mode: Screening
2. Exposant Sobol: 6 (= 2^6 = 64 points)
3. Tous les paramètres cochés
4. LANCER
```

**Données à collecter** :
- [ ] **Temps total** affiché à la fin
- [ ] **Message de timing** dans la console (première image) :
  ```
  --- Analyse détaillée des temps d'exécution (en ms, pour une image) ---
    - Niveau de bruit détecté: XXX
    - Seuil de bruit configuré: XXX
    - Stratégie: Denoising OPTIMISÉ ou COMPLET
    - Étape 1_line_removal: XX ms
    - Étape 2_normalization: XX ms
    - Étape 3_denoising: XX ms
    - Étape 4_binarization: XX ms
    - Étape 5_ocr_tesseract: XX ms
    - Étape 6_sharp_contrast: XX ms
    - TEMPS TOTAL par image: XX ms
  ```

**Calculs à faire** :
```
Temps par trial = Temps total / 64
Speedup = (Temps par image × 24) / Temps par trial
Efficacité = Speedup / Nombre de workers
```

---

### **Test 2 : Vérification de la Stratégie de Denoising**

**Question clé** : Quelle stratégie est appliquée majoritairement ?

**Analyse** :
- Si **"Denoising OPTIMISÉ (searchWindowSize=15)"** → Images plutôt propres
  - **Action** : `noise_threshold` peut probablement être **fixé**
  - Gain : Réduction de l'espace de recherche (7 paramètres → 6)

- Si **"Denoising COMPLET (searchWindowSize=21)"** → Images bruitées
  - **Action** : `noise_threshold` doit être **optimisé**
  - Important de trouver le seuil optimal

- Si **mélange des deux** → Variabilité importante
  - **Action** : `noise_threshold` est **crucial** à optimiser
  - Permet d'adapter la stratégie image par image

---

### **Test 3 : Impact du Nombre de Workers**

**Objectif** : Vérifier si 24 threads ou 18 threads sont utilisés

**Procédure** :
```python
# Modifier temporairement gui_optimizer_v3_ultim.py ligne 587
# Test A (actuel) :
optimal_workers = int(os.cpu_count() * 1.5)  # = 36, limité à 24 images

# Test B (tous les threads) :
optimal_workers = os.cpu_count()  # = 24

# Test C (cores physiques seulement) :
optimal_workers = os.cpu_count() // 2  # = 12
```

**Mesurer** : Temps pour 64 points avec chaque configuration

**Hypothèse** :
- Config A (×1.5) et B (×2.0) devraient donner le **même résultat** avec 24 images
- Config C devrait être **~20-30% plus lent**

---

## 📈 Projections Actuelles

### **Avec 1.5s par trial (24 images) :**

| Scénario | Points | Temps Estimé |
|----------|--------|--------------|
| Screening rapide | 2^7 = 128 | 3.2 min |
| Screening moyen | 2^8 = 256 | 6.4 min |
| Screening complet | 2^9 = 512 | **12.8 min** ⭐ |
| Screening large | 2^10 = 1024 | 25.6 min |
| Optuna 500 trials | 500 | **12.5 min** ⭐ |

**Conclusion** : Vous pouvez faire un screening complet de 512 points en moins de 13 minutes ! 🚀

---

## 🎯 Stratégie Recommandée Post-Tests

### **Scénario A : Images Propres (Denoising OPTIMISÉ majoritaire)**

**Workflow recommandé** :
1. **Fixer les paramètres peu influents** :
   ```
   denoise_h = 9.0 (fixe)
   noise_threshold = 100.0 (fixe)
   dilate_iter = 2 (fixe)
   ```

2. **Screening ciblé** (2^9 = 512 points) sur :
   ```
   line_h_size, line_v_size, norm_kernel, bin_block_size, bin_c
   → 5 paramètres seulement
   ```

3. **Analyser** avec `analyze_screening.py`

4. **Optimisation fine** (Optuna 200-300 trials) sur les 3 paramètres les plus influents

**Gain temps** : 5 paramètres au lieu de 8 = **optimisation 50% plus rapide**

---

### **Scénario B : Images Bruitées (Denoising COMPLET majoritaire)**

**Workflow recommandé** :
1. **Screening complet** (2^10 = 1024 points) sur **tous les paramètres**

2. **Analyser** pour identifier :
   - Les 4-5 paramètres les plus influents
   - Les corrélations entre `denoise_h` et `noise_threshold`

3. **Optimisation NSGA-II** (Optuna, 500 trials) :
   - Multi-objectif : Maximiser OCR + Minimiser temps de traitement
   - Tous les paramètres influents

**Gain** : Compréhension complète de l'espace + optimum robuste

---

### **Scénario C : Variabilité Importante (Mix stratégies)**

**Workflow recommandé** :
1. **Screening stratifié** :
   - Séparer les images en 2 groupes (propres vs bruitées)
   - Screening séparé sur chaque groupe

2. **Paramètres adaptatifs** :
   - Optimiser `noise_threshold` pour classifier automatiquement
   - Deux jeux de paramètres optimaux (un par catégorie)

**Gain** : Adaptabilité maximale aux différents types d'images

---

## 🔍 Analyses Complémentaires à Faire

### **Après le premier screening de 512 points :**

1. **Ouvrir le CSV** et vérifier :
   ```python
   import pandas as pd
   df = pd.read_csv('screening_sobol_9_*.csv', sep=';')

   # Dispersion des scores
   print(f"Amplitude : {df['score_tesseract'].max() - df['score_tesseract'].min():.2f}%")
   print(f"Écart-type : {df['score_tesseract'].std():.2f}%")

   # Si amplitude > 10% → optimisation vaut vraiment la peine
   # Si amplitude < 5% → plateau atteint, paramètres moins critiques
   ```

2. **Analyser avec le script** :
   ```bash
   python analyze_screening.py screening_sobol_9_*.csv
   ```

3. **Regarder les graphiques** :
   - `main_effects.png` : Hiérarchie des paramètres
   - `top4_effects_detail.png` : Tendances (linéaires, plateaux, optimums locaux)
   - `correlations_target.png` : Paramètres positivement/négativement corrélés

4. **Lire le rapport** : `rapport_analyse_*.txt`
   - Section "RECOMMANDATIONS" : Paramètres à optimiser vs fixer

---

## 💡 Optimisations Potentielles Supplémentaires

### **Si vous voulez aller encore plus vite (Phase 2) :**

1. **GPU OpenCV (UMat)** :
   - Gain estimé : +10-15% sur GaussianBlur, threshold
   - Complexité : Faible (déjà OpenCL activé)

2. **Batch Tesseract** :
   - Grouper les 24 images en 1 appel Tesseract
   - Gain estimé : +5-10% (réduction overhead)
   - Complexité : Moyenne

3. **Cache preprocessing** :
   - Si les mêmes images sont utilisées, cacher les étapes 1-2 (line removal, normalization)
   - Gain estimé : +20-30% si applicable
   - Complexité : Moyenne

**Total Phase 2** : Gain additionnel de +30-40%
**Temps par trial** : 1.5s → **1.0s** (objectif ambitieux)

---

## 📋 Checklist Complète

### **Tests à faire :**
- [ ] Screening 64 points et collecter les timings détaillés
- [ ] Noter la stratégie de denoising appliquée
- [ ] Vérifier le nombre de workers utilisés (log)
- [ ] Screening 512 points pour analyse sérieuse

### **Analyses :**
- [ ] Exécuter `analyze_screening.py` sur les résultats
- [ ] Identifier les 3-4 paramètres clés
- [ ] Vérifier si `noise_threshold` est influent ou fixable
- [ ] Calculer le speedup et l'efficacité du parallélisme

### **Optimisations :**
- [ ] Optimisation ciblée sur les paramètres clés uniquement
- [ ] Comparer les résultats avec/sans paramètres fixés
- [ ] (Optionnel) Implémenter Phase 2 si gain supplémentaire nécessaire

---

## 🎓 Questions pour Guider l'Analyse

1. **Quelle est l'amplitude des scores** dans le screening ?
   - Si > 10% : Optimisation très profitable
   - Si < 5% : Plateau, gains marginaux

2. **Quels paramètres dominent** dans l'analyse ?
   - Si 2-3 paramètres > 80% de l'effet : Focus sur eux
   - Si distribution homogène : Tous importants (interactions complexes)

3. **Y a-t-il des corrélations fortes** entre paramètres ?
   - Si oui : Optimiser ensemble ou choisir l'un des deux
   - Si non : Indépendance, bon signe pour l'optimisation

4. **Les meilleurs scores sont-ils** :
   - Au centre de l'espace ? → Bon, optimum trouvable
   - Aux extrémités ? → Élargir les plages Min/Max
   - Dispersés aléatoirement ? → Espace complexe, utiliser NSGA-II

---

## 📞 Prochaines Étapes

1. **Faire les tests** listés ci-dessus
2. **Partager les résultats** (timings + CSV du screening)
3. **Analyser ensemble** les graphiques et recommandations
4. **Ajuster la stratégie** selon les résultats
5. **Lancer l'optimisation finale** sur les paramètres clés

---

**Bon courage pour ces tests ! Les résultats seront très intéressants ! 🚀**

---

*Fichier créé le 2025-11-27 - À mettre à jour avec les résultats réels*
