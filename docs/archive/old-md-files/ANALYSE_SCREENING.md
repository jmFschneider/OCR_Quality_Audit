# Guide d'Analyse des Résultats de Screening

## 📊 Objectif

Ce guide explique comment analyser les résultats du mode **Screening Sobol** pour identifier les paramètres les plus influents sur la qualité OCR.

---

## 🚀 Workflow Complet

### **Étape 1 : Lancer le Screening**

Dans la GUI :
```
1. Sélectionner Mode : "Screening"
2. Choisir Exposant Sobol : 9 ou 10 (512 ou 1024 points)
3. Cocher TOUS les paramètres à tester
4. Cliquer "LANCER"
```

Résultat : Un fichier CSV `screening_sobol_9_YYYYMMDD_HHMMSS.csv`

---

### **Étape 2 : Installer les dépendances d'analyse**

```bash
pip install -r requirements_analysis.txt
```

Ou manuellement :
```bash
pip install pandas numpy matplotlib seaborn
```

---

### **Étape 3 : Lancer l'analyse**

```bash
python analyze_screening.py screening_sobol_9_20250127_143052.csv
```

---

## 📈 Résultats de l'Analyse

Le script génère un dossier `analysis_screening_sobol_9_YYYYMMDD_HHMMSS/` contenant :

### **1. Graphiques**

- **`main_effects.png`** : Classement visuel des paramètres par impact
  - Les 3 paramètres les plus influents sont en couleur coral
  - Plus la barre est longue, plus le paramètre est important

- **`top4_effects_detail.png`** : Effets détaillés des 4 paramètres principaux
  - Montre comment le score varie avec chaque paramètre
  - Permet de voir les tendances (linéaire, plateau, optimal local)

- **`correlations_target.png`** : Corrélations avec le score OCR
  - Rouge = corrélation positive (augmenter le paramètre améliore le score)
  - Bleu = corrélation négative (augmenter le paramètre dégrade le score)

- **`correlations_params.png`** : Corrélations entre paramètres
  - Détecte si certains paramètres sont redondants

- **`score_distribution.png`** : Histogramme des scores obtenus
  - Montre la dispersion des résultats
  - Ligne rouge = moyenne, orange = médiane

### **2. Rapport Texte**

**`rapport_analyse_YYYYMMDD_HHMMSS.txt`** contient :

- Statistiques descriptives (moyenne, min, max, écart-type)
- Classement des paramètres par influence
- Corrélations détaillées
- **Recommandations** :
  - Quels paramètres optimiser en priorité
  - Quels paramètres peuvent être fixés

---

## 🎯 Interpréter les Résultats

### **Effets Principaux (Main Effects)**

**Effet = Variabilité du score quand on change le paramètre**

- **Effet > 5** : Paramètre **TRÈS influent** → À optimiser en priorité
- **Effet 2-5** : Paramètre **modérément influent** → À inclure dans l'optimisation
- **Effet < 2** : Paramètre **peu influent** → Peut être fixé à sa valeur par défaut

**Exemple :**
```
noise_threshold     | Effet:  8.45 | Amplitude: 15.32%
denoise_h           | Effet:  7.21 | Amplitude: 13.87%
bin_c               | Effet:  3.12 | Amplitude:  6.45%
line_h_size         | Effet:  1.23 | Amplitude:  2.10%  ← Peu influent
```

→ Conclusion : Concentrez l'optimisation sur `noise_threshold` et `denoise_h`

### **Corrélations**

**Corrélation avec le score :**
- **|r| > 0.5** : Fort impact (positif ou négatif)
- **|r| < 0.2** : Faible impact

**Corrélation entre paramètres :**
- **|r| > 0.5** : Paramètres redondants → Optimiser l'un ou l'autre, pas les deux

**Exemple :**
```
Corrélations avec score_tesseract :
  📈 denoise_h        : +0.723  ← Fort impact positif
  📈 noise_threshold  : -0.612  ← Fort impact négatif
     bin_c            : +0.145  ← Faible impact

Corrélation entre paramètres :
  denoise_h ↔ noise_threshold : -0.68  ← Redondance !
```

→ Conclusion : Ces deux paramètres sont liés, optimiser les deux ensemble

---

## 💡 Recommandations Post-Analyse

### **Cas 1 : Tous les paramètres sont influents**

→ Lancer une optimisation avec **Optuna (NSGA-II)** pour gérer les interactions complexes

### **Cas 2 : 3-4 paramètres dominent**

→ **Fixer** les paramètres peu influents, **optimiser** les autres avec Scipy ou Optuna TPE

### **Cas 3 : Les meilleurs scores sont aux extrémités des plages**

→ **Élargir les plages Min/Max** et relancer un screening

### **Cas 4 : Plateau (pas de variation claire)**

→ Le problème n'est peut-être pas dans les paramètres testés
→ Vérifier la qualité des images source

---

## 🔬 Exemple d'Analyse Avancée (Python)

Si vous voulez aller plus loin, voici comment charger et analyser le CSV manuellement :

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('screening_sobol_9_20250127_143052.csv', sep=';')

# Top 10 combinaisons
top10 = df.nlargest(10, 'score_tesseract')
print(top10[['score_tesseract', 'denoise_h', 'noise_threshold', 'bin_c']])

# Scatter plot 2D
plt.scatter(df['denoise_h'], df['score_tesseract'], alpha=0.5)
plt.xlabel('denoise_h')
plt.ylabel('Score Tesseract (%)')
plt.show()

# Détecter les interactions (ex: denoise_h × noise_threshold)
df['interaction'] = df['denoise_h'] * df['noise_threshold']
print(df[['interaction', 'score_tesseract']].corr())
```

---

## ⚙️ Configuration du Script d'Analyse

Le script `analyze_screening.py` peut être modifié si besoin :

- **Ligne 44** : Changer `n_bins=10` pour plus/moins de granularité
- **Ligne 89** : Changer le seuil de corrélation forte (actuellement 0.5)
- **Ligne 112** : Personnaliser les couleurs des graphiques

---

## 🆘 Troubleshooting

**Erreur : "No module named 'pandas'"**
```bash
pip install pandas numpy matplotlib seaborn
```

**Graphiques flous**
→ Augmenter le DPI dans le script (ligne 125 : `dpi=150` → `dpi=300`)

**Trop de paramètres sur les graphiques**
→ Le script affiche automatiquement les top 4, modifiable ligne 135

---

## 📚 Références

- **Séquence de Sobol** : https://en.wikipedia.org/wiki/Sobol_sequence
- **Design of Experiments** : https://en.wikipedia.org/wiki/Design_of_experiments
- **Analyse de sensibilité** : https://en.wikipedia.org/wiki/Sensitivity_analysis

---

## 🎓 Pour Aller Plus Loin

Après le screening, vous pouvez :

1. **Analyse factorielle** : Identifier les interactions de second ordre
2. **Response Surface Methodology (RSM)** : Modéliser la surface de réponse
3. **ANOVA** : Test statistique des effets principaux
4. **Kriging / Gaussian Process** : Interpolation pour trouver l'optimum global

Le screening Sobol est la **première étape** d'une analyse rigoureuse !

---

**Bon courage dans vos optimisations ! 🚀**
