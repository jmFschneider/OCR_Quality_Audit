# Changelog - Sélecteur d'exposant Sobol

## 📅 Date: 2025-12-02

## ✨ Fonctionnalité ajoutée : Sélecteur d'exposant Sobol (2^n)

### Problème résolu
- ❌ Ancien système : Saisie directe du nombre de points (ex: 32)
- ❌ Pas de validation visuelle
- ❌ Utilisateur peut entrer des valeurs non-optimales (ex: 30, 50)
- ❌ Pas d'aide pour choisir le bon nombre de points

### Solution implémentée
- ✅ Nouveau système : Saisie de l'exposant n pour calculer 2^n points
- ✅ Label dynamique affichant le nombre de points calculé
- ✅ Validation automatique avec alerte si valeur trop élevée
- ✅ Interface intuitive avec mise à jour en temps réel

## 🖥️ Interface graphique

### Avant
```
┌──────────────────────────────────┐
│ Points Sobol: [32]               │
│ [▶️ Lancer Sobol] [⏹️ Arrêter]   │
└──────────────────────────────────┘
```

### Après
```
┌─────────────────────────────────────────┐
│ Exposant Sobol (2^n): [5] = 32 points  │
│ [▶️ Lancer Sobol]  [⏹️ Arrêter]        │
└─────────────────────────────────────────┘
```

### Comportement dynamique

#### Saisie valide
```
[3] = 8 points        (noir)
[5] = 32 points       (noir)
[10] = 1024 points    (noir)
```

#### Saisie invalide
```
[abc] = Invalide      (rouge)
[1.5] = Invalide      (rouge)
```

#### Valeur trop élevée
```
[17] ! > 65536        (rouge)
[20] ! > 65536        (rouge)
```

## 📝 Code modifié

### 1. Zone de saisie (gui_main.py:110-127)

```python
# Ancien code
ttk.Label(opt_frame, text="Points Sobol:").grid(row=0, column=0, padx=5)
self.sobol_points = ttk.Entry(opt_frame, width=10)
self.sobol_points.insert(0, "32")
self.sobol_points.grid(row=0, column=1, padx=5)

# Nouveau code
ttk.Label(opt_frame, text="Exposant Sobol (2^n):").grid(row=0, column=0, padx=5)

# Variable avec callback pour mise à jour dynamique
self.sobol_exponent_var = tk.StringVar(value="5")
self.sobol_exponent_var.trace_add("write", self.update_sobol_points_label)

self.sobol_exponent_entry = ttk.Entry(opt_frame, width=5, textvariable=self.sobol_exponent_var)
self.sobol_exponent_entry.grid(row=0, column=1, padx=2)

# Label dynamique
self.sobol_points_label = ttk.Label(opt_frame, text="= 32 points")
self.sobol_points_label.grid(row=0, column=2, padx=5)
```

### 2. Fonction de mise à jour (gui_main.py:154-164)

```python
def update_sobol_points_label(self, *args):
    """Met à jour le label affichant le nombre de points Sobol (2^n)."""
    try:
        exponent = int(self.sobol_exponent_var.get())
        if exponent > 16:  # Limite pour éviter les très grands nombres
            self.sobol_points_label.config(text="! > 65536", foreground="red")
            return
        n_points = 2**exponent
        self.sobol_points_label.config(text=f"= {n_points} points", foreground="black")
    except ValueError:
        self.sobol_points_label.config(text="= Invalide", foreground="red")
```

### 3. Fonction run_sobol (gui_main.py:206-220)

```python
# Ancien code
def run_sobol(self):
    try:
        n_points = int(self.sobol_points.get())
    except:
        self.log("❌ Nombre de points invalide")
        return

    self.log(f"🚀 Démarrage Sobol avec {n_points} points")

# Nouveau code
def run_sobol(self):
    try:
        exponent = int(self.sobol_exponent_var.get())
        if exponent > 16:
            self.log("❌ Exposant trop élevé (max 16 = 65536 points)")
            return
        n_points = 2**exponent
    except:
        self.log("❌ Exposant Sobol invalide")
        return

    self.log(f"🚀 Démarrage Sobol avec 2^{exponent} = {n_points} points")
```

## 📊 Valeurs recommandées

| Exposant | Points | Temps* | Usage |
|----------|--------|--------|-------|
| 3 | 8 | ~15s | Test rapide |
| 5 | 32 | ~1 min | Exploration rapide ⭐ |
| 6 | 64 | ~2 min | Exploration moyenne |
| 7 | 128 | ~4 min | Exploration standard ⭐ |
| 8 | 256 | ~8 min | Exploration complète ⭐ |
| 10 | 1024 | ~30 min | Screening exhaustif |
| 12 | 4096 | ~2h | Analyse approfondie |

*Pour 2 images avec GPU CUDA

⭐ = Valeurs recommandées

## 🔒 Protection des limites

### Limite dans l'interface
```python
if exponent > 16:
    self.sobol_points_label.config(text="! > 65536", foreground="red")
```

### Limite dans l'exécution
```python
if exponent > 16:
    self.log("❌ Exposant trop élevé (max 16 = 65536 points)")
    return
```

### Pourquoi 2^16 ?
- **2^16 = 65536 points** : Limite raisonnable
- Au-delà : Risque de mémoire insuffisante
- Temps d'exécution : > 18 heures pour 24 images

## 🧪 Test de validation

### Script de test
```bash
python3 test_sobol_exponent.py
```

### Résultat attendu
```
======================================================================
TEST SÉLECTEUR EXPOSANT SOBOL
======================================================================

1. TEST DU CALCUL D'EXPOSANT:
   2^ 3 =      8 points
   2^ 5 =     32 points
   2^ 7 =    128 points
   2^10 =   1024 points
   2^12 =   4096 points

   Valeur limite:
   2^16 = 65536 points (max recommandé)
   2^17 = 131072 points (trop élevé)

4. TEST SCREENING SOBOL AVEC DIFFÉRENTS EXPOSANTS:
   Test avec 2^2 = 4 points:
   ✅ Screening terminé
   📁 Fichier: screening_sobol_4pts_*.csv

RECOMMANDATIONS:
  • Exploration rapide    : 2^5 = 32 points    (~1 min)
  • Exploration standard  : 2^7 = 128 points   (~4 min)
  • Exploration complète  : 2^8 = 256 points   (~8 min)
  • Screening exhaustif   : 2^10 = 1024 points (~30 min)
======================================================================
```

## 📈 Avantages du système d'exposant

### 1. Valeurs optimales
Les séquences de Sobol sont conçues pour les puissances de 2 :
```
✅ 2^5 = 32 points   (optimal)
❌ 30 points         (sous-optimal)
```

### 2. Couverture de l'espace
```
8 points (2^3)       32 points (2^5)       128 points (2^7)
● ● ● ●              ● ● ● ● ● ● ● ●      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
● ● ● ●              ● ● ● ● ● ● ● ●      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                     ● ● ● ● ● ● ● ●      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                     ● ● ● ● ● ● ● ●      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
```

### 3. Échelle intuitive
```
2^n  →  Doublement du nombre de points à chaque incrément
3 → 8
4 → 16
5 → 32   (standard)
6 → 64
7 → 128  (recommandé)
8 → 256  (complet)
```

## 🎓 Arrière-plan théorique

### Séquences de Sobol
Les séquences de Sobol génèrent des points quasi-aléatoires qui couvrent uniformément l'espace.
La qualité de cette couverture est optimale pour des nombres de points = 2^n.

### Propriétés mathématiques
1. **Structure binaire** : Base 2 intrinsèque
2. **Convergence** : O(1/N) pour 2^n points
3. **Discrepance** : Minimale pour puissances de 2

## 📁 Fichiers

### Modifiés
- `gui_main.py` : +17 lignes (interface + fonction update_sobol_points_label)

### Nouveaux
- `test_sobol_exponent.py` : Script de test (121 lignes)
- `README_SOBOL_EXPONENT.md` : Documentation détaillée (389 lignes)
- `CHANGELOG_SOBOL_EXPONENT.md` : Ce fichier

## 📊 Exemples de logs

### Avant
```
🚀 Démarrage Sobol avec 32 points
```

### Après
```
🚀 Démarrage Sobol avec 2^5 = 32 points
🚀 Screening Sobol en cours (2^5 = 32 points)...
```

## 💡 Guide d'utilisation

### Premiers pas
1. Ouvrir l'interface : `python3 gui_main.py`
2. Charger les images
3. Saisir l'exposant : **5** (pour 32 points)
4. Observer le label : "= 32 points"
5. Cliquer sur "▶️ Lancer Sobol"

### Exploration progressive
1. **Test rapide** : Exposant 3 (8 points, ~15s)
2. **Si prometteur** : Exposant 5 (32 points, ~1 min)
3. **Raffiner** : Exposant 7 (128 points, ~4 min)
4. **Finaliser** : Exposant 8 (256 points, ~8 min)

### Estimation du temps
```python
Temps ≈ 2^n × nb_images × 1 seconde

Exemple avec 24 images et n=7 :
Temps ≈ 2^7 × 24 × 1s = 128 × 24 = 3072s ≈ 51 min
```

## ✅ Checklist de validation

- [x] Interface avec exposant implémentée
- [x] Variable tk.StringVar avec trace_add configurée
- [x] Label dynamique affichant le nombre de points
- [x] Validation des limites (max 2^16)
- [x] Changement de couleur (noir/rouge)
- [x] Fonction update_sobol_points_label créée
- [x] Fonction run_sobol modifiée pour utiliser l'exposant
- [x] Affichage "2^n = X points" dans les logs
- [x] Test unitaire créé et validé
- [x] Documentation complète créée
- [x] Compatible GPU CUDA

## 🚀 Prochaines étapes suggérées

1. Ajouter des boutons rapides pour valeurs communes (5, 7, 8)
2. Afficher l'estimation du temps total
3. Graphique de progression avec ETA
4. Sauvegarde de l'exposant préféré dans les préférences
