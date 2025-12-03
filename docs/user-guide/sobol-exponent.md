# Sélecteur d'exposant Sobol (2^n)

## 📊 Vue d'ensemble

Le système de sélection des points Sobol utilise maintenant un **exposant** (n) au lieu d'un nombre fixe.
Le nombre de points est calculé automatiquement : **Nombre de points = 2^n**

## 🎯 Avantages du système d'exposant

### 1. Séquences optimales
Les séquences de Sobol sont conçues pour les puissances de 2 :
- ✅ **2^5 = 32 points** : Couverture optimale de l'espace
- ❌ **30 points** : Couverture sous-optimale

### 2. Échelle intuitive
```
2^3  = 8       points  (test rapide)
2^5  = 32      points  (exploration rapide)
2^7  = 128     points  (exploration standard)
2^8  = 256     points  (exploration complète)
2^10 = 1024    points  (screening exhaustif)
```

### 3. Label dynamique
L'interface affiche automatiquement le nombre de points calculé :
```
Exposant Sobol (2^n): [5] = 32 points
```

## 🖥️ Interface graphique

### Zone de saisie

```
┌─────────────────────────────────────────┐
│  Exposant Sobol (2^n): [5] = 32 points  │
│  [▶️ Lancer Sobol]  [⏹️ Arrêter]        │
└─────────────────────────────────────────┘
```

### Comportement dynamique

1. **Saisie valide** : Le label affiche le nombre de points en noir
   ```
   [5] = 32 points
   [7] = 128 points
   ```

2. **Saisie invalide** : Le label affiche "Invalide" en rouge
   ```
   [abc] = Invalide
   ```

3. **Valeur trop élevée** : Le label affiche un avertissement en rouge
   ```
   [17] ! > 65536
   ```

## 📝 Utilisation

### Dans l'interface graphique

1. Entrer l'exposant dans le champ (ex: 5)
2. Le label se met à jour automatiquement : "= 32 points"
3. Cliquer sur "▶️ Lancer Sobol"
4. Le log affiche : "🚀 Démarrage Sobol avec 2^5 = 32 points"

### Valeurs recommandées

| Exposant | Points | Temps estimé* | Usage |
|----------|--------|---------------|-------|
| 3 | 8 | ~15s | Test rapide |
| 5 | 32 | ~1 min | Exploration rapide |
| 6 | 64 | ~2 min | Exploration moyenne |
| 7 | 128 | ~4 min | Exploration standard |
| 8 | 256 | ~8 min | Exploration complète |
| 10 | 1024 | ~30 min | Screening exhaustif |
| 12 | 4096 | ~2h | Analyse approfondie |

*Pour 2 images. Multiplier par (nb_images/2) pour estimer.

## ⚠️ Limites

### Limite technique
- **Maximum recommandé** : 2^16 = 65536 points
- Au-delà : Risque de mémoire insuffisante et temps d'exécution très long

### Protection dans l'interface
```python
if exponent > 16:
    self.log("❌ Exposant trop élevé (max 16 = 65536 points)")
    return
```

## 🔧 Code technique

### Fonction de mise à jour du label

```python
def update_sobol_points_label(self, *args):
    """Met à jour le label affichant le nombre de points Sobol (2^n)."""
    try:
        exponent = int(self.sobol_exponent_var.get())
        if exponent > 16:
            self.sobol_points_label.config(text="! > 65536", foreground="red")
            return
        n_points = 2**exponent
        self.sobol_points_label.config(text=f"= {n_points} points", foreground="black")
    except ValueError:
        self.sobol_points_label.config(text="= Invalide", foreground="red")
```

### Calcul du nombre de points

```python
def run_sobol(self):
    try:
        exponent = int(self.sobol_exponent_var.get())
        if exponent > 16:
            self.log("❌ Exposant trop élevé (max 16 = 65536 points)")
            return
        n_points = 2**exponent  # Calcul automatique
    except:
        self.log("❌ Exposant Sobol invalide")
        return

    self.log(f"🚀 Démarrage Sobol avec 2^{exponent} = {n_points} points")
```

## 📊 Estimation des temps

### Formule
```
Temps total ≈ 2^n × nb_images × 1s
```

### Exemples (pour 24 images)
```
2^5  = 32 points    → 32 × 24 × 1s  ≈ 13 min
2^7  = 128 points   → 128 × 24 × 1s ≈ 51 min
2^8  = 256 points   → 256 × 24 × 1s ≈ 1h42
2^10 = 1024 points  → 1024 × 24 × 1s ≈ 6h50
```

## 🧪 Test

### Script de test
```bash
python3 test_sobol_exponent.py
```

**Vérifie :**
1. Calcul 2^n pour différents exposants
2. Valeurs limites (2^16, 2^17)
3. Screening Sobol avec 2^2 = 4 points
4. Estimation des temps

**Résultat attendu :**
```
1. TEST DU CALCUL D'EXPOSANT:
   2^ 3 =      8 points
   2^ 5 =     32 points
   2^ 7 =    128 points
   2^10 =   1024 points

✅ Screening terminé
📁 Fichier: screening_sobol_4pts_*.csv

RECOMMANDATIONS:
  • Exploration rapide    : 2^5 = 32 points    (~1 min)
  • Exploration standard  : 2^7 = 128 points   (~4 min)
  • Exploration complète  : 2^8 = 256 points   (~8 min)
```

## 🔄 Comparaison avant/après

### Avant (nombre fixe)
```
Points Sobol: [32]
[▶️ Lancer Sobol]
```
- ❌ Pas de validation visuelle
- ❌ Utilisateur peut entrer n'importe quel nombre
- ❌ Pas de guidage sur les valeurs optimales

### Après (exposant 2^n)
```
Exposant Sobol (2^n): [5] = 32 points
[▶️ Lancer Sobol]
```
- ✅ Label dynamique avec validation visuelle
- ✅ Valeurs optimales (puissances de 2)
- ✅ Alerte si valeur trop élevée
- ✅ Affichage "2^n = X points" dans les logs

## 📁 Fichiers modifiés

### gui_main.py
- **Ligne 110-127** : Zone de saisie avec exposant et label dynamique
- **Ligne 154-164** : Fonction `update_sobol_points_label()`
- **Ligne 206-220** : Fonction `run_sobol()` avec calcul 2^n

### Nouveaux fichiers
- `test_sobol_exponent.py` : Script de test complet
- `README_SOBOL_EXPONENT.md` : Cette documentation

## 💡 Conseils d'utilisation

### Pour débutants
Commencer avec **2^5 = 32 points** :
- Temps raisonnable (~1-2 min)
- Donne une première idée des paramètres optimaux
- Permet de valider que tout fonctionne

### Pour exploration
Utiliser **2^7 = 128 points** :
- Bon compromis temps/qualité
- Couverture suffisante de l'espace paramétrique
- Standard pour la plupart des cas

### Pour production
Utiliser **2^8 à 2^10** :
- Résultats fiables et reproductibles
- Analyse statistique valide
- Identification des paramètres optimaux

## 🎓 Arrière-plan théorique

### Pourquoi les puissances de 2 ?

Les séquences de Sobol génèrent des points quasi-aléatoires qui couvrent uniformément l'espace.
La qualité de cette couverture est optimale pour des nombres de points = 2^n car :

1. **Structure binaire** : Les séquences de Sobol utilisent une base 2
2. **Propriétés mathématiques** : Garanties de couverture uniforme
3. **Convergence** : Meilleure vitesse de convergence

### Comparaison couverture

```
30 points (non-optimal)     32 points (2^5, optimal)
     ●  ●    ●   ●                ● ● ● ●
   ●    ●  ●     ●              ● ● ● ●
     ●    ●   ●  ●              ● ● ● ●
   ●  ●     ●    ●              ● ● ● ●
```

Les 32 points couvrent mieux l'espace que 30 points arbitraires.

## ✅ Validation

- [x] Interface avec exposant implémentée
- [x] Label dynamique fonctionnel
- [x] Validation des limites (max 16)
- [x] Calcul automatique 2^n
- [x] Affichage dans les logs
- [x] Test unitaire créé
- [x] Documentation créée
- [x] Compatible GPU CUDA
