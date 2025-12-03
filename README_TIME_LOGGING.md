# Système de logging des temps de traitement

## 📊 Vue d'ensemble

Au lieu d'afficher les temps dans le terminal, le système sauvegarde maintenant toutes les mesures dans un fichier CSV pour analyse post-traitement.

## 🎯 Fonctionnalités

### 1. Sauvegarde automatique des temps
- **Format CSV** avec séparateur `;` pour Excel
- **Buffer de 50 mesures** pour optimiser les I/O disque
- **Timestamp** pour chaque mesure
- **Scores** inclus pour corrélation temps/qualité

### 2. Données enregistrées par image
Pour chaque image traitée :
- `timestamp` : Date et heure de la mesure
- `point_id` : Numéro du point Sobol
- `image_id` : Numéro de l'image dans le batch
- `temps_total_ms` : Temps total de traitement
- `temps_cuda_ms` : Temps du traitement CUDA
- `temps_tesseract_ms` : Temps de Tesseract
- `temps_sharpness_ms` : Temps calcul netteté
- `temps_contrast_ms` : Temps calcul contraste
- `score_tesseract` : Score OCR obtenu
- `score_sharpness` : Netteté obtenue
- `score_contrast` : Contraste obtenu

### 3. Script d'analyse automatique
- Calcul de **moyennes**, **médianes**, **min**, **max**, **écart-types**
- Statistiques **par point Sobol**
- Statistiques **par image**
- **Répartition des temps** en pourcentage
- **Recommandations** d'optimisation
- **Estimations** pour différents volumes

## 📁 Fichiers générés

### Format du nom
```
timing_log_YYYYMMDD_HHMMSS.csv
```

**Exemple:**
```
timing_log_20251203_114222.csv
```

### Structure du fichier CSV
```csv
timestamp;point_id;image_id;temps_total_ms;temps_cuda_ms;temps_tesseract_ms;temps_sharpness_ms;temps_contrast_ms;score_tesseract;score_sharpness;score_contrast
2025-12-03 11:42:23.553;1;0;894.77;179.54;709.21;4.39;1.64;45.12;13165.93;61.37
2025-12-03 11:42:24.679;1;1;1125.9;186.35;933.61;4.21;1.73;52.2;17726.13;65.2
```

## 🚀 Utilisation

### 1. Lancer un screening avec logging
```python
import optimizer

best_params, csv_file = optimizer.run_sobol_screening(
    images=images,
    baseline_scores=baselines,
    n_points=32,
    param_ranges=param_ranges,
    fixed_params=fixed_params,
    enable_time_logging=True  # Active le logging (défaut: True)
)
```

**Sortie console:**
```
📊 Logging des temps activé: timing_log_20251203_114222.csv
🔍 SCREENING SOBOL: Génération de 32 points
...
✅ Logging des temps fermé: timing_log_20251203_114222.csv
```

### 2. Analyser les résultats
```bash
# Analyse du fichier le plus récent
python3 analyser_temps.py

# Ou analyser un fichier spécifique
python3 analyser_temps.py timing_log_20251203_114222.csv
```

### 3. Désactiver le logging (si nécessaire)
```python
best_params, csv_file = optimizer.run_sobol_screening(
    ...,
    enable_time_logging=False  # Pas de fichier créé
)
```

## 📊 Exemple d'analyse

### Statistiques globales
```
Métrique                    Min        Max    Moyenne    Médiane   Écart-type
--------------------------------------------------------------------------
Temps total               894.8     1140.7     1014.5     1012.9        121.2
Temps CUDA                177.4      199.5      190.2      192.7          8.3
Temps Tesseract           696.0      940.8      818.7      823.2        118.5
Temps Netteté               3.1        5.1        4.2        4.4          0.7
Temps Contraste             0.7        1.7        1.4        1.7          0.4
```

### Répartition des temps
```
Composant              Temps (ms)   % du total
--------------------------------------------
CUDA (traitement):          190.2        18.7%
Tesseract (OCR):            818.7        80.7%
Netteté:                      4.2         0.4%
Contraste:                    1.4         0.1%
```

**Conclusion:** Tesseract représente **80.7%** du temps total (goulot d'étranglement)

### Statistiques par point Sobol
```
 Point  Nb images    Temps total moy     CUDA moy     Tess moy
----------------------------------------------------------------------
     1          2             1010.3 ms        182.9 ms        821.4 ms
     2          2             1019.9 ms        187.7 ms        827.1 ms
     3          2             1021.2 ms        196.4 ms        818.4 ms
     4          2             1006.7 ms        193.8 ms        807.9 ms
```

### Statistiques par image
```
 Image   Nb mesures    Temps total moy     CUDA moy     Tess moy
----------------------------------------------------------------------
     0            4              901.4 ms        188.0 ms        708.4 ms
     1            4             1127.6 ms        192.3 ms        929.1 ms
```

### Recommandations
```
Analyse du goulot d'étranglement:
⚠️  Tesseract représente 80.7% du temps total
   → Envisager un OCR avec support GPU (EasyOCR, PaddleOCR)
   → Ou paralléliser Tesseract sur plusieurs images
```

### Estimations de temps
```
Temps moyen par image: 1014 ms

Nb images    Nb points       Temps estimé
---------------------------------------
2            32                    1.1min
10           32                    5.4min
24           32                   13.0min
2            128                   4.3min
24           128                  51.9min
24           256                     1.7h
```

## 🔧 Classe TimeLogger

### Initialisation
```python
logger = TimeLogger(
    enabled=True,              # Si False, aucun fichier créé
    filename=None              # Auto-généré si None
)
```

### Enregistrement d'une mesure
```python
logger.log(
    point_id=1,               # ID du point Sobol
    image_id=0,               # ID de l'image
    temps_total=1000.5,       # ms
    temps_cuda=200.3,         # ms
    temps_tess=750.2,         # ms
    temps_sharp=4.5,          # ms
    temps_cont=1.5,           # ms
    score_tess=85.5,          # Score Tesseract
    score_sharp=15000.2,      # Netteté
    score_cont=65.3           # Contraste
)
```

### Fermeture
```python
logger.close()  # Flush final et message de confirmation
```

## 📈 Analyse avancée avec pandas

Si vous voulez faire votre propre analyse :

```python
import pandas as pd

# Charger le fichier
df = pd.read_csv('timing_log_20251203_114222.csv', sep=';')

# Statistiques descriptives
print(df.describe())

# Moyenne par point
moyennes = df.groupby('point_id')[['temps_total_ms', 'temps_cuda_ms', 'temps_tesseract_ms']].mean()
print(moyennes)

# Corrélation temps vs scores
corr = df[['temps_total_ms', 'score_tesseract', 'score_sharpness']].corr()
print(corr)

# Graphique
import matplotlib.pyplot as plt

df.plot(x='point_id', y=['temps_cuda_ms', 'temps_tesseract_ms'], kind='bar')
plt.title('Répartition des temps par point')
plt.ylabel('Temps (ms)')
plt.show()
```

## 🎯 Cas d'usage

### 1. Optimiser les paramètres CUDA
Identifier les points Sobol qui prennent le plus de temps CUDA :
```bash
python3 analyser_temps.py timing_log_*.csv | grep "CUDA moy"
```

### 2. Comparer différents runs
```bash
# Run 1 : avec denoising
python3 analyser_temps.py timing_log_20251203_114222.csv > run1.txt

# Run 2 : sans denoising
python3 analyser_temps.py timing_log_20251203_120045.csv > run2.txt

# Comparer
diff run1.txt run2.txt
```

### 3. Estimer le temps pour production
```python
import statistics

# Charger les temps mesurés
with open('timing_log_20251203_114222.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=';')
    temps = [float(row['temps_total_ms']) for row in reader]

avg_ms = statistics.mean(temps)
nb_images_prod = 1000
nb_points = 256

temps_total_h = (nb_images_prod * nb_points * avg_ms) / 1000 / 3600
print(f"Temps estimé: {temps_total_h:.1f} heures")
```

## ⚙️ Configuration

### Taille du buffer
Modifier dans `optimizer.py` :
```python
class TimeLogger:
    def __init__(self, enabled=True, filename=None):
        ...
        self.buffer_size = 50  # Valeur par défaut
```

**Recommandations:**
- `buffer_size = 10` : Flush fréquent (moins de risque de perte)
- `buffer_size = 50` : Bon compromis (défaut)
- `buffer_size = 100` : Maximum de performance

### Format du timestamp
Modifier dans `optimizer.py` ligne 58 :
```python
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
```

## 🧪 Tests

### Test rapide
```bash
python3 test_time_logging.py
```

**Vérifie:**
1. Création du fichier timing_log_*.csv
2. Enregistrement des 8 mesures (4 points × 2 images)
3. Analyse automatique avec statistiques

### Test manuel
```python
import optimizer

# Créer un logger
logger = optimizer.TimeLogger(enabled=True)

# Enregistrer une mesure
logger.log(1, 0, 1000, 200, 750, 5, 2, 85, 15000, 65)

# Fermer
logger.close()
```

## 📝 Notes techniques

### Gestion des erreurs
Le logger capture les erreurs d'écriture sans bloquer le screening :
```python
except Exception as e:
    print(f"⚠️ Erreur écriture timing log: {e}")
```

### Thread-safety
Le logger n'est **pas thread-safe**. Il est conçu pour un usage séquentiel en mode GPU.
En mode CPU multiprocessing, le logging n'est pas actif.

### Performance
- **Overhead** : < 0.5ms par mesure
- **Buffer** : Minimise les I/O disque
- **Impact** : Négligeable sur les performances globales

## 🎓 Recommandations

### Pour exploration rapide
```python
enable_time_logging=False  # Désactiver pour gagner du temps
```

### Pour production
```python
enable_time_logging=True   # Activer pour traçabilité
```

### Pour debug
```python
# Analyser immédiatement après
best_params, csv = optimizer.run_sobol_screening(...)
os.system("python3 analyser_temps.py")
```

## ✅ Avantages vs affichage terminal

| Critère | Terminal | Fichier CSV |
|---------|----------|-------------|
| Persistance | ❌ Perdu après fermeture | ✅ Sauvegardé |
| Analyse | ❌ Difficile | ✅ Facile avec pandas |
| Statistiques | ❌ Impossible | ✅ Automatique |
| Comparaison | ❌ Impossible | ✅ Entre différents runs |
| Corrélations | ❌ Impossible | ✅ Temps vs scores |
| Graphiques | ❌ Impossible | ✅ Avec matplotlib |

## 🚀 Évolutions futures possibles

1. **Logging en mode CPU** : Activer aussi pour le multiprocessing
2. **Dashboard temps réel** : Interface web pour suivre les temps
3. **Alertes** : Notification si temps > seuil
4. **Compression** : Compresser les anciens fichiers (.csv.gz)
5. **Base de données** : SQLite pour requêtes complexes
