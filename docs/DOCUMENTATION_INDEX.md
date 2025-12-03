# Index de la Documentation - OCR Quality Audit

## 📖 Vue d'ensemble

Cette documentation est organisée en 4 catégories principales pour faciliter la navigation.

## 🎯 Guides Utilisateur

Documentation pour l'utilisation quotidienne de l'outil.

### [Sobol Screening](user-guide/sobol-screening.md)
- Utilisation de l'optimisation Sobol
- Format des fichiers CSV générés
- Optimisations GPU/CPU
- **À lire en premier** pour comprendre le fonctionnement

### [Exposant Sobol (2^n)](user-guide/sobol-exponent.md)
- Sélecteur d'exposant dans l'interface
- Valeurs recommandées selon le contexte
- Label dynamique et validation
- Estimations de temps

### [Logging des Temps](user-guide/time-logging.md)
- Système de sauvegarde CSV des performances
- Utilisation de `analyser_temps.py`
- Statistiques et recommandations
- Format des fichiers de timing

### [Mesure des Temps](user-guide/timing-measurement.md)
- Analyse détaillée des temps de traitement
- Identification des goulots d'étranglement
- Comparaison GPU vs CPU
- Profiling détaillé

## 🔧 Documentation Technique

Documentation pour les développeurs et contributeurs.

### [Résumé de Modularisation](technical/modularization-summary.md)
- Architecture modulaire (pipeline.py, optimizer.py, gui_main.py)
- Améliorations de performance
- Tests et validation
- Guide de migration depuis l'ancien code

### [Corrections Appliquées](technical/CORRECTIONS_APPLIED.md)
- Historique des corrections de bugs
- Résolution des problèmes d'intégration
- Adaptations du code

## 📋 Changelogs

Historique détaillé des modifications.

### [Exposant Sobol](changelogs/sobol-exponent.md)
- Implémentation du système 2^n
- Modifications de l'interface
- Tests et validation

### [Time Logging](changelogs/time-logging.md)
- Système de logging CSV
- Classe TimeLogger
- Script d'analyse automatique

### [Timing](changelogs/timing.md)
- Mesure des temps de traitement
- Fonctions _timed
- Integration dans le pipeline

## 📚 Archives

Documentation obsolète conservée pour référence historique.

### [old-md-files/](archive/old-md-files/)
- Anciens README et guides
- Documentation des phases 1-2
- Fichiers de configuration obsolètes

### [ubuntu-migration/](archive/ubuntu-migration/)
- Guide d'installation Ubuntu
- Compilation OpenCV avec CUDA
- Scripts de build
- Migration depuis OpenCL

## 🧪 Tests

Documentation des tests disponibles dans `/tests/`

| Fichier | Description |
|---------|-------------|
| `test_time_logging.py` | Validation du système de logging CSV |
| `test_timing.py` | Test des mesures de temps |
| `test_sobol_exponent.py` | Test du sélecteur d'exposant |
| `test_sobol_integration.py` | Test d'intégration complète |
| `test_corrections.py` | Validation des corrections |

## 🚀 Démarrage Rapide

### Nouveau utilisateur
1. Lire le [README principal](../README.md)
2. Suivre les instructions d'installation
3. Lire [Sobol Screening](user-guide/sobol-screening.md)
4. Lancer `python3 gui_main.py`

### Développeur
1. Lire [Résumé de Modularisation](technical/modularization-summary.md)
2. Consulter [Corrections Appliquées](technical/CORRECTIONS_APPLIED.md)
3. Examiner les tests dans `/tests/`
4. Consulter les changelogs pour l'historique

### Analyse de Performance
1. Lire [Logging des Temps](user-guide/time-logging.md)
2. Exécuter un screening Sobol
3. Analyser avec `python3 analyser_temps.py`
4. Consulter [Mesure des Temps](user-guide/timing-measurement.md)

## 📊 Schéma de Navigation

```
docs/
├── DOCUMENTATION_INDEX.md  ← Vous êtes ici
│
├── user-guide/             ← Pour utiliser l'outil
│   ├── sobol-screening.md
│   ├── sobol-exponent.md
│   ├── time-logging.md
│   └── timing-measurement.md
│
├── technical/              ← Pour développer/contribuer
│   ├── modularization-summary.md
│   └── CORRECTIONS_APPLIED.md
│
├── changelogs/             ← Historique des changements
│   ├── sobol-exponent.md
│   ├── time-logging.md
│   └── timing.md
│
└── archive/                ← Référence historique
    ├── old-md-files/
    └── ubuntu-migration/
```

## 🔗 Liens Rapides

### Documentation Principale
- **[README.md](../README.md)** - Point d'entrée principal
- **[Installation](../README.md#-installation)** - Guide d'installation
- **[Démarrage Rapide](../README.md#-démarrage-rapide)** - Premier lancement

### Guides Essentiels
- **[Sobol Screening](user-guide/sobol-screening.md)** - Optimisation
- **[Time Logging](user-guide/time-logging.md)** - Analyse de performance
- **[Modularization](technical/modularization-summary.md)** - Architecture

### Support
- **[Résolution de Problèmes](../README.md#-résolution-de-problèmes)** - Bugs courants
- **[GitHub Issues](https://github.com/jmFschneider/OCR_Quality_Audit/issues)** - Support communautaire

## 📝 Contribution à la Documentation

Pour améliorer cette documentation :

1. **Guides utilisateur** → `docs/user-guide/`
2. **Documentation technique** → `docs/technical/`
3. **Changelogs** → `docs/changelogs/`
4. **Archives** → Ne pas modifier (historique)

### Standards de Documentation

- **Format** : Markdown avec syntax highlighting
- **Langue** : Français (code en anglais)
- **Structure** : Titre H1, sections H2/H3, exemples de code
- **Liens** : Relatifs pour navigation interne
- **Exemples** : Inclure des cas d'usage concrets

## 🎓 Recommandations de Lecture

### Parcours Utilisateur
1. README principal
2. Sobol Screening
3. Exposant Sobol (2^n)
4. Time Logging

### Parcours Développeur
1. Modularization Summary
2. Corrections Applied
3. Changelogs (tous)
4. Code source (pipeline.py, optimizer.py)

### Parcours Performance
1. Mesure des Temps
2. Time Logging
3. analyser_temps.py
4. Tests de timing

---

**Dernière mise à jour** : 2025-12-03
**Version** : 3.0 (Architecture modulaire)
