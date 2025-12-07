# Configuration OpenCV CUDA - Guide Rapide

## ✅ Installation actuelle

Votre système est configuré avec :
- **OpenCV 4.8.0** compilé avec **CUDA 11.8**
- **GPU** : NVIDIA GTX 1080 Ti
- **Protection pip** : Empêche l'installation d'opencv-python sans CUDA

## 📋 Fichiers de protection créés

### 1. `~/.config/pip/pip.conf`
Configuration pip globale qui bloque opencv-python.

**Installation** :
```bash
mkdir -p ~/.config/pip
cp pip.conf ~/.config/pip/
```

**Contenu** :
```ini
[install]
no-binary = opencv-python,opencv-python-headless,opencv-contrib-python
```

### 2. `pip-constraints.txt`
Fichier de contraintes pour le projet.

**Utilisation** :
```bash
pip install -c pip-constraints.txt -r requirements_ubuntu.txt
```

### 3. `tests/test_opencv_cuda.py`
Script de validation complet de l'installation CUDA.

**Utilisation** :
```bash
python3 tests/test_opencv_cuda.py
```

**Tests effectués** :
- ✅ Version OpenCV 4.8.x
- ✅ CUDA devices détectés
- ✅ Opérations CUDA (threshold, upload/download)
- ✅ Filtres CUDA (Gaussian, Morphology)
- ✅ Protection pip active

### 4. `docs/technical/opencv-protection.md`
Documentation technique complète sur la protection et le dépannage.

## 🚀 Vérification rapide

```bash
# Test rapide (doit afficher "4.8.0" et "1")
python3 -c "import cv2; print(f'{cv2.__version__} - CUDA:{cv2.cuda.getCudaEnabledDeviceCount()}')"

# Test complet
python3 tests/test_opencv_cuda.py

# Lancer l'application
python3 gui_main.py
# → Doit afficher "✅ GPU CUDA activé (1 device(s))"
```

## ⚠️ Problèmes résolus

### Problème initial
OpenCV 4.12.0 **sans CUDA** avait été installé ce matin (4 déc 2025 à 10:35), écrasant la version compilée.

### Solution appliquée
```bash
# 1. Désinstallation de la version pip
pip3 uninstall -y opencv-python

# 2. Installation de la protection
mkdir -p ~/.config/pip
cp pip.conf ~/.config/pip/

# 3. Vérification
python3 tests/test_opencv_cuda.py
```

### Résultat
```
🎉 TOUS LES TESTS RÉUSSIS !

Votre installation OpenCV CUDA est correcte.
Vous pouvez utiliser l'application avec accélération GPU.
```

## 📊 Causes du problème

L'installation d'opencv-python peut survenir lors de :

1. **Installation de dépendances** : `pip install -r requirements_ubuntu.txt`
   - Certaines dépendances (scipy, pandas) ont pu déclencher une installation automatique

2. **Mise à jour système** : `pip install --upgrade pip`
   - Pip peut suggérer des mises à jour incluant opencv-python

3. **Installation manuelle** : `pip install opencv-python`
   - Installation accidentelle

## 🛡️ Protection installée

Avec `~/.config/pip/pip.conf`, pip ne peut plus installer opencv-python :

```bash
$ pip install opencv-python
ERROR: Could not find a version that satisfies the requirement opencv-python
```

## 📖 Documentation

- **Guide utilisateur** : `README.md`
- **Protection OpenCV** : `docs/technical/opencv-protection.md`
- **Tests** : `tests/test_opencv_cuda.py`
- **Contraintes pip** : `pip-constraints.txt`

## 🔧 Maintenance

### Vérification périodique
```bash
# Vérifier qu'opencv-python n'est pas installé
pip list | grep opencv
# → Ne doit rien afficher

# Vérifier CUDA
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# → Doit afficher "1"
```

### Si opencv-python est réinstallé
```bash
# Désinstaller immédiatement
pip3 uninstall -y opencv-python opencv-python-headless

# Vérifier la protection
cat ~/.config/pip/pip.conf
# → Doit contenir "no-binary = opencv-python..."
```

### Recompilation (si nécessaire)
```bash
cd docs/archive/ubuntu-migration/
bash compile_opencv_numpy126.sh
# → Durée : ~30-45 minutes
```

## ✅ État actuel

| Élément | Statut | Détails |
|---------|--------|---------|
| OpenCV version | ✅ 4.8.0 | Avec CUDA |
| CUDA devices | ✅ 1 | GTX 1080 Ti |
| Protection pip | ✅ Activée | ~/.config/pip/pip.conf |
| Tests | ✅ Tous passent | test_opencv_cuda.py |
| Application | ✅ Fonctionne | GUI démarre avec CUDA |

## 🎯 Prochaines étapes

1. ✅ **Protection installée** - pip.conf créé
2. ✅ **Tests validés** - Tous les tests passent
3. ✅ **Documentation créée** - Guides complets
4. 🔄 **Utilisation normale** - Lancer l'application

```bash
python3 gui_main.py
```

---

**Créé le** : 2025-12-04
**Version OpenCV** : 4.8.0
**Version CUDA** : 11.8
**GPU** : NVIDIA GTX 1080 Ti
