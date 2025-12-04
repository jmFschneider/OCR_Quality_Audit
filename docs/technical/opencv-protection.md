# Protection OpenCV CUDA

## 🎯 Problème

Le projet utilise **OpenCV 4.8.0 compilé avec CUDA** pour l'accélération GPU. Cependant, pip peut automatiquement installer `opencv-python` (version sans CUDA) lors de l'installation d'autres dépendances, ce qui écrase notre version compilée.

## 🛡️ Solutions de protection

### 1. Configuration pip globale (Recommandé)

**Fichier** : `~/.config/pip/pip.conf`

```ini
[install]
no-binary = opencv-python,opencv-python-headless,opencv-contrib-python
```

**Effet** : Empêche pip d'installer les versions binaires d'OpenCV (force compilation depuis sources, ce qui échouera et bloquera l'installation).

### 2. Fichier de contraintes du projet

**Fichier** : `pip-constraints.txt`

```txt
opencv-python==999.0.0
opencv-python-headless==999.0.0
opencv-contrib-python==999.0.0
```

**Utilisation** :
```bash
pip install -c pip-constraints.txt -r requirements_ubuntu.txt
```

**Effet** : Spécifie une version impossible (999.0.0) qui n'existe pas, forçant pip à échouer si un package demande opencv-python.

### 3. Vérification au démarrage

Le fichier `pipeline.py` détecte automatiquement si CUDA est disponible :

```python
USE_CUDA = False
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    USE_CUDA = count > 0
except:
    USE_CUDA = False
```

**Message attendu au démarrage** :
```
✅ GPU CUDA activé (1 device(s))
```

**Si vous voyez** :
```
⚠️ Mode CPU uniquement
```

→ OpenCV sans CUDA a été installé par erreur.

## 🔧 Restauration en cas de problème

### Vérifier la version OpenCV installée

```bash
python3 -c "import cv2; print(f'Version: {cv2.__version__}'); print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

**Sortie attendue** :
```
Version: 4.8.0
CUDA: 1
```

### Supprimer opencv-python erroné

```bash
# Désinstaller toutes les versions pip d'OpenCV
pip3 uninstall -y opencv-python opencv-python-headless opencv-contrib-python

# Vérifier que la version système est chargée
python3 -c "import cv2; print(cv2.__file__)"
# Doit afficher: /usr/lib/python3/dist-packages/cv2/__init__.py
# OU: /usr/local/lib/python3.10/dist-packages/cv2/__init__.py
```

## 📦 Où est OpenCV CUDA ?

### Bibliothèques natives (C++)
```
/usr/local/lib/libopencv_*.so.4.8.0
```

Exemples :
- `libopencv_core.so.4.8.0`
- `libopencv_cudaarithm.so.4.8.0`
- `libopencv_cudafilters.so.4.8.0`
- etc.

### Binding Python
```
/usr/lib/python3/dist-packages/cv2/
```
OU
```
/usr/local/lib/python3.10/dist-packages/cv2/
```

### Headers et configuration
```
/usr/local/include/opencv4/
/usr/local/share/opencv4/
/usr/local/lib/cmake/opencv4/
```

## 🧪 Test de validation

```bash
# Script de test complet
python3 << 'EOF'
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"Chargé depuis: {cv2.__file__}")

# Test CUDA
cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
print(f"CUDA devices: {cuda_devices}")

if cuda_devices > 0:
    print("✅ CUDA activé")

    # Test création GpuMat
    test_img = np.ones((100, 100), dtype=np.uint8) * 128
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(test_img)
    print(f"✅ GpuMat créé: {gpu_mat.size()}")

    # Test threshold CUDA
    _, gpu_result = cv2.cuda.threshold(gpu_mat, 100, 255, cv2.THRESH_BINARY)
    result = gpu_result.download()
    print(f"✅ Threshold CUDA OK: {result.shape}")

    print("\n🎉 OpenCV CUDA fonctionne parfaitement !")
else:
    print("❌ CUDA NON activé - version pip installée par erreur")
    print("   Solution: pip3 uninstall opencv-python")
EOF
```

## 📝 Procédure d'installation pour nouveaux utilisateurs

### 1. Installer les dépendances système
```bash
sudo apt install tesseract-ocr tesseract-ocr-fra python3-tk
```

### 2. Compiler OpenCV avec CUDA
```bash
# Utiliser le script de compilation
cd docs/archive/ubuntu-migration/
bash compile_opencv_numpy126.sh
```

### 3. Installer les dépendances Python AVEC protection
```bash
# Installer pip.conf d'abord
mkdir -p ~/.config/pip
cp pip.conf ~/.config/pip/

# Puis installer les dépendances
pip install -c pip-constraints.txt -r requirements_ubuntu.txt
```

### 4. Vérifier l'installation
```bash
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} - CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')"
```

## ⚠️ Pièges courants

### Piège 1 : Dépendance transitive
Certains packages (ex: scikit-image, albumentations) peuvent demander opencv-python comme dépendance.

**Solution** : Installer ces packages avec `--no-deps` puis installer manuellement leurs autres dépendances :
```bash
pip install --no-deps scikit-image
pip install numpy scipy pillow  # dépendances de scikit-image
```

### Piège 2 : requirements.txt d'un autre projet
Si vous utilisez un requirements.txt d'un autre projet qui spécifie opencv-python :

**Solution** : Créer un requirements local sans opencv-python :
```bash
grep -v opencv-python other_requirements.txt > requirements_local.txt
pip install -r requirements_local.txt
```

### Piège 3 : Notebooks Jupyter
Jupyter peut réinstaller opencv-python lors de `!pip install` dans une cellule.

**Solution** : Toujours vérifier après installation :
```python
import cv2
assert cv2.cuda.getCudaEnabledDeviceCount() > 0, "CUDA non disponible!"
```

## 🔍 Débogage

### Vérifier quels packages dépendent d'opencv
```bash
pip show opencv-python 2>/dev/null && echo "⚠️ opencv-python est installé !" || echo "✅ opencv-python non présent"
```

### Vérifier l'ordre de chargement Python
```bash
python3 -c "import sys; print('\n'.join(sys.path))"
```

L'ordre de priorité est :
1. `~/.local/lib/python3.10/site-packages` (install --user)
2. `/usr/local/lib/python3.10/dist-packages` (install système)
3. `/usr/lib/python3/dist-packages` (packages Ubuntu)

### Forcer le rechargement
```bash
# Supprimer le cache Python
rm -rf ~/.cache/pip
rm -rf __pycache__
rm -rf .pytest_cache

# Réimporter
python3 -c "import importlib; import cv2; importlib.reload(cv2); print(cv2.__version__)"
```

## 📚 Références

- [Compilation OpenCV avec CUDA](../archive/ubuntu-migration/PHASE3_OPENCV_CUDA_UBUNTU.md)
- [Configuration pip](https://pip.pypa.io/en/stable/topics/configuration/)
- [Contraintes pip](https://pip.pypa.io/en/stable/user_guide/#constraints-files)
- [OpenCV CUDA modules](https://docs.opencv.org/4.8.0/d1/d1a/group__cuda.html)

## ✅ Checklist de validation

Après installation, vérifier :

- [ ] `python3 -c "import cv2; print(cv2.__version__)"` → 4.8.0
- [ ] `python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"` → 1
- [ ] `pip list | grep opencv` → Aucun résultat
- [ ] `cat ~/.config/pip/pip.conf` → Contient `no-binary = opencv-python`
- [ ] `python3 gui_main.py` → Message "✅ GPU CUDA activé"
- [ ] `nvidia-smi` → GPU visible et utilisé

---

**Dernière mise à jour** : 2025-12-04
**Version OpenCV CUDA** : 4.8.0
**Version CUDA** : 11.8
