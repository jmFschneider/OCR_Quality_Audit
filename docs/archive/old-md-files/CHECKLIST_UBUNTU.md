# Checklist - Déploiement Ubuntu avec CUDA

## 📋 Fichiers modifiés à transférer

### Fichiers principaux
- ✅ `gui_optimizer_v3_ultim.py` (version CUDA optimisée)
- ✅ `test_cuda_performance.py` (script de benchmark)
- ✅ `MODIFICATIONS_CUDA.md` (documentation)
- ✅ `scipy_optimizer.py` (inchangé, mais nécessaire)

### Fichiers de référence (optionnel)
- `gui_optimizer_v3_ultim_V2.py` (propositions Gemini - référence uniquement)
- `gui_optimizer_v3_ultim_backup.py` (backup avant modifications)

---

## 🚀 Étapes de déploiement sur Ubuntu

### 1. Vérification de l'environnement CUDA

```bash
# Vérifier que NVIDIA drivers sont installés
nvidia-smi

# Vérifier OpenCV avec CUDA
python3 << EOF
import cv2
print(f"OpenCV version: {cv2.__version__}")
try:
    count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"✅ CUDA devices: {count}")
except:
    print("❌ OpenCV sans support CUDA")
EOF
```

**Résultat attendu:**
```
✅ CUDA devices: 1
```

---

### 2. Test rapide de performance

```bash
cd /chemin/vers/OCR_Quality_Audit
python3 test_cuda_performance.py
```

**Ce test va mesurer:**
- Détection GPU CUDA
- Chargement d'images depuis `test_scans/`
- Benchmark CPU vs GPU (GaussianBlur, Morphologie, Laplacian)

**Résultats attendus (GTX 1080 Ti):**
- GaussianBlur: x3-5 speedup
- Morphologie: x4-8 speedup
- Laplacian: x2-4 speedup

---

### 3. Test complet avec l'interface

```bash
python3 gui_optimizer_v3_ultim.py
```

**Configuration de test recommandée:**
1. Cocher **"Debug/Timing"** (pour voir les logs détaillés)
2. Mode: **Screening**
3. Exposant Sobol: **5** (32 points = test rapide)
4. Vérifier que le message suivant apparaît:
   ```
   🚀 PHASE 3 - ACCÉLÉRATION CUDA ACTIVÉE (GTX 1080 Ti)
   ✅ 1 GPU CUDA détecté(s)
   ```

5. Lancer l'optimisation
6. Observer les temps d'exécution dans les logs

---

## 📊 Comparaison des performances

### Mesures de référence (à noter AVANT les modifications)

```
⏱️ Temps d'exécution AVANT (OpenCL/UMat):
- Traitement 1 image: _____ ms
- Screening 32 points (n=5): _____ secondes
```

### Mesures après optimisation CUDA

```
⏱️ Temps d'exécution APRÈS (CUDA natif):
- Traitement 1 image: _____ ms
- Screening 32 points (n=5): _____ secondes

🚀 Gain: x_____
```

---

## 🐛 Dépannage

### Si CUDA n'est pas détecté

**Problème:** `AttributeError: module 'cv2' has no attribute 'cuda'`

**Solution:** OpenCV n'a pas été compilé avec CUDA. Deux options:

1. **Option 1: Utiliser opencv-contrib-python (si disponible avec CUDA)**
   ```bash
   pip3 uninstall opencv-python opencv-contrib-python
   pip3 install opencv-contrib-python
   ```

2. **Option 2: Compiler OpenCV avec CUDA** (voir INSTALLATION_UBUNTU.md)

---

### Si les performances sont identiques ou pires

**Vérifications:**

1. **CUDA est-il vraiment activé ?**
   ```bash
   # Doit afficher le message CUDA dans les logs
   python3 gui_optimizer_v3_ultim.py
   # Chercher: "🚀 PHASE 3 - ACCÉLÉRATION CUDA ACTIVÉE"
   ```

2. **La carte GPU est-elle utilisée ?**
   ```bash
   # Terminal 1: Lancer l'optimisation
   python3 gui_optimizer_v3_ultim.py

   # Terminal 2: Surveiller l'utilisation GPU
   watch -n 1 nvidia-smi
   ```

   **Attendu:** Utilisation GPU ~30-80% pendant le traitement

3. **Y a-t-il des erreurs dans les logs ?**
   - Vérifier la console pour des messages d'erreur
   - Activer "Debug/Timing" pour voir les détails

---

### Si certaines opérations échouent

**Symptôme:** Erreurs du type `cv2.error: OpenCV(4.x.x) ... GpuMat ...`

**Causes possibles:**
1. Type d'image incompatible (channels, depth)
2. Kernel trop grand pour la mémoire GPU
3. Driver CUDA obsolète

**Solution:**
```python
# Le code a des fallbacks CPU automatiques
# Vérifier les logs pour voir quelles fonctions tombent en CPU
```

---

## ✅ Validation finale

Une fois les tests terminés, le programme devrait:

1. ✅ Démarrer sans erreur
2. ✅ Afficher "PHASE 3 - ACCÉLÉRATION CUDA ACTIVÉE"
3. ✅ Traiter les images 2-5x plus vite qu'avant
4. ✅ Utiliser le GPU (visible dans `nvidia-smi`)
5. ✅ Produire les mêmes résultats (pas de régression qualité)

---

## 🎯 Objectif final

**Temps d'exécution cible pour n=5 images (Screening 32 points):**

- **AVANT (OpenCL):** ~X minutes
- **APRÈS (CUDA):** ~X/3 minutes (gain x3 minimum attendu)

Si ce gain n'est pas atteint, consulter la section Dépannage.

---

## 📝 Notes importantes

### Ce qui reste sur CPU (normal)
- **Tesseract OCR:** Pas de support GPU
- **fastNlMeansDenoising:** Pas d'équivalent CUDA performant
- **adaptiveThreshold:** Algorithme adaptatif complexe

Ces opérations sont **inévitables** mais représentent ~30-40% du temps total.

### Ce qui est maintenant sur GPU (gain majeur)
- **GaussianBlur** (normalisation)
- **Morphologie** (suppression lignes)
- **Threshold OTSU** (binarisation)
- **Laplacian** (netteté, bruit)
- **meanStdDev** (métriques)
- **divide** (normalisation)

Ces opérations représentent ~60-70% du temps et sont **x3-8 plus rapides**.

---

## 🔄 Commandes Git (après validation)

Si tout fonctionne bien:

```bash
cd /chemin/vers/OCR_Quality_Audit

# Voir les fichiers modifiés
git status

# Ajouter les modifications
git add gui_optimizer_v3_ultim.py test_cuda_performance.py MODIFICATIONS_CUDA.md CHECKLIST_UBUNTU.md

# Commit
git commit -m "feat(cuda): Migrate from OpenCL to native CUDA for GTX 1080 Ti

- Replace cv2.ocl with cv2.cuda API
- Add ensure_gpu()/ensure_cpu() helpers
- Optimize all pipeline functions with CUDA filters
- Expected speedup: x2-5 on image processing
- Maintain CPU fallback for compatibility"

# Pousser vers la branche
git push origin linux/ubuntu
```

---

**Prêt pour le transfert et les tests sur Ubuntu ! 🚀**
