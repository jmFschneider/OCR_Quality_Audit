# Documentation du Moteur "Haute Fidélité" (Blur + CLAHE)

Ce document détaille le fonctionnement et les paramètres du moteur de traitement d'image **Haute Fidélité (High Fidelity)**, mis en place pour optimiser la reconnaissance optique de caractères (OCR) et l'analyse par IA.

Contrairement au moteur "Standard" qui vise une binarisation stricte (noir et blanc), ce moteur privilégie la **conservation des nuances de gris** et de la texture, tout en nettoyant agressivement le bruit et les artefacts (lignes, tâches). Il est particulièrement adapté aux moteurs OCR modernes (Tesseract LSTM) et aux modèles de vision (Gemini, GPT-4V).

## Vue d'ensemble du Pipeline

Le traitement suit 4 étapes séquentielles :
1.  **Suppression des Lignes (Inpainting)** : Détection des lignes de tableau/soulignement et reconstruction intelligente de l'image sous la ligne.
2.  **Débruitage (Denoising)** : Lissage avancé pour supprimer le bruit de scan (grain) sans flouter les bords des lettres.
3.  **Normalisation de Fond** : Uniformisation de l'éclairage pour avoir un fond parfaitement blanc et du texte sombre.
4.  **CLAHE Final** : Rehaussement local du contraste pour faire ressortir les caractères faibles.

---

## Détail des Paramètres

Voici les paramètres configurables du moteur, leur rôle et leur impact.

### 1. Suppression des Lignes (Inpainting)

Cette étape évite que les lignes (tableaux, soulignements) ne "collent" aux lettres, ce qui perturbe l'OCR. Elle utilise une technique d'inpainting (reconstruction) pour effacer la ligne proprement.

| Paramètre | Code | Description & Impact | Valeurs Typiques |
| :--- | :--- | :--- | :--- |
| **Largeur Ligne Horiz.** | `inp_line_h` | Longueur minimale pour qu'un trait horizontal soit considéré comme une ligne à effacer. <br>**Impact :** Trop bas = efface des parties de lettres (ex: la barre du 'T'). Trop haut = rate les lignes courtes. | `30` - `50` px |
| **Hauteur Ligne Vert.** | `inp_line_v` | Hauteur minimale pour qu'un trait vertical soit considéré comme une ligne. <br>**Impact :** Similaire à l'horizontal, pour les colonnes de tableaux. | `30` - `50` px |

### 2. Débruitage (Denoising)

Utilise l'algorithme *Non-Local Means Denoising*, excellent mais coûteux en calcul.

| Paramètre | Code | Description & Impact | Valeurs Typiques |
| :--- | :--- | :--- | :--- |
| **Force Denoising** | `denoise_h` | Intensité du filtrage du bruit. Contrôle le paramètre `h` de l'algo NLMeans. <br>**Impact :** <br>• Faible (< 5) : Laisse du grain.<br>• Élevé (> 15) : "Plastifie" l'image, risque d'effacer les points sur les i ou les virgules. | `10.0` - `15.0` |

### 3. Normalisation de Fond (Background Normalization)

Cette étape estime le fond de l'image pour le soustraire. Elle simule une page parfaitement plate et éclairée uniformément.
*Formule : Résultat = (Image / Fond) normalisé.*

| Paramètre | Code | Description & Impact | Valeurs Typiques |
| :--- | :--- | :--- | :--- |
| **Dilatation Fond** | `bg_dilate` | Élargit les zones sombres (texte) avant l'estimation du fond pour les ignorer. <br>**Impact :** Permet d'ignorer le texte épais lors du calcul du fond. Doit être assez grand pour couvrir l'épaisseur d'un trait de caractère. | `3` - `9` px |
| **Flou Fond (Blur)** | `bg_blur` | Taille du flou médian appliqué pour lisser l'estimation du fond. <br>**Impact :** <br>• Petit : Le "fond" suit trop le texte, risque d'effacer le texte par soustraction.<br>• Grand : Le fond est très lisse, corrige bien les ombres globales mais gère mal les tâches locales. | `15` - `31` px |

### 4. Rehaussement de Contraste (CLAHE)

*Contrast Limited Adaptive Histogram Equalization*. Améliore le contraste localement plutôt que globalement.

| Paramètre | Code | Description & Impact | Valeurs Typiques |
| :--- | :--- | :--- | :--- |
| **CLAHE Clip Limit** | `clahe_clip` | Seuil de limitation du contraste. Empêche d'amplifier le bruit dans les zones unies. <br>**Impact :** <br>• Faible (1.0) : Peu d'effet.<br>• Élevé (> 4.0) : Très fort contraste, mais fait ressortir le grain du papier comme du bruit. | `1.5` - `3.0` |
| **CLAHE Grid Size** | `clahe_tile` | Taille de la grille (ex: 8x8) pour l'égalisation locale. <br>**Impact :** Définit la "localité" du contraste.<br>• Petit (4) : S'adapte à de très petites variations.<br>• Grand (16) : Plus global, évite les variations brusques de luminosité. | `8` (standard) |

---

## Pourquoi "Blur + CLAHE" ?

La synergie entre ces deux étapes est le cœur de ce moteur :

1.  **Le Blur (Denoising + Normalisation)** nettoie l'image. Si on appliquait le CLAHE directement sur l'image brute, le bruit de fond serait amplifié massivement, rendant le texte illisible.
2.  **Le CLAHE** intervient *après* le nettoyage pour redonner du "punch" aux caractères qui auraient pu être un peu affadis par le lissage.

C'est cet équilibre (Lisser le bruit VS Accentuer le texte) que nous cherchons à optimiser via le Screening Sobol.
