import cv2
import numpy as np
import pytesseract
import optuna
# --- AJOUTER CET IMPORT EN HAUT DU FICHIER ---
from optuna.samplers import QMCSampler, TPESampler

import os
from glob import glob
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog


# --- CONFIGURATION FIXE ---
INPUT_FOLDER = 'test_scans'
# Si vous êtes sous Windows, décommentez et ajustez la ligne suivante si Tesseract n'est pas dans le PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- LOGIQUE CORE OCR ---

def remove_lines_param(gray_image, h_size, v_size, dilate_iter):
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))

    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    mask_lignes = cv2.addWeighted(detect_horizontal, 1, detect_vertical, 1, 0.0)

    # Paramètre ajouté : dilate_iter (combien on grossit le masque)
    mask_lignes = cv2.dilate(mask_lignes, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=dilate_iter)

    resultat = gray_image.copy()
    resultat[mask_lignes > 0] = 255
    return resultat


def normalisation_division(image_gray, kernel_size):
    # Paramètre ajouté : kernel_size (taille du flou pour estimer le fond)
    # Doit être impair
    if kernel_size % 2 == 0: kernel_size += 1
    fond = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
    return cv2.divide(image_gray, fond, scale=255)


def evaluer_score_tesseract(image):
    try:
        # Optimisation vitesse : on réduit l'image si elle est gigantesque pour le test
        if image.shape[1] > 2500:
            scale = 0.5
            image = cv2.resize(image, None, fx=scale, fy=scale)

        data = pytesseract.image_to_data(image, config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(x) for x in data['conf'] if int(x) != -1]
        return sum(confs) / len(confs) if confs else 0
    except:
        return 0


def pipeline_complet(image, params):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 1. Suppression lignes (avec nouveau param dilate_iter)
    no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])

    # 2. Normalisation (avec nouveau param norm_kernel)
    norm = normalisation_division(no_lines, params['norm_kernel'])

    # 3. Denoising
    if params['denoise_h'] > 0:
        denoised = cv2.fastNlMeansDenoising(norm, None, h=params['denoise_h'], templateWindowSize=7,
                                            searchWindowSize=21)
    else:
        denoised = norm

    # 4. Binarisation
    final = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        params['bin_block_size'],
        params['bin_c']
    )
    return final


# --- LOGIQUE OPTUNA DANS LA GUI ---

def run_optimization(gui_app, n_trials, param_ranges):
    def objective(trial):
        # Mapping des paramètres (Identique à avant)
        params = {
            'line_h_size': trial.suggest_int('line_h_size', param_ranges['line_h'][0], param_ranges['line_h'][1]),
            'line_v_size': trial.suggest_int('line_v_size', param_ranges['line_v'][0], param_ranges['line_v'][1]),
            'dilate_iter': trial.suggest_int('dilate_iter', param_ranges['dilate_iter'][0],
                                             param_ranges['dilate_iter'][1]),
            'norm_kernel': trial.suggest_int('norm_kernel_base', param_ranges['norm_kernel'][0],
                                             param_ranges['norm_kernel'][1]) * 2 + 1,
            'denoise_h': trial.suggest_float('denoise_h', param_ranges['denoise_h'][0], param_ranges['denoise_h'][1]),
            'bin_block_base': trial.suggest_int('bin_block_base', param_ranges['bin_block'][0],
                                                param_ranges['bin_block'][1]),
            'bin_c': trial.suggest_float('bin_c', param_ranges['bin_c'][0], param_ranges['bin_c'][1])
        }
        params['bin_block_size'] = params['bin_block_base'] * 2 + 1

        scores = []
        fichiers = glob(os.path.join(INPUT_FOLDER, '*.*'))[:10]

        for f in fichiers:
            img = cv2.imread(f)
            if img is None: continue
            processed_img = pipeline_complet(img, params)
            score = evaluer_score_tesseract(processed_img)
            scores.append(score)

        mean_score = sum(scores) / len(scores) if scores else 0
        gui_app.on_trial_finish(trial, mean_score, params)
        return mean_score

    try:
        # --- C'EST ICI QUE LA MAGIE OPÈRE ---

        # 1. Configuration du Sampler SOBOL
        # Le QMCSampler utilise des séquences de Sobol pour couvrir l'espace uniformément
        # C'est beaucoup plus efficace que le Random pour l'initialisation
        sampler = QMCSampler(qmc_type='sobol')

        # NOTE : Si vous préférez l'intelligence Bayésienne (TPE) mais avec un démarrage large
        # vous pouvez utiliser ceci à la place :
        # sampler = TPESampler(n_startup_trials=50, multivariate=True)
        # Mais pour répondre à votre demande Sobol stricte, on garde QMCSampler.

        study = optuna.create_study(direction='maximize', sampler=sampler)

        # 2. WARM START (Injection des connaissances actuelles)
        # On injecte les paramètres qui vous ont donné le 53% (ou proche)
        # Cela force l'algo à évaluer ce point, et Sobol explorera autour et ailleurs.
        # (Adaptez ces valeurs avec celles de votre colonne "Valeur Optimale" actuelle !)
        print("Injection du point de départ connu (Warm Start)...")
        study.enqueue_trial({
            'line_h_size': 30,  # <--- Mettez vos valeurs gagnantes ici
            'line_v_size': 51,
            'dilate_iter': 1,
            'norm_kernel_base': 12,  # (Rappel: Valeur base = (25-1)/2 par exemple)
            'denoise_h': 9.58,
            'bin_block_base': 5,  # (Rappel: Valeur base = (11-1)/2)
            'bin_c': 15.59
        })

        gui_app.update_status(f"Démarrage Sobol ({n_trials} essais)...")
        study.optimize(objective, n_trials=n_trials)
        gui_app.update_status(f"✅ Terminé ! Meilleur score : {study.best_value:.2f}%")

    except Exception as e:
        gui_app.update_status(f"Erreur: {e}")
        print(e)  # Pour le debug console

# --- CLASSE GUI TKINTER ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR V2 - Mode Avancé")
        master.geometry("900x700")

        self.best_score_so_far = 0.0
        self.param_entries = {}
        self.optimal_labels = {}  # Stockage des labels de la colonne "Optimal"

        # Définition des paramètres : (Min, Max, Défaut)
        self.default_params = {
            'line_h': (20, 100, 30),  # Longueur ligne horizontale
            'line_v': (20, 100, 50),  # Longueur ligne verticale
            'dilate_iter': (1, 3, 1),  # [NOUVEAU] Épaisseur suppression lignes
            'norm_kernel': (15, 60, 25),  # [NOUVEAU] Taille flou normalisation (base)
            'denoise_h': (2.0, 20.0, 9.0),  # Force nettoyage
            'bin_block': (5, 55, 15),  # Taille bloc binarisation (base)
            'bin_c': (2.0, 25.0, 15.0)  # Constante binarisation
        }

        self.create_widgets()

    def create_widgets(self):
        # Style
        style = ttk.Style()
        style.configure("Bold.TLabel", font=('Helvetica', 9, 'bold'))

        # -- Zone Paramètres --
        param_frame = ttk.LabelFrame(self.master, text="Espace de Recherche (Paramètres)")
        param_frame.pack(padx=10, pady=10, fill="x")

        # En-têtes
        headers = ["Paramètre", "Min", "Max", "Défaut (Start)", "🏆 Valeur Optimale"]
        for col, text in enumerate(headers):
            ttk.Label(param_frame, text=text, style="Bold.TLabel").grid(row=0, column=col, padx=10, pady=5)

        # Génération des lignes
        row = 1
        for name, (min_val, max_val, default_val) in self.default_params.items():
            ttk.Label(param_frame, text=name).grid(row=row, column=0, padx=5, pady=2, sticky="e")

            self.param_entries[name] = {}

            # Min
            e_min = ttk.Entry(param_frame, width=8)
            e_min.insert(0, str(min_val))
            e_min.grid(row=row, column=1)
            self.param_entries[name]['min'] = e_min

            # Max
            e_max = ttk.Entry(param_frame, width=8)
            e_max.insert(0, str(max_val))
            e_max.grid(row=row, column=2)
            self.param_entries[name]['max'] = e_max

            # Défaut (Indicatif)
            lbl_def = ttk.Label(param_frame, text=str(default_val), foreground="gray")
            lbl_def.grid(row=row, column=3)

            # 🏆 COLONNE OPTIMALE (Vide au début)
            lbl_opt = ttk.Label(param_frame, text="-", foreground="blue", font=('Helvetica', 10, 'bold'))
            lbl_opt.grid(row=row, column=4, padx=10)
            self.optimal_labels[name] = lbl_opt

            row += 1

        # -- Zone Contrôle --
        ctrl_frame = ttk.Frame(self.master)
        ctrl_frame.pack(pady=5)

        ttk.Label(ctrl_frame, text="Nombre d'essais (Trials):").pack(side="left", padx=5)
        self.trials_entry = ttk.Entry(ctrl_frame, width=8)
        self.trials_entry.insert(0, "500")
        self.trials_entry.pack(side="left")

        self.btn_start = ttk.Button(ctrl_frame, text="▶ LANCER L'OPTIMISATION", command=self.start_optimization)
        self.btn_start.pack(side="left", padx=20)

        # -- Zone Log --
        self.status_label = ttk.Label(self.master, text="En attente...", font=('Helvetica', 10))
        self.status_label.pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(self.master, width=80, height=15)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)

    def get_param_ranges(self):
        ranges = {}
        for name, entries in self.param_entries.items():
            try:
                ranges[name] = (float(entries['min'].get()), float(entries['max'].get()))
            except ValueError:
                return None
        return ranges

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def on_trial_finish(self, trial, score, params):
        # Appelé depuis le thread d'optimisation
        msg = f"[Trial {trial.number}] Score: {score:.2f}%"
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

        # Si nouveau record
        if score > self.best_score_so_far:
            self.best_score_so_far = score
            self.update_status(f"🔥 NOUVEAU RECORD : {score:.2f}% (Trial {trial.number})")

            # MISE À JOUR DE LA COLONNE OPTIMALE
            # On utilise .after pour être thread-safe avec Tkinter
            self.master.after(0, self.update_optimal_display, params)

    def update_optimal_display(self, params):
        # Met à jour les labels bleus dans le tableau
        # On doit mapper les noms de params d'Optuna vers les noms de l'UI

        # Mapping direct
        mappings = {
            'line_h': params.get('line_h_size'),
            'line_v': params.get('line_v_size'),
            'dilate_iter': params.get('dilate_iter'),
            'denoise_h': params.get('denoise_h'),
            'bin_c': params.get('bin_c')
        }

        # Mapping avec calcul
        if 'norm_kernel_base' in params:
            mappings['norm_kernel'] = params['norm_kernel_base'] * 2 + 1

        if 'bin_block_base' in params:
            mappings['bin_block'] = params['bin_block_base'] * 2 + 1

        # Mise à jour graphique
        for name, value in mappings.items():
            if name in self.optimal_labels and value is not None:
                txt = f"{value:.2f}" if isinstance(value, float) else f"{value}"
                self.optimal_labels[name].config(text=txt)

    def start_optimization(self):
        ranges = self.get_param_ranges()
        if not ranges:
            self.update_status("Erreur dans les valeurs Min/Max")
            return

        try:
            n_trials = int(self.trials_entry.get())
        except:
            return

        self.btn_start.config(state="disabled")
        self.log_text.delete('1.0', tk.END)
        self.best_score_so_far = 0.0

        # Reset des labels optimaux
        for lbl in self.optimal_labels.values():
            lbl.config(text="-")

        thread = threading.Thread(target=run_optimization, args=(self, n_trials, ranges))
        thread.start()


if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()