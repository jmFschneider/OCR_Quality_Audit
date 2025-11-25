import cv2
import numpy as np
import pytesseract
import optuna
import os
from glob import glob
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog
from functools import partial

# --- CONFIGURATION FIXE ---
INPUT_FOLDER = 'test_scans'
# Si vous êtes sous Windows, décommentez et ajustez la ligne suivante si Tesseract n'est pas dans le PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- LOGIQUE CORE OCR (Fonctions de l'ancien script) ---

def remove_lines_param(gray_image, h_size, v_size):
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    mask_lignes = cv2.addWeighted(detect_horizontal, 1, detect_vertical, 1, 0.0)
    mask_lignes = cv2.dilate(mask_lignes, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    resultat = gray_image.copy()
    resultat[mask_lignes > 0] = 255
    return resultat


def normalisation_division(image_gray):
    fond = cv2.GaussianBlur(image_gray, (51, 51), 0)
    return cv2.divide(image_gray, fond, scale=255)


def evaluer_score_tesseract(image):
    try:
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

    no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'])
    norm = normalisation_division(no_lines)

    if params['denoise_h'] > 0:
        denoised = cv2.fastNlMeansDenoising(norm, None, h=params['denoise_h'], templateWindowSize=7,
                                            searchWindowSize=21)
    else:
        denoised = norm

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
    """ Exécute Optuna dans un thread séparé """

    def objective(trial):
        # Définition de l'espace de recherche (basé sur les plages entrées par l'utilisateur)

        # Lignes (Integers)
        params = {
            'line_h_size': trial.suggest_int('line_h_size', param_ranges['line_h'][0], param_ranges['line_h'][1]),
            'line_v_size': trial.suggest_int('line_v_size', param_ranges['line_v'][0], param_ranges['line_v'][1]),

            # Nettoyage bruit (Float)
            'denoise_h': trial.suggest_float('denoise_h', param_ranges['denoise_h'][0], param_ranges['denoise_h'][1]),

            # Binarisation (Integers et Float)
            'bin_block_base': trial.suggest_int('bin_block_base', param_ranges['bin_block'][0],
                                                param_ranges['bin_block'][1]),
            'bin_c': trial.suggest_float('bin_c', param_ranges['bin_c'][0], param_ranges['bin_c'][1])
        }

        # Calcul du block_size réel (impair)
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

        # Mise à jour du log dans la GUI (Score global)
        gui_app.update_log(
            f"Essai {trial.number}: Score Moyen = {mean_score:.2f}% (h={params['denoise_h']:.1f}, BSize={params['bin_block_size']}, C={params['bin_c']:.1f})")

        return mean_score

    # Démarrage de l'étude Optuna
    gui_app.update_status("Recherche en cours...")
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        gui_app.display_results(study)
    except Exception as e:
        gui_app.update_status(f"Erreur: {e}")


# --- CLASSE GUI TKINTER ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR Bayésien")

        self.param_entries = {}
        self.default_params = {
            'line_h': (20, 80, 50),
            'line_v': (20, 80, 50),
            'denoise_h': (2.0, 15.0, 7.0),
            'bin_block': (5, 45, 15),
            'bin_c': (2.0, 20.0, 10.0)
        }

        self.create_widgets()

    def create_widgets(self):
        # Cadre Paramètres
        param_frame = ttk.LabelFrame(self.master, text="Paramètres d'Optimisation")
        param_frame.pack(padx=10, pady=10, fill="x")

        # Entrées : Nom, Min, Max, Défaut
        row = 0
        ttk.Label(param_frame, text="Paramètre").grid(row=row, column=0, padx=5, pady=2)
        ttk.Label(param_frame, text="Min").grid(row=row, column=1, padx=5, pady=2)
        ttk.Label(param_frame, text="Max").grid(row=row, column=2, padx=5, pady=2)
        ttk.Label(param_frame, text="Défaut").grid(row=row, column=3, padx=5, pady=2)
        row += 1

        for name, (min_val, max_val, default_val) in self.default_params.items():
            ttk.Label(param_frame, text=name).grid(row=row, column=0, padx=5, pady=2, sticky="w")

            self.param_entries[name] = {}

            # Min
            min_entry = ttk.Entry(param_frame, width=8)
            min_entry.insert(0, str(min_val))
            min_entry.grid(row=row, column=1, padx=5, pady=2)
            self.param_entries[name]['min'] = min_entry

            # Max
            max_entry = ttk.Entry(param_frame, width=8)
            max_entry.insert(0, str(max_val))
            max_entry.grid(row=row, column=2, padx=5, pady=2)
            self.param_entries[name]['max'] = max_entry

            # Défaut
            ttk.Label(param_frame, text=str(default_val)).grid(row=row, column=3, padx=5, pady=2)

            row += 1

        # Nombre d'essais
        ttk.Label(param_frame, text="Nb d'Essais (n_trials)").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.trials_entry = ttk.Entry(param_frame, width=8)
        self.trials_entry.insert(0, "50")
        self.trials_entry.grid(row=row, column=1, padx=5, pady=5)

        # Bouton Démarrer
        self.start_button = ttk.Button(self.master, text="🚀 Démarrer l'Optimisation", command=self.start_optimization)
        self.start_button.pack(pady=10)

        # Cadre Statut et Log
        log_frame = ttk.LabelFrame(self.master, text="Statut & Historique des Scores (éléments séparés)")
        log_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.status_label = ttk.Label(log_frame, text="Prêt. (Mettez les images dans scans_input_test)")
        self.status_label.pack(fill="x", padx=5, pady=2)

        self.log_text = scrolledtext.ScrolledText(log_frame, width=60, height=10)
        self.log_text.pack(padx=5, pady=5, fill="both", expand=True)

        # Cadre Résultats Finaux
        result_frame = ttk.LabelFrame(self.master, text="Résultats Optimaux (Score Global)")
        result_frame.pack(padx=10, pady=10, fill="x")

        self.best_score_label = ttk.Label(result_frame, text="Meilleur Score: N/A")
        self.best_score_label.pack(padx=5, pady=2, anchor="w")
        self.best_params_label = ttk.Label(result_frame, text="Meilleurs Paramètres: N/A")
        self.best_params_label.pack(padx=5, pady=2, anchor="w")

    def get_param_ranges(self):
        ranges = {}
        for name, entries in self.param_entries.items():
            try:
                min_val = float(entries['min'].get())
                max_val = float(entries['max'].get())
                ranges[name] = (min_val, max_val)
            except ValueError:
                self.update_status(f"Erreur: Valeur non valide pour {name}")
                return None
        return ranges

    def update_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()  # Force l'actualisation

    def update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.yview(tk.END)  # Scroll automatique
        self.master.update_idletasks()

    def display_results(self, study):
        best = study.best_params

        # Calcul du block_size réel
        real_block_size = best['bin_block_base'] * 2 + 1

        self.best_score_label.config(text=f"Meilleur Score: {study.best_value:.2f}% (Score Global)")

        params_str = (
            f"Lignes H/V: {best['line_h_size']}/{best['line_v_size']} | "
            f"Denoise h: {best['denoise_h']:.2f} | "
            f"BlockSize: {real_block_size} | "
            f"Const C: {best['bin_c']:.2f}"
        )
        self.best_params_label.config(text=f"Meilleurs Paramètres: {params_str}")
        self.update_status("✅ OPTIMISATION TERMINÉE!")

    def start_optimization(self):
        param_ranges = self.get_param_ranges()
        if not param_ranges: return

        try:
            n_trials = int(self.trials_entry.get())
        except ValueError:
            self.update_status("Erreur: Le nombre d'essais doit être un entier.")
            return

        # Vider les résultats précédents
        self.log_text.delete('1.0', tk.END)
        self.best_score_label.config(text="Meilleur Score: Recherche...")
        self.best_params_label.config(text="Meilleurs Paramètres: En cours...")

        # Démarrer l'optimisation dans un thread séparé
        # Cela empêche l'interface de se figer pendant les 50 essais
        thread = threading.Thread(target=run_optimization, args=(self, n_trials, param_ranges))
        thread.start()


if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Dossier créé : {INPUT_FOLDER}. Veuillez y placer vos images de test.")

    # Tentative d'initialisation de PyTesseract pour les erreurs précoces
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        root = tk.Tk()
        root.withdraw()  # Cache la fenêtre principale inutile
        simpledialog.messagebox.showerror("Erreur Tesseract",
                                          "Tesseract n'est pas trouvé. Veuillez vérifier le PATH ou décommenter la ligne pytesseract.pytesseract.tesseract_cmd = r'...' dans le script.")
        exit()

    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()