import cv2
import numpy as np
import pytesseract
import optuna
from optuna.samplers import TPESampler, QMCSampler, NSGAIISampler
import os
from glob import glob
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FOLDER = 'test_scans'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- CORE FUNCTIONS (UNCHANGED) ---

def get_sharpness(image):
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_contrast(image):
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    return gray.std()

def get_tesseract_score(image):
    try:
        if image.shape[1] > 2500: image = cv2.resize(image, None, fx=0.5, fy=0.5)
        data = pytesseract.image_to_data(image, config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(x) for x in data['conf'] if int(x) != -1]
        return sum(confs) / len(confs) if confs else 0
    except: return 0

def evaluer_toutes_metriques(image):
    return get_tesseract_score(image), get_sharpness(image), get_contrast(image)

def remove_lines_param(gray_image, h_size, v_size, dilate_iter):
    thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    h_detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_detect = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    mask = cv2.addWeighted(h_detect, 1, v_detect, 1, 0.0)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=dilate_iter)
    result = gray_image.copy()
    result[mask > 0] = 255
    return result

def normalisation_division(image_gray, kernel_size):
    if kernel_size % 2 == 0: kernel_size += 1
    fond = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
    return cv2.divide(image_gray, fond, scale=255)

def pipeline_complet(image, params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])
    norm = normalisation_division(no_lines, params['norm_kernel'])
    denoised = cv2.fastNlMeansDenoising(norm, None, h=params['denoise_h'], templateWindowSize=7, searchWindowSize=21) if params['denoise_h'] > 0 else norm
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])


# --- OPTUNA & LOGGING ---

def run_optimization(gui_app, n_trials, param_ranges, fixed_params, algo_choice):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"resultats_metrics_{algo_choice}_{timestamp}.csv"
    
    # Build headers dynamically based on what's being optimized and what's fixed
    all_param_names = list(param_ranges.keys()) + list(fixed_params.keys())
    # Correctly map param names to their CSV header names
    header_map = {'line_h': 'line_h', 'line_v': 'line_v', 'norm_kernel': 'norm_kernel', 'denoise_h': 'denoise_h', 'bin_block': 'bin_block_size', 'bin_c': 'bin_c'}
    dynamic_headers = [header_map[p] for p in all_param_names if p in header_map]

    csv_headers = ['trial_id', 'score_tesseract', 'score_nettete', 'score_contraste'] + dynamic_headers
    if 'dilate_iter' not in fixed_params: csv_headers.append('dilate_iter')


    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)
    gui_app.update_status(f"Log détaillé : {csv_filename}")

    def objective(trial):
        params = fixed_params.copy()

        # Suggest active parameters
        if 'line_h' in param_ranges: params['line_h_size'] = trial.suggest_int('line_h_size', param_ranges['line_h'][0], param_ranges['line_h'][1])
        if 'line_v' in param_ranges: params['line_v_size'] = trial.suggest_int('line_v_size', param_ranges['line_v'][0], param_ranges['line_v'][1])
        if 'norm_kernel' in param_ranges: params['norm_kernel'] = trial.suggest_int('norm_kernel_base', param_ranges['norm_kernel'][0], param_ranges['norm_kernel'][1]) * 2 + 1
        if 'denoise_h' in param_ranges: params['denoise_h'] = trial.suggest_float('denoise_h', param_ranges['denoise_h'][0], param_ranges['denoise_h'][1])
        if 'bin_block' in param_ranges:
            base_val = trial.suggest_int('bin_block_base', param_ranges['bin_block'][0], param_ranges['bin_block'][1])
            params['bin_block_size'] = base_val * 2 + 1
        if 'bin_c' in param_ranges: params['bin_c'] = trial.suggest_float('bin_c', param_ranges['bin_c'][0], param_ranges['bin_c'][1])
        
        list_tess, list_sharp, list_cont = [], [], []
        fichiers = glob(os.path.join(INPUT_FOLDER, '*.*'))[:10]
        for f in fichiers:
            img = cv2.imread(f)
            if img is None: continue
            processed_img = pipeline_complet(img, params)
            tess, sharp, cont = evaluer_toutes_metriques(processed_img)
            list_tess.append(tess); list_sharp.append(sharp); list_cont.append(cont)

        avg_tess = sum(list_tess) / len(list_tess) if list_tess else 0
        avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
        avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0

        try:
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                # Build row dynamically
                row_data = [trial.number, round(avg_tess, 4), round(avg_sharp, 2), round(avg_cont, 2)]
                if 'line_h_size' in params: row_data.append(params['line_h_size'])
                if 'line_v_size' in params: row_data.append(params['line_v_size'])
                if 'norm_kernel' in params: row_data.append(params['norm_kernel'])
                if 'denoise_h' in params: row_data.append(round(params['denoise_h'], 2))
                if 'bin_block_size' in params: row_data.append(params['bin_block_size'])
                if 'bin_c' in params: row_data.append(round(params['bin_c'], 2))
                if 'dilate_iter' in params: row_data.append(params['dilate_iter'])
                writer.writerow(row_data)
        except Exception as e: print(f"Erreur CSV: {e}")

        gui_app.on_trial_finish(trial, avg_tess, avg_sharp, avg_cont, params)
        return avg_tess

    sampler = TPESampler(n_startup_trials=20)
    if algo_choice == "Sobol (Quasi-Monte Carlo)": sampler = QMCSampler(qmc_type='sobol')
    elif algo_choice == "NSGA-II (Génétique)": sampler = NSGAIISampler(population_size=50, mutation_prob=0.05)
    
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    gui_app.update_status(f"✅ Terminé ! Meilleur Tesseract : {study.best_value:.2f}%")


# --- GUI ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR V5 - Sélection de Paramètres")
        master.geometry("1000x750")

        self.best_score_so_far = 0.0
        self.param_entries = {}
        self.optimal_labels = {}
        self.default_params = {
            'line_h': (30, 70, 45), 'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75), 'denoise_h': (2.0, 20.0, 9.0),
            'bin_block': (30, 100, 60), 'bin_c': (10, 25.0, 15.0)
        }
        self.param_enabled_vars = {name: tk.BooleanVar(value=True) for name in self.default_params}
        
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Bold.TLabel", font=('Helvetica', 9, 'bold'))

        param_frame = ttk.LabelFrame(self.master, text="Espace de Recherche")
        param_frame.pack(padx=10, pady=10, fill="x")

        headers = ["Actif", "Paramètre", "Min", "Max", "Défaut", "🏆 Optimal"]
        for col, text in enumerate(headers):
            ttk.Label(param_frame, text=text, style="Bold.TLabel").grid(row=0, column=col, padx=5, pady=5)

        row = 1
        for name, (min_val, max_val, default_val) in self.default_params.items():
            # Checkbox
            chk = ttk.Checkbutton(param_frame, variable=self.param_enabled_vars[name])
            chk.grid(row=row, column=0, padx=5)
            
            ttk.Label(param_frame, text=name).grid(row=row, column=1, padx=5, pady=2, sticky="e")
            self.param_entries[name] = {}
            for i, val in enumerate([min_val, max_val]):
                entry = ttk.Entry(param_frame, width=8)
                entry.insert(0, str(val))
                entry.grid(row=row, column=2 + i)
                self.param_entries[name][['min', 'max'][i]] = entry
            
            ttk.Label(param_frame, text=str(default_val), foreground="gray").grid(row=row, column=4)
            lbl_opt = ttk.Label(param_frame, text="-", foreground="blue", font=('Helvetica', 10, 'bold'))
            lbl_opt.grid(row=row, column=5, padx=10)
            self.optimal_labels[name] = lbl_opt
            row += 1
            
        ttk.Label(param_frame, text="dilate_iter").grid(row=row, column=1, padx=5, pady=2, sticky="e")
        ttk.Label(param_frame, text="FIXE", foreground="green", font=('Helvetica', 9, 'bold')).grid(row=row, column=0, padx=5)
        ttk.Label(param_frame, text="2", foreground="gray").grid(row=row, column=4)
        row += 1

        ctrl_frame = ttk.LabelFrame(self.master, text="Configuration")
        ctrl_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Label(ctrl_frame, text="Algorithme :").pack(side="left", padx=5)
        self.algo_var = tk.StringVar(value="NSGA-II (Génétique)")
        self.algo_combo = ttk.Combobox(ctrl_frame, textvariable=self.algo_var, state="readonly", width=25, values=("TPE (Bayésien)", "Sobol (Quasi-Monte Carlo)", "NSGA-II (Génétique)"))
        self.algo_combo.pack(side="left", padx=5)
        
        ttk.Label(ctrl_frame, text="Nb Essais :").pack(side="left", padx=10)
        self.trials_entry = ttk.Entry(ctrl_frame, width=8); self.trials_entry.insert(0, "100")
        self.trials_entry.pack(side="left")
        
        self.btn_start = ttk.Button(ctrl_frame, text="▶ LANCER", command=self.start_optimization)
        self.btn_start.pack(side="left", padx=20)
        
        self.status_label = ttk.Label(self.master, text="Prêt.", font=('Helvetica', 10, 'italic'))
        self.status_label.pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.master, width=90, height=12)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)

    def get_optim_config(self):
        active_ranges = {}
        fixed_params = {'dilate_iter': 2} # Always fixed
        for name, (min_val, max_val, default_val) in self.default_params.items():
            if self.param_enabled_vars[name].get():
                try:
                    active_ranges[name] = (float(self.param_entries[name]['min'].get()), float(self.param_entries[name]['max'].get()))
                except ValueError: return None, None
            else:
                # Assign default value for fixed params, handling derived params correctly
                if name == 'norm_kernel': fixed_params[name] = default_val * 2 + 1
                elif name == 'bin_block': fixed_params['bin_block_size'] = default_val * 2 + 1
                elif name == 'line_h': fixed_params['line_h_size'] = default_val
                elif name == 'line_v': fixed_params['line_v_size'] = default_val
                else: fixed_params[name] = default_val
        return active_ranges, fixed_params

    def update_status(self, msg): self.status_label.config(text=msg)

    def on_trial_finish(self, trial, t_score, s_score, c_score, params):
        msg = f"[Trial {trial.number}] Tess: {t_score:.2f}% | Netteté: {s_score:.1f} | Contraste: {c_score:.1f}"
        self.log_text.insert(tk.END, msg + "\n"); self.log_text.see(tk.END)
        if t_score > self.best_score_so_far:
            self.best_score_so_far = t_score
            self.update_status(f"🔥 RECORD TESSERACT : {t_score:.2f}% (Trial {trial.number})")
            self.master.after(0, self.update_optimal_display, params)

    def update_optimal_display(self, params):
        mappings = {'line_h': params.get('line_h_size'), 'line_v': params.get('line_v_size'),
                    'denoise_h': params.get('denoise_h'), 'bin_c': params.get('bin_c')}
        if 'norm_kernel' in params: mappings['norm_kernel'] = (params['norm_kernel']-1)//2
        if 'bin_block_size' in params: mappings['bin_block'] = (params['bin_block_size']-1)//2
        for name, value in mappings.items():
            if name in self.optimal_labels and value is not None:
                self.optimal_labels[name].config(text=f"{value:.2f}" if isinstance(value, float) else f"{value}")

    def start_optimization(self):
        active_ranges, fixed_params = self.get_optim_config()
        if active_ranges is None: messagebox.showerror("Erreur de Saisie", "Vérifiez que les Min/Max sont des nombres valides."); return
        try:
            n_trials = int(self.trials_entry.get())
            algo = self.algo_var.get()
        except ValueError: messagebox.showerror("Erreur de Saisie", "Le nombre d'essais doit être un entier."); return

        self.btn_start.config(state="disabled")
        self.log_text.delete('1.0', tk.END)
        self.best_score_so_far = 0.0
        for lbl in self.optimal_labels.values(): lbl.config(text="-")

        thread = threading.Thread(target=run_optimization, args=(self, n_trials, active_ranges, fixed_params, algo))
        thread.start()

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()