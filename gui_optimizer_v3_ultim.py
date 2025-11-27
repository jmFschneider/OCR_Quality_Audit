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
import multiprocessing
from itertools import repeat

# --- CONFIGURATION ---
INPUT_FOLDER = 'test_scans'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Activer les optimisations OpenCV
cv2.setUseOptimized(True)

# Tenter d'activer OpenCL pour l'accélération GPU si disponible
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL activé pour OpenCV (accélération GPU potentielle).")
else:
    print("OpenCL non disponible ou non activé pour OpenCV.")

# --- CORE FUNCTIONS (UNCHANGED) ---

def get_sharpness(image):
    gray = image # Image is already grayscale
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_contrast(image):
    gray = image # Image is already grayscale
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
    gray = image # Image is already grayscale
    no_lines = remove_lines_param(gray, params['line_h_size'], params['line_v_size'], params['dilate_iter'])
    norm = normalisation_division(no_lines, params['norm_kernel'])
    denoised = cv2.fastNlMeansDenoising(norm, None, h=params['denoise_h'], templateWindowSize=7, searchWindowSize=21) if params['denoise_h'] > 0 else norm
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params['bin_block_size'], params['bin_c'])






import scipy_optimizer

def process_image_data_wrapper(args):
    """
    Wrapper function to process a single image's data. Takes a tuple of (image_data, params)
    as input to be compatible with pool.map.
    """
    # FORCER LE MONO-THREADING : Crucial pour les performances en multiprocessing.
    # Empêche les bibliothèques sous-jacentes (OpenCV, Tesseract/OpenMP, MKL)
    # de créer leurs propres threads, ce qui provoquerait une sur-sollicitation.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    cv2.setNumThreads(1)

    img, params = args
    if img is None:
        return None
    processed_img = pipeline_complet(img, params)
    return evaluer_toutes_metriques(processed_img)

# --- OPTUNA & LOGGING ---

def run_optuna_optimization(gui_app, n_trials, param_ranges, fixed_params, algo_choice):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"resultats_metrics_{algo_choice}_{timestamp}.csv"
    
    all_param_names = list(param_ranges.keys())
    
    header_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'denoise_h': 'denoise_h', 'bin_block': 'bin_block_size', 'bin_c': 'bin_c'}
    dynamic_headers = [header_map[p] for p in all_param_names if p in header_map]
    csv_headers = ['trial_id', 'score_tesseract', 'score_nettete', 'score_contraste'] + dynamic_headers

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)
    gui_app.update_log_from_thread(f"Log détaillé (legacy) : {csv_filename}")

    def objective(trial):
        if gui_app.cancellation_requested.is_set():
            raise optuna.exceptions.TrialPruned("Annulation demandée par l'utilisateur.")

        params = fixed_params.copy()
        
        # Build params dict for this trial
        current_params = {}
        if 'line_h' in param_ranges: current_params['line_h_size'] = trial.suggest_int('line_h_size', int(param_ranges['line_h'][0]), int(param_ranges['line_h'][1]))
        if 'line_v' in param_ranges: current_params['line_v_size'] = trial.suggest_int('line_v_size', int(param_ranges['line_v'][0]), int(param_ranges['line_v'][1]))
        if 'norm_kernel' in param_ranges: current_params['norm_kernel'] = trial.suggest_int('norm_kernel_base', int(param_ranges['norm_kernel'][0]), int(param_ranges['norm_kernel'][1])) * 2 + 1
        if 'denoise_h' in param_ranges: current_params['denoise_h'] = trial.suggest_float('denoise_h', param_ranges['denoise_h'][0], param_ranges['denoise_h'][1])
        if 'bin_block' in param_ranges:
            base_val = trial.suggest_int('bin_block_base', int(param_ranges['bin_block'][0]), int(param_ranges['bin_block'][1]))
            current_params['bin_block_size'] = base_val * 2 + 1
        if 'bin_c' in param_ranges: current_params['bin_c'] = trial.suggest_float('bin_c', param_ranges['bin_c'][0], param_ranges['bin_c'][1])
        
        params.update(current_params)

        avg_tess, avg_sharp, avg_cont = gui_app.evaluate_pipeline(params)

        try:
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                row_data = [trial.number, round(avg_tess, 4), round(avg_sharp, 2), round(avg_cont, 2)]
                for header in dynamic_headers:
                    row_data.append(params.get(header))
                writer.writerow(row_data)
        except Exception as e: 
            gui_app.update_log_from_thread(f"Erreur CSV (legacy): {e}")

        gui_app.on_trial_finish(trial.number, avg_tess, avg_sharp, avg_cont, params)
        return avg_tess

    def cancellation_callback(study, trial):
        if gui_app.cancellation_requested.is_set():
            study.stop()

    sampler_map = {
        "TPE (Bayésien)": TPESampler(n_startup_trials=20),
        "Sobol (Quasi-Monte Carlo)": QMCSampler(qmc_type='sobol', scramble=True),
        "NSGA-II (Génétique)": NSGAIISampler(population_size=50, mutation_prob=0.05)
    }
    sampler = sampler_map.get(algo_choice, TPESampler())
    
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[cancellation_callback])

    if gui_app.cancellation_requested.is_set():
        gui_app.update_status_from_thread("⏹️ Optuna annulé ! Sauvegarde des résultats...")
    else:
        gui_app.update_status_from_thread(f"✅ Optuna terminé ! Meilleur Tesseract : {study.best_value:.2f}%")
    
    gui_app.finalize_run()


# --- GUI ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR V7 - Parallélisé")
        master.geometry("1000x800")

        self.best_score_so_far = 0.0
        self.trial_count = 0
        self.param_entries = {}
        self.optimal_labels = {}
        self.loaded_images = [] # To store pre-loaded images
        self.default_params = {
            'line_h': (30, 70, 45), 'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75), 'denoise_h': (2.0, 20.0, 9.0),
            'bin_block': (30, 100, 60), 'bin_c': (10, 25.0, 15.0)
        }
        self.param_enabled_vars = {name: tk.BooleanVar(value=True) for name in self.default_params}
        self.cancellation_requested = threading.Event()
        self.results_data = []
        
        self.optuna_algos = ["TPE (Bayésien)", "Sobol (Quasi-Monte Carlo)", "NSGA-II (Génétique)"]
        self.scipy_algos = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

        self.image_files = []
        self.create_widgets()
        self.refresh_image_list()

    def pre_load_images(self):
        """Loads all images from the input folder into memory, in grayscale."""
        self.update_log_from_thread("Pré-chargement des images en mémoire (en niveaux de gris)...")
        self.loaded_images = []
        if not self.image_files:
            messagebox.showwarning("Aucune image", f"Aucune image trouvée dans le dossier {INPUT_FOLDER}. Cliquez sur 🔄 pour rafraîchir.")
            return
            
        for f in self.image_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.loaded_images.append(img)
        self.update_log_from_thread(f"{len(self.loaded_images)} images chargées en mémoire.")

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Bold.TLabel", font=('Helvetica', 9, 'bold'))

        param_frame = ttk.LabelFrame(self.master, text="Espace de Recherche des Paramètres")
        param_frame.pack(padx=10, pady=10, fill="x")
        headers = ["Actif", "Paramètre", "Min", "Max", "Défaut", "🏆 Optimal"]
        for col, text in enumerate(headers):
            ttk.Label(param_frame, text=text, style="Bold.TLabel").grid(row=0, column=col, padx=5, pady=5)
        row = 1
        for name, (min_val, max_val, default_val) in self.default_params.items():
            ttk.Checkbutton(param_frame, variable=self.param_enabled_vars[name]).grid(row=row, column=0, padx=5)
            ttk.Label(param_frame, text=name).grid(row=row, column=1, padx=5, pady=2, sticky="e")
            self.param_entries[name] = {}
            for i, val in enumerate([min_val, max_val]):
                entry = ttk.Entry(param_frame, width=8); entry.insert(0, str(val)); entry.grid(row=row, column=2 + i)
                self.param_entries[name][['min', 'max'][i]] = entry
            ttk.Label(param_frame, text=str(default_val), foreground="gray").grid(row=row, column=4)
            lbl_opt = ttk.Label(param_frame, text="-", foreground="blue", font=('Helvetica', 10, 'bold')); lbl_opt.grid(row=row, column=5, padx=10)
            self.optimal_labels[name] = lbl_opt
            row += 1
        ttk.Label(param_frame, text="dilate_iter").grid(row=row, column=1, padx=5, pady=2, sticky="e")
        ttk.Label(param_frame, text="FIXE", foreground="green", font=('Helvetica', 9, 'bold')).grid(row=row, column=0, padx=5)
        ttk.Label(param_frame, text="2", foreground="gray").grid(row=row, column=4)
        
        ctrl_frame = ttk.LabelFrame(self.master, text="Configuration de l'Optimisation")
        ctrl_frame.pack(padx=10, pady=5, fill="x")

        # --- Utilisation de GRID pour une disposition stable ---
        ctrl_frame.columnconfigure(2, weight=1)

        # Colonne 0: Bibliothèque
        ttk.Label(ctrl_frame, text="Bibliothèque :").grid(row=0, column=0, sticky="w", padx=5)
        self.lib_var = tk.StringVar(value="Optuna")
        self.lib_combo = ttk.Combobox(ctrl_frame, textvariable=self.lib_var, state="readonly", width=10, values=["Optuna", "Scipy"])
        self.lib_combo.grid(row=0, column=0, sticky="w", padx=(80,5))
        self.lib_combo.bind("<<ComboboxSelected>>", self.on_library_select)
        
        # Colonne 1: Algorithme
        ttk.Label(ctrl_frame, text="Algorithme :").grid(row=0, column=1, sticky="w", padx=5)
        self.algo_var = tk.StringVar(value=self.optuna_algos[0])
        self.algo_combo = ttk.Combobox(ctrl_frame, textvariable=self.algo_var, state="readonly", width=25, values=self.optuna_algos)
        self.algo_combo.grid(row=0, column=1, sticky="w", padx=(70,5))

        # Colonne 2: Options dynamiques (Frames non encore placées)
        self.optuna_frame = ttk.Frame(ctrl_frame)
        ttk.Label(self.optuna_frame, text="Nb Essais :").pack(side="left")
        self.trials_entry = ttk.Entry(self.optuna_frame, width=8); self.trials_entry.insert(0, "100")
        self.trials_entry.pack(side="left", padx=5)
        
        self.scipy_frame = ttk.Frame(ctrl_frame)
        ttk.Label(self.scipy_frame, text="Exposant Sobol (2^n):").pack(side="left")
        self.sobol_exponent_var = tk.StringVar(value="5")
        self.sobol_exponent_var.trace_add("write", self.update_sobol_points_label)
        self.sobol_exponent_entry = ttk.Entry(self.scipy_frame, width=5, textvariable=self.sobol_exponent_var)
        self.sobol_exponent_entry.pack(side="left", padx=2)
        self.sobol_points_label = ttk.Label(self.scipy_frame, text="= 32 points")
        self.sobol_points_label.pack(side="left", padx=(0,10))
        ttk.Label(self.scipy_frame, text="Itérations/point:").pack(side="left")
        self.scipy_iter_entry = ttk.Entry(self.scipy_frame, width=8); self.scipy_iter_entry.insert(0, "15")
        self.scipy_iter_entry.pack(side="left", padx=5)

        # Colonne 3: Compteur d'images
        img_frame = ttk.Frame(ctrl_frame)
        img_frame.grid(row=0, column=3, sticky="e", padx=5)
        self.image_count_label = ttk.Label(img_frame, text="Images: 0")
        self.image_count_label.pack(side="left")
        self.btn_refresh_images = ttk.Button(img_frame, text="🔄", width=3, command=self.refresh_image_list)
        self.btn_refresh_images.pack(side="left", padx=5)

        # Colonne 4: Boutons d'action
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=0, column=4, sticky="e", padx=5)
        self.btn_start = ttk.Button(btn_frame, text="▶ LANCER", command=self.start_optimization)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(btn_frame, text="⏹ ANNULER", command=self.request_cancellation, state="disabled")
        self.btn_cancel.pack(side="left", padx=5)
        
        self.status_label = ttk.Label(self.master, text="Prêt.", font=('Helvetica', 10, 'italic'))
        self.status_label.pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.master, width=90, height=12)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.on_library_select(None)

    def on_library_select(self, event):
        # On cache d'abord les deux frames d'options
        self.scipy_frame.grid_remove()
        self.optuna_frame.grid_remove()

        lib = self.lib_var.get()
        if lib == "Optuna":
            self.algo_combo.config(values=self.optuna_algos)
            self.algo_var.set(self.optuna_algos[0])
            self.optuna_frame.grid(row=0, column=2, sticky="w")
        else:
            self.algo_combo.config(values=self.scipy_algos)
            self.algo_var.set(self.scipy_algos[0])
            self.scipy_frame.grid(row=0, column=2, sticky="w")


    def update_sobol_points_label(self, *args):
        try:
            exponent = int(self.sobol_exponent_var.get())
            if exponent > 16: # Limite pour éviter les très grands nombres
                self.sobol_points_label.config(text="! > 65536")
                return
            n_points = 2**exponent
            self.sobol_points_label.config(text=f"= {n_points} points")
        except ValueError:
            self.sobol_points_label.config(text="= Invalide")

    def refresh_image_list(self):
        """Met à jour la liste des fichiers image et l'affichage dans l'UI."""
        self.image_files = glob(os.path.join(INPUT_FOLDER, '*.*'))
        self.image_count_label.config(text=f"Images: {len(self.image_files)}")

    def get_optim_config(self):
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}
        param_order = []
        for name in self.default_params.keys():
            if self.param_enabled_vars[name].get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())
                    active_ranges[name] = (min_val, max_val)
                    param_order.append(name)
                except ValueError: return None, None, None
            else:
                default_val = self.default_params[name][2]
                if name == 'norm_kernel': fixed_params['norm_kernel'] = int(default_val) * 2 + 1
                elif name == 'bin_block': fixed_params['bin_block_size'] = int(default_val) * 2 + 1
                elif name == 'line_h': fixed_params['line_h_size'] = default_val
                elif name == 'line_v': fixed_params['line_v_size'] = default_val
                else: fixed_params[name] = default_val
        return active_ranges, fixed_params, param_order
    
    def evaluate_pipeline(self, params):
        if not self.loaded_images:
            return 0, 0, 0

        pool_args = zip(self.loaded_images, repeat(params))
        
        # Limiter la taille du pool au nombre de tâches si < cpu_count
        pool_size = min(len(self.loaded_images), os.cpu_count())
        
        try:
            with multiprocessing.Pool(processes=pool_size) as pool:
                results = pool.map(process_image_data_wrapper, pool_args)
            
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return 0, 0, 0

            list_tess, list_sharp, list_cont = zip(*valid_results)

            avg_tess = sum(list_tess) / len(list_tess) if list_tess else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_tess, avg_sharp, avg_cont

        except Exception as e:
            print(f"Erreur de multiprocessing, passage en mode séquentiel: {e}")
            list_tess, list_sharp, list_cont = [], [], []
            for img in self.loaded_images:
                processed_img = pipeline_complet(img, params)
                tess, sharp, cont = evaluer_toutes_metriques(processed_img)
                list_tess.append(tess); list_sharp.append(sharp); list_cont.append(cont)
            
            avg_tess = sum(list_tess) / len(list_tess) if list_tess else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_tess, avg_sharp, avg_cont

    def start_optimization(self):
        self.cancellation_requested.clear()
        self.results_data.clear()
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

        active_ranges, fixed_params, param_order = self.get_optim_config()
        if active_ranges is None: 
            messagebox.showerror("Erreur de Saisie", "Vérifiez que les Min/Max sont des nombres valides.")
            self.finalize_run()
            return

        self.log_text.delete('1.0', tk.END)
        self.best_score_so_far = 0.0
        self.trial_count = 0
        for lbl in self.optimal_labels.values(): lbl.config(text="-")

        # Pre-load images once before starting the thread
        self.pre_load_images()

        library = self.lib_var.get()
        algo = self.algo_var.get()

        if library == "Optuna":
            try:
                n_trials = int(self.trials_entry.get())
                thread = threading.Thread(target=run_optuna_optimization, args=(self, n_trials, active_ranges, fixed_params, algo))
                thread.start()
            except ValueError: 
                messagebox.showerror("Erreur de Saisie", "Le nombre d'essais doit être un entier.")
                self.finalize_run()
        
        elif library == "Scipy":
            try:
                exponent = int(self.sobol_exponent_var.get())
                n_sobol = 2**exponent
                n_iter = int(self.scipy_iter_entry.get())
                bounds = [active_ranges[p] for p in param_order]

                def objective_for_scipy(param_values):
                    # DEBUG print(f"DEBUG: Scipy objective called with {param_values}")
                    if self.cancellation_requested.is_set(): return 0 # Stop early

                    self.trial_count += 1
                    params = fixed_params.copy()
                    current_params = {}
                    for i, name in enumerate(param_order):
                        val = param_values[i]
                        p_name_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'bin_block': 'bin_block_size', 'denoise_h': 'denoise_h', 'bin_c': 'bin_c'}
                        
                        mapped_name = p_name_map.get(name)
                        if not mapped_name: continue

                        if name in ['norm_kernel', 'bin_block']:
                             current_params[mapped_name] = int(val) * 2 + 1
                        elif name in ['line_h', 'line_v']:
                             current_params[mapped_name] = int(val)
                        else:
                             current_params[mapped_name] = val
                    
                    params.update(current_params)
                    t_score, s_score, c_score = self.evaluate_pipeline(params)
                    self.on_trial_finish(self.trial_count, t_score, s_score, c_score, params)
                    return -t_score

                thread = threading.Thread(target=self.run_scipy_thread, args=(objective_for_scipy, bounds, algo, n_sobol, n_iter))
                thread.start()

            except ValueError:
                messagebox.showerror("Erreur de Saisie", "Les points Sobol et Itérations doivent être des entiers.")
                self.finalize_run()

    def run_scipy_thread(self, objective, bounds, algo, n_sobol, n_iter):
        result = scipy_optimizer.run_scipy_optimization(objective, bounds, algo, n_sobol, n_iter, self.update_log_from_thread, self.cancellation_requested)
        
        # Si aucun résultat n'a été trouvé (par ex. annulation immédiate)
        if result.get("x") is None:
            self.update_status_from_thread("⏹️ Scipy annulé, aucun résultat à afficher.")
            self.finalize_run()
            return

        best_params_values = result.x
        
        # Reconstruct the full params dict to display it
        _, fixed_params, param_order = self.get_optim_config()
        final_params = fixed_params.copy()
        current_params = {}
        for i, name in enumerate(param_order):
            # Vérifier si l'indice est valide pour best_params_values
            if i >= len(best_params_values):
                continue
            val = best_params_values[i]
            p_name_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'bin_block': 'bin_block_size', 'denoise_h': 'denoise_h', 'bin_c': 'bin_c'}
            mapped_name = p_name_map.get(name)
            if not mapped_name: continue

            if name in ['norm_kernel', 'bin_block']:
                 current_params[mapped_name] = int(val) * 2 + 1
            else:
                 current_params[mapped_name] = val
        final_params.update(current_params)

        if self.cancellation_requested.is_set():
            self.update_status_from_thread(f"⏹️ Scipy annulé ! Sauvegarde des résultats...")
        else:
            self.update_status_from_thread(f"✅ Scipy terminé ! Meilleur score (négatif) : {result.fun:.4f}")

        self.master.after(0, self.update_optimal_display, final_params)
        self.finalize_run()

    def request_cancellation(self):
        """Signale l'annulation de l'optimisation en cours."""
        self.cancellation_requested.set()
        self.update_status_from_thread("🛑 Annulation demandée...")
        self.btn_cancel.config(state="disabled")

    def save_results_to_csv(self):
        """Sauvegarde les résultats collectés dans un fichier CSV."""
        if not self.results_data:
            self.update_log_from_thread("Aucune donnée à sauvegarder.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optim_results_{timestamp}.csv"
        
        # Les en-têtes sont basés sur les clés du premier résultat
        first_result = self.results_data[0]
        param_headers = list(first_result.get('params', {}).keys())
        score_headers = list(first_result.get('scores', {}).keys())
        headers = param_headers + score_headers

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(headers)
                for result in self.results_data:
                    # Assure l'ordre des valeurs
                    param_values = [result.get('params', {}).get(h) for h in param_headers]
                    score_values = [result.get('scores', {}).get(h) for h in score_headers]
                    writer.writerow(param_values + score_values)
            self.update_log_from_thread(f"📈 Résultats sauvegardés dans : {filename}")
        except Exception as e:
            self.update_log_from_thread(f"Erreur lors de la sauvegarde CSV : {e}")

    def update_log_from_thread(self, msg):
        self.master.after(0, self.log_text.insert, tk.END, msg + "\n")
        self.master.after(0, self.log_text.see, tk.END)

    def update_status_from_thread(self, msg):
        self.master.after(0, self.status_label.config, {'text': msg})
        
    def finalize_run(self):
        """Remet l'interface en état 'Prêt' et sauvegarde les résultats."""
        self.master.after(0, self.btn_start.config, {'state': 'normal'})
        self.master.after(0, self.btn_cancel.config, {'state': 'disabled'})
        self.master.after(0, self.save_results_to_csv)

    def on_trial_finish(self, trial_num, t_score, s_score, c_score, params):
        msg = f"[Essai {trial_num}] Tess: {t_score:.2f}% | Netteté: {s_score:.1f} | Contraste: {c_score:.1f}"
        self.update_log_from_thread(msg)

        # Enregistrement des données pour le CSV
        trial_data = {
            'params': params.copy(),
            'scores': {
                'tesseract': round(t_score, 4),
                'nettete': round(s_score, 2),
                'contraste': round(c_score, 2)
            }
        }
        self.results_data.append(trial_data)

        if t_score > self.best_score_so_far:
            self.best_score_so_far = t_score
            self.update_status_from_thread(f"🔥 RECORD TESSERACT : {t_score:.2f}% (Essai {trial_num})")
            self.master.after(0, self.update_optimal_display, params)

    def update_optimal_display(self, params):
        param_map_to_gui = {
            'line_h_size': 'line_h', 'line_v_size': 'line_v',
            'denoise_h': 'denoise_h', 'bin_c': 'bin_c',
            'norm_kernel': 'norm_kernel', 'bin_block_size': 'bin_block'
        }
        for p_name, p_val in params.items():
            gui_name = param_map_to_gui.get(p_name)
            if gui_name in self.optimal_labels:
                if gui_name == 'norm_kernel': value_to_display = (p_val - 1) // 2
                elif gui_name == 'bin_block': value_to_display = (p_val - 1) // 2
                else: value_to_display = p_val
                
                self.optimal_labels[gui_name].config(text=f"{value_to_display:.2f}" if isinstance(value_to_display, float) else f"{value_to_display}")



if __name__ == "__main__":



    multiprocessing.freeze_support()



    if not os.path.exists(INPUT_FOLDER): os.makedirs(INPUT_FOLDER)



    root = tk.Tk()



    app = OptimizerGUI(root)



    root.mainloop()




