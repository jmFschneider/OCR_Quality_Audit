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






import scipy_optimizer

def process_single_file_wrapper(args):
    """
    Wrapper function to process a single file. Takes a tuple of (file_path, params)
    as input to be compatible with pool.starmap.
    """
    file_path, params = args
    img = cv2.imread(file_path)
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
    gui_app.update_log_from_thread(f"Log détaillé : {csv_filename}")

    def objective(trial):
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
            gui_app.update_log_from_thread(f"Erreur CSV: {e}")

        gui_app.on_trial_finish(trial.number, avg_tess, avg_sharp, avg_cont, params)
        return avg_tess

    sampler_map = {
        "TPE (Bayésien)": TPESampler(n_startup_trials=20),
        "Sobol (Quasi-Monte Carlo)": QMCSampler(qmc_type='sobol', scramble=True),
        "NSGA-II (Génétique)": NSGAIISampler(population_size=50, mutation_prob=0.05)
    }
    sampler = sampler_map.get(algo_choice, TPESampler())
    
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    gui_app.update_status_from_thread(f"✅ Terminé ! Meilleur Tesseract : {study.best_value:.2f}%")
    gui_app.enable_start_button()


# --- GUI ---

class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("🔍 Optimiseur OCR V6 - Optuna & Scipy")
        master.geometry("1000x800")

        self.best_score_so_far = 0.0
        self.trial_count = 0
        self.param_entries = {}
        self.optimal_labels = {}
        self.default_params = {
            'line_h': (30, 70, 45), 'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75), 'denoise_h': (2.0, 20.0, 9.0),
            'bin_block': (30, 100, 60), 'bin_c': (10, 25.0, 15.0)
        }
        self.param_enabled_vars = {name: tk.BooleanVar(value=True) for name in self.default_params}
        
        self.optuna_algos = ["TPE (Bayésien)", "Sobol (Quasi-Monte Carlo)", "NSGA-II (Génétique)"]
        self.scipy_algos = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

        self.create_widgets()

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

        ttk.Label(ctrl_frame, text="Bibliothèque :").pack(side="left", padx=(5,0))
        self.lib_var = tk.StringVar(value="Optuna")
        self.lib_combo = ttk.Combobox(ctrl_frame, textvariable=self.lib_var, state="readonly", width=10, values=["Optuna", "Scipy"])
        self.lib_combo.pack(side="left", padx=(5,15))
        self.lib_combo.bind("<<ComboboxSelected>>", self.on_library_select)
        
        ttk.Label(ctrl_frame, text="Algorithme :").pack(side="left", padx=5)
        self.algo_var = tk.StringVar(value=self.optuna_algos[0])
        self.algo_combo = ttk.Combobox(ctrl_frame, textvariable=self.algo_var, state="readonly", width=25, values=self.optuna_algos)
        self.algo_combo.pack(side="left", padx=5)

        self.optuna_frame = ttk.Frame(ctrl_frame)
        self.optuna_frame.pack(side="left", padx=5)
        ttk.Label(self.optuna_frame, text="Nb Essais :").pack(side="left", padx=(10,0))
        self.trials_entry = ttk.Entry(self.optuna_frame, width=8); self.trials_entry.insert(0, "100")
        self.trials_entry.pack(side="left", padx=5)
        
        self.scipy_frame = ttk.Frame(ctrl_frame)
        ttk.Label(self.scipy_frame, text="Points Sobol:").pack(side="left", padx=(10,0))
        self.sobol_points_entry = ttk.Entry(self.scipy_frame, width=8); self.sobol_points_entry.insert(0, "20")
        self.sobol_points_entry.pack(side="left", padx=5)
        ttk.Label(self.scipy_frame, text="Itérations/point:").pack(side="left", padx=(10,0))
        self.scipy_iter_entry = ttk.Entry(self.scipy_frame, width=8); self.scipy_iter_entry.insert(0, "15")
        self.scipy_iter_entry.pack(side="left", padx=5)

        self.btn_start = ttk.Button(ctrl_frame, text="▶ LANCER", command=self.start_optimization)
        self.btn_start.pack(side="right", padx=20)
        
        self.status_label = ttk.Label(self.master, text="Prêt.", font=('Helvetica', 10, 'italic'))
        self.status_label.pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(self.master, width=90, height=12)
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.on_library_select(None)

    def on_library_select(self, event):
        lib = self.lib_var.get()
        if lib == "Optuna":
            self.algo_combo.config(values=self.optuna_algos)
            self.algo_var.set(self.optuna_algos[0])
            self.optuna_frame.pack(side="left", padx=5)
            self.scipy_frame.pack_forget()
        else:
            self.algo_combo.config(values=self.scipy_algos)
            self.algo_var.set(self.scipy_algos[0])
            self.scipy_frame.pack(side="left", padx=5)
            self.optuna_frame.pack_forget()

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
        fichiers = glob(os.path.join(INPUT_FOLDER, '*.*'))[:5] # Limit files for speed
        
        # Prepare arguments for the multiprocessing pool
        pool_args = zip(fichiers, repeat(params))
        
        try:
            # Create a pool of workers
            with multiprocessing.Pool(os.cpu_count()) as pool:
                results = pool.map(process_single_file_wrapper, pool_args)
            
            # Filter out None results from failed image reads
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                return 0, 0, 0

            # Unzip results
            list_tess, list_sharp, list_cont = zip(*valid_results)

            avg_tess = sum(list_tess) / len(list_tess) if list_tess else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_tess, avg_sharp, avg_cont

        except Exception as e:
            # Fallback to sequential processing in case of multiprocessing error
            print(f"Erreur de multiprocessing, passage en mode séquentiel: {e}")
            list_tess, list_sharp, list_cont = [], [], []
            for f in fichiers:
                img = cv2.imread(f)
                if img is None: continue
                processed_img = pipeline_complet(img, params)
                tess, sharp, cont = evaluer_toutes_metriques(processed_img)
                list_tess.append(tess); list_sharp.append(sharp); list_cont.append(cont)
            
            avg_tess = sum(list_tess) / len(list_tess) if list_tess else 0
            avg_sharp = sum(list_sharp) / len(list_sharp) if list_sharp else 0
            avg_cont = sum(list_cont) / len(list_cont) if list_cont else 0
            return avg_tess, avg_sharp, avg_cont

    def start_optimization(self):
        active_ranges, fixed_params, param_order = self.get_optim_config()
        if active_ranges is None: messagebox.showerror("Erreur de Saisie", "Vérifiez que les Min/Max sont des nombres valides."); return

        self.btn_start.config(state="disabled")
        self.log_text.delete('1.0', tk.END)
        self.best_score_so_far = 0.0
        self.trial_count = 0
        for lbl in self.optimal_labels.values(): lbl.config(text="-")

        library = self.lib_var.get()
        algo = self.algo_var.get()

        if library == "Optuna":
            try:
                n_trials = int(self.trials_entry.get())
                thread = threading.Thread(target=run_optuna_optimization, args=(self, n_trials, active_ranges, fixed_params, algo))
                thread.start()
            except ValueError: 
                messagebox.showerror("Erreur de Saisie", "Le nombre d'essais doit être un entier.")
                self.enable_start_button()
        
        elif library == "Scipy":
            try:
                n_sobol = int(self.sobol_points_entry.get())
                n_iter = int(self.scipy_iter_entry.get())
                bounds = [active_ranges[p] for p in param_order]

                def objective_for_scipy(param_values):
                    # DEBUG print(f"DEBUG: Scipy objective called with {param_values}")
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
                self.enable_start_button()

    def run_scipy_thread(self, objective, bounds, algo, n_sobol, n_iter):
        result = scipy_optimizer.run_scipy_optimization(objective, bounds, algo, n_sobol, n_iter, self.update_log_from_thread)
        best_params_values = result.x
        
        # Reconstruct the full params dict to display it
        _, fixed_params, param_order = self.get_optim_config()
        final_params = fixed_params.copy()
        current_params = {}
        for i, name in enumerate(param_order):
            val = best_params_values[i]
            p_name_map = {'line_h': 'line_h_size', 'line_v': 'line_v_size', 'norm_kernel': 'norm_kernel', 'bin_block': 'bin_block_size', 'denoise_h': 'denoise_h', 'bin_c': 'bin_c'}
            mapped_name = p_name_map.get(name)
            if not mapped_name: continue

            if name in ['norm_kernel', 'bin_block']:
                 current_params[mapped_name] = int(val) * 2 + 1
            else:
                 current_params[mapped_name] = val
        final_params.update(current_params)

        self.update_status_from_thread(f"✅ Scipy terminé ! Meilleur score (négatif) : {result.fun:.4f}")
        self.master.after(0, self.update_optimal_display, final_params)
        self.enable_start_button()

    def update_log_from_thread(self, msg):
        self.master.after(0, self.log_text.insert, tk.END, msg + "\n")
        self.master.after(0, self.log_text.see, tk.END)

    def update_status_from_thread(self, msg):
        self.master.after(0, self.status_label.config, {'text': msg})
        
    def enable_start_button(self):
        self.master.after(0, self.btn_start.config, {'state': 'normal'})

    def on_trial_finish(self, trial_num, t_score, s_score, c_score, params):
        msg = f"[Essai {trial_num}] Tess: {t_score:.2f}% | Netteté: {s_score:.1f} | Contraste: {c_score:.1f}"
        self.update_log_from_thread(msg)
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




