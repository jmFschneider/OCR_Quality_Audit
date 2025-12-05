#!/usr/bin/env python3
"""
GUI Optimizer - Version modulaire
Utilise pipeline.py et optimizer.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
import os
from glob import glob
import threading
from datetime import datetime
import csv
import platform
import multiprocessing

# Imports des modules locaux
import pipeline
import optimizer
import scipy_optimizer

# Configuration
INPUT_FOLDER = "test_scans"


class OptimizerGUI:
    def __init__(self, master):
        self.master = master
        master.title(f"🔍 Optimiseur OCR - {'GPU CUDA' if pipeline.USE_CUDA else 'CPU'}")
        master.geometry("1000x700")

        # Données
        self.loaded_images = []
        self.baseline_scores = []
        self.image_files = []
        self.results_data = []
        self.cancellation_requested = threading.Event()

        # Paramètres par défaut (min, max, default)
        self.default_params = {
            'line_h': (30, 70, 45),
            'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75),
            'denoise_h': (2.0, 20.0, 9.0),
            'noise_threshold': (20.0, 500.0, 100.0),
            'bin_block': (30, 100, 60),
            'bin_c': (10, 25.0, 15.0)
        }

        self.param_enabled_vars = {
            name: tk.BooleanVar(value=True) 
            for name in self.default_params
        }

        self.create_widgets()
        self.refresh_image_list()

    def create_widgets(self):
        """Crée l'interface."""
        # Frame principale
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Info GPU/CPU
        mode_text = f"Mode: {'✅ GPU CUDA + Tesseract' if pipeline.USE_CUDA else '⚠️  CPU + Tesseract'}"
        mode_label = ttk.Label(
            main_frame,
            text=mode_text,
            font=("Arial", 12, "bold")
        )
        mode_label.grid(row=0, column=0, columnspan=3, pady=5)

        # Images
        images_frame = ttk.LabelFrame(main_frame, text="📁 Images", padding="5")
        images_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.image_count_label = ttk.Label(images_frame, text="Aucune image")
        self.image_count_label.grid(row=0, column=0, padx=5)

        ttk.Button(images_frame, text="🔄 Rafraîchir", 
                  command=self.refresh_image_list).grid(row=0, column=1, padx=5)
        
        ttk.Button(images_frame, text="📥 Charger en mémoire", 
                  command=self.load_images_threaded).grid(row=0, column=2, padx=5)

        # Paramètres
        params_frame = ttk.LabelFrame(main_frame, text="⚙️ Paramètres", padding="5")
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.param_entries = {}
        for row, (name, (min_val, max_val, default)) in enumerate(self.default_params.items()):
            ttk.Checkbutton(params_frame, variable=self.param_enabled_vars[name]).grid(row=row, column=0)
            ttk.Label(params_frame, text=name).grid(row=row, column=1, sticky=tk.W)
            
            # Min
            min_entry = ttk.Entry(params_frame, width=8)
            min_entry.insert(0, str(min_val))
            min_entry.grid(row=row, column=2, padx=2)
            
            # Max
            max_entry = ttk.Entry(params_frame, width=8)
            max_entry.insert(0, str(max_val))
            max_entry.grid(row=row, column=3, padx=2)

            self.param_entries[name] = {'min': min_entry, 'max': max_entry}

        # Boutons optimisation
        opt_frame = ttk.LabelFrame(main_frame, text="🚀 Optimisation", padding="5")
        opt_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Ligne 1: Mode et Algorithme
        ttk.Label(opt_frame, text="Mode:").grid(row=0, column=0, sticky="w", padx=5)
        self.mode_var = tk.StringVar(value="Screening")
        self.mode_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.mode_var,
            state="readonly",
            width=10,
            values=["Screening", "SciPy"]
        )
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_select)

        ttk.Label(opt_frame, text="Algorithme:").grid(row=0, column=2, sticky="w", padx=(15, 5))

        # Algorithmes par mode
        self.scipy_algos = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP']

        self.algo_var = tk.StringVar(value="Sobol DoE")
        self.algo_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.algo_var,
            state="readonly",
            width=18,
            values=["Sobol DoE"]
        )
        self.algo_combo.grid(row=0, column=3, sticky="w", padx=5)

        # Ligne 2: Options spécifiques au mode
        # Frame pour options Screening (Sobol)
        self.screening_frame = ttk.Frame(opt_frame)
        self.screening_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)

        ttk.Label(self.screening_frame, text="Exposant Sobol (2^n):").pack(side="left", padx=5)
        self.sobol_exponent_var = tk.StringVar(value="5")
        self.sobol_exponent_var.trace_add("write", self.update_sobol_points_label)
        self.sobol_exponent_entry = ttk.Entry(self.screening_frame, width=5, textvariable=self.sobol_exponent_var)
        self.sobol_exponent_entry.pack(side="left", padx=2)
        self.sobol_points_label = ttk.Label(self.screening_frame, text="= 32 points")
        self.sobol_points_label.pack(side="left", padx=5)

        # Frame pour options SciPy
        self.scipy_frame = ttk.Frame(opt_frame)

        ttk.Label(self.scipy_frame, text="Itérations:").pack(side="left", padx=5)
        self.scipy_iter_var = tk.StringVar(value="15")
        self.scipy_iter_entry = ttk.Entry(self.scipy_frame, width=8, textvariable=self.scipy_iter_var)
        self.scipy_iter_entry.pack(side="left", padx=5)

        # Ligne 3: Boutons d'action
        button_frame = ttk.Frame(opt_frame)
        button_frame.grid(row=2, column=0, columnspan=4, sticky="w", pady=5)

        self.btn_start = ttk.Button(button_frame, text="▶️ Lancer",
                  command=self.start_optimization)
        self.btn_start.pack(side="left", padx=5)

        self.btn_cancel = ttk.Button(button_frame, text="⏹️ Arrêter",
                  command=self.cancel_optimization, state="disabled")
        self.btn_cancel.pack(side="left", padx=5)

        # Initialiser l'affichage
        self.on_mode_select(None)

        # Logs
        log_frame = ttk.LabelFrame(main_frame, text="📋 Logs", padding="5")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Barre de progression
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.grid(row=0, column=0, sticky=tk.W, padx=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)

        # Status
        self.status_label = ttk.Label(main_frame, text="Prêt", relief=tk.SUNKEN)
        self.status_label.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Configuration grille
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        progress_frame.columnconfigure(0, weight=1)

    def log(self, message):
        """Ajoute un message au log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()

    def update_sobol_points_label(self, *args):
        """Met à jour le label affichant le nombre de points Sobol (2^n)."""
        try:
            exponent = int(self.sobol_exponent_var.get())
            if exponent > 16:  # Limite pour éviter les très grands nombres
                self.sobol_points_label.config(text="! > 65536", foreground="red")
                return
            n_points = 2**exponent
            self.sobol_points_label.config(text=f"= {n_points} points", foreground="black")
        except ValueError:
            self.sobol_points_label.config(text="= Invalide", foreground="red")

    def on_mode_select(self, event):
        """Gère le changement de mode d'optimisation."""
        # Cache les deux frames d'options
        self.scipy_frame.grid_remove()
        self.screening_frame.grid_remove()

        mode = self.mode_var.get()
        if mode == "Screening":
            self.algo_combo.config(values=["Sobol DoE"])
            self.algo_var.set("Sobol DoE")
            self.screening_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)
        else:  # SciPy
            self.algo_combo.config(values=self.scipy_algos)
            self.algo_var.set(self.scipy_algos[0])
            self.scipy_frame.grid(row=1, column=0, columnspan=4, sticky="w", pady=5)

    def refresh_image_list(self):
        """Rafraîchit la liste des images."""
        self.image_files = glob(os.path.join(INPUT_FOLDER, "*.jpg"))
        self.image_count_label.config(text=f"{len(self.image_files)} images trouvées")
        self.log(f"📁 {len(self.image_files)} images dans {INPUT_FOLDER}")

    def load_images_threaded(self):
        """Charge les images dans un thread."""
        threading.Thread(target=self.load_images, daemon=True).start()

    def load_images(self):
        """Charge toutes les images en mémoire."""
        import numpy as np

        total_images = len(self.image_files)
        self.log("📥 Chargement des images...")

        # Initialiser la barre de progression
        self.progress_bar['maximum'] = total_images + 1  # +1 pour le calcul baseline
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Chargement des images...")

        self.loaded_images = []
        self.baseline_scores = []

        # Charger les images avec progression
        for idx, f in enumerate(self.image_files, 1):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Garantir uint8 pour compatibilité GPU
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                self.loaded_images.append(img)

            # Mettre à jour la progression
            self.progress_bar['value'] = idx
            self.progress_label.config(text=f"Chargement: {idx}/{total_images} images")
            self.master.update_idletasks()

        # Calcul des scores baseline avec multiprocessing
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        self.log(f"🔍 Calcul des scores baseline...")
        self.progress_label.config(text=f"Calcul baseline (parallèle: {cpu_count} workers)...")
        self.master.update_idletasks()

        import time
        t0 = time.time()
        self.baseline_scores = optimizer.calculate_baseline_scores(self.loaded_images)
        t_baseline = time.time() - t0

        # Compléter la progression
        self.progress_bar['value'] = total_images + 1
        self.progress_label.config(text="✅ Chargement terminé")

        self.log(f"✅ {len(self.loaded_images)} images chargées")
        self.log(f"✅ {len(self.baseline_scores)} scores baseline calculés en {t_baseline:.1f}s")
        self.log(f"   (Traitement parallèle: {cpu_count} workers)")
        self.log(f"✅ Chargement initial terminé")

        # Réinitialiser la barre après 2 secondes
        self.master.after(2000, lambda: self.progress_bar.config(value=0))
        self.master.after(2000, lambda: self.progress_label.config(text=""))

    def start_optimization(self):
        """Lance l'optimisation (Screening ou SciPy) dans un thread."""
        if not self.loaded_images:
            messagebox.showwarning("Attention", "Chargez d'abord les images !")
            return

        mode = self.mode_var.get()
        if mode == "Screening":
            threading.Thread(target=self.run_sobol, daemon=True).start()
        else:  # SciPy
            threading.Thread(target=self.run_scipy, daemon=True).start()

    def run_sobol(self):
        """Exécute l'optimisation Sobol."""
        try:
            exponent = int(self.sobol_exponent_var.get())
            if exponent > 16:
                self.log("❌ Exposant trop élevé (max 16 = 65536 points)")
                return
            n_points = 2**exponent
        except:
            self.log("❌ Exposant Sobol invalide")
            return

        self.cancellation_requested.clear()
        self.status_label.config(text=f"🚀 Screening Sobol en cours (2^{exponent} = {n_points} points)...")
        self.log(f"\n🚀 Démarrage Sobol avec 2^{exponent} = {n_points} points")

        # Récupérer les ranges actifs
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}  # Paramètre fixe

        for name, enabled_var in self.param_enabled_vars.items():
            if enabled_var.get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())
                    active_ranges[name] = (min_val, max_val)
                except:
                    self.log(f"❌ Valeurs invalides pour {name}")
                    return
            else:
                # Paramètres désactivés = valeurs par défaut fixes
                default_val = self.default_params[name][2]
                if name == 'norm_kernel':
                    fixed_params['norm_kernel'] = int(default_val) * 2 + 1
                elif name == 'bin_block':
                    fixed_params['bin_block_size'] = int(default_val) * 2 + 1
                elif name == 'line_h':
                    fixed_params['line_h_size'] = default_val
                elif name == 'line_v':
                    fixed_params['line_v_size'] = default_val
                else:
                    fixed_params[name] = default_val

        if not active_ranges:
            self.log("❌ Aucun paramètre actif à optimiser")
            return

        self.log(f"📊 Paramètres actifs: {list(active_ranges.keys())}")
        self.log(f"🔒 Paramètres fixes: {list(fixed_params.keys())}")

        # Initialiser la barre de progression pour Sobol
        self.master.after(0, lambda: self.progress_bar.config(maximum=n_points, value=0))
        self.master.after(0, lambda: self.progress_label.config(text=f"Screening Sobol: 0/{n_points} points"))

        # Callback pour mise à jour de la GUI
        def sobol_callback(point_idx, scores_dict, params_dict):
            msg = (f"[Point {point_idx+1}] Delta: {scores_dict['tesseract_delta']:.2f}% | "
                   f"Tess: {scores_dict['tesseract']:.2f}% | "
                   f"Netteté: {scores_dict['nettete']:.1f} | "
                   f"Contraste: {scores_dict['contraste']:.1f}")

            # Mise à jour thread-safe
            self.master.after(0, self.log, msg)

            # Mettre à jour la barre de progression
            self.master.after(0, lambda: self.progress_bar.config(value=point_idx + 1))
            self.master.after(0, lambda: self.progress_label.config(
                text=f"Screening Sobol: {point_idx + 1}/{n_points} points"
            ))

        # Option pour afficher les temps détaillés (peut ralentir l'UI)
        verbose_timing = False  # Mettre True pour debug

        # Lancer le screening Sobol
        try:
            best_params, csv_file = optimizer.run_sobol_screening(
                images=self.loaded_images,
                baseline_scores=self.baseline_scores,
                n_points=n_points,
                param_ranges=active_ranges,
                fixed_params=fixed_params,
                callback=sobol_callback,
                cancellation_event=self.cancellation_requested,
                verbose_timing=verbose_timing
            )

            if best_params:
                self.log(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS:")
                for key, val in best_params.items():
                    self.log(f"   {key}: {val}")
                self.log(f"\n📁 Résultats sauvegardés dans: {csv_file}")
                self.log(f"✅ Screening Sobol terminé - {n_points} points évalués")
                self.status_label.config(text=f"✅ Sobol terminé! Résultats: {csv_file}")

                # Mettre à jour la barre de progression
                self.progress_label.config(text=f"✅ Screening terminé - {n_points} points")

                # Réinitialiser après 3 secondes
                self.master.after(3000, lambda: self.progress_bar.config(value=0))
                self.master.after(3000, lambda: self.progress_label.config(text=""))
            else:
                self.log("⚠️ Aucun meilleur paramètre trouvé (screening annulé?)")
                self.status_label.config(text="⏹️ Screening annulé")
                self.progress_label.config(text="⏹️ Annulé")

                # Réinitialiser après 2 secondes
                self.master.after(2000, lambda: self.progress_bar.config(value=0))
                self.master.after(2000, lambda: self.progress_label.config(text=""))

        except Exception as e:
            self.log(f"❌ Erreur pendant le screening: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.status_label.config(text="❌ Erreur screening")
            self.progress_label.config(text="❌ Erreur")

            # Réinitialiser après 2 secondes
            self.master.after(2000, lambda: self.progress_bar.config(value=0))
            self.master.after(2000, lambda: self.progress_label.config(text=""))

    def run_scipy(self):
        """Exécute l'optimisation SciPy."""
        # Récupérer les paramètres SciPy
        algorithm = self.algo_var.get()
        try:
            n_iterations = int(self.scipy_iter_var.get())
            if n_iterations < 1:
                self.log("❌ Le nombre d'itérations doit être >= 1")
                return
        except:
            self.log("❌ Nombre d'itérations invalide")
            return

        self.cancellation_requested.clear()
        self.status_label.config(text=f"🚀 Optimisation {algorithm} en cours...")
        self.log(f"\n🚀 Démarrage SciPy avec {algorithm} ({n_iterations} itérations)")

        # Récupérer les ranges actifs (même logique que Sobol)
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}

        for name, enabled_var in self.param_enabled_vars.items():
            if enabled_var.get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())
                    active_ranges[name] = (min_val, max_val)
                except:
                    self.log(f"❌ Valeurs invalides pour {name}")
                    return
            else:
                # Paramètres désactivés = valeurs par défaut fixes
                default_val = self.default_params[name][2]
                if name == 'norm_kernel':
                    fixed_params['norm_kernel'] = int(default_val) * 2 + 1
                elif name == 'bin_block':
                    fixed_params['bin_block_size'] = int(default_val) * 2 + 1
                elif name == 'line_h':
                    fixed_params['line_h_size'] = default_val
                elif name == 'line_v':
                    fixed_params['line_v_size'] = default_val
                else:
                    fixed_params[name] = default_val

        if not active_ranges:
            self.log("❌ Aucun paramètre actif à optimiser")
            return

        self.log(f"📊 Paramètres actifs: {list(active_ranges.keys())}")
        self.log(f"🔒 Paramètres fixes: {list(fixed_params.keys())}")

        # Convertir active_ranges en liste de bounds pour scipy
        param_names = list(active_ranges.keys())
        bounds = [active_ranges[name] for name in param_names]

        # Fonction objectif pour SciPy
        def objective_func(params_array):
            """Fonction objectif à minimiser."""
            # Convertir array en dict de paramètres
            params_dict = dict(zip(param_names, params_array))

            # Convertir noms courts en noms complets avec règles de transformation
            full_params = fixed_params.copy()
            for name, val in params_dict.items():
                if name == 'norm_kernel':
                    full_params['norm_kernel'] = int(val) * 2 + 1
                elif name == 'bin_block':
                    full_params['bin_block_size'] = int(val) * 2 + 1
                elif name == 'line_h':
                    full_params['line_h_size'] = val
                elif name == 'line_v':
                    full_params['line_v_size'] = val
                else:
                    full_params[name] = val

            # Évaluer avec optimizer
            result = optimizer.evaluate_pipeline(
                self.loaded_images,
                self.baseline_scores,
                full_params,
                cancellation_event=self.cancellation_requested
            )

            # SciPy minimise, donc on retourne -delta pour maximiser
            return -result['tesseract_delta']

        # Callback pour mise à jour GUI
        def update_callback(msg):
            self.master.after(0, self.log, msg)

        # Initialiser barre de progression (indéterminée pour SciPy)
        self.master.after(0, lambda: self.progress_bar.config(mode='indeterminate'))
        self.master.after(0, lambda: self.progress_bar.start(10))
        self.master.after(0, lambda: self.progress_label.config(text=f"Optimisation {algorithm}..."))

        # Lancer optimisation SciPy
        try:
            best_result = scipy_optimizer.run_scipy_optimization(
                objective_func=objective_func,
                bounds=bounds,
                algorithm=algorithm,
                n_sobol_points=3,  # 3 points de départ
                n_iterations=n_iterations,
                update_callback=update_callback,
                cancellation_event=self.cancellation_requested
            )

            # Arrêter la barre indéterminée
            self.master.after(0, lambda: self.progress_bar.stop())
            self.master.after(0, lambda: self.progress_bar.config(mode='determinate', value=100))

            if best_result and best_result['x'] is not None:
                # Convertir array en dict de paramètres
                best_params_array = best_result['x']
                best_params = dict(zip(param_names, best_params_array))

                # Convertir en paramètres complets
                full_params = fixed_params.copy()
                for name, val in best_params.items():
                    if name == 'norm_kernel':
                        full_params['norm_kernel'] = int(val) * 2 + 1
                    elif name == 'bin_block':
                        full_params['bin_block_size'] = int(val) * 2 + 1
                    elif name == 'line_h':
                        full_params['line_h_size'] = val
                    elif name == 'line_v':
                        full_params['line_v_size'] = val
                    else:
                        full_params[name] = val

                self.log(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS:")
                for key, val in full_params.items():
                    self.log(f"   {key}: {val}")
                self.log(f"   Score objectif: {best_result['fun']:.4f}")
                self.log(f"   Delta Tesseract: {-best_result['fun']:.2f}%")
                self.log(f"\n✅ Optimisation {algorithm} terminée")
                self.status_label.config(text=f"✅ {algorithm} terminé!")
                self.progress_label.config(text=f"✅ Optimisation terminée")

                # Réinitialiser après 3 secondes
                self.master.after(3000, lambda: self.progress_bar.config(value=0))
                self.master.after(3000, lambda: self.progress_label.config(text=""))
            else:
                self.log("⚠️ Optimisation annulée ou aucun résultat trouvé")
                self.status_label.config(text="⏹️ Optimisation annulée")
                self.progress_label.config(text="⏹️ Annulé")

                # Réinitialiser après 2 secondes
                self.master.after(2000, lambda: self.progress_bar.config(value=0))
                self.master.after(2000, lambda: self.progress_label.config(text=""))

        except Exception as e:
            # Arrêter la barre indéterminée
            self.master.after(0, lambda: self.progress_bar.stop())
            self.master.after(0, lambda: self.progress_bar.config(mode='determinate'))

            self.log(f"❌ Erreur pendant l'optimisation: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.status_label.config(text="❌ Erreur optimisation")
            self.progress_label.config(text="❌ Erreur")

            # Réinitialiser après 2 secondes
            self.master.after(2000, lambda: self.progress_bar.config(value=0))
            self.master.after(2000, lambda: self.progress_label.config(text=""))

    def cancel_optimization(self):
        """Annule l'optimisation en cours."""
        self.cancellation_requested.set()
        self.log("⏹️ Annulation demandée...")


def main():
    """Point d'entrée principal."""
    print("[DEBUG] Démarrage de l'application...")
    
    # Configuration multiprocessing
    if platform.system() != 'Windows':
        try:
            multiprocessing.set_start_method('spawn')
            print("[DEBUG] multiprocessing.set_start_method('spawn') OK")
        except RuntimeError:
            pass

    multiprocessing.freeze_support()

    # Vérifier dossier
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"[DEBUG] Dossier {INPUT_FOLDER} créé")

    # Lancer GUI
    root = tk.Tk()
    app = OptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
