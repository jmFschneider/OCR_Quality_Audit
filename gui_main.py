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
        self.param_entries = {} # Stocke les widgets Entry

        # --- Définition des paramètres pour chaque moteur ---
        self.params_standard = {
            'line_h': (30, 70, 45),
            'line_v': (40, 120, 50),
            'norm_kernel': (40, 100, 75),
            'denoise_h': (2.0, 20.0, 9.0),
            'noise_threshold': (20.0, 500.0, 100.0),
            'bin_block': (30, 100, 60),
            'bin_c': (10, 25.0, 15.0)
        }

        self.params_hq = {
            'inp_line_h': (20, 100, 40),
            'inp_line_v': (20, 100, 40),
            'denoise_h': (5.0, 20.0, 12.0),
            'bg_dilate': (3, 15, 7),
            'bg_blur': (11, 51, 21),
            'clahe_clip': (1.0, 5.0, 2.0),
            'clahe_tile': (4, 16, 8)
        }

        # Moteur par défaut
        self.current_engine = "Standard (Binarisation)"
        self.current_params_def = self.params_standard

        # Variables de contrôle (sera rempli dynamiquement)
        self.param_enabled_vars = {} 

        self.create_widgets()
        self.refresh_image_list()

    def create_widgets(self):
        """Crée l'interface."""
        # Frame principale
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- En-tête : Info GPU + Moteur ---
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Selecteur Moteur
        ttk.Label(header_frame, text="Moteur:").pack(side="left", padx=5)
        self.engine_var = tk.StringVar(value=self.current_engine)
        self.engine_combo = ttk.Combobox(
            header_frame, 
            textvariable=self.engine_var,
            state="readonly",
            width=25,
            values=["Standard (Binarisation)", "Haute Fidélité (Blur+CLAHE)"]
        )
        self.engine_combo.pack(side="left", padx=5)
        self.engine_combo.bind("<<ComboboxSelected>>", self.on_engine_change)

        # Info GPU
        mode_text = f"{'✅ GPU CUDA' if pipeline.USE_CUDA else '⚠️ CPU Mode'}"
        ttk.Label(header_frame, text=f"| {mode_text}", font=("Arial", 10, "bold")).pack(side="left", padx=10)

        # Images
        images_frame = ttk.LabelFrame(main_frame, text="📁 Images", padding="5")
        images_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.image_count_label = ttk.Label(images_frame, text="Aucune image")
        self.image_count_label.grid(row=0, column=0, padx=5)

        ttk.Button(images_frame, text="🔄 Rafraîchir", 
                  command=self.refresh_image_list).grid(row=0, column=1, padx=5)
        
        ttk.Button(images_frame, text="📥 Charger en mémoire", 
                  command=self.load_images_threaded).grid(row=0, column=2, padx=5)

        # Paramètres (Conteneur vide au départ, rempli par update_param_ui)
        self.params_frame = ttk.LabelFrame(main_frame, text="⚙️ Paramètres", padding="5")
        self.params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Initialisation de l'UI des paramètres
        self.update_param_ui()

        # Boutons optimisation
        opt_frame = ttk.LabelFrame(main_frame, text="🚀 Optimisation", padding="5")
        opt_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # ... (reste des widgets optimisations identique, juste indentation ajustée si besoin)
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

        # Objectif (Cible)
        ttk.Label(opt_frame, text="Cible:").grid(row=0, column=2, sticky="w", padx=(15, 5))
        self.target_var = tk.StringVar(value="Tesseract Delta")
        self.target_combo = ttk.Combobox(
            opt_frame, 
            textvariable=self.target_var,
            state="readonly", 
            width=15,
            values=["Tesseract Delta", "CNR (Gemini)"]
        )
        self.target_combo.grid(row=0, column=3, sticky="w", padx=5)

        ttk.Label(opt_frame, text="Algo:").grid(row=0, column=4, sticky="w", padx=(15, 5))

        # Algorithmes par mode
        # Avec bounds (contraints): L-BFGS-B, TNC, SLSQP
        # Sans bounds (non contraints): Nelder-Mead, Powell, CG, BFGS, COBYLA
        self.scipy_algos = ['L-BFGS-B', 'TNC', 'SLSQP', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'COBYLA']

        # Descriptions des algorithmes
        self.algo_descriptions = {
            'Sobol DoE': 'Design of Experiments - Exploration exhaustive de l\'espace',
            'L-BFGS-B': '✓ Bounds | Quasi-Newton | Mémoire limitée | Efficace haute dimension',
            'TNC': '✓ Bounds | Newton tronqué | Robuste | Bon pour problèmes bruités',
            'SLSQP': '✓ Bounds | Programmation quadratique | Support contraintes linéaires',
            'Nelder-Mead': '✗ Bounds | Simplexe | Sans gradient | Robuste mais lent',
            'Powell': '✗ Bounds | Direction conjuguée | Sans gradient | Rapide localement',
            'CG': '✗ Bounds | Gradient conjugué | Nécessite gradient | Efficace',
            'BFGS': '✗ Bounds | Quasi-Newton | Nécessite gradient | Très efficace',
            'COBYLA': '✗ Bounds | Approximation linéaire | Sans gradient | Support contraintes'
        }

        self.algo_var = tk.StringVar(value="Sobol DoE")
        self.algo_combo = ttk.Combobox(
            opt_frame,
            textvariable=self.algo_var,
            state="readonly",
            width=18,
            values=["Sobol DoE"]
        )
        self.algo_combo.grid(row=0, column=5, sticky="w", padx=5)
        self.algo_combo.bind("<<ComboboxSelected>>", self.on_algo_select)

        # Label pour afficher les caractéristiques de l'algorithme
        self.algo_info_label = ttk.Label(
            opt_frame,
            text=self.algo_descriptions['Sobol DoE'],
            font=("Arial", 9),
            foreground="#555555"
        )
        self.algo_info_label.grid(row=0, column=6, sticky="w", padx=10)

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

        ttk.Label(self.scipy_frame, text="Log tous les:").pack(side="left", padx=(15, 5))
        self.scipy_log_freq_var = tk.StringVar(value="10")
        self.scipy_log_freq_combo = ttk.Combobox(
            self.scipy_frame,
            textvariable=self.scipy_log_freq_var,
            state="readonly",
            width=8,
            values=["1", "5", "10", "20", "50", "100"]
        )
        self.scipy_log_freq_combo.pack(side="left", padx=2)
        ttk.Label(self.scipy_frame, text="points").pack(side="left", padx=2)

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

    def on_engine_change(self, event):
        """Gère le changement de moteur de traitement."""
        selection = self.engine_var.get()
        if selection == "Haute Fidélité (Blur+CLAHE)":
            self.current_params_def = self.params_hq
        else:
            self.current_params_def = self.params_standard
            
        self.log(f"⚙️ Changement moteur: {selection}")
        self.update_param_ui()

    def update_param_ui(self):
        """Reconstruit la liste des paramètres."""
        # Nettoyer les anciens widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        self.param_entries = {}
        self.param_enabled_vars = {}

        # En-têtes
        ttk.Label(self.params_frame, text="Actif", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=2)
        ttk.Label(self.params_frame, text="Paramètre", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky=tk.W, padx=2)
        ttk.Label(self.params_frame, text="Min", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=2)
        ttk.Label(self.params_frame, text="Max", font=("Arial", 9, "bold")).grid(row=0, column=3, padx=2)
        ttk.Label(self.params_frame, text="Valeur fixe", font=("Arial", 9, "bold")).grid(row=0, column=4, padx=2)

        info_label = ttk.Label(
            self.params_frame,
            text="ℹ️ Coché = optimise | Décoché = fixe",
            font=("Arial", 8),
            foreground="gray"
        )
        info_label.grid(row=0, column=5, sticky="w", padx=10)

        # Générer les lignes
        for idx, (name, (min_val, max_val, default)) in enumerate(self.current_params_def.items()):
            row = idx + 1
            
            # Variable boolean pour checkbutton
            self.param_enabled_vars[name] = tk.BooleanVar(value=True)

            # Checkbox
            checkbox = ttk.Checkbutton(
                self.params_frame,
                variable=self.param_enabled_vars[name],
                command=lambda n=name: self.toggle_param_fields(n)
            )
            checkbox.grid(row=row, column=0)
            ttk.Label(self.params_frame, text=name).grid(row=row, column=1, sticky=tk.W)

            # Min
            min_entry = ttk.Entry(self.params_frame, width=8)
            min_entry.insert(0, str(min_val))
            min_entry.grid(row=row, column=2, padx=2)

            # Max
            max_entry = ttk.Entry(self.params_frame, width=8)
            max_entry.insert(0, str(max_val))
            max_entry.grid(row=row, column=3, padx=2)

            # Valeur fixe
            fixed_entry = ttk.Entry(self.params_frame, width=8)
            fixed_entry.insert(0, str(default))
            fixed_entry.grid(row=row, column=4, padx=2)

            self.param_entries[name] = {'min': min_entry, 'max': max_entry, 'fixed': fixed_entry}
            
            # Appliquer état initial
            self.toggle_param_fields(name)

    def log(self, message):
        """Ajoute un message au log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()

    def toggle_param_fields(self, param_name):
        """Active/désactive les champs Min/Max/Fixed selon l'état de la checkbox."""
        is_enabled = self.param_enabled_vars[param_name].get()
        entries = self.param_entries[param_name]

        if is_enabled:
            # Paramètre actif: Min/Max actifs, Fixed grisé
            entries['min'].config(state='normal')
            entries['max'].config(state='normal')
            entries['fixed'].config(state='disabled')
        else:
            # Paramètre inactif: Min/Max grisés, Fixed actif
            entries['min'].config(state='disabled')
            entries['max'].config(state='disabled')
            entries['fixed'].config(state='normal')

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

        # Mettre à jour la description de l'algorithme
        self.on_algo_select(None)

    def on_algo_select(self, event):
        """Met à jour la description de l'algorithme sélectionné."""
        algo = self.algo_var.get()
        if algo in self.algo_descriptions:
            self.algo_info_label.config(text=self.algo_descriptions[algo])
        else:
            self.algo_info_label.config(text="")

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

        # Activer/désactiver les boutons
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

        # Déterminer mode
        engine_mode = 'blur_clahe' if self.engine_var.get() == "Haute Fidélité (Blur+CLAHE)" else 'standard'
        
        self.status_label.config(text=f"🚀 Screening Sobol en cours ({engine_mode}, 2^{exponent} = {n_points} points)...")
        self.log(f"\n🚀 Démarrage Sobol ({engine_mode}) avec 2^{exponent} = {n_points} points")

        # Récupérer les ranges actifs
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}  # Paramètre fixe commun (pour Standard)
        
        # Pour Blur_CLAHE, on n'a pas forcément dilate_iter, mais ça ne gêne pas s'il est ignoré.

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
                # Paramètres désactivés = valeurs fixes définies par l'utilisateur
                try:
                    fixed_val = float(self.param_entries[name]['fixed'].get())
                except:
                    self.log(f"❌ Valeur fixe invalide pour {name}")
                    return

                # Logique de mapping spécifique selon le nom du paramètre (si nécessaire)
                # Note: optimizer.py gère déjà la conversion int/odd pour norm_kernel, bin_block etc.
                # Ici on passe juste la valeur brute.
                # SAUF pour les noms "Legacy" du mode Standard qui sont transformés dans run_sobol_screening
                
                # On peut passer directement la valeur brute, l'optimizer s'occupera du formattage
                # sauf si on veut être explicite ici.
                fixed_params[name] = fixed_val

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
                   f"Netteté: {scores_dict['nettete']:.0f} | "
                   f"CNR: {scores_dict['cnr']:.2f}")

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
                verbose_timing=verbose_timing,
                pipeline_mode=engine_mode
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

        finally:
            # Réactiver les boutons dans tous les cas
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")

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

        try:
            log_frequency = int(self.scipy_log_freq_var.get())
        except:
            log_frequency = 10  # Valeur par défaut

        self.cancellation_requested.clear()

        # Activer/désactiver les boutons
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")

        # Déterminer mode
        engine_mode = 'blur_clahe' if self.engine_var.get() == "Haute Fidélité (Blur+CLAHE)" else 'standard'

        self.status_label.config(text=f"🚀 Optimisation {algorithm} ({engine_mode}) en cours...")
        self.log(f"\n🚀 Démarrage SciPy {algorithm} ({engine_mode}) ({n_iterations} itérations)")

        # Récupérer les ranges actifs
        active_ranges = {}
        fixed_params = {'dilate_iter': 2}

        for name, enabled_var in self.param_enabled_vars.items():
            if enabled_var.get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())

                    # Validation: min doit être < max
                    if min_val >= max_val:
                        self.log(f"❌ Pour {name}: min ({min_val}) doit être < max ({max_val})")
                        return

                    active_ranges[name] = (min_val, max_val)
                except:
                    self.log(f"❌ Valeurs invalides pour {name}")
                    return
            else:
                # Paramètres désactivés = valeurs fixes
                try:
                    fixed_val = float(self.param_entries[name]['fixed'].get())
                    fixed_params[name] = fixed_val
                except:
                    self.log(f"❌ Valeur fixe invalide pour {name}")
                    return

        if not active_ranges:
            self.log("❌ Aucun paramètre actif à optimiser")
            return

        self.log(f"📊 Paramètres actifs: {list(active_ranges.keys())}")
        self.log(f"🔒 Paramètres fixes: {list(fixed_params.keys())}")
        self.log(f"🎯 Objectif: MAXIMISER {self.target_var.get()}")

        # Convertir active_ranges en liste de bounds pour scipy
        param_names = list(active_ranges.keys())
        bounds = [active_ranges[name] for name in param_names]

        # Variables pour le suivi de l'optimisation
        eval_count = [0]
        best_score = [float('-inf')]
        best_metrics = [None]
        
        target_metric = self.target_var.get()

        # Fonction objectif pour SciPy
        def objective_func(params_array):
            """Fonction objectif à minimiser."""
            eval_count[0] += 1
            current_eval = eval_count[0]

            # Convertir array en dict de paramètres
            params_dict = dict(zip(param_names, params_array))

            # Reconstruire les paramètres complets
            full_params = fixed_params.copy()
            for name, val in params_dict.items():
                full_params[name] = val

            # Application des transformations (int, odd, etc.)
            final_params = full_params.copy()
            for k, v in final_params.items():
                if k in ['norm_kernel', 'bin_block', 'bg_dilate', 'bg_blur', 'bin_block_size']:
                    final_params[k] = int(v) * 2 + 1
                elif k in ['line_h', 'line_v', 'inp_line_h', 'inp_line_v', 'clahe_tile', 
                           'line_h_size', 'line_v_size']:
                    final_params[k] = int(v)

            # Mapping legacy standard mode
            if engine_mode == 'standard':
                if 'line_h' in final_params: final_params['line_h_size'] = final_params.pop('line_h')
                if 'line_v' in final_params: final_params['line_v_size'] = final_params.pop('line_v')
                if 'bin_block' in final_params: final_params['bin_block_size'] = final_params.pop('bin_block')

            # Évaluer
            avg_delta, avg_abs, avg_sharp, avg_cnr = optimizer.evaluate_pipeline(
                self.loaded_images,
                self.baseline_scores,
                final_params,
                pipeline_mode=engine_mode
            )
            
            # Choix de la métrique cible
            if target_metric == "CNR (Gemini)":
                current_score = avg_cnr
            else:
                current_score = avg_delta

            # Déterminer si c'est une amélioration
            is_improvement = False
            if current_score > best_score[0]:
                is_improvement = True
                best_score[0] = current_score
                best_metrics[0] = {
                    'delta': avg_delta,
                    'tesseract': avg_abs,
                    'sharpness': avg_sharp,
                    'cnr': avg_cnr,
                    'params': params_dict.copy()
                }

            # Logger
            should_log = (current_eval % log_frequency == 0 or current_eval == 1 or is_improvement)

            if should_log:
                trend = "🆕 MEILLEUR!" if is_improvement else "📊"
                msg = (f"{trend} [Eval {current_eval}] Score ({target_metric}): {current_score:.2f} | "
                       f"DeltaTess: {avg_delta:+.2f}% | CNR: {avg_cnr:.2f}")
                self.master.after(0, self.log, msg)

            return -current_score

        # Callback pour mise à jour GUI
        def update_callback(msg):
            self.master.after(0, self.log, msg)

        # Initialiser barre de progression
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
                self.log(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS ({eval_count[0]} évaluations):")
                # Afficher les paramètres bruts (optimisés)
                best_params_array = best_result['x']
                best_params = dict(zip(param_names, best_params_array))
                
                for key, val in best_params.items():
                    self.log(f"   {key}: {val:.4f}")
                    
                self.log(f"\n📊 MÉTRIQUES:")
                if best_metrics[0]:
                    self.log(f"   Delta Tesseract: {best_metrics[0]['delta']:+.2f}%")
                    self.log(f"   Score Tesseract: {best_metrics[0]['tesseract']:.2f}%")
                    self.log(f"   Score CNR: {best_metrics[0]['cnr']:.2f}")
                else:
                    self.log(f"   Score Cible: {-best_result['fun']:.2f}")
                    
                self.log(f"\n✅ Optimisation {algorithm} terminée")
                self.status_label.config(text=f"✅ {algorithm} terminé!")
                self.progress_label.config(text=f"✅ Optimisation terminée")

                self.master.after(3000, lambda: self.progress_bar.config(value=0))
                self.master.after(3000, lambda: self.progress_label.config(text=""))
            else:
                self.log(f"⚠️ Optimisation annulée ou aucun résultat")
                self.status_label.config(text="⏹️ Optimisation annulée")
                self.progress_label.config(text="⏹️ Annulé")
                self.master.after(2000, lambda: self.progress_bar.config(value=0))

        except Exception as e:
            self.master.after(0, lambda: self.progress_bar.stop())
            self.master.after(0, lambda: self.progress_bar.config(mode='determinate'))
            self.log(f"❌ Erreur: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.status_label.config(text="❌ Erreur optimisation")
            self.master.after(2000, lambda: self.progress_bar.config(value=0))

        finally:
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")

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
