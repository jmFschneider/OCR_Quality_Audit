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
        mode_label = ttk.Label(
            main_frame, 
            text=f"Mode: {'✅ GPU CUDA' if pipeline.USE_CUDA else '⚠️  CPU'}",
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

        ttk.Label(opt_frame, text="Exposant Sobol (2^n):").grid(row=0, column=0, padx=5)

        # Variable pour l'exposant avec callback pour mise à jour dynamique
        self.sobol_exponent_var = tk.StringVar(value="5")
        self.sobol_exponent_var.trace_add("write", self.update_sobol_points_label)

        self.sobol_exponent_entry = ttk.Entry(opt_frame, width=5, textvariable=self.sobol_exponent_var)
        self.sobol_exponent_entry.grid(row=0, column=1, padx=2)

        # Label dynamique pour afficher le nombre de points
        self.sobol_points_label = ttk.Label(opt_frame, text="= 32 points")
        self.sobol_points_label.grid(row=0, column=2, padx=5)

        ttk.Button(opt_frame, text="▶️ Lancer Sobol",
                  command=self.start_sobol).grid(row=0, column=3, padx=5)

        ttk.Button(opt_frame, text="⏹️ Arrêter",
                  command=self.cancel_optimization).grid(row=0, column=4, padx=5)
        
        # Sélecteur de moteur OCR
        ttk.Label(opt_frame, text="Moteur OCR:").grid(row=1, column=0, padx=5, pady=5)
        self.ocr_engine_var = tk.StringVar(value="Tesseract")
        self.ocr_combo = ttk.Combobox(opt_frame, textvariable=self.ocr_engine_var, state="readonly", width=10, values=["Tesseract", "RapidOCR"])
        self.ocr_combo.grid(row=1, column=1, padx=2, pady=5, columnspan=2, sticky=tk.W)
        self.ocr_combo.bind("<<ComboboxSelected>>", self.on_ocr_engine_select)

        # Logs
        log_frame = ttk.LabelFrame(main_frame, text="📋 Logs", padding="5")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status
        self.status_label = ttk.Label(main_frame, text="Prêt", relief=tk.SUNKEN)
        self.status_label.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Configuration grille
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Initialiser l'état du moteur OCR
        self.on_ocr_engine_select(None)

    def on_ocr_engine_select(self, event):
        """Met à jour le moteur OCR sélectionné et l'enregistre dans l'environnement pour les workers."""
        selection = self.ocr_engine_var.get()
        
        if selection == "RapidOCR":
            if not pipeline.RAPIDOCR_AVAILABLE:
                messagebox.showwarning("RapidOCR Manquant", "La bibliothèque 'rapidocr-onnxruntime' n'est pas installée.\nRetour à Tesseract.")
                self.ocr_engine_var.set("Tesseract")
                os.environ['OCR_USE_RAPIDOCR'] = '0'
            else:
                os.environ['OCR_USE_RAPIDOCR'] = '1'
        else: # Tesseract
            os.environ['OCR_USE_RAPIDOCR'] = '0'
        
        # Mettre à jour la variable globale dans le module pipeline
        pipeline.USE_RAPID_OCR = (os.environ['OCR_USE_RAPIDOCR'] == '1')
        self.log(f"Moteur OCR défini sur: {selection}")

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
        self.log("📥 Chargement des images...")
        self.loaded_images = []
        self.baseline_scores = []

        for f in self.image_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Garantir uint8 pour compatibilité GPU
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                self.loaded_images.append(img)

        # Calcul des scores baseline
        self.log(f"🔍 Calcul des scores baseline...")
        self.baseline_scores = optimizer.calculate_baseline_scores(self.loaded_images)

        self.log(f"✅ {len(self.loaded_images)} images chargées")
        self.log(f"✅ {len(self.baseline_scores)} scores baseline calculés")

    def start_sobol(self):
        """Lance l'optimisation Sobol dans un thread."""
        if not self.loaded_images:
            messagebox.showwarning("Attention", "Chargez d'abord les images !")
            return

        threading.Thread(target=self.run_sobol, daemon=True).start()

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

        # Callback pour mise à jour de la GUI
        def sobol_callback(point_idx, scores_dict, params_dict):
            # Déterminer le moteur OCR actif pour le log
            ocr_engine_name = "RapidOCR" if pipeline.USE_RAPID_OCR else "Tesseract"
            
            msg = (f"[OCR: {ocr_engine_name}] [Point {point_idx+1}] Delta: {scores_dict['tesseract_delta']:.2f}% | "
                   f"Tess: {scores_dict['tesseract']:.2f}% | "
                   f"Netteté: {scores_dict['nettete']:.1f} | "
                   f"Contraste: {scores_dict['contraste']:.1f}")

            # Mise à jour thread-safe
            self.master.after(0, self.log, msg)

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
                self.status_label.config(text=f"✅ Sobol terminé! Résultats: {csv_file}")
            else:
                self.log("⚠️ Aucun meilleur paramètre trouvé (screening annulé?)")
                self.status_label.config(text="⏹️ Screening annulé")

        except Exception as e:
            self.log(f"❌ Erreur pendant le screening: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.status_label.config(text="❌ Erreur screening")

    def cancel_optimization(self):
        """Annule l'optimisation en cours."""
        self.cancellation_requested.set()
        self.log("⏹️ Annulation demandée...")


def main():
    """Point d'entrée principal."""
    print("[DEBUG] Démarrage de l'application...")

    # Initialiser CUDA avant multiprocessing
    print("[DEBUG] Initialisation CUDA...")
    pipeline._init_cuda()

    # Configuration multiprocessing (CRITIQUE pour Windows)
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[DEBUG] multiprocessing.set_start_method('spawn') OK")
    except RuntimeError as e:
        print(f"[DEBUG] multiprocessing déjà configuré: {e}")

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
