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

        ttk.Label(opt_frame, text="Points Sobol:").grid(row=0, column=0, padx=5)
        self.sobol_points = ttk.Entry(opt_frame, width=10)
        self.sobol_points.insert(0, "32")
        self.sobol_points.grid(row=0, column=1, padx=5)

        ttk.Button(opt_frame, text="▶️ Lancer Sobol", 
                  command=self.start_sobol).grid(row=0, column=2, padx=5)

        ttk.Button(opt_frame, text="⏹️ Arrêter", 
                  command=self.cancel_optimization).grid(row=0, column=3, padx=5)

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

    def log(self, message):
        """Ajoute un message au log."""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.update()

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
        self.log("📥 Chargement des images...")
        self.loaded_images = []
        self.baseline_scores = []

        for f in self.image_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.dtype == 'uint8':
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
            n_points = int(self.sobol_points.get())
        except:
            self.log("❌ Nombre de points invalide")
            return

        self.cancellation_requested.clear()
        self.log(f"\n🚀 Démarrage Sobol avec {n_points} points")

        # Récupérer les ranges actifs
        active_ranges = {}
        for name, enabled_var in self.param_enabled_vars.items():
            if enabled_var.get():
                try:
                    min_val = float(self.param_entries[name]['min'].get())
                    max_val = float(self.param_entries[name]['max'].get())
                    active_ranges[name] = (min_val, max_val)
                except:
                    self.log(f"❌ Valeurs invalides pour {name}")
                    return

        # TODO: Implémenter Sobol screening avec optimizer.evaluate_pipeline()
        self.log("⚠️  Sobol screening à implémenter")
        self.log("✅ Terminé")

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
