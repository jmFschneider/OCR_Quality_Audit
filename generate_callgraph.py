import tkinter as tk
from gui_optimizer_v3_ultim import OptimizerGUI, run_optimization
from pycallgraph2 import PyCallGraph, Config
from pycallgraph2.output import GraphvizOutput
from pycallgraph2 import GlobbingFilter

# --- Sortie graphe ---
graphviz = GraphvizOutput()
graphviz.output_file = 'callgraph_core2.png'

config = Config()
config.trace_return_value = True
config.trace_filter = GlobbingFilter(
    include=['gui_optimizer_v3_ultim.*'],
    exclude=['tkinter.*', 'threading.*', 'builtins.*', 'os.*', 'cv2.*', 'numpy.*', 'pytesseract.*', 'optuna.*']
)

with PyCallGraph(output=graphviz, config=config):
    # Créer un vrai root mais le cacher immédiatement
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre

    app = OptimizerGUI(root)

    # Récupérer config active et fixed params
    active_ranges, fixed_params = app.get_optim_config()

    # Lancer un trial minimal pour générer les appels
    run_optimization(app, n_trials=1, param_ranges=active_ranges, fixed_params=fixed_params, algo_choice="TPE (Bayésien)")
