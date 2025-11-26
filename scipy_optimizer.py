# Ce fichier contiendra la logique d'optimisation avec Scipy.
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

# Wrapper pour la fonction objectif pour compter les appels
class ObjectiveWrapper:
    def __init__(self, objective_func):
        self.objective_func = objective_func
        self.evaluations = 0
        self.start_time = None

    def __call__(self, params):
        if self.start_time is None:
            self.start_time = time.time()
        
        self.evaluations += 1
        return self.objective_func(params)

    def get_stats(self):
        return self.evaluations, time.time() - self.start_time if self.start_time else 0

def run_scipy_optimization(objective_func, bounds, algorithm, n_sobol_points, n_iterations, update_callback=None):
    """
    Exécute l'optimisation avec Scipy à partir de points de départ générés par une séquence de Sobol.

    :param objective_func: La fonction à optimiser.
    :param bounds: Les limites des paramètres.
    :param algorithm: L'algorithme Scipy à utiliser.
    :param n_sobol_points: Le nombre de points de départ à générer.
    :param n_iterations: Le nombre maximum d'itérations pour chaque optimisation.
    :param update_callback: Callback pour mettre à jour l'UI avec la progression.
    :return: Le meilleur résultat trouvé.
    """
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    
    # Générateur de séquence de Sobol
    sobol_sampler = qmc.Sobol(d=len(bounds), scramble=True)
    sobol_points = sobol_sampler.random(n=n_sobol_points)
    scaled_sobol_points = qmc.scale(sobol_points, lower_bounds, upper_bounds)

    best_result = {"fun": float('inf'), "x": None}
    
    wrapped_objective = ObjectiveWrapper(objective_func)

    for i, start_point in enumerate(scaled_sobol_points):
        if update_callback:
            update_callback(f"Optimisation depuis le point de départ {i+1}/{n_sobol_points}...")

        result = minimize(
            wrapped_objective,
            x0=start_point,
            method=algorithm,
            bounds=bounds,
            options={'maxiter': n_iterations}
        )

        if result.fun < best_result["fun"]:
            best_result = result
            if update_callback:
                update_callback(f"Nouveau meilleur résultat trouvé : {result.fun:.4f}")

    evals, duration = wrapped_objective.get_stats()
    if update_callback:
        update_callback(f"Optimisation Scipy terminée. Total: {evals} évaluations en {duration:.2f} secondes.")
        
    return best_result
