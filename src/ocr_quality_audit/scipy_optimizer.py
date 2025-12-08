# Ce fichier contiendra la logique d'optimisation avec Scipy.
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

# Wrapper pour la fonction objectif pour compter les appels
class ObjectiveWrapper:
    def __init__(self, objective_func, cancellation_event=None):
        self.objective_func = objective_func
        self.cancellation_event = cancellation_event
        self.evaluations = 0
        self.start_time = None

    def __call__(self, params):
        # Vérification de l'annulation au début de chaque évaluation
        if self.cancellation_event and self.cancellation_event.is_set():
            # Le bloc try/except autour de minimize attrapera cette exception
            raise StopOptimization("Annulation détectée dans l'objectif.")

        if self.start_time is None:
            self.start_time = time.time()
        
        self.evaluations += 1
        return self.objective_func(params)

    def get_stats(self):
        return self.evaluations, time.time() - self.start_time if self.start_time else 0

# Exception personnalisée pour arrêter l'optimisation proprement
class StopOptimization(Exception):
    pass

def run_scipy_optimization(objective_func, bounds, algorithm, n_sobol_points, n_iterations, update_callback=None, cancellation_event=None):
    """
    Exécute l'optimisation avec Scipy à partir de points de départ générés par une séquence de Sobol.

    Algorithmes supportant les bounds: L-BFGS-B, TNC, SLSQP
    Algorithmes sans bounds: Nelder-Mead, Powell, CG, BFGS, COBYLA
    """
    # Algorithmes qui supportent les contraintes de bornes
    BOUNDED_ALGORITHMS = {'L-BFGS-B', 'TNC', 'SLSQP'}

    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    sobol_sampler = qmc.Sobol(d=len(bounds), scramble=True)
    sobol_points = sobol_sampler.random(n=n_sobol_points)
    scaled_sobol_points = qmc.scale(sobol_points, lower_bounds, upper_bounds)

    best_result = {"fun": float('inf'), "x": None}

    # Le wrapper vérifie maintenant l'annulation lui-même
    wrapped_objective = ObjectiveWrapper(objective_func, cancellation_event)

    # Le callback reste une sécurité supplémentaire
    def callback(xk):
        if cancellation_event and cancellation_event.is_set():
            raise StopOptimization("Annulation détectée dans le callback.")

    # Vérifier si l'algorithme supporte les bounds
    use_bounds = algorithm in BOUNDED_ALGORITHMS

    if not use_bounds and update_callback:
        update_callback(f"ℹ️  {algorithm} ne supporte pas les contraintes de bornes (recherche non contrainte)")

    for i, start_point in enumerate(scaled_sobol_points):
        if cancellation_event and cancellation_event.is_set():
            if update_callback:
                update_callback("Optimisation Scipy annulée avant un nouveau point de départ.")
            break

        if update_callback:
            update_callback(f"Optimisation depuis le point de départ {i+1}/{n_sobol_points}...")

        try:
            # Préparer les options de minimize
            minimize_kwargs = {
                'fun': wrapped_objective,
                'x0': start_point,
                'method': algorithm,
                'options': {'maxiter': n_iterations},
                'callback': callback if cancellation_event else None
            }

            # Ajouter bounds seulement si l'algorithme le supporte
            if use_bounds:
                minimize_kwargs['bounds'] = bounds

            result = minimize(**minimize_kwargs)

            if result.fun < best_result["fun"]:
                best_result = result
                if update_callback:
                    update_callback(f"Nouveau meilleur résultat trouvé : {result.fun:.4f}")
        except StopOptimization as e:
            if update_callback:
                update_callback(f"Arrêt de l'optimisation locale ({e})")
            break # Sortir de la boucle for principale

    evals, duration = wrapped_objective.get_stats()
    if update_callback:
        if cancellation_event and cancellation_event.is_set():
             update_callback(f"Scipy annulé. Total: {evals} évaluations en {duration:.2f} secondes.")
        else:
             update_callback(f"Optimisation Scipy terminée. Total: {evals} évaluations en {duration:.2f} secondes.")

    return best_result
