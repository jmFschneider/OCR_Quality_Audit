"""
optimizer.py - Logique d'optimisation des paramètres
Gestion des algorithmes d'optimisation (Sobol, Optuna, SciPy)
Correction & durcissement (CUDA/CPU + multiprocessing)
"""

import multiprocessing
from itertools import repeat, zip_longest
import os
import time
import csv
from datetime import datetime
import pipeline


# ============================================================
# LOGGER DE TEMPS (pour analyse post-traitement)
# ============================================================

class TimeLogger:
    """Enregistre les temps de traitement dans un fichier CSV."""

    def __init__(self, enabled=True, filename=None):
        """
        Args:
            enabled: Si False, désactive le logging (pas de fichier créé)
            filename: Nom du fichier CSV (auto-généré si None)
        """
        self.enabled = enabled
        self.filename = filename
        self.buffer = []
        self.buffer_size = 50  # Flush tous les 50 enregistrements

        if self.enabled:
            if self.filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.filename = f"timing_log_{timestamp}.csv"

            # Créer le fichier avec les en-têtes
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([
                    'timestamp', 'point_id', 'image_id',
                    'temps_total_ms', 'temps_cuda_ms',
                    'temps_tesseract_ms', 'temps_sharpness_ms', 'temps_contrast_ms',
                    'score_tesseract', 'score_sharpness', 'score_contrast'
                ])

            print(f"📊 Logging des temps activé: {self.filename}")

    def log(self, point_id, image_id, temps_total, temps_cuda,
            temps_tess, temps_sharp, temps_cont,
            score_tess=None, score_sharp=None, score_cont=None):
        """Enregistre une mesure de temps."""
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        self.buffer.append([
            timestamp, point_id, image_id,
            round(temps_total, 2),
            round(temps_cuda, 2),
            round(temps_tess, 2),
            round(temps_sharp, 2),
            round(temps_cont, 2),
            round(score_tess, 2) if score_tess is not None else '',
            round(score_sharp, 2) if score_sharp is not None else '',
            round(score_cont, 2) if score_cont is not None else ''
        ])

        # Flush si buffer plein
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Écrit le buffer dans le fichier."""
        if not self.enabled or not self.buffer:
            return

        try:
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(self.buffer)
            self.buffer = []
        except Exception as e:
            print(f"⚠️ Erreur écriture timing log: {e}")

    def close(self):
        """Flush final et fermeture."""
        self.flush()
        if self.enabled:
            print(f"✅ Logging des temps fermé: {self.filename}")


# Instance globale du logger (sera initialisée par run_sobol_screening)
_time_logger = None


# ============================================================
# WORKERS MULTIPROCESSING (MODE CPU)
# ============================================================

def process_image_fast(args):
    """Worker CPU pour multiprocessing."""
    # Mono-thread pour ne pas surcharger les workers
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    try:
        if len(args) == 4:
            img, params, baseline_score, pipeline_mode = args
        else:
            img, params, baseline_score = args
            pipeline_mode = 'standard'
    except ValueError:
        return None

    if img is None:
        return None

    # Traitement
    if pipeline_mode == 'blur_clahe':
        processed_img = pipeline.pipeline_blur_clahe(img, params)
    else:
        processed_img = pipeline.pipeline_complet(img, params)

    # Metrics
    score_tess = pipeline.get_tesseract_score(processed_img)
    baseline_score = baseline_score if baseline_score is not None else 0.0
    score_delta = score_tess - baseline_score
    score_sharp = pipeline.get_sharpness(processed_img)
    score_cnr = pipeline.get_cnr_quality(processed_img) # CNR

    return score_delta, score_tess, score_sharp, score_cnr


# ============================================================
# EVALUATION DU pipeline (GPU ou CPU)
# ============================================================

def evaluate_pipeline(images, baseline_scores, params, point_id=0, pipeline_mode='standard'):
    """Évalue le pipeline sur un ensemble d'images.

    Args:
        images: Liste d'images (numpy arrays)
        baseline_scores: Liste des scores de base (OCR sur images originales)
        params: Dictionnaire de paramètres du pipeline
        point_id: ID du point d'optimisation (pour le logging)
        pipeline_mode: 'standard' ou 'blur_clahe'

    Returns:
        (avg_delta, avg_abs, avg_sharp, avg_cont)
    """

    if not images:
        return 0, 0, 0, 0

    # STRATÉGIE ADAPTATIVE :
    # - Si CUDA : traitement séquentiel sur GPU (Uniquement pour mode standard pour l'instant)
    # - Si CPU ou Blur_CLAHE : multiprocessing parallèle

    # Note: Blur_CLAHE utilise des fonctions CPU (inpaint), donc on force le CPU multiprocessing
    # sauf si on a une implémentation GPU complète.
    use_cuda_path = pipeline.USE_CUDA and pipeline_mode == 'standard'

    if use_cuda_path:
        # MODE GPU : Pipeline séquentiel + Métriques parallèles
        # Le GPU ne peut pas être partagé, mais le calcul OCR CPU peut être parallélisé

        import time
        list_delta, list_abs, list_sharp, list_cont = [], [], [], []

        # PHASE 1: Pipeline CUDA (séquentiel - obligatoire)
        processed_images = []
        t0_pipeline = time.time()

        for img in images:
            processed = pipeline.pipeline_complet(img, params)
            processed_images.append(processed)

        t_pipeline_total = (time.time() - t0_pipeline) * 1000
        t_pipeline_avg = t_pipeline_total / len(images) if images else 0

        # PHASE 2: Métriques OCR (parallèle - multiprocessing)
        t0_metrics = time.time()
        metrics_results = pipeline.evaluer_toutes_metriques_batch(processed_images)
        t_metrics_total = (time.time() - t0_metrics) * 1000

        # PHASE 3: Accumulation des résultats
        global _time_logger

        for i, (tess_abs, sharp, cnr_score, t_tess, t_metrics, _) in enumerate(metrics_results):
            baseline = (
                baseline_scores[i] if i < len(baseline_scores)
                else 0.0
            )

            # Logger les temps (si activé)
            if _time_logger is not None:
                # Temps CUDA : moyenne du batch (approximation)
                t_cuda_cpu = t_pipeline_avg
                t_total = t_cuda_cpu + t_tess + t_metrics

                _time_logger.log(
                    point_id=point_id,
                    image_id=i,
                    temps_total=t_total,
                    temps_cuda=t_cuda_cpu,
                    temps_tess=t_tess,
                    temps_sharp=t_metrics, # Renommé implicitement
                    temps_cont=0,
                    score_tess=tess_abs,
                    score_sharp=sharp,
                    score_cont=cnr_score # C'est le CNR
                )

            # Accumulation résultats
            list_abs.append(tess_abs)
            list_delta.append(tess_abs - baseline)
            list_sharp.append(sharp)
            list_cont.append(cnr_score) # list_cont stocke maintenant CNR

    # ======================
    # MODE CPU (multiprocessing)
    # ======================
    else:
        # zip_longest sécurise les tailles différentes
        pool_args = zip_longest(images, repeat(params), baseline_scores, repeat(pipeline_mode), fillvalue=None)

        # Limite raisonnable : ne jamais dépasser os.cpu_count()
        max_workers = os.cpu_count()
        pool_size = min(len(images), max_workers)

        try:
            with multiprocessing.Pool(processes=pool_size) as pool:
                results = pool.map(process_image_fast, pool_args)

            valid = [r for r in results if r is not None]
            if not valid:
                return 0, 0, 0, 0

            list_delta, list_abs, list_sharp, list_cont = zip(*valid)

        except Exception as e:
            print(f"[optimizer] Erreur multiprocessing → fallback séquentiel ({e})")
            list_delta, list_abs, list_sharp, list_cont = [], [], [], []

            for i, img in enumerate(images):
                baseline = baseline_scores[i] if i < len(baseline_scores) else 0.0
                
                if pipeline_mode == 'blur_clahe':
                    processed_img = pipeline.pipeline_blur_clahe(img, params)
                else:
                    processed_img = pipeline.pipeline_complet(img, params)
                    
                tess_abs = pipeline.get_tesseract_score(processed_img)
                cnr_val = pipeline.get_cnr_quality(processed_img) # CNR explicit

                list_abs.append(tess_abs)
                list_delta.append(tess_abs - baseline)
                list_sharp.append(pipeline.get_sharpness(processed_img))
                list_cont.append(cnr_val)

    # ======================
    # MOYENNES
    # ======================
    return (
        sum(list_delta) / len(list_delta),
        sum(list_abs) / len(list_abs),
        sum(list_sharp) / len(list_sharp),
        sum(list_cont) / len(list_cont) # Moyenne CNR
    )


# ============================================================
# CALCUL DES SCORES BASELINE
# ============================================================

def calculate_baseline_scores(images, use_multiprocessing=True):
    """Calcule les scores OCR des images originales.

    Args:
        images: Liste d'images (numpy arrays)
        use_multiprocessing: Si True, utilise traitement parallèle (défaut: True)

    Returns:
        Liste des scores baseline
    """
    if use_multiprocessing and len(images) > 1:
        # Traitement parallèle (2-3x plus rapide)
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        max_workers = min(mp.cpu_count(), len(images))
        print(f"🚀 Calcul baseline: {len(images)} images avec {max_workers} workers")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            baseline_scores = list(executor.map(pipeline.get_tesseract_score, images))

        return baseline_scores
    else:
        # Traitement séquentiel (fallback)
        baseline_scores = []
        for img in images:
            try:
                score = pipeline.get_tesseract_score(img)
            except Exception:
                score = 0.0
            baseline_scores.append(score)
        return baseline_scores


# ============================================================
# UTILITAIRES PARAMÈTRES
# ============================================================

def build_params(line_h, line_v, norm_kernel_base, denoise_h, noise_threshold,
                bin_block_base, bin_c, dilate_iter=2):
    """Construit un dictionnaire de paramètres pour le pipeline.

    Args:
        line_h: Taille kernel horizontal pour suppression lignes
        line_v: Taille kernel vertical pour suppression lignes
        norm_kernel_base: Base pour norm_kernel (sera transformé en impair)
        denoise_h: Paramètre h pour denoising
        noise_threshold: Seuil pour denoising adaptatif
        bin_block_base: Base pour bin_block_size (sera transformé en impair)
        bin_c: Constante pour binarisation adaptative
        dilate_iter: Nombre d'itérations de dilatation

    Returns:
        Dictionnaire de paramètres
    """
    return {
        'line_h_size': int(line_h),
        'line_v_size': int(line_v),
        'dilate_iter': int(dilate_iter),
        'norm_kernel': int(norm_kernel_base) * 2 + 1,  # Toujours impair
        'denoise_h': float(denoise_h),
        'noise_threshold': float(noise_threshold),
        'bin_block_size': int(bin_block_base) * 2 + 1,  # Toujours impair
        'bin_c': float(bin_c)
    }


def params_to_tuple(params):
    """Convertit un dict de paramètres en tuple ordonné."""
    return (
        params['line_h_size'],
        params['line_v_size'],
        (params['norm_kernel'] - 1) // 2,  # Retour à la base
        params['denoise_h'],
        params['noise_threshold'],
        (params['bin_block_size'] - 1) // 2,  # Retour à la base
        params['bin_c'],
        params['dilate_iter']
    )


# ============================================================
# SCREENING SOBOL (Design of Experiments)
# ============================================================

def run_sobol_screening(images, baseline_scores, n_points, param_ranges,
                       fixed_params, callback=None, cancellation_event=None,
                       verbose_timing=True, enable_time_logging=True,
                       pipeline_mode='standard'):
    """Screening Sobol pur (Design of Experiments).

    Génère n_points avec une séquence Sobol et évalue tous sans optimisation.
    Sauvegarde tous les résultats dans un CSV pour analyse ultérieure.

    Args:
        images: Liste d'images chargées en mémoire
        baseline_scores: Scores OCR des images originales
        n_points: Nombre de points à évaluer
        param_ranges: Dict des ranges de paramètres actifs
                     ex: {'line_h': (30, 70), 'norm_kernel': (40, 100), ...}
        fixed_params: Dict des paramètres fixes (ex: {'dilate_iter': 2})
        callback: Fonction appelée après chaque point (optionnel)
                  callback(point_idx, scores_dict, params_dict)
        cancellation_event: threading.Event() pour annulation (optionnel)
        verbose_timing: DÉPRÉCIÉ - Les temps sont maintenant sauvegardés dans un CSV
        enable_time_logging: Si True, sauvegarde les temps dans un fichier CSV
        pipeline_mode: 'standard' ou 'blur_clahe'

    Returns:
        Tuple (best_params_dict, csv_filename)
    """
    from scipy.stats import qmc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"screening_sobol_{pipeline_mode}_{n_points}pts_{timestamp}.csv"

    # Initialiser le logger de temps
    global _time_logger
    if enable_time_logging:
        _time_logger = TimeLogger(enabled=True)
    else:
        _time_logger = None

    print(f"\n🔍 SCREENING SOBOL ({pipeline_mode}): Génération de {n_points} points")

    # Préparer les bornes pour Sobol
    param_names = list(param_ranges.keys())
    lower_bounds = [param_ranges[p][0] for p in param_names]
    upper_bounds = [param_ranges[p][1] for p in param_names]

    # Générer séquence Sobol
    sampler = qmc.Sobol(d=len(param_names), scramble=True)
    sobol_samples = sampler.random(n=n_points)
    scaled_samples = qmc.scale(sobol_samples, lower_bounds, upper_bounds)

    # Préparer le CSV
    header_map = {
        'line_h': 'line_h_size',
        'line_v': 'line_v_size',
        'norm_kernel': 'norm_kernel',
        'denoise_h': 'denoise_h',
        'noise_threshold': 'noise_threshold',
        'bin_block': 'bin_block_size',
        'bin_c': 'bin_c',
        # Nouveaux params Blur/CLAHE
        'inp_line_h': 'inp_line_h',
        'inp_line_v': 'inp_line_v',
        'bg_dilate': 'bg_dilate',
        'bg_blur': 'bg_blur',
        'clahe_clip': 'clahe_clip',
        'clahe_tile': 'clahe_tile'
    }

    csv_headers = ['point_id', 'score_tesseract_delta', 'score_tesseract',
                   'score_nettete', 'score_cnr']
    for p in param_names:
        csv_headers.append(header_map.get(p, p))

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)

    print(f"📄 Fichier de résultats: {csv_filename}")

    # ============================================================
    # SAUVEGARDE CONFIGURATION RUN
    # ============================================================
    config_filename = csv_filename.replace(".csv", "_config.csv")
    try:
        with open(config_filename, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(["Category", "Key", "Value_Min_or_Fixed", "Value_Max"])
            
            # 1. Info Globales
            w.writerow(["Info", "Pipeline Mode", pipeline_mode, ""])
            w.writerow(["Info", "Algorithm", "Sobol Screening", ""])
            w.writerow(["Info", "N Points", n_points, ""])
            w.writerow(["Info", "Timestamp", timestamp, ""])
            
            # 2. Paramètres Actifs (Ranges)
            for k, v in param_ranges.items():
                w.writerow(["Range", k, v[0], v[1]])
                
            # 3. Paramètres Fixes
            for k, v in fixed_params.items():
                w.writerow(["Fixed", k, v, ""])

            # 4. Baseline Stats
            if baseline_scores:
                b_mean = sum(baseline_scores) / len(baseline_scores)
                b_min = min(baseline_scores)
                b_max = max(baseline_scores)
                w.writerow(["Baseline", "Mean Score", f"{b_mean:.2f}", ""])
                w.writerow(["Baseline", "Min Score", f"{b_min:.2f}", ""])
                w.writerow(["Baseline", "Max Score", f"{b_max:.2f}", ""])
                w.writerow(["Baseline", "N Images", len(baseline_scores), ""])
            else:
                w.writerow(["Baseline", "Stats", "N/A", ""])

        print(f"📄 Fichier de configuration: {config_filename}")
        
    except Exception as e:
        print(f"⚠️ Erreur sauvegarde config: {e}")

    # Évaluer chaque point
    best_score = 0
    best_params = None

    csv_buffer = []
    BATCH_SIZE = 50  # Écriture par lots pour performance

    for idx, sample in enumerate(scaled_samples):
        # Vérifier annulation
        if cancellation_event and cancellation_event.is_set():
            print("⚠️ Screening annulé par l'utilisateur")
            break

        # Construire params dict
        params = fixed_params.copy()
        for i, param_name in enumerate(param_names):
            val = sample[i]
            # Conversion int/float selon le type de param
            if param_name in ['norm_kernel', 'bin_block', 'bg_dilate', 'bg_blur']:
                # Doit être impair
                params[param_name] = int(val) * 2 + 1
            elif param_name in ['line_h', 'line_v', 'inp_line_h', 'inp_line_v', 'clahe_tile']:
                params[param_name] = int(val)
            elif param_name == 'bin_block_size': # alias
                params['bin_block_size'] = int(val) * 2 + 1
            elif param_name == 'line_h_size': # alias
                params['line_h_size'] = int(val)
            elif param_name == 'line_v_size': # alias
                params['line_v_size'] = int(val)
            else:
                # Floats: denoise_h, noise_threshold, bin_c, clahe_clip
                params[param_name] = val

        # Map ancien noms vers nouveaux noms attendus par pipeline (si nécessaire)
        # Pour standard: line_h -> line_h_size
        if pipeline_mode == 'standard':
            if 'line_h' in params: params['line_h_size'] = params.pop('line_h')
            if 'line_v' in params: params['line_v_size'] = params.pop('line_v')
            if 'bin_block' in params: params['bin_block_size'] = params.pop('bin_block')
            
        # Pour blur_clahe, les noms clés dans params correspondent déjà aux args de la fonction 
        # (inp_line_h, etc.) sauf si on a fait des mappings dans GUI.
        # On assume que GUI envoie 'inp_line_h' etc.

        # Évaluer
        avg_delta, avg_abs, avg_sharp, avg_cont = evaluate_pipeline(
            images, baseline_scores, params, point_id=idx+1, pipeline_mode=pipeline_mode
        )

        # Ajouter au buffer CSV
        row_data = [idx + 1, avg_delta, avg_abs, avg_sharp, avg_cont]
        for p in param_names:
            # Récupérer la valeur "transformée" ou brute
            # Attention: ici on veut logger la valeur "source" (celle de l'optimiseur) 
            # ou la valeur "finale" ? Le header correspond aux noms optimizer.
            # On va logger la valeur telle qu'utilisée par le pipeline si possible, 
            # ou la valeur du sample.
            # Par simplicité, on log la valeur params[mapped_name] si dispo, sinon sample.
            
            # Mapping inverse rapide pour retrouver la clé dans 'params'
            mapped_key = p
            if p == 'line_h': mapped_key = 'line_h_size'
            elif p == 'line_v': mapped_key = 'line_v_size'
            elif p == 'bin_block': mapped_key = 'bin_block_size'
            
            val_to_log = params.get(mapped_key, sample[param_names.index(p)])
            row_data.append(val_to_log)

        csv_buffer.append(row_data)

        # Écriture par lots (Batching pour performance)
        if len(csv_buffer) >= BATCH_SIZE:
            try:
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerows(csv_buffer)
                csv_buffer = []
            except Exception as e:
                print(f"Erreur écriture CSV batch: {e}")

        # Suivi du meilleur
        if avg_delta > best_score:
            best_score = avg_delta
            best_params = params.copy()
            print(f"🔥 Point {idx+1}/{n_points}: Nouveau meilleur gain = {avg_delta:.2f}%")
        else:
            if (idx + 1) % 50 == 0:  # Log tous les 50 points
                print(f"   Point {idx+1}/{n_points}: Gain = {avg_delta:.2f}%")

        # Callback optionnel pour mise à jour GUI
        if callback:
            scores_dict = {
                'tesseract_delta': avg_delta,
                'tesseract': avg_abs,
                'nettete': avg_sharp,
                'cnr': avg_cont  # avg_cont contient maintenant le CNR
            }
            callback(idx, scores_dict, params)

    # Vider le reste du buffer
    if csv_buffer:
        try:
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(csv_buffer)
        except Exception as e:
            print(f"Erreur écriture CSV final: {e}")

    print(f"\n✅ Screening terminé! Meilleur gain: {best_score:.2f}%")
    print(f"📊 {n_points} points évalués et sauvegardés dans {csv_filename}")

    # Fermer le logger de temps
    if _time_logger is not None:
        _time_logger.close()

    return best_params, csv_filename
