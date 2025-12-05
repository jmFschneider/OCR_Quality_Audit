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
        img, params, baseline_score = args
    except ValueError:
        return None


    if img is None:
        return None

    # Traitement
    processed_img = pipeline.pipeline_complet(img, params)

    # Metrics
    score_tess = pipeline.get_tesseract_score(processed_img)
    baseline_score = baseline_score if baseline_score is not None else 0.0
    score_delta = score_tess - baseline_score
    score_sharp = pipeline.get_sharpness(processed_img)
    score_cont = pipeline.get_contrast(processed_img)

    return score_delta, score_tess, score_sharp, score_cont


# ============================================================
# EVALUATION DU pipeline (GPU ou CPU)
# ============================================================

def evaluate_pipeline(images, baseline_scores, params, point_id=0):
    """Évalue le pipeline sur un ensemble d'images.

    Args:
        images: Liste d'images (numpy arrays)
        baseline_scores: Liste des scores de base (OCR sur images originales)
        params: Dictionnaire de paramètres du pipeline
        point_id: ID du point d'optimisation (pour le logging)

    Returns:
        (avg_delta, avg_abs, avg_sharp, avg_cont)
    """

    if not images:
        return 0, 0, 0, 0

    # STRATÉGIE ADAPTATIVE :
    # - Si CUDA : traitement séquentiel sur GPU
    # - Si CPU : multiprocessing parallèle

    if pipeline.USE_CUDA:
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

        for i, (tess_abs, sharp, cont, t_tess, t_sharp, t_cont) in enumerate(metrics_results):
            baseline = (
                baseline_scores[i] if i < len(baseline_scores)
                else 0.0
            )

            # Logger les temps (si activé)
            if _time_logger is not None:
                # Temps CUDA : moyenne du batch (approximation)
                t_cuda_cpu = t_pipeline_avg
                t_total = t_cuda_cpu + t_tess + t_sharp + t_cont

                _time_logger.log(
                    point_id=point_id,
                    image_id=i,
                    temps_total=t_total,
                    temps_cuda=t_cuda_cpu,
                    temps_tess=t_tess,
                    temps_sharp=t_sharp,
                    temps_cont=t_cont,
                    score_tess=tess_abs,
                    score_sharp=sharp,
                    score_cont=cont
                )

            # Accumulation résultats
            list_abs.append(tess_abs)
            list_delta.append(tess_abs - baseline)
            list_sharp.append(sharp)
            list_cont.append(cont)

    # ======================
    # MODE CPU (multiprocessing)
    # ======================
    else:
        # zip_longest sécurise les tailles différentes
        pool_args = zip_longest(images, repeat(params), baseline_scores, fillvalue=0.0)

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
            print(f"[optimizer_chat] Erreur multiprocessing → fallback séquentiel : {e}")
            list_delta, list_abs, list_sharp, list_cont = [], [], [], []

            for i, img in enumerate(images):
                baseline = baseline_scores[i] if i < len(baseline_scores) else 0.0
                processed_img = pipeline.pipeline_complet(img, params)
                tess_abs = pipeline.get_tesseract_score(processed_img)

                list_abs.append(tess_abs)
                list_delta.append(tess_abs - baseline)
                list_sharp.append(pipeline.get_sharpness(processed_img))
                list_cont.append(pipeline.get_contrast(processed_img))

    # ======================
    # MOYENNES
    # ======================
    return (
        sum(list_delta) / len(list_delta),
        sum(list_abs) / len(list_abs),
        sum(list_sharp) / len(list_sharp),
        sum(list_cont) / len(list_cont)
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

def _evaluate_single_point_worker(args):
    """Worker pour évaluer un seul point (utilisé par le multiprocessing)."""
    (idx, sample, param_names, param_ranges, fixed_params,
     images, baseline_scores) = args

    # Construire params dict
    params = fixed_params.copy()
    for i, param_name in enumerate(param_names):
        val = sample[i]
        if param_name == 'norm_kernel':
            params['norm_kernel'] = int(val) * 2 + 1
        elif param_name == 'bin_block':
            params['bin_block_size'] = int(val) * 2 + 1
        elif param_name == 'line_h':
            params['line_h_size'] = int(val)
        elif param_name == 'line_v':
            params['line_v_size'] = int(val)
        elif param_name in ['denoise_h', 'noise_threshold', 'bin_c']:
            params[param_name] = val
        else:
            params[param_name] = val

    # Évaluer le pipeline
    avg_delta, avg_abs, avg_sharp, avg_cont = evaluate_pipeline(
        images, baseline_scores, params, point_id=idx+1
    )

    return (idx, avg_delta, avg_abs, avg_sharp, avg_cont, params)


def run_sobol_screening(images, baseline_scores, n_points, param_ranges,
                       fixed_params, callback=None, cancellation_event=None,
                       verbose_timing=True, enable_time_logging=True,
                       points_per_batch=None):
    """Screening Sobol pur (Design of Experiments) avec traitement parallèle des points.

    Génère n_points avec une séquence Sobol et évalue tous sans optimisation.
    Traite plusieurs points en parallèle pour maximiser l'utilisation CPU.
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
        points_per_batch: Nombre de points à traiter en parallèle (None = auto)

    Returns:
        Tuple (best_params_dict, csv_filename)
    """
    from scipy.stats import qmc
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"screening_sobol_{n_points}pts_{timestamp}.csv"

    # Initialiser le logger de temps
    global _time_logger
    if enable_time_logging:
        _time_logger = TimeLogger(enabled=True)
    else:
        _time_logger = None

    print(f"\n🔍 SCREENING SOBOL: Génération de {n_points} points")

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
        'bin_c': 'bin_c'
    }

    csv_headers = ['point_id', 'score_tesseract_delta', 'score_tesseract',
                   'score_nettete', 'score_contraste']
    for p in param_names:
        csv_headers.append(header_map.get(p, p))

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(csv_headers)

    print(f"📄 Fichier de résultats: {csv_filename}")

    # Calculer le nombre de points à traiter en parallèle
    if points_per_batch is None:
        # Auto: nombre de cores / nombre d'images par point
        points_per_batch = max(1, mp.cpu_count() // len(images))

    print(f"⚡ Traitement par batches: {points_per_batch} points en parallèle")
    print(f"💻 Utilisation CPU estimée: {points_per_batch * len(images)} workers max")

    # Évaluer chaque point (par batches parallèles)
    best_score = 0
    best_params = None

    csv_buffer = []
    CSV_BATCH_SIZE = 50  # Écriture par lots pour performance

    # Préparer les arguments pour tous les points
    all_point_args = [
        (idx, sample, param_names, param_ranges, fixed_params, images, baseline_scores)
        for idx, sample in enumerate(scaled_samples)
    ]

    # Traiter par batches
    for batch_start in range(0, len(all_point_args), points_per_batch):
        # Vérifier annulation
        if cancellation_event and cancellation_event.is_set():
            print("⚠️ Screening annulé par l'utilisateur")
            break

        batch_end = min(batch_start + points_per_batch, len(all_point_args))
        batch_args = all_point_args[batch_start:batch_end]

        # Traiter ce batch en parallèle
        with ThreadPoolExecutor(max_workers=points_per_batch) as executor:
            batch_results = list(executor.map(_evaluate_single_point_worker, batch_args))

        # Traiter les résultats du batch
        for idx, avg_delta, avg_abs, avg_sharp, avg_cont, params in batch_results:
            # Ajouter au buffer CSV
            row_data = [idx + 1, avg_delta, avg_abs, avg_sharp, avg_cont]
            for p in param_names:
                if p == 'norm_kernel':
                    row_data.append(params.get('norm_kernel'))
                elif p == 'bin_block':
                    row_data.append(params.get('bin_block_size'))
                elif p == 'line_h':
                    row_data.append(params.get('line_h_size'))
                elif p == 'line_v':
                    row_data.append(params.get('line_v_size'))
                else:
                    row_data.append(params.get(p))

            csv_buffer.append(row_data)

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
                    'contraste': avg_cont
                }
                callback(idx, scores_dict, params)

        # Écriture par lots (Batching pour performance)
        if len(csv_buffer) >= CSV_BATCH_SIZE:
            try:
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerows(csv_buffer)
                csv_buffer = []
            except Exception as e:
                print(f"Erreur écriture CSV batch: {e}")

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
