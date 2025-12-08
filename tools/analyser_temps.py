#!/usr/bin/env python3
"""
Analyseur de fichiers de timing
Calcule des statistiques sur les temps de traitement enregistrés
"""

import csv
import sys
from glob import glob
import statistics

def analyser_fichier_timing(filename):
    """Analyse un fichier de timing et affiche les statistiques."""

    print("="*70)
    print(f"ANALYSE DU FICHIER: {filename}")
    print("="*70)

    # Lire le fichier
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                data.append({
                    'point_id': int(row['point_id']),
                    'image_id': int(row['image_id']),
                    'temps_total': float(row['temps_total_ms']),
                    'temps_cuda': float(row['temps_cuda_ms']),
                    'temps_tess': float(row['temps_tesseract_ms']),
                    'temps_sharp': float(row['temps_sharpness_ms']),
                    'temps_cont': float(row['temps_contrast_ms']),
                    'score_tess': float(row['score_tesseract']) if row['score_tesseract'] else None,
                    'score_sharp': float(row['score_sharpness']) if row['score_sharpness'] else None,
                    'score_cont': float(row['score_contrast']) if row['score_contrast'] else None,
                })
            except (ValueError, KeyError) as e:
                print(f"⚠️ Ligne ignorée: {e}")
                continue

    if not data:
        print("❌ Aucune donnée valide trouvée")
        return

    print(f"\n📊 {len(data)} mesures chargées")

    # Extraire les données
    temps_total = [d['temps_total'] for d in data]
    temps_cuda = [d['temps_cuda'] for d in data]
    temps_tess = [d['temps_tess'] for d in data]
    temps_sharp = [d['temps_sharp'] for d in data]
    temps_cont = [d['temps_cont'] for d in data]

    # Statistiques globales
    print("\n" + "="*70)
    print("STATISTIQUES GLOBALES (tous les points et images)")
    print("="*70)

    print(f"\n{'Métrique':<20} {'Min':>10} {'Max':>10} {'Moyenne':>10} {'Médiane':>10} {'Écart-type':>12}")
    print("-"*74)

    metriques = [
        ("Temps total", temps_total),
        ("Temps CUDA", temps_cuda),
        ("Temps Tesseract", temps_tess),
        ("Temps Netteté", temps_sharp),
        ("Temps Contraste", temps_cont),
    ]

    for nom, valeurs in metriques:
        if valeurs:
            print(f"{nom:<20} {min(valeurs):>10.1f} {max(valeurs):>10.1f} "
                  f"{statistics.mean(valeurs):>10.1f} {statistics.median(valeurs):>10.1f} "
                  f"{statistics.stdev(valeurs):>12.1f}")

    # Répartition en pourcentage
    print("\n" + "="*70)
    print("RÉPARTITION DES TEMPS (en % du temps total moyen)")
    print("="*70)

    avg_total = statistics.mean(temps_total)
    avg_cuda = statistics.mean(temps_cuda)
    avg_tess = statistics.mean(temps_tess)
    avg_sharp = statistics.mean(temps_sharp)
    avg_cont = statistics.mean(temps_cont)

    print(f"\nTemps total moyen: {avg_total:.1f} ms\n")
    print(f"{'Composant':<20} {'Temps (ms)':>12} {'% du total':>12}")
    print("-"*44)
    print(f"{'CUDA (traitement):':<20} {avg_cuda:>12.1f} {(avg_cuda/avg_total*100):>11.1f}%")
    print(f"{'Tesseract (OCR):':<20} {avg_tess:>12.1f} {(avg_tess/avg_total*100):>11.1f}%")
    print(f"{'Netteté:':<20} {avg_sharp:>12.1f} {(avg_sharp/avg_total*100):>11.1f}%")
    print(f"{'Contraste:':<20} {avg_cont:>12.1f} {(avg_cont/avg_total*100):>11.1f}%")

    # Statistiques par point
    points = sorted(set(d['point_id'] for d in data))
    if len(points) > 1:
        print("\n" + "="*70)
        print("STATISTIQUES PAR POINT SOBOL")
        print("="*70)

        print(f"\n{'Point':>6} {'Nb images':>10} {'Temps total moy':>18} {'CUDA moy':>12} {'Tess moy':>12}")
        print("-"*70)

        for point in points:
            point_data = [d for d in data if d['point_id'] == point]
            nb_images = len(point_data)
            avg_total_point = statistics.mean([d['temps_total'] for d in point_data])
            avg_cuda_point = statistics.mean([d['temps_cuda'] for d in point_data])
            avg_tess_point = statistics.mean([d['temps_tess'] for d in point_data])

            print(f"{point:>6} {nb_images:>10} {avg_total_point:>18.1f} ms {avg_cuda_point:>12.1f} ms {avg_tess_point:>12.1f} ms")

    # Statistiques par image
    images = sorted(set(d['image_id'] for d in data))
    if len(images) > 1:
        print("\n" + "="*70)
        print("STATISTIQUES PAR IMAGE")
        print("="*70)

        print(f"\n{'Image':>6} {'Nb mesures':>12} {'Temps total moy':>18} {'CUDA moy':>12} {'Tess moy':>12}")
        print("-"*70)

        for img_id in images:
            img_data = [d for d in data if d['image_id'] == img_id]
            nb_mesures = len(img_data)
            avg_total_img = statistics.mean([d['temps_total'] for d in img_data])
            avg_cuda_img = statistics.mean([d['temps_cuda'] for d in img_data])
            avg_tess_img = statistics.mean([d['temps_tess'] for d in img_data])

            print(f"{img_id:>6} {nb_mesures:>12} {avg_total_img:>18.1f} ms {avg_cuda_img:>12.1f} ms {avg_tess_img:>12.1f} ms")

    # Recommandations
    print("\n" + "="*70)
    print("RECOMMANDATIONS D'OPTIMISATION")
    print("="*70)

    pct_cuda = (avg_cuda / avg_total) * 100
    pct_tess = (avg_tess / avg_total) * 100

    print("\nAnalyse du goulot d'étranglement:")
    if pct_tess > 70:
        print(f"⚠️  Tesseract représente {pct_tess:.1f}% du temps total")
        print("   → Envisager un OCR avec support GPU (EasyOCR, PaddleOCR)")
        print("   → Ou paralléliser Tesseract sur plusieurs images")
    elif pct_cuda > 50:
        print(f"⚠️  Traitement CUDA représente {pct_cuda:.1f}% du temps total")
        print("   → Optimiser les kernels CUDA")
        print("   → Réduire les transferts GPU↔CPU")
    else:
        print("✅ Temps bien réparti entre CUDA et Tesseract")

    # Estimation pour différents volumes
    print("\n" + "="*70)
    print("ESTIMATIONS DE TEMPS POUR DIFFÉRENTS VOLUMES")
    print("="*70)

    print(f"\nTemps moyen par image: {avg_total:.0f} ms")
    print(f"\n{'Nb images':<12} {'Nb points':<12} {'Temps estimé':>15}")
    print("-"*39)

    volumes = [
        (2, 32),
        (10, 32),
        (24, 32),
        (2, 128),
        (24, 128),
        (24, 256),
    ]

    for nb_img, nb_pts in volumes:
        temps_s = (nb_img * nb_pts * avg_total) / 1000
        if temps_s < 60:
            temps_str = f"{temps_s:.0f}s"
        elif temps_s < 3600:
            temps_str = f"{temps_s/60:.1f}min"
        else:
            temps_str = f"{temps_s/3600:.1f}h"
        print(f"{nb_img:<12} {nb_pts:<12} {temps_str:>15}")

    print("\n" + "="*70)


def main():
    """Point d'entrée principal."""

    # Chercher les fichiers de timing
    timing_files = sorted(glob("timing_log_*.csv"), reverse=True)

    if not timing_files:
        print("❌ Aucun fichier timing_log_*.csv trouvé")
        sys.exit(1)

    # Si un fichier est spécifié en argument, l'utiliser
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if not filename.endswith('.csv'):
            filename += '.csv'
    else:
        # Sinon, utiliser le plus récent
        filename = timing_files[0]
        if len(timing_files) > 1:
            print(f"\n📁 {len(timing_files)} fichiers trouvés, analyse du plus récent:")
            print(f"   {filename}")
            print(f"\n💡 Utilisez: python3 analyser_temps.py <fichier> pour analyser un autre fichier\n")

    try:
        analyser_fichier_timing(filename)
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
