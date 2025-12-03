import time
import pytesseract
import cv2
import numpy as np
from pathlib import Path

# Configuration du chemin Tesseract (ajustez si nécessaire)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def benchmark_tesseract(image_path, num_iterations=5):
  """
  Benchmark des différentes configurations Tesseract

  Args:
      image_path: Chemin vers l'image de test
      num_iterations: Nombre d'itérations pour moyenner les résultats
  """

  # Configurations à tester
  configs = [
      ('Actuel (psm 6)', '--oem 1 --psm 6'),
      ('Actuel + français', '--oem 1 --psm 6 -l fra'),
      ('PSM 3 (auto)', '--oem 1 --psm 3'),
      ('PSM 3 + français', '--oem 1 --psm 3 -l fra'),
      ('PSM 11 (sparse)', '--oem 1 --psm 11'),
      ('PSM 11 + français', '--oem 1 --psm 11 -l fra'),
  ]

  # Charger l'image
  img = cv2.imread(str(image_path))
  if img is None:
      print(f"❌ Erreur: Impossible de charger l'image {image_path}")
      return

  # Convertir en niveaux de gris
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  print(f"\n{'='*70}")
  print(f"Benchmark Tesseract OCR")
  print(f"{'='*70}")
  print(f"Image: {Path(image_path).name}")
  print(f"Dimensions: {img.shape[1]}x{img.shape[0]} px")
  print(f"Itérations: {num_iterations}")
  print(f"{'='*70}\n")

  results = []

  for name, config in configs:
      times = []
      word_counts = []
      avg_confidences = []

      print(f"Test: {name:25s} ", end='', flush=True)

      for i in range(num_iterations):
          try:
              start = time.perf_counter()
              data = pytesseract.image_to_data(
                  gray,
                  config=config,
                  output_type=pytesseract.Output.DICT
              )
              elapsed = time.perf_counter() - start
              times.append(elapsed * 1000)  # Conversion en ms

              # Compter les mots détectés
              words = [w for w in data.get('text', []) if w.strip()]
              word_counts.append(len(words))

              # Calculer la confiance moyenne
              confs = [int(c) for c in data.get('conf', []) if c != '-1']
              if confs:
                  avg_confidences.append(np.mean(confs))

              print('.', end='', flush=True)

          except Exception as e:
              print(f"\n❌ Erreur avec {name}: {e}")
              break

      if times:
          avg_time = np.mean(times)
          std_time = np.std(times)
          avg_words = np.mean(word_counts) if word_counts else 0
          avg_conf = np.mean(avg_confidences) if avg_confidences else 0

          results.append({
              'name': name,
              'avg_time': avg_time,
              'std_time': std_time,
              'avg_words': avg_words,
              'avg_conf': avg_conf
          })

          print(f" ✓ {avg_time:6.1f} ms (±{std_time:4.1f})")

  # Afficher les résultats comparatifs
  print(f"\n{'='*70}")
  print(f"{'Configuration':<25} {'Temps (ms)':<15} {'Mots':<10} {'Conf %':<10} {'Vitesse'}")
  print(f"{'='*70}")

  baseline_time = results[0]['avg_time'] if results else 1

  for r in results:
      relative_speed = (baseline_time / r['avg_time']) * 100
      speed_indicator = '⚡' * min(5, int(relative_speed / 20))

      print(f"{r['name']:<25} "
            f"{r['avg_time']:6.1f} (±{r['std_time']:4.1f})  "
            f"{r['avg_words']:5.0f}     "
            f"{r['avg_conf']:5.1f}%    "
            f"{relative_speed:5.1f}% {speed_indicator}")

  # Recommandations
  print(f"\n{'='*70}")
  print("💡 Recommandations:")
  print(f"{'='*70}")

  fastest = min(results, key=lambda x: x['avg_time'])
  most_words = max(results, key=lambda x: x['avg_words'])
  best_conf = max(results, key=lambda x: x['avg_conf'])

  print(f"⚡ Plus rapide      : {fastest['name']} ({fastest['avg_time']:.1f} ms)")
  print(f"📝 Plus de mots    : {most_words['name']} ({most_words['avg_words']:.0f} mots)")
  print(f"🎯 Meilleure conf. : {best_conf['name']} ({best_conf['avg_conf']:.1f}%)")

  # Calcul du meilleur compromis (score combiné)
  for r in results:
      # Score = vitesse + qualité (confiance) - pénalité si peu de mots
      time_score = (baseline_time / r['avg_time']) * 50  # 50 points max
      conf_score = (r['avg_conf'] / 100) * 30  # 30 points max
      word_score = min(20, (r['avg_words'] / most_words['avg_words']) * 20)  # 20 points max
      r['total_score'] = time_score + conf_score + word_score

  best_overall = max(results, key=lambda x: x['total_score'])
  print(f"⚖️  Meilleur compromis : {best_overall['name']}")
  print(f"{'='*70}\n")


def main():
  """Point d'entrée principal"""
  import sys

  if len(sys.argv) < 2:
      print("Usage: python benchmark_tesseract.py <chemin_image> [iterations]")
      print("\nExemple:")
      print("  python benchmark_tesseract.py mon_image.png")
      print("  python benchmark_tesseract.py mon_image.png 10")
      sys.exit(1)

  image_path = sys.argv[1]
  iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5

  if not Path(image_path).exists():
      print(f"❌ Erreur: Le fichier {image_path} n'existe pas")
      sys.exit(1)

  benchmark_tesseract(image_path, iterations)


if __name__ == "__main__":
  main()