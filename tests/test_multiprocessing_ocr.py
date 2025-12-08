"""
Test de diagnostic multiprocessing avec OCR réel
Vérifie si l'appel OCR bloque dans les workers
"""

import multiprocessing
import numpy as np
import cv2
import os


def test_ocr_worker(worker_id):
    """Worker qui tente un vrai appel OCR"""
    print(f"Worker {worker_id} - Démarré", flush=True)

    try:
        # Import des modules
        print(f"Worker {worker_id} - Import pipeline...", flush=True)
        import pipeline
        print(f"Worker {worker_id} - pipeline importé (USE_RAPID_OCR={pipeline.USE_RAPID_OCR})", flush=True)

        # Créer une image de test simple
        print(f"Worker {worker_id} - Création image de test...", flush=True)
        test_img = np.ones((100, 300), dtype=np.uint8) * 255
        cv2.putText(test_img, "TEST", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        print(f"Worker {worker_id} - Image créée", flush=True)

        # APPEL OCR (C'EST ICI QUE ÇA BLOQUE PROBABLEMENT)
        print(f"Worker {worker_id} - Appel get_tesseract_score()...", flush=True)
        score = pipeline.get_tesseract_score(test_img)
        print(f"Worker {worker_id} - OCR OK! Score: {score}", flush=True)

        return f"Worker {worker_id}: Success (score={score})"

    except Exception as e:
        print(f"Worker {worker_id} - ERREUR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return f"Worker {worker_id}: Error - {e}"


def main():
    print("=" * 60)
    print("Test multiprocessing avec OCR réel")
    print("=" * 60)

    # Configuration multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("✓ spawn configuré")
    except RuntimeError as e:
        print(f"⚠ {e}")

    multiprocessing.freeze_support()

    # Choisir le moteur OCR
    print("\nQuel moteur OCR voulez-vous tester?")
    print("1. Tesseract (défaut)")
    print("2. RapidOCR")
    choice = input("Choix [1/2]: ").strip() or "1"

    if choice == "2":
        os.environ['OCR_USE_RAPIDOCR'] = '1'
        print("✓ RapidOCR sélectionné")
    else:
        os.environ['OCR_USE_RAPIDOCR'] = '0'
        print("✓ Tesseract sélectionné")

    print("\nLancement de 2 workers avec OCR réel...")
    print("⏱️  Timeout de 60 secondes")
    print("(Si ça bloque, Ctrl+C et notez le dernier message)\n")

    try:
        with multiprocessing.Pool(processes=2) as pool:
            print("✓ Pool créé")

            # Lancer les workers
            results = pool.map_async(test_ocr_worker, [1, 2])

            # Attendre avec timeout de 60 secondes
            final_results = results.get(timeout=60)

            print("\n" + "=" * 60)
            print("RÉSULTATS:")
            for r in final_results:
                print(f"  {r}")
            print("=" * 60)
            print("\n✅ Test réussi! L'OCR fonctionne en multiprocessing.")

    except multiprocessing.TimeoutError:
        print("\n" + "=" * 60)
        print("❌ TIMEOUT après 60 secondes!")
        print("=" * 60)
        print("\nLE PROBLÈME EST IDENTIFIÉ:")
        print("- L'appel à get_tesseract_score() ou get_rapidocr_score()")
        print("  bloque dans les workers multiprocessing")
        print("\nSOLUTIONS POSSIBLES:")
        print("1. Désactiver le multiprocessing (mode séquentiel)")
        print("2. Utiliser un autre moteur OCR")
        print("3. Ajouter un timeout sur les appels OCR")

    except KeyboardInterrupt:
        print("\n⚠ Interrompu par Ctrl+C")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
