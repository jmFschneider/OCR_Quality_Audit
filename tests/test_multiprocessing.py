"""
Test de diagnostic multiprocessing
Vérifie où le programme bloque exactement
"""

import multiprocessing
import time
import sys

def test_worker(x):
    """Worker de test minimaliste"""
    print(f"Worker {x} - Démarré", flush=True)

    try:
        # Test 1: Import basique
        print(f"Worker {x} - Import cv2...", flush=True)
        import cv2
        print(f"Worker {x} - cv2 OK", flush=True)

        # Test 2: Import pipeline (C'EST ICI QUE ÇA BLOQUE PROBABLEMENT)
        print(f"Worker {x} - Import pipeline...", flush=True)
        import pipeline
        print(f"Worker {x} - pipeline OK", flush=True)

        # Test 3: Import optimizer
        print(f"Worker {x} - Import optimizer...", flush=True)
        import optimizer
        print(f"Worker {x} - optimizer OK", flush=True)

        print(f"Worker {x} - Terminé avec succès!", flush=True)
        return f"Success {x}"

    except Exception as e:
        print(f"Worker {x} - ERREUR: {e}", flush=True)
        return f"Error {x}: {e}"


def main():
    print("=" * 60)
    print("Test de diagnostic multiprocessing")
    print("=" * 60)

    # Configuration multiprocessing (comme dans gui_main.py)
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("✓ multiprocessing.set_start_method('spawn') configuré")
    except RuntimeError as e:
        print(f"⚠ multiprocessing déjà configuré: {e}")

    multiprocessing.freeze_support()
    print("✓ multiprocessing.freeze_support() appelé")

    print("\nLancement de 2 workers de test...")
    print("(Si ça bloque, appuyez sur Ctrl+C et notez où ça s'arrête)\n")

    try:
        with multiprocessing.Pool(processes=2) as pool:
            print("✓ Pool créé")

            # Timeout de 30 secondes
            results = pool.map_async(test_worker, [1, 2])

            # Attendre avec timeout
            final_results = results.get(timeout=30)

            print("\n" + "=" * 60)
            print("RÉSULTATS:")
            for r in final_results:
                print(f"  {r}")
            print("=" * 60)
            print("\n✅ Test réussi! Le multiprocessing fonctionne correctement.")

    except multiprocessing.TimeoutError:
        print("\n❌ TIMEOUT! Le pool a bloqué pendant plus de 30 secondes.")
        print("Le problème vient probablement de l'import de pipeline.py ou optimizer.py")

    except KeyboardInterrupt:
        print("\n⚠ Interrompu par l'utilisateur (Ctrl+C)")
        print("Notez le dernier message affiché pour identifier où ça bloque")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
