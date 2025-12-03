# Gemini Project State: OCR Quality Audit Optimizer

## 1. Project Objective

The primary goal of this project is to create a tool that can systematically find the optimal image preprocessing parameters to improve the quality of Tesseract OCR results. The tool provides a GUI to configure and run optimization tasks.

## 2. Key Files & Structure

*   `gui_optimizer_v3_ultim.py`: This is the main application file. It contains:
    *   The Tkinter-based Graphical User Interface (GUI).
    *   The logic for orchestrating optimization runs.
    *   The core image processing pipeline (`pipeline_complet`).
    *   The objective functions for the optimizers.

*   `scipy_optimizer.py`: This module holds the specific implementation for running an optimization using the Scipy library. It is called by the main GUI when "Scipy" is selected.

*   `README.md`: Contains the setup, usage, and general documentation for the project.

*   `test_scans/`: The input directory where users should place all the images they want to use for evaluating the OCR quality during an optimization run.

## 3. Core Functionality & Performance

The application evaluates a set of image processing parameters by running them against a collection of images and returns a score based on Tesseract's confidence. An optimizer (either Optuna or Scipy) then uses this score to suggest a new, better set of parameters.

### Optimization Libraries:
*   **Optuna**: Supports TPE, Sobol (QMC), and NSGA-II samplers.
*   **Scipy**: Uses an initial set of points from a Sobol sequence and runs a local optimizer (e.g., L-BFGS-B, Nelder-Mead) from each point.

### CPU Parallelization Management:
To maximize performance on multi-core CPUs, the application parallelizes the evaluation of the image set.
1.  **Dynamic Process Pool:** A `multiprocessing.Pool` is used to distribute the processing of each image to a different CPU core. The size of this pool is dynamically adjusted to `min(number_of_images, cpu_core_count)` to avoid creating unnecessary processes.
2.  **Thread Oversubscription Prevention:** A major performance bottleneck was identified and resolved. To prevent the main multiprocessing pool from competing with the implicit multi-threading of underlying libraries (like OpenCV and Tesseract/OpenMP), each worker process is now forced to be single-threaded by setting environment variables (`OMP_NUM_THREADS=1`) and library-specific commands (`cv2.setNumThreads(1)`). This ensures efficient and predictable scaling on multi-core machines.

## 4. New Features (as of Nov 26, 2025)

*   **Cancellable Optimizations:** A "Cancel" button in the GUI allows for gracefully stopping an ongoing optimization. Both Optuna and Scipy loops will be interrupted.
*   **Automatic CSV Export:** At the end of every run (whether completed or cancelled), all tested parameters and their corresponding scores are automatically saved to a new, timestamped CSV file in the root directory (e.g., `optim_results_YYYYMMDD_HHMMSS.csv`).
*   **GUI Enhancements:**
    *   **Sobol Exponent Input:** For Scipy, the GUI now asks for an exponent (`m`) to calculate the number of Sobol points (`n=2^m`), ensuring the power-of-2 requirement is met for better sequence properties.
    *   **Image Counter:** The UI now displays the number of images found in the `test_scans` directory and includes a refresh button.
    *   **Robust Layout:** The main control bar's layout has been refactored using a `grid` geometry manager to prevent UI elements from disappearing or overlapping, providing a more stable user experience.

## 5. Current Status

The project is functional. It has been significantly enhanced with the addition of cancellation, automatic saving, and major UI/performance fixes that allow it to scale properly on multi-core hardware. The code is ready for the next development steps or for use.