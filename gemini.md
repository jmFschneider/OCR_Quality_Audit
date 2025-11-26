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

## 3. Core Functionality

The application evaluates a set of image processing parameters by running them against a collection of images and returns a score based on Tesseract's confidence. An optimizer (either Optuna or Scipy) then uses this score to suggest a new, better set of parameters.

### Optimization Libraries:
*   **Optuna**: Supports TPE, Sobol (QMC), and NSGA-II samplers.
*   **Scipy**: Uses an initial set of points from a Sobol sequence and runs a local optimizer (e.g., L-BFGS-B, Nelder-Mead) from each point.

### Performance Optimizations:
To handle potentially slow evaluations, two key performance strategies have been implemented:

1.  **Image Pre-loading:** At the start of an optimization, all images from the `test_scans` folder are loaded into RAM. This prevents the disk from being a bottleneck, as worker processes no longer need to read files.
2.  **CPU Parallelization:** The evaluation of the image set is parallelized across all available CPU cores using Python's `multiprocessing` module. Each image in the pre-loaded set is assigned to a different core for processing.

## 4. Current Status & Last Actions

*   The implementation of both Optuna and Scipy optimizers is complete.
*   The performance bottleneck on multi-core CPUs was diagnosed and resolved by implementing the image pre-loading strategy, which prevents I/O contention.
*   All project changes, including the creation of `scipy_optimizer.py` and the `README.md`, have been committed and pushed to the `main` branch of the remote GitHub repository.
*   The project is considered functional and ready for use or further development.
