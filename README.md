# OCR Quality Audit Optimizer

This project provides a Graphical User Interface (GUI) tool to optimize image preprocessing parameters for Optical Character Recognition (OCR) using Tesseract. It helps in finding the best combination of parameters to maximize OCR accuracy, sharpness, and contrast of processed images.

## Features

-   **Interactive GUI:** User-friendly interface built with Tkinter for easy parameter selection and optimization control.
-   **Multiple Optimization Libraries:**
    -   **Optuna:** Supports various samplers like TPE (Tree-structured Parzen Estimator), Sobol (Quasi-Monte Carlo), and NSGA-II (Multi-objective Genetic Algorithm).
    -   **Scipy:** Integrates several algorithms such as Nelder-Mead, L-BFGS-B, TNC, COBYLA, and SLSQP for local optimization, starting from Sobol-sequenced initial points.
-   **CPU Parallelization:** Accelerates image evaluation by processing multiple images concurrently across available CPU cores.
-   **Customizable Parameters:** Optimize parameters for line removal, normalization, denoising, and adaptive binarization.
-   **Real-time Feedback:** Displays optimization progress, best scores, and optimal parameter sets.
-   **Metric Logging:** Saves detailed results of each trial to a CSV file.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd OCR_Quality_Audit
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install opencv-python-headless numpy pytesseract optuna scipy
    ```
    *Note: `tkinter` is usually included with Python.*

4.  **Install Tesseract OCR:**
    Download and install Tesseract OCR from https://tesseract-ocr.github.io/tessdoc/Installation.html.

5.  **Configure Tesseract Path:**
    Update the `pytesseract.pytesseract.tesseract_cmd` variable in `gui_optimizer_v3_ultim.py` to point to your Tesseract executable (e.g., `r'C:\Program Files\Tesseract-OCR\tesseract.exe'` for Windows).

## Usage

1.  **Place your input images** in the `test_scans` directory.
2.  **Run the GUI application:**
    ```bash
    python gui_optimizer_v3_ultim.py
    ```
3.  **Select your desired optimization library** (Optuna or Scipy), algorithm, and configure the optimization parameters (e.g., number of trials, Sobol points, iterations).
4.  **Click "LANCER"** to start the optimization.

## Project Structure

-   `gui_optimizer_v3_ultim.py`: The main GUI application and optimization orchestrator.
-   `scipy_optimizer.py`: Contains the core logic for Scipy-based optimization.
-   `test_scans/`: Directory for input image files.
-   `README.md`: This file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
