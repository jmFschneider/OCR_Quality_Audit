#!/bin/bash
# Script de compilation OpenCV avec NumPy 1.26.4
set -e

# Configuration environnement CUDA
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Variables Python
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")
PYTHON3_EXECUTABLE=$(which python3)
PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

echo "=========================================================================="
echo "  Compilation OpenCV 4.8.0 avec CUDA 11.8 et NumPy 1.21.5"
echo "=========================================================================="
echo ""
echo "Configuration Python:"
echo "  Executable: $PYTHON3_EXECUTABLE"
echo "  NumPy include: $NUMPY_INC"
echo "  Python include: $PYTHON3_INCLUDE_DIR"
echo "  Python packages: $PYTHON3_PACKAGES_PATH"
echo ""

# Vérifier NumPy version
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo "NumPy version: $NUMPY_VERSION"
if [ "$NUMPY_VERSION" != "1.21.5" ]; then
    echo "ERREUR: NumPy doit être 1.21.5, trouvé: $NUMPY_VERSION"
    exit 1
fi
echo ""

# Aller dans le répertoire de build
cd ~/opencv_build/opencv/build

echo "=== Étape 1/3: Configuration CMake ==="
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=6.1 \
      -D CUDA_ARCH_PTX="" \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=ON \
      -D WITH_CUDNN=OFF \
      -D OPENCV_DNN_CUDA=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_TBB=ON \
      -D WITH_OPENCL=OFF \
      -D WITH_OPENGL=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$PYTHON3_EXECUTABLE \
      -D PYTHON3_INCLUDE_DIR=$PYTHON3_INCLUDE_DIR \
      -D PYTHON3_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH \
      -D PYTHON3_NUMPY_INCLUDE_DIRS=$NUMPY_INC \
      .. | tee cmake_output.log

echo ""
echo "=== Vérification configuration CMake ==="

# Vérifier CUDA
if grep -q "WITH_CUDA:BOOL=ON" CMakeCache.txt; then
    CUDA_VER=$(grep "CUDA_VERSION" CMakeCache.txt | grep -v "INTERNAL" | head -1 | cut -d= -f2)
    echo "✓ CUDA activé (version $CUDA_VER)"
else
    echo "ERREUR: CUDA non activé"
    exit 1
fi

# Vérifier Python bindings
if grep -q "BUILD_opencv_python3:BOOL=ON" CMakeCache.txt; then
    echo "✓ Bindings Python3 activés"
else
    echo "ERREUR: Bindings Python3 non activés"
    exit 1
fi

# Vérifier NumPy
if grep -q "PYTHON3_NUMPY_INCLUDE_DIRS" CMakeCache.txt; then
    NUMPY_PATH=$(grep "PYTHON3_NUMPY_INCLUDE_DIRS" CMakeCache.txt | cut -d= -f2)
    echo "✓ NumPy include path: $NUMPY_PATH"
else
    echo "ERREUR: NumPy include path non trouvé"
    exit 1
fi

echo ""
echo "=== Étape 2/3: Compilation (make -j12) ==="
echo "Durée estimée: 45-60 minutes"
echo ""
make -j12 2>&1 | tee make_output.log

echo ""
echo "=== Étape 3/3: Installation ==="
echo "Cette étape nécessite sudo..."
echo "Exécutez manuellement: cd ~/opencv_build/opencv/build && sudo make install"
echo ""
echo "Puis créez le symlink dans le venv:"
echo "ln -sf /usr/local/lib/python3.10/dist-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so \\"
echo "       ~/.venv/lib/python3.10/site-packages/cv2.so"
echo ""
echo "=========================================================================="
echo "  Compilation terminée avec succès !"
echo "=========================================================================="
