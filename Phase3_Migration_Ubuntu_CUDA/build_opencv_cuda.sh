#!/bin/bash

################################################################################
# Script de compilation automatique OpenCV 4.8.0 avec CUDA
# Pour Ubuntu 20.04/22.04 + NVIDIA RTX 1080
#
# Usage: ./build_opencv_cuda.sh
# Durée: ~45-60 minutes (12 cores)
################################################################################

set -e  # Arrêt en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "=========================================================================="
echo "  Compilation OpenCV 4.8.0 avec CUDA pour OCR Quality Audit - Phase 3"
echo "=========================================================================="
echo -e "${NC}"

# Variables
OPENCV_VERSION="4.8.0"
CUDA_ARCH_BIN="6.1"  # GTX 1080 = Pascal 6.1
NUM_CORES=$(nproc)
BUILD_DIR="$HOME/opencv_build"

################################################################################
# Étape 1 : Vérifications préalables
################################################################################

echo -e "${YELLOW}[1/7] Vérification de la configuration...${NC}"

# Vérifier Ubuntu
UBUNTU_VERSION=$(lsb_release -rs)
if [[ ! "$UBUNTU_VERSION" =~ ^(20.04|22.04)$ ]]; then
    echo -e "${RED}ERREUR: Ce script est conçu pour Ubuntu 20.04 ou 22.04${NC}"
    echo "Votre version: $UBUNTU_VERSION"
    read -p "Continuer quand même ? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Ubuntu $UBUNTU_VERSION détecté${NC}"
fi

# Vérifier NVIDIA GPU
if ! lspci | grep -qi nvidia; then
    echo -e "${RED}ERREUR: Aucune carte NVIDIA détectée${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Carte NVIDIA détectée${NC}"

# Vérifier driver NVIDIA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠ nvidia-smi non trouvé. Installation des drivers NVIDIA...${NC}"
    sudo ubuntu-drivers autoinstall
    echo -e "${YELLOW}Redémarrage nécessaire après installation des drivers.${NC}"
    read -p "Redémarrer maintenant ? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        echo -e "${RED}Veuillez redémarrer manuellement puis relancer ce script.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Driver NVIDIA OK ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader))${NC}"

################################################################################
# Étape 2 : Installation CUDA Toolkit
################################################################################

echo -e "\n${YELLOW}[2/7] Installation CUDA Toolkit 11.8...${NC}"

# Exporter PATH CUDA immédiatement dans ce script (ne pas utiliser .bashrc)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    echo -e "${GREEN}✓ CUDA déjà installé (version $CUDA_VERSION)${NC}"
else
    echo "Installation du dépôt NVIDIA..."
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt update

    echo "Installation CUDA Toolkit 11.8..."
    sudo apt install -y cuda-toolkit-11-8

    # Réexporter PATH après installation
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

    # Ajouter à .bashrc pour les sessions futures (optionnel)
    if ! grep -q "/usr/local/cuda-11.8/bin" ~/.bashrc; then
        echo '' >> ~/.bashrc
        echo '# CUDA Toolkit 11.8' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi

    echo -e "${GREEN}✓ CUDA Toolkit installé${NC}"
fi

# Vérification CRITIQUE que nvcc est accessible
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERREUR: nvcc toujours introuvable après installation CUDA${NC}"
    echo -e "${YELLOW}PATH actuel: $PATH${NC}"
    echo -e "${YELLOW}LD_LIBRARY_PATH actuel: $LD_LIBRARY_PATH${NC}"
    echo ""
    echo "Solutions possibles:"
    echo "  1. Vérifier que CUDA est bien installé: ls -la /usr/local/ | grep cuda"
    echo "  2. Installer manuellement: sudo apt install cuda-toolkit-11-8"
    echo "  3. Relancer le script après installation"
    exit 1
fi

echo -e "${GREEN}✓ nvcc accessible: $(which nvcc)${NC}"

################################################################################
# Étape 3 : Installation des dépendances
################################################################################

echo -e "\n${YELLOW}[3/7] Installation des dépendances système...${NC}"

sudo apt update
sudo apt install -y \
    build-essential cmake git pkg-config unzip \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev gfortran \
    python3-dev python3-pip python3-venv \
    python3-numpy \
    libtbb2 libtbb-dev

echo -e "${GREEN}✓ Dépendances système installées${NC}"

# Vérifier NumPy (CRITIQUE pour les bindings Python)
echo "Vérification de NumPy..."
if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${YELLOW}Installation de NumPy via pip3...${NC}"
    pip3 install numpy
fi
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo -e "${GREEN}✓ NumPy $NUMPY_VERSION disponible${NC}"

################################################################################
# Étape 4 : Téléchargement sources OpenCV
################################################################################

echo -e "\n${YELLOW}[4/7] Téléchargement OpenCV $OPENCV_VERSION...${NC}"

# Créer répertoire de build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Télécharger OpenCV si nécessaire
if [ ! -d "opencv" ]; then
    echo "Téléchargement opencv..."
    wget -q --show-progress -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
    unzip -q opencv.zip
    mv opencv-$OPENCV_VERSION opencv
    rm opencv.zip
else
    echo -e "${GREEN}✓ Sources OpenCV déjà présentes${NC}"
fi

# Télécharger opencv_contrib si nécessaire
if [ ! -d "opencv_contrib" ]; then
    echo "Téléchargement opencv_contrib..."
    wget -q --show-progress -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
    unzip -q opencv_contrib.zip
    mv opencv_contrib-$OPENCV_VERSION opencv_contrib
    rm opencv_contrib.zip
else
    echo -e "${GREEN}✓ Sources opencv_contrib déjà présentes${NC}"
fi

echo -e "${GREEN}✓ Sources téléchargées${NC}"

################################################################################
# Étape 5 : Configuration CMake
################################################################################

echo -e "\n${YELLOW}[5/7] Configuration CMake avec CUDA...${NC}"

cd "$BUILD_DIR/opencv"
rm -rf build
mkdir build && cd build

# Détecter Python
PYTHON3_EXECUTABLE=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Fix distutils pour Python 3.10+ (deprecated depuis Python 3.10, supprimé en 3.12)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo -e "${YELLOW}Python $PYTHON_VERSION détecté - Utilisation de sysconfig${NC}"
    PYTHON3_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    PYTHON3_PACKAGES_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
else
    echo -e "${YELLOW}Python $PYTHON_VERSION détecté - Utilisation de distutils${NC}"
    PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
    PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
fi

# 🚨 AJOUT CRITIQUE pour les bindings Python avec NumPy
NUMPY_INC=$(python3 -c "import numpy; print(numpy.get_include())")

echo "Configuration Python:"
echo "  Version: $PYTHON_VERSION"
echo "  Executable: $PYTHON3_EXECUTABLE"
echo "  Include: $PYTHON3_INCLUDE_DIR"
echo "  Packages: $PYTHON3_PACKAGES_PATH"

# Vérifier que nvcc est toujours accessible avant CMake
echo "Vérification finale de l'environnement CUDA avant CMake..."
echo "  nvcc: $(which nvcc)"
echo "  Version: $(nvcc --version | grep release)"

# Configuration CMake avec logging détaillé
echo -e "${YELLOW}Lancement de CMake...${NC}"
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH="$BUILD_DIR/opencv_contrib/modules" \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=OFF \
      -D OPENCV_DNN_CUDA=OFF \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_ARCH_BIN=$CUDA_ARCH_BIN \
      -D CUDA_ARCH_PTX=$CUDA_ARCH_BIN \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_OPENGL=ON \
      -D WITH_OPENCL=ON \
      -D WITH_TBB=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$PYTHON3_EXECUTABLE \
      -D PYTHON3_INCLUDE_DIR=$PYTHON3_INCLUDE_DIR \
      -D PYTHON3_PACKAGES_PATH=$PYTHON3_PACKAGES_PATH \
	  -D PYTHON3_NUMPY_INCLUDE_DIRS=$NUMPY_INC \
      .. | tee cmake_output.log

# Vérifications détaillées de la configuration
echo -e "\n${YELLOW}Vérification de la configuration CMake...${NC}"

# Vérifier CUDA
if grep -q "NVIDIA CUDA.*YES" CMakeCache.txt; then
    CUDA_VER=$(grep "CUDA_VERSION" CMakeCache.txt | grep -v "INTERNAL" | head -1 | cut -d= -f2)
    echo -e "${GREEN}✓ CUDA activé (version $CUDA_VER)${NC}"
else
    echo -e "${RED}ERREUR: CUDA non activé dans CMake${NC}"
    echo -e "${YELLOW}Détails de la configuration CUDA:${NC}"
    grep -i cuda cmake_output.log | head -20
    echo ""
    echo "Vérifiez que:"
    echo "  1. nvcc est accessible: which nvcc"
    echo "  2. CUDA Toolkit est installé: ls /usr/local/cuda-11.8/"
    echo "  3. PATH et LD_LIBRARY_PATH sont corrects"
    exit 1
fi

# Vérifier GPU Architecture
if grep -q "CUDA_ARCH_BIN:STRING=$CUDA_ARCH_BIN" CMakeCache.txt; then
    echo -e "${GREEN}✓ Architecture GPU: $CUDA_ARCH_BIN (Pascal - GTX 1080)${NC}"
else
    echo -e "${YELLOW}⚠ Architecture GPU non définie correctement${NC}"
fi

# Vérifier Python bindings
if grep -q "BUILD_opencv_python3:BOOL=ON" CMakeCache.txt; then
    echo -e "${GREEN}✓ Bindings Python3 activés${NC}"
else
    echo -e "${RED}ERREUR: Bindings Python3 non activés${NC}"
    exit 1
fi

# Vérifier NumPy
if grep -q "PYTHON3_NUMPY_INCLUDE_DIRS" CMakeCache.txt; then
    NUMPY_PATH=$(grep "PYTHON3_NUMPY_INCLUDE_DIRS" CMakeCache.txt | cut -d= -f2)
    echo -e "${GREEN}✓ NumPy détecté: $NUMPY_PATH${NC}"
else
    echo -e "${YELLOW}⚠ NumPy non détecté - Les bindings Python pourraient ne pas se construire${NC}"
fi

echo -e "${GREEN}✓ Configuration CMake validée${NC}"

################################################################################
# Étape 6 : Compilation
################################################################################

echo -e "\n${YELLOW}[6/7] Compilation OpenCV (utilisant $NUM_CORES cœurs)...${NC}"
echo -e "${YELLOW}Cela va prendre 20-30 minutes...${NC}"

start_time=$(date +%s)

# Compilation avec tous les cœurs
if ! make -j$NUM_CORES; then
    echo -e "${RED}Erreur lors de la compilation avec -j$NUM_CORES${NC}"
    echo -e "${YELLOW}Nouvelle tentative avec moins de threads...${NC}"
    make -j4
fi

end_time=$(date +%s)
duration=$((end_time - start_time))
echo -e "${GREEN}✓ Compilation terminée en $((duration/60)) min $((duration%60)) s${NC}"

################################################################################
# Étape 7 : Installation
################################################################################

echo -e "\n${YELLOW}[7/7] Installation d'OpenCV...${NC}"

sudo make install
sudo ldconfig

echo -e "${GREEN}✓ OpenCV installé${NC}"

################################################################################
# Vérification finale
################################################################################

echo -e "\n${YELLOW}Vérification de l'installation...${NC}"

# Test Python
if python3 -c "import cv2" 2>/dev/null; then
    CV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)")
    echo -e "${GREEN}✓ OpenCV $CV_VERSION importable en Python${NC}"
else
    echo -e "${RED}ERREUR: Impossible d'importer cv2 en Python${NC}"
    exit 1
fi

# Test CUDA
CUDA_COUNT=$(python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null || echo "0")
if [ "$CUDA_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ CUDA activé ($CUDA_COUNT GPU détecté)${NC}"
else
    echo -e "${RED}ERREUR: CUDA non détecté dans OpenCV${NC}"
    echo "Vérifiez que les bibliothèques CUDA sont bien liées:"
    ldd $(python3 -c "import cv2; print(cv2.__file__)") | grep cuda
    exit 1
fi

################################################################################
# Résumé
################################################################################

echo -e "\n${GREEN}"
echo "=========================================================================="
echo "  ✓ Compilation OpenCV avec CUDA terminée avec succès !"
echo "=========================================================================="
echo -e "${NC}"
echo "Version OpenCV: $CV_VERSION"
echo "CUDA: Activé ($CUDA_COUNT GPU)"
echo "Temps total: $((duration/60)) minutes"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo "  1. Testez avec: python3 test_cuda.py"
echo "  2. Lisez PHASE3_OPENCV_CUDA_UBUNTU.md pour adapter votre code"
echo "  3. Lancez vos optimisations sur images 300 DPI !"
echo ""
echo -e "${GREEN}Répertoire de build conservé dans: $BUILD_DIR${NC}"
echo -e "${YELLOW}Vous pouvez le supprimer pour libérer ~5 GB:${NC}"
echo "  rm -rf $BUILD_DIR"
echo ""
