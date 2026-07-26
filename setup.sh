#!/bin/bash

ENV_NAME="spec2vec_env"
HARD_REQUIREMENTS_PATH="SPEC2VEC/software_requirements/spec2vec_requirements_hard.txt"

echo "============================================="
echo "1. Creating Conda environment '$ENV_NAME' with Python 3.10.19..."
echo "============================================="
conda create -n $ENV_NAME python=3.10.19 -y

echo "============================================="
echo "2. Installing NumPy (required for build dependencies)..."
echo "============================================="
conda run -n $ENV_NAME pip install numpy==2.2.5

echo "============================================="
echo "3. Installing complex C-extensions (without build isolation)..."
echo "============================================="
conda run -n $ENV_NAME pip install pyfastnoisesimd==0.4.2 --no-build-isolation

echo "============================================="
echo "4. Installing remaining requirements..."
echo "============================================="
conda run -n $ENV_NAME pip install -r $HARD_REQUIREMENTS_PATH
conda run -n $ENV_NAME pip install pyradiomics==3.0.1 --no-build-isolation

echo "============================================="
echo "Installation Complete! 🎉"
echo "To start using Spec2VEC, activate your environment by running:"
echo "conda activate $ENV_NAME"
echo "============================================="
