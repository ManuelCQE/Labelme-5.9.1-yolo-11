#!/bin/bash
# install.sh — Installation LabelMe 5.9.1-Yolo-DWPose
# Linux standalone — aucun prérequis sauf curl et bash

set -e

echo "============================================================"
echo " Installation LabelMe 5.9.1-Yolo-DWPose"
echo "============================================================"
echo

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_DIR="$ROOT/miniconda"
ENV_DIR="$CONDA_DIR/envs/labelme-env"
PYTHON_EXE="$ENV_DIR/bin/python"
PIP_EXE="$ENV_DIR/bin/pip"
CONDA_EXE="$CONDA_DIR/bin/conda"

# ── Détection GPU et choix PyTorch ───────────────────────────────────────
detect_torch_index() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "     Pas de GPU NVIDIA detecte — installation PyTorch CPU."
        echo "     Note : l'inference sera tres lente sans GPU."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_LABEL="CPU"
        return
    fi

    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "     Driver NVIDIA detecte : $DRIVER_VER"
    DRIVER_MAJOR=$(echo "$DRIVER_VER" | cut -d'.' -f1)

    if [ "$DRIVER_MAJOR" -lt 452 ] 2>/dev/null; then
        echo "     Driver trop ancien (< 452) — installation PyTorch CPU."
        echo "     Mettez a jour vos drivers NVIDIA pour profiter du GPU."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_LABEL="CPU (driver trop ancien)"
    elif [ "$DRIVER_MAJOR" -lt 526 ] 2>/dev/null; then
        echo "     Driver compatible CUDA 11.8 — installation PyTorch cu118..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_LABEL="CUDA 11.8"
    elif [ "$DRIVER_MAJOR" -lt 536 ] 2>/dev/null; then
        echo "     Driver compatible CUDA 12.1 — installation PyTorch cu121..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_LABEL="CUDA 12.1"
    else
        echo "     Driver compatible CUDA 12.6 — installation PyTorch cu126..."
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        TORCH_LABEL="CUDA 12.6"
    fi
}

# ── Fonction install dépendances ─────────────────────────────────────────
install_deps() {
    echo
    echo "[4/5] Detection GPU et installation PyTorch..."
    detect_torch_index
    echo "     PyTorch : $TORCH_LABEL"

    "$PIP_EXE" install torch==2.9.0 torchvision==0.24.0 \
        --index-url "$TORCH_INDEX"

    "$PIP_EXE" install -r "$ROOT/requirements.txt"
    "$PIP_EXE" install -e "$ROOT"
    echo "[OK] Dependances installees."
}

# Déjà installé ?
if [ -f "$PYTHON_EXE" ]; then
    echo "[OK] Environnement deja installe. Mise a jour des dependances..."
    install_deps
    exit 0
fi

# ── Étape 1 : Télécharger Miniconda ──────────────────────────────────────
echo "[1/5] Telechargement de Miniconda..."
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="$ROOT/miniconda_installer.sh"

if [ ! -f "$MINICONDA_INSTALLER" ]; then
    curl -L --progress-bar "$MINICONDA_URL" -o "$MINICONDA_INSTALLER"
fi
chmod +x "$MINICONDA_INSTALLER"
echo "[OK] Miniconda telecharge."

# ── Étape 2 : Installer Miniconda en mode portable ───────────────────────
echo
echo "[2/5] Installation de Miniconda dans $CONDA_DIR..."
bash "$MINICONDA_INSTALLER" -b -p "$CONDA_DIR"
rm "$MINICONDA_INSTALLER"
echo "[OK] Miniconda installe."

# ── Étape 3 : Créer l'environnement Python 3.10 + pip ────────────────────
echo
echo "[3/5] Creation de l'environnement labelme-env (Python 3.10)..."
"$CONDA_EXE" create -y -p "$ENV_DIR" python=3.10 pip
echo "[OK] Environnement cree."

# ── Étape 4 : Installer les dépendances ──────────────────────────────────
install_deps

# ── Étape 5 : Télécharger les modèles ────────────────────────────────────
echo
echo "[5/5] Telechargement des modeles YOLO11 + DWPose..."
"$PYTHON_EXE" "$ROOT/download_models.py"

echo
echo "============================================================"
echo " Installation terminee ! [$TORCH_LABEL]"
echo " Lancez LabelMe avec : bash launch.sh"
echo "============================================================"
echo
