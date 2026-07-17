#!/bin/bash
# launch.sh — Lance LabelMe + pipeline YOLO11

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_EXE="$ROOT/miniconda/envs/labelme-env/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "[ERREUR] Installation non trouvee."
    echo "Veuillez d'abord executer : bash install.sh"
    exit 1
fi

"$PYTHON_EXE" -m labelme
