#!/usr/bin/env bash
# Setup a Python venv for evaluating PaddleOCR (the OCR engine behind
# BytesByJay/Indic-OCR) on the maNipravALa line eval split.
#
# Usage:  training/scripts/indic-ocr/setup_env.sh [VENV_DIR] [REPO_CLONE_DIR]
# Defaults: /tmp/opencode/indicocr-venv and /tmp/opencode/indic-ocr
#
# Kept out of the main pyenv on purpose: PaddleOCR 2.7.x needs the 2023-era
# numpy/opencv stack, which conflicts with the kraken/torch environment.
set -euo pipefail
VENV="${1:-/tmp/opencode/indicocr-venv}"
ICLONE="${2:-/tmp/opencode/indic-ocr}"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel setuptools

# paddleocr 2.7.3 == the API version Indic-OCR's src/ocr_engine.py is written
# against (PaddleOCR(use_angle_cls=..., lang=..., use_gpu=..., show_log=False)
# and .ocr(img, cls=True)). paddlepaddle 2.6.2 is the last 2.x CPU build.
"$VENV/bin/pip" install -q paddlepaddle==2.6.2 paddleocr==2.7.3

# Pin the rest of the stack to versions compatible with numpy 1.x
# (2025+ releases of these pin numpy>=2, which breaks the paddle wheels).
"$VENV/bin/pip" uninstall -q -y opencv-python opencv-python-headless opencv-contrib-python || true
"$VENV/bin/pip" install -q \
    'numpy==1.26.4' 'scipy==1.11.4' 'contourpy==1.2.1' 'matplotlib==3.8.4' \
    'scikit-image==0.22.0' 'pandas==2.2.3' 'tifffile==2024.8.30' 'imageio==2.34.2' \
    'opencv-contrib-python==4.6.0.66' 'jiwer'

if [ ! -d "$ICLONE" ]; then
    git clone --depth 1 https://github.com/BytesByJay/Indic-OCR "$ICLONE"
fi

echo "venv:       $VENV"
echo "indic-repo: $ICLONE"
echo "ok: $("$VENV/bin/python" -c 'import paddleocr, paddle, numpy, cv2; print("paddleocr", paddleocr.__version__, "| paddle", paddle.__version__, "| numpy", numpy.__version__, "| cv2", cv2.__version__)')"
