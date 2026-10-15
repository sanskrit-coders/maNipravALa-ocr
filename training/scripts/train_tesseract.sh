#!/bin/bash
# Fine-tune Tesseract LSTM for manipravala (tam+san) from synthetic + real line data.
# Requires: tesseract training tools (text2image, lstmtraining, etc. in /tmp/tess-root/usr/bin),
#   tam+san traineddata, and training/local/tessdata/*.png+*.box+*.txt (from export_tesseract.py).
# Eval: evaluation/ pages NEVER used here (see evaluation/eval_index.jsonl).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
TESS=/tmp/tess-root/usr/bin/tesseract
TEDATA=/tmp/tess-root/usr/share/tesseract-ocr/5/tessdata
OUT=$REPO/training/tess_model
mkdir -p "$OUT"
# 1) Generate lstmf files (needs .box + .tif; our export gives .png+.box -> convert png to tif)
#    tesseract <img> <outbase> --psm 6 lstm.train  (requires 'lstm.train' config; on Debian it ships
#    with tesseract-ocr package under tessdata/configs/ -- if missing, install tesseract-ocr fully with sudo
#    or on Colab: apt install tesseract-ocr tesseract-ocr-tam tesseract-ocr-san)
# 2) Make lstmf list, then fine-tune from tam+san:
#    combine_tessdata -e $TEDATA/tam.traineddata $OUT/tam.lstm
#    lstmtraining --model_output $OUT/manipravala --continue_from $OUT/tam.lstm \
#      --traineddata $TEDATA/tam.traineddata --train_listfile $OUT/list.txt --eval_listfile $OUT/elist.txt \
#      --max_iterations 5000 --target_error_rate 0.01
# 3) Evaluate ONLY on evaluation/ (see training/scripts/evaluate.py), never train on it.
echo "See comments in $0 for full tesstrain commands. Pilot provides training/local/tessdata (synth 50 lines)."
echo "Run full training on Colab/GPU or sudo machine with tesseract-ocr installed; keep evaluation/ strictly held out."
