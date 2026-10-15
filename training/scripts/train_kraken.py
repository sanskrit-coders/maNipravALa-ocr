"""Fine-tune Kraken (ketos) for manipravala line OCR.

Train data: training/local/kraken/*.xml + *.png (from export_kraken.py, synth pilot; real after manual line GT).
Eval data: evaluation/ ONLY (never train on it) — segment eval pages to lines for CER, or page-level via evaluate.py.

Install (needs torch; use GPU/Colab — CPU install timed out here after 300s):
  pip install kraken torch torchvision
  # existing Grantha starter model: doc_curation/doc_curation/pdf/data/grantha_kraken.mlmodel
  # (Kraken v4 .mlmodel; for v5 use .ckpt — convert or start from kraken base model)

Train (example):
  ketos train -f page -t training/local/kraken/*.xml \
    -e evaluation/kraken_eval/*.xml \
    -o training/local/kraken_model --resize new --codec PASETO --max-epochs 50 --early-stopping 10
  # --base-model <grantha.mlmodel> to fine-tune instead of from scratch.

Inference baseline (zero-shot with grantha model) for comparison:
  kraken -i evaluation/images/p0043.jpg evaluation/pred_kraken/p0043.txt segment -bl ocr -m <grantha.mlmodel>
Then: python3 training/scripts/evaluate.py --pred evaluation/pred_kraken --out evaluation/results_kraken.json

Note: kraken install (torch) exceeded 300s timeout on this CPU box — run on Colab/GPU.
Keep training/ and evaluation/ strictly separate (no eval images in ketos train list).
"""
print(__doc__)
