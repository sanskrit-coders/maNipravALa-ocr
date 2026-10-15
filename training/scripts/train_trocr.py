"""Fine-tune TrOCR / vision-encoder-decoder for manipravala lines (needs GPU).

Train pairs: training/local/hf_trocr/metadata.jsonl + images/ (from export_hf.py, synth pilot).
Eval pairs: evaluation/hf_eval/ (separate, build from evaluation/gt lines — TODO manual line split;
for now page-level eval via training/scripts/evaluate.py on full-page TrOCR preds).

Install (GPU/Colab; too heavy for this CPU box):
  pip install torch transformers datasets jiwer pillow

Model: microsoft/trocr-base-printed (Latin-pretrained; expect poor zero-shot on Tamil/Deva —
baseline comparison will show this) or facebook/nougat-base (better for Indic? test both).
Tokenizer: TrOCRProcessor (handles unicode Tamil+Deva as-is).

Pseudo:
  from datasets import load_dataset
  ds = load_dataset('imagefolder', data_dir='training/local/hf_trocr')  # with metadata.jsonl
  processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
  model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
  # resize token embeddings if needed (Tamil chars already in vocab as unicode? verify)
  # Seq2SeqTrainer with CER metric (jiwer), predict on evaluation/hf_eval, compare.
Zero-shot inference for baseline table:
  pipeline image-to-text on evaluation/images/p0042-44.jpg (CPU possible for 3 pages, slow;
  run on Colab GPU). Save to evaluation/pred_trocr/{page}.txt, then evaluate.py.
Keep evaluation/ strictly held out from Trainer train split.
"""
print(__doc__)
