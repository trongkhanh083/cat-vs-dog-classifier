#!/usr/bin/env bash
# scripts/run_training.sh
set -e
export PYTHONPATH="$(pwd)"   # so `python -m src.train` can import your src/ package

# Load defaults from configs/default.yaml
RAW_DIR=$(grep '^raw_dir:' configs/default.yaml    | awk '{print $2}')
DATA_DIR=$(grep '^data_dir:' configs/default.yaml   | awk '{print $2}')
CKPT_PATH=$(grep '^ckpt_path:' configs/default.yaml| awk '{print $2}')
EPOCHS=$(grep '^epochs:' configs/default.yaml      | awk '{print $2}')
BATCH_SIZE=$(grep '^batch_size:' configs/default.yaml | awk '{print $2}')
TEST_DIR=$(grep '^test_dir:' configs/default.yaml  | awk '{print $2}')
NUM_IMAGES=$(grep '^num_images:' configs/default.yaml | awk '{print $2}')

echo "➜ Training with:"
echo "   raw_dir=$RAW_DIR"
echo "   data_dir=$DATA_DIR"
echo "   ckpt_path=$CKPT_PATH"
echo "   epochs=$EPOCHS, batch_size=$BATCH_SIZE"
echo

# 1) Train (also does prepare_data → split & fit)
python -m src.train \
  --raw_dir    "$RAW_DIR" \
  --data_dir   "$DATA_DIR" \
  --ckpt_path  "$CKPT_PATH" \
  --epochs     "$EPOCHS" \
  --batch_size "$BATCH_SIZE"

echo && echo "➜ Evaluation:"
# 2) Evaluate on test set (reads the same ckpt_path)
python -m src.evaluate \
  --data_dir  "$DATA_DIR" \
  --ckpt_path "$CKPT_PATH"

echo && echo "➜ Prediction:"
# 3) Predict on a handful of samples
python -m src.predict \
  --ckpt_path  "$CKPT_PATH" \
  --test_dir   "$TEST_DIR" \
  --num_images "$NUM_IMAGES"

echo
echo "All done! You finished the Cat vs Dog Classifier Project. Enjoy your result!"