#!/usr/bin/env bash
set -euo pipefail

python -m scripts.train \
  --data-path dataset/FinTruthQA.csv \
  --task QUES_RELEVANCE \
  --model-name bert-base-chinese \
  --output-dir outputs/ques_relevance \
  --epochs 30 \
  --batch-size 32 \
  --max-length 256 \
  --plot
