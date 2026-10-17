# Frozen pilot protocol

This is a reproducibility pilot, not a reproduction of the thesis benchmark.

## Data

Use the eight original COCO train2017 photographs and existing bounding-box class annotations distributed in Ultralytics COCO8. Convert boxes to a set of image-level class IDs. Do not add labels after looking at predictions. COCO8's train/val directory names do not make these images held out from RAM pretraining.

Archive: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip

Archive SHA256: `54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9`.

Use the exact intersection of the canonical 80 COCO names and RAM's English vocabulary, ignoring case and surrounding whitespace only. Record excluded categories and annotation coverage. This restriction is shared by all baselines. COCO labels cover objects, not every valid scene attribute; unmatched tags are not necessarily false in an open-vocabulary sense.

## Models and comparisons

- RAM Swin-L, 384 pixels, official ImageNet preprocessing; extract actual sigmoid logits before thresholding.
- OpenAI CLIP ViT-B/32, its own official preprocessing, fixed prompt `a photo of a {tag}.`.
- RAM confidence ranking; CLIP cosine ranking; equal-weight per-image z-score fusion; fusion followed by greedy MMR with relevance weight 0.8.
- Rank at most five labels; precision denominator remains five. Compute image-level P@5, recall@5 and F1@5, then arithmetic means. These are not class-wise mAP.
- No training, parameter search, manually revised truth or best-image selection on this pilot.

Retain model/data/source hashes, raw per-label scores, every image's predictions, timings and failures. Loading time and warm-up are separate from inference time. A completed model run is required before reporting model quality. Unit-test arrays are mathematical fixtures, never experimental observations.

## Interpretation

Eight training-source photographs can reveal integration errors and compare behavior on these examples. They cannot establish generalization, statistical superiority or current state of the art. A subsequent experiment needs a larger independently annotated held-out set, a fixed tuning split, class-wise metrics and uncertainty estimates.
