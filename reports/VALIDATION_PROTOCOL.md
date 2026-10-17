# Larger validation protocol — frozen before model inference

Use existing COCO val2017 instance annotations, not model-generated reference labels. Select the 512 images with the smallest hexadecimal SHA256 of `ram-clip-validation-v1:<numeric image_id>` from all 5,000 validation images. The first 128 constitute a development subset; the remaining 384 constitute the evaluation subset. Retain failed downloads and inference errors explicitly; do not replace difficult images.

Primary comparison: the same fixed 69-name RAM/COCO vocabulary intersection as the eight-image pilot. Preserve every original reference category and report excluded-label coverage. The eleven excluded COCO categories mean this is not full 80-category performance. Do not introduce post-result synonym mappings.

RAM, CLIP, alpha=0.5 fusion and fusion+MMR (relevance weight=0.8) all use the same images and vocabulary. No tuning is performed in this run. The development partition is reserved for future controlled experiments. Report primary evaluation metrics on the 384 evaluation images separately from development.

Primary endpoint: mean per-image F1@5. Also report mean P@5 and recall@5, per-image paired differences against RAM, and a paired image bootstrap 95% percentile interval for the mean difference (10,000 resamples, seed 20261017). Bootstrap samples resample actual observed image results; they are uncertainty estimates, not new model observations or synthetic reference data.

Keep the original annotations, selected IDs, image hashes, complete raw scores and failure logs. Verify cached results against the complete manifest before resuming. No data are excluded for poor model predictions. Report reference-label coverage and the number of images with no remaining in-vocabulary positives.

These COCO validation images are separate from the earlier COCO8 training images. They are not guaranteed absent from pretrained models' data. This experiment therefore cannot establish zero-shot performance, novelty, state of the art or the thesis's full benchmark claims. It tests whether the earlier reranking outcome persists on a larger fixed sample.
