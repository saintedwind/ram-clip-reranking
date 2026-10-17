# Real pretrained RAM/CLIP pilot — 2026-10-17

All eight original COCO8 photographs decoded successfully. Their source annotations were frozen before inference; no images or reference labels were generated. The checkpoint passed SHA256 verification against the canonical OpenAI CLIP URL hash. The canonical endpoint timed out, so identical bytes were obtained through the `jinaai/clip-models` Hugging Face mirror.

## Observations

| Model | Images | Candidate labels | Mean P@5 | Mean recall@5 | Mean F1@5 |
|---|---:|---:|---:|---:|---:|
| RAM Swin-L | 8 | 69 | 0.3000 | 0.9583 | 0.4435 |
| CLIP ViT-B/32 | 8 | 69 | 0.2500 | 0.8333 | 0.3720 |
| Equal-weight fusion | 8 | 69 | 0.3000 | 0.9583 | 0.4435 |
| Fusion + greedy MMR | 8 | 69 | 0.3000 | 0.9583 | 0.4435 |

The full run is recorded in `full-pilot.json`. Both checkpoints passed their canonical SHA256 checks. RAM's captured logits agreed with the official thresholded predictions on every image. **Neither fusion nor MMR improved these metrics over RAM.** MMR changed the ordering of the vase image's candidates but did not recover an additional annotated class. No parameter search was performed after observing this outcome.

Raw cosine vectors, top-five predictions, image and annotation hashes, per-image metrics and measured CPU timings are in `clip-pilot.json`. Image-encoding times after one warm-up were approximately 0.075–0.130 seconds on this machine, excluding model loading and text encoding. These are a single run's observations, not a latency service-level guarantee.

The vocabulary intersection excludes 11 of the 80 canonical COCO category names. In these eight images, `potted plant` is excluded from two annotations. This reduces 15 image/category-positive annotations to 13 within the evaluation vocabulary. The excluded labels remain visible in the raw artifact. No synonym mapping was added after inspecting results.

Examples of mistakes are retained: the dog image ranks `bed` ahead of `dog`; the horse image omits `person` from its top five. High recall with modest precision here means that forcing five predictions often adds unrelated categories to sparsely labelled images. It does not establish calibration or broad image-tagging quality.

## Limits and remaining work

COCO8 comprises COCO train2017 images. Training-source overlap and the small sample prevent generalization and zero-shot claims. Object annotations are not exhaustive scene descriptions. The equal-weight fusion and MMR parameters remain frozen; they have not been tuned on these results.

The RAM checkpoint transfer initially truncated and was resumed with validated byte ranges; inference began only after the complete file matched its canonical hash. RAM took roughly 3.65–4.68 seconds per image after one warm-up, excluding loading and preprocessing. The two models ran sequentially to reduce peak memory.

The published code and protocol are a completed small integration pilot, not a reproduction of the thesis's broad benchmark claims. The next scientific experiment should use a larger independently annotated held-out collection and a separate tuning split. The present observations provide no basis for advertising a general accuracy improvement.
