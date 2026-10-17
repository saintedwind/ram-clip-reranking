# Real pretrained CLIP pilot — 2026-10-17

All eight original COCO8 photographs decoded successfully. Their source annotations were frozen before inference; no images or reference labels were generated. The checkpoint passed SHA256 verification against the canonical OpenAI CLIP URL hash. The canonical endpoint timed out, so identical bytes were obtained through the `jinaai/clip-models` Hugging Face mirror.

## Observations

| Model | Images | Candidate labels | Mean P@5 | Mean recall@5 | Mean F1@5 |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 | 8 | 69 | 0.2500 | 0.8333 | 0.3720 |

Raw cosine vectors, top-five predictions, image and annotation hashes, per-image metrics and measured CPU timings are in `clip-pilot.json`. Image-encoding times after one warm-up were approximately 0.075–0.130 seconds on this machine, excluding model loading and text encoding. These are a single run's observations, not a latency service-level guarantee.

The vocabulary intersection excludes 11 of the 80 canonical COCO category names. In these eight images, `potted plant` is excluded from two annotations. This reduces 15 image/category-positive annotations to 13 within the evaluation vocabulary. The excluded labels remain visible in the raw artifact. No synonym mapping was added after inspecting results.

Examples of mistakes are retained: the dog image ranks `bed` ahead of `dog`; the horse image omits `person` from its top five. High recall with modest precision here means that forcing five predictions often adds unrelated categories to sparsely labelled images. It does not establish calibration or broad image-tagging quality.

## Limits and remaining work

COCO8 comprises COCO train2017 images. Training-source overlap and the small sample prevent generalization and zero-shot claims. Object annotations are not exhaustive scene descriptions. The equal-weight fusion and MMR parameters remain frozen; they have not been tuned on these results.

RAM, fusion and MMR model-comparison results are **not yet available** because the RAM checkpoint transfer remains incomplete. The ranking tests do not substitute for this missing experiment. The published code and protocol are an experimental reproducibility release, not a completed reproduction of thesis claims.
