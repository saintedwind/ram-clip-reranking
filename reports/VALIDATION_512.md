# COCO val2017: 512-image validation

The protocol and deterministic image selection were recorded before model inference. 128 development images and 384 evaluation images were processed with the same frozen checkpoints and parameters. The primary results below use only the 384 evaluation images. No tuning was performed.

## Primary endpoint and supporting metrics

| Method | Mean P@5 | Mean recall@5 | Mean F1@5 |
|---|---:|---:|---:|
| RAM | 0.4297 | 0.8548 | 0.5294 |
| CLIP | 0.3151 | 0.6729 | 0.3974 |
| Fusion | 0.4167 | 0.8393 | 0.5161 |
| Fusion + MMR | 0.4156 | 0.8387 | 0.5152 |

## Paired differences from RAM

| Method | F1 difference (percentage points) | 95% bootstrap interval | Improved / worse / tied images |
|---|---:|---|---|
| CLIP | -13.204 | [-15.086, -11.330] | 13 / 169 / 202 |
| Fusion | -1.328 | [-2.314, -0.389] | 22 / 43 / 319 |
| Fusion + MMR | -1.424 | [-2.415, -0.475] | 22 / 45 / 317 |

Intervals use 10,000 paired image bootstrap resamples, seed 20261017. They are per-comparison percentile intervals, without a multiple-comparison adjustment. They do not measure training variation, annotation uncertainty or out-of-domain performance.

- Fusion: The paired effect is negative on this cohort.
- Fusion + MMR: The paired effect is negative on this cohort.

## Coverage and limits

The evaluation references contain 1177 image/category positives; 141 fall outside the 69-name vocabulary. The retained reference coverage is 88.02%. 13 evaluation images have no in-vocabulary positive reference. All images remain in the primary denominator. No image was replaced because of its predictions.

The complete 512-image selection covers all 80 original COCO classes, but scoring is restricted to the 69 exact-name matches. Rare-category estimates are unstable. These are COCO validation images, not guaranteed absent from model pretraining. This is not full COCO-80 performance, a zero-shot claim or a reproduction of the thesis benchmark.

## Exploratory class AP

This analysis was declared separately while inference was running. It averages non-interpolated AP over 69 of the 69 candidate classes with positive evaluation references. Every class support and AP are retained in the JSON artifact. No cross-image confidence is invented for MMR.

| Method | Mean AP over present shared classes |
|---|---:|
| RAM | 0.8039 |
| CLIP | 0.5024 |
| Fusion | 0.7446 |

## Measured CPU feature-stage latency

| Stage | Median seconds/image | 95th percentile |
|---|---:|---:|
| RAM | 4.813 | 5.577 |
| CLIP | 0.255 | 0.286 |

One warm-up preceded each stage. Timings exclude model loading and image preprocessing; they do not include fusion/MMR postprocessing or constitute an end-to-end latency guarantee. RAM used independently computed candidate queries after native-output equivalence checks.

## Error counts

The following counts use the unchanged source annotations; references are not revised after inspecting predictions.

### RAM

Most frequent unmatched predictions: person (117), cup (88), car (63), dog (56), bench (48), couch (38), chair (38), bird (37), bowl (36), bus (32).

Most frequent missed references: handbag (24), bottle (21), cup (15), sports ball (15), bowl (10), backpack (9), chair (9), spoon (8), vase (7), book (7).

### CLIP

Most frequent unmatched predictions: frisbee (80), sandwich (52), bowl (42), bench (36), tie (35), skateboard (35), oven (34), bus (32), hair drier (32), kite (32).

Most frequent missed references: person (103), cup (33), bottle (30), chair (30), car (20), handbag (20), book (15), bowl (15), backpack (14), vase (12).

### Fusion

Most frequent unmatched predictions: person (92), bench (50), car (50), cup (44), frisbee (41), bowl (38), bus (36), dog (26), couch (24), chair (24).

Most frequent missed references: bottle (23), cup (22), handbag (22), chair (17), backpack (11), book (9), bowl (9), vase (9), spoon (8), wine glass (7).

### Fusion + MMR

Most frequent unmatched predictions: person (91), bench (52), car (49), cup (43), frisbee (43), bus (36), bowl (35), dog (26), couch (24), chair (24).

Most frequent missed references: bottle (24), cup (22), handbag (22), chair (17), backpack (11), book (10), bowl (9), vase (9), spoon (8), wine glass (7).

## Reproduction artifacts

`validation-512.json` contains every image, raw scores, embeddings, predictions and timings. `validation-512-summary.json` contains both partitions and paired intervals. `validation-512-manifest.json` and `validation-registration.json` identify the source data, protocol and executed code. The CSV is an evaluation-only convenience export, not a replacement for the raw run.
