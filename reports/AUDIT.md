# Audit of the supplied notebook and thesis

Reviewed sources: the user-confirmed `final_code_(4)_(2).ipynb` (92 cells), local `ram.pdf`, the RAM++ paper, and `gholamhosseini-thesis.pdf`, especially methods/results pages 45–65. Original files remain unchanged; PDFs containing personal front matter are not distributed.

## Findings

- The notebook calls `inference_ram`, whose official output is English and Chinese tag strings. Treating the second value as numeric confidence is incorrect. Later cells replace both RAM and BLIP scores with constant 0.5. Those constants are placeholders, not observations.
- The supplied implementation builds the RAM checkpoint, while the thesis describes RAM++ in several sections. Results must name the actual model and checkpoint; the architectures and weight files are not interchangeable.
- An attention block is attached to the model without a demonstrated call in its inference path or trained weights. Attaching a module alone cannot change predictions.
- Two reranker definitions coexist. One accepts `mmr_lambda` but never uses it. Another scans a fixed score order and checks an MMR sign; it does not repeatedly maximize MMR over remaining candidates.
- The singleton flattening can turn a one-tag list into a string and then process characters as tags. A score/tag length mismatch is silently truncated. Both behaviors should raise or preserve explicit shapes.
- Ground truth is revised after examining output. This contaminates the comparison; labels must come from a fixed source before inference.
- A low Spearman rank correlation only shows a changed ordering. It does not establish better labels. Alphabetical/vocabulary order is not a confidence-ranked RAM baseline.
- Caption-word splitting creates fragments, drops phrases and has no per-label confidence. No BLIP word confidence will be invented in the new baseline.
- RAM's official transform uses ImageNet normalization. The original notebook uses CLIP normalization for RAM input; each model must retain its own preprocessing.
- The thesis's broad benchmark, mAP gain, unseen-category and latency claims lack corresponding reproducible raw runs in the supplied notebook. They are not adopted as verified results. A fixed score-fusion layer also does not modify the frozen RAM attention maps.
- Installing incompatible library versions and adding a dummy dependency function hides failures. The revised runner uses an isolated environment and explicit versions.

## Revised evaluation boundary

Use real pretrained outputs and original image annotations. Report RAM-only, CLIP-only, score fusion and fusion plus greedy MMR on the same candidate vocabulary and images. Fixed alpha/lambda are declared before results. No synthetic image, fabricated score, model-authored ground truth or post-result label revision is allowed in evaluation.

Small downloaded subsets are smoke tests, not a reproduction of the thesis's full benchmarks. COCO8 derives from COCO training images and may overlap RAM pretraining, so it cannot support a zero-shot or unseen-category claim. Mathematical unit-test fixtures are explicitly separated from model-evaluation data.
