# RAM–CLIP tag reranking

An audit and reproducibility study of an MSc image-tagging notebook. The experiment asks whether fixed CLIP relevance scores and diversity selection improve the ordering of RAM tags on an explicitly defined object vocabulary.

The original code had confidence-extraction and evaluation errors. This revision separates a tested ranking implementation from a pretrained-model experiment. It does not carry forward unverified accuracy numbers from the thesis.

## Status

The ranking regression tests pass. Eight real COCO photographs and their existing annotations have been downloaded and fingerprinted. **Pretrained inference is not yet verified:** the RAM download was truncated and is being resumed; the canonical CLIP weight endpoint timed out. No model quality score is reported until both weights pass their checksums and the complete experiment runs.

Read the [audit](reports/AUDIT.md), [fixed protocol](reports/PROTOCOL.md) and [data manifest](reports/data-manifest.json). The manifest contains hashes and existing object labels, not redistributed photographs.

## Ranking core

```sh
python -m pip install -e ".[test]"
python -m pytest -q
```

Fusion standardizes each image's RAM confidence and CLIP cosine vectors before combining them with equal weight. These normalized scores are ranking values, not calibrated probabilities. Greedy maximum marginal relevance recomputes the best remaining candidate after each selection; relevance weight is fixed at 0.8.

## Model experiment

The experimental runner requires PyTorch 2.8, torchvision 0.23, timm 0.4.12, transformers 4.38.2, fairscale, ftfy, regex, and the official RAM/CLIP source trees. This integration remains under validation. CPU execution with the original RAM checkpoint needs several GB of storage and memory.

Pinned sources:

- [RAM](https://github.com/xinyu1205/recognize-anything/tree/7cb804a8609e9f4b1a50b7f31436d2df40bb9481)
- [CLIP](https://github.com/openai/CLIP/tree/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1)
- RAM weight revision: `xinyu1205/recognize_anything_model@ebc52dc741e86466202a5ab8ab22eae6e7d48bf1`

With those source directories on `PYTHONPATH`, verified weights and a local `bert-base-uncased` tokenizer:

```sh
python -m tag_research.experiment --data /path/to/coco8 --ram /path/to/ram_swin_large_14m.pth --clip /path/to/ViT-B-32.pt --tokenizer /path/to/bert-tokenizer --output results/pilot.json
```

The runner refuses mismatched weight hashes and records per-image scores, predictions and timings. It captures RAM's actual classifier logits and checks agreement with the official thresholded tags. Each model retains its own preprocessing. It never substitutes placeholder scores or inferred ground truth.

## Scientific scope

This code uses **RAM**, not RAM++. RAM already incorporates CLIP supervision in its original design; the additional CLIP reranking here is a separate inference-stage experiment. A frozen reranker does not train an attention decoder, alter the backbone's attention maps or establish recognition of unseen classes.

COCO8 comes from COCO train2017 and may overlap pretraining. Eight images are an integration pilot, not evidence of generalization. COCO annotations also omit many valid scene attributes. No claim of state-of-the-art performance or thesis benchmark reproduction is made.

## References

- [Recognize Anything: A Strong Image Tagging Model](https://arxiv.org/abs/2306.03514)
- [Open-Set Image Tagging with Multi-Grained Text Supervision (RAM++)](https://arxiv.org/abs/2310.15200)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [SigLIP 2](https://arxiv.org/abs/2502.14786), a relevant newer comparison to consider in a separate experiment; not evaluated here.
- [COCO8 dataset documentation](https://docs.ultralytics.com/datasets/detect/coco8/)

Model code, weights and data retain their upstream terms. Original thesis PDFs, personal front matter and third-party image files are not included in the public package.
