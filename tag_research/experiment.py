"""CPU pilot. Requires official RAM and CLIP source on PYTHONPATH."""
import argparse
import gc
import json
import platform
import time
from pathlib import Path
import numpy as np
from .data import manifest, COCO_NAMES, sha256
from .rerank import fuse, select_mmr, metrics

RAM_SHA = '15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--ram', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Fail before importing or allocating a model if an artifact is incomplete.
    if sha256(args.ram) != RAM_SHA or sha256(args.clip) != CLIP_SHA:
        raise ValueError('Model checksum mismatch; inference refused')
    rows = manifest(args.data)
    import torch
    from PIL import Image
    from ram import get_transform
    from ram.models.ram import RAM
    from ram.models.utils import interpolate_relative_pos_embed
    torch.manual_seed(0)
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = RAM(image_size=384, vit='swin_l', text_encoder_type=args.tokenizer)
    checkpoint = torch.load(args.ram, map_location='cpu', weights_only=True, mmap=True)
    state = checkpoint['model']
    for key in list(state):
        if 'relative_position_bias_table' in key:
            state[key] = interpolate_relative_pos_embed(state[key], 23 ** 2, param_name=key)
        elif 'relative_position_index' in key or 'attn_mask' in key:
            del state[key]
        elif 'vision_multi' in key:
            state[key.replace('vision_multi', 'tagging_head')] = state.pop(key)
    incompatible = model.load_state_dict(state, strict=False, assign=True)
    allowed = ('relative_position_index', 'attn_mask')
    missing = [key for key in incompatible.missing_keys if not any(x in key for x in allowed)]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f'Unverified checkpoint mapping: {missing}, {incompatible.unexpected_keys}')
    del checkpoint, state
    # These modules are not used by generate_tag; retain the official tagging path.
    del model.tag_encoder, model.text_decoder
    gc.collect()
    model.eval()
    names = [str(x).strip().lower() for x in model.tag_list]
    candidates = [x for x in COCO_NAMES if x in names]
    indices = [names.index(x) for x in candidates]
    transform = get_transform(384)
    observed = []
    hook = model.fc.register_forward_hook(lambda module, inputs, output: observed.append(output.detach()))
    load_seconds = time.perf_counter() - start
    scores = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            with Image.open(args.data / row['image']) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            if i == 0:
                model.generate_tag(tensor)
                observed.clear()
            start = time.perf_counter()
            official, _ = model.generate_tag(tensor)
            elapsed = time.perf_counter() - start
            if len(observed) != 1:
                raise RuntimeError('Expected one classifier output')
            probabilities = observed.pop().squeeze().sigmoid().cpu().numpy()
            selected = [names[j] for j in range(len(names))
                        if probabilities[j] > float(model.class_threshold[j])
                        and j not in model.delete_tag_index]
            if selected != [x.strip().lower() for x in official[0].split(' | ') if x.strip()]:
                raise RuntimeError('Captured logits disagree with official prediction')
            scores.append(probabilities[indices].tolist())
            row['ram_seconds'] = elapsed
            print(f"RAM {i + 1}/{len(rows)}: {elapsed:.2f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.ram.json').write_text(json.dumps({
        'stage': 'RAM inference only; CLIP and comparisons not yet completed',
        'ram_sha256': RAM_SHA, 'candidates': candidates,
        'images': rows, 'ram_scores': scores}, indent=2), encoding='utf-8')
    hook.remove()
    del model, tensor, observed
    gc.collect()
    import clip
    encoder, preprocess = clip.load(str(args.clip), device='cpu', jit=False)
    encoder.eval()
    with torch.inference_mode():
        text = encoder.encode_text(clip.tokenize([f'a photo of a {tag}.' for tag in candidates])).float()
        text = text / text.norm(dim=-1, keepdim=True)
        for i, row in enumerate(rows):
            with Image.open(args.data / row['image']) as image:
                tensor = preprocess(image.convert('RGB')).unsqueeze(0)
            if i == 0:
                encoder.encode_image(tensor)
            start = time.perf_counter()
            feature = encoder.encode_image(tensor).float()
            feature = feature / feature.norm(dim=-1, keepdim=True)
            cosine = (feature @ text.T).squeeze(0).cpu().numpy()
            row['clip_seconds'] = time.perf_counter() - start
            ram_scores = np.asarray(scores[i])
            combined = fuse(ram_scores, cosine)
            orders = {'ram': np.argsort(-ram_scores, kind='stable').tolist(),
                      'clip': np.argsort(-cosine, kind='stable').tolist(),
                      'fusion': np.argsort(-combined, kind='stable').tolist(),
                      'mmr': select_mmr(combined, text.cpu().numpy(), 5)}
            truth = set(row['truth']) & set(candidates)
            row.update({'ram_scores': scores[i], 'clip_cosines': cosine.tolist(),
                        'excluded_truth': sorted(set(row['truth']) - truth),
                        'predictions': {method: [candidates[j] for j in order[:5]] for method, order in orders.items()},
                        'metrics': {method: metrics([candidates[j] for j in order], truth) for method, order in orders.items()}})
            print(f"CLIP {i + 1}/{len(rows)}: {row['clip_seconds']:.2f}s", flush=True)
    result = {'scope': 'Eight-image COCO train-source smoke test; not a held-out benchmark',
              'ram_sha256': RAM_SHA, 'clip_sha256': CLIP_SHA, 'ram_load_seconds': load_seconds,
              'candidates': candidates, 'excluded_categories': sorted(set(COCO_NAMES) - set(candidates)),
              'torch_version': torch.__version__, 'images': rows,
              'python_version': platform.python_version(), 'device': 'cpu',
              'parameters': {'alpha': 0.5, 'mmr_relevance_weight': 0.8, 'k': 5},
              'mean_metrics': {method: {metric: float(np.mean([r['metrics'][method][metric] for r in rows]))
                               for metric in ('p_at_k', 'recall_at_k', 'f1_at_k')}
                               for method in ('ram', 'clip', 'fusion', 'mmr')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
