"""Ontology-aligned inference, forked from the frozen v0.2 runner.

Historical experiment.py remains unchanged for registered reproducibility.
CLIP prompts and reference labels retain canonical COCO names.
"""
from .ontology import align, ALIASES, UNRESOLVED
import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .data import manifest, COCO_NAMES, sha256
from .rerank import fuse, select_mmr, metrics

RAM_SHA = '15c729c793af28b9d107c69f85836a1356d76ea830d4714699fb62e55fcc08ed'
CLIP_SHA = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'

def subset_probabilities(model, image, indices):
    """Independent RAM label queries; no label-to-label attention is allowed."""
    import torch
    if any(hasattr(layer, 'attention') for layer in model.tagging_head.encoder.layer):
        raise ValueError('Subset queries require RAM head without label self-attention')
    labels = torch.relu(model.wordvec_proj(model.label_embed[indices]))
    features = model.image_proj(model.visual_encoder(image))
    mask = torch.ones(features.shape[:-1], dtype=torch.long, device=features.device)
    output = model.tagging_head(encoder_embeds=labels.unsqueeze(0).expand(image.shape[0], -1, -1),
        encoder_hidden_states=features, encoder_attention_mask=mask, return_dict=False, mode='tagging')
    return model.fc(output[0]).squeeze(-1).sigmoid()

def main():
    started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--ram', type=Path, required=True)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Frozen manifest for a larger COCO validation subset')
    parser.add_argument('--query-subset', action='store_true', help='Compute only independent candidate queries; verify native agreement first')
    args = parser.parse_args()
    # Fail before importing or allocating a model if an artifact is incomplete.
    if sha256(args.ram) != RAM_SHA or sha256(args.clip) != CLIP_SHA:
        raise ValueError('Model checksum mismatch; inference refused')
    if args.manifest:
        document = json.loads(args.manifest.read_text(encoding='utf-8'))
        rows = document['images']
        if not rows or len({r['image'] for r in rows}) != len(rows):
            raise ValueError('Empty or duplicate manifest')
        for row in rows:
            path = (args.data / row['image']).resolve()
            if not path.is_relative_to(args.data.resolve()) or sha256(path) != row['sha256']:
                raise ValueError('Manifest image path/hash mismatch')
            if not set(row['truth']) <= set(COCO_NAMES):
                raise ValueError('Unknown reference category')
    else:
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
    candidates, indices = align(names)
    transform = get_transform(384)
    observed = []
    hook = model.fc.register_forward_hook(lambda module, inputs, output: observed.append(output.detach()))
    load_seconds = time.perf_counter() - start
    scores = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.with_suffix('.ram.json')
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        signature = lambda items: [(r['image'], r['sha256'], r['truth']) for r in items]
        if (cached['ram_sha256'] != RAM_SHA or cached['candidates'] != candidates
                or cached.get('query_subset', False) != args.query_subset
                or signature(cached['images']) != signature(rows)):
            raise ValueError('Cached RAM results belong to a different experiment')
        scores = cached['ram_scores']
        if len(scores) > len(rows) or any(len(s) != len(candidates) for s in scores):
            raise ValueError('Invalid cached score dimensions')
        values = np.asarray(scores)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Invalid cached probabilities')
        for i in range(len(scores)):
            rows[i]['ram_seconds'] = cached['images'][i]['ram_seconds']
        print(f'Resuming from {len(scores)} cached RAM images', flush=True)
    warmed_up = False
    native_check_error = None
    with torch.inference_mode():
        for i, row in enumerate(rows):
            if i < len(scores):
                continue
            with Image.open(args.data / row['image']) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            if not warmed_up:
                model.generate_tag(tensor)
                if args.query_subset:
                    native = observed.pop().squeeze().sigmoid().cpu().numpy()[indices]
                    optimized = subset_probabilities(model, tensor, indices).squeeze(0).cpu().numpy()
                    native_check_error = float(np.max(np.abs(native - optimized)))
                    if not np.allclose(native, optimized, atol=1e-6, rtol=1e-5):
                        raise RuntimeError('Subset queries disagree with full native RAM')
                observed.clear()
                warmed_up = True
            start = time.perf_counter()
            if args.query_subset:
                probabilities = subset_probabilities(model, tensor, indices).squeeze(0).cpu().numpy()
                observed.clear()
                elapsed = time.perf_counter() - start
                scores.append(probabilities.tolist())
                row['ram_seconds'] = elapsed
                print(f"RAM subset {i + 1}/{len(rows)}: {elapsed:.2f}s", flush=True)
                if (i + 1) % 8 == 0:
                    temporary = cache_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'ram_sha256': RAM_SHA, 'candidates': candidates, 'query_subset': args.query_subset,
                        'images': rows, 'ram_scores': scores}), encoding='utf-8')
                    temporary.replace(cache_path)
                continue
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
            if (i + 1) % 8 == 0:
                temporary = cache_path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'ram_sha256': RAM_SHA, 'candidates': candidates, 'query_subset': args.query_subset,
                    'images': rows, 'ram_scores': scores}), encoding='utf-8')
                temporary.replace(cache_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.ram.json').write_text(json.dumps({
        'stage': 'RAM inference only; CLIP and comparisons not yet completed',
        'ram_sha256': RAM_SHA, 'candidates': candidates,
        'query_subset': args.query_subset,
        'images': rows, 'ram_scores': scores}, indent=2), encoding='utf-8')
    hook.remove()
    del model, observed
    if 'tensor' in locals():
        del tensor
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
    result = {'scope': ('COCO val2017 deterministic subset; pretraining overlap unknown; no zero-shot claim'
                       if args.manifest else 'Eight-image COCO train-source smoke test; not a held-out benchmark'),
              'manifest_sha256': sha256(args.manifest) if args.manifest else None,
              'started_at_utc': started_at,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'ram_sha256': RAM_SHA, 'clip_sha256': CLIP_SHA, 'ram_load_seconds': load_seconds,
              'ontology_aliases': ALIASES, 'ontology_unresolved': UNRESOLVED,
              'candidates': candidates, 'excluded_categories': sorted(set(COCO_NAMES) - set(candidates)),
              'torch_version': torch.__version__, 'images': rows,
              'python_version': platform.python_version(), 'device': 'cpu',
              'parameters': {'alpha': 0.5, 'mmr_relevance_weight': 0.8, 'k': 5},
              'query_subset': args.query_subset, 'native_subset_max_abs_error': native_check_error,
              'text_embeddings': text.cpu().numpy().tolist(),
              'mean_metrics': {method: {metric: float(np.mean([r['metrics'][method][metric] for r in rows]))
                               for metric in ('p_at_k', 'recall_at_k', 'f1_at_k')}
                               for method in ('ram', 'clip', 'fusion', 'mmr')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
