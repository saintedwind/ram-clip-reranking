"""Run the CLIP-only stage while keeping the pilot vocabulary fixed."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from .data import COCO_NAMES, manifest, sha256
from .rerank import metrics
from .experiment import CLIP_SHA

def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--ram-vocabulary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if sha256(args.clip) != CLIP_SHA:
        raise ValueError('CLIP weight checksum mismatch')
    rows = manifest(args.data)
    vocabulary = {line.strip().lower() for line in args.ram_vocabulary.read_text(encoding='utf-8').splitlines()}
    candidates = [name for name in COCO_NAMES if name in vocabulary]
    import torch
    import clip
    from PIL import Image
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model, preprocess = clip.load(str(args.clip), device='cpu', jit=False)
    model.eval()
    with torch.inference_mode():
        text = model.encode_text(clip.tokenize([f'a photo of a {name}.' for name in candidates])).float()
        text /= text.norm(dim=-1, keepdim=True)
        for i, row in enumerate(rows):
            with Image.open(args.data / row['image']) as image:
                tensor = preprocess(image.convert('RGB')).unsqueeze(0)
            if i == 0:
                model.encode_image(tensor)
            start = time.perf_counter()
            feature = model.encode_image(tensor).float()
            feature /= feature.norm(dim=-1, keepdim=True)
            cosine = (feature @ text.T).squeeze(0).cpu().numpy()
            row['seconds'] = time.perf_counter() - start
            order = np.argsort(-cosine, kind='stable')
            ranking = [candidates[j] for j in order]
            truth = set(row['truth']) & set(candidates)
            row.update({'cosines': cosine.tolist(), 'predictions': ranking[:5],
                        'excluded_truth': sorted(set(row['truth']) - truth),
                        'metrics': metrics(ranking, truth)})
            print(row['image'], row['predictions'], flush=True)
    result = {'scope': 'CLIP-only integration pilot; eight COCO train-source images',
              'checkpoint_sha256': CLIP_SHA, 'ram_vocabulary_sha256': sha256(args.ram_vocabulary),
              'torch_version': torch.__version__, 'device': 'cpu', 'prompt': 'a photo of a {tag}.',
              'candidates': candidates, 'images': rows,
              'mean_metrics': {key: float(np.mean([r['metrics'][key] for r in rows]))
                               for key in ('p_at_k', 'recall_at_k', 'f1_at_k')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['mean_metrics'], flush=True)

if __name__ == '__main__':
    main()
