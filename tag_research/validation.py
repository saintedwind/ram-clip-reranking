"""Paired uncertainty estimates from recorded, real image-level outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np

def paired_interval(differences, seed=20261017, resamples=10000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite paired observations required')
    if type(resamples) is not int or resamples < 1:
        raise ValueError('Positive integer resample count required')
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 250):
        count = min(250, resamples - start)
        means[start:start + count] = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    return {'mean_difference': float(values.mean()),
            'bootstrap_95_percentile_interval': np.quantile(means, [.025, .975]).tolist(),
            'improved_images': int((values > 1e-12).sum()),
            'worse_images': int((values < -1e-12).sum()),
            'tied_images': int((np.abs(values) <= 1e-12).sum())}

def summarize(run):
    result = {'endpoint': 'Mean per-image F1@5', 'bootstrap_seed': 20261017,
              'bootstrap_resamples': 10000, 'partitions': {}}
    for split in ('development', 'evaluation'):
        rows = [r for r in run['images'] if r.get('split') == split]
        if len(rows) < 2:
            raise ValueError('Missing predefined partition')
        part = {'images': len(rows),
                'category_image_counts': {name: sum(name in r['truth'] for r in rows)
                                         for name in sorted({label for r in rows for label in r['truth']})},
                'reference_positives': sum(len(r['truth']) for r in rows),
                'excluded_reference_positives': sum(len(r['excluded_truth']) for r in rows),
                'images_without_in_vocabulary_positives': sum(not (set(r['truth']) & set(run['candidates'])) for r in rows),
                'mean_metrics': {}, 'paired_f1_against_ram': {}}
        for method in ('ram', 'clip', 'fusion', 'mmr'):
            part['mean_metrics'][method] = {metric: float(np.mean([r['metrics'][method][metric] for r in rows]))
                 for metric in ('p_at_k', 'recall_at_k', 'f1_at_k')}
            if method != 'ram':
                differences = [r['metrics'][method]['f1_at_k'] - r['metrics']['ram']['f1_at_k'] for r in rows]
                part['paired_f1_against_ram'][method] = paired_interval(differences)
        result['partitions'][split] = part
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = summarize(json.loads(args.run.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
