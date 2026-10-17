"""Exploratory class-wise non-interpolated AP; separate from the primary endpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from .rerank import fuse

def average_precision(truth, scores):
    truth = np.asarray(truth)
    scores = np.asarray(scores, dtype=float)
    if truth.ndim != 1 or truth.shape != scores.shape or not len(truth):
        raise ValueError('Aligned nonempty vectors required')
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError('Binary truth and finite scores required')
    if not truth.sum():
        return None
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    true_positives = np.cumsum(truth[order])
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    precision = true_positives[ends] / (ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives[ends]]) / truth.sum()
    return float(np.sum(precision * recall_increments))

def summarize_class_ap(run):
    rows = [r for r in run['images'] if r.get('split') == 'evaluation']
    if not rows:
        raise ValueError('No evaluation partition')
    names = run['candidates']
    truth = np.asarray([[name in row['truth'] for name in names] for row in rows])
    vectors = {'ram': np.asarray([r['ram_scores'] for r in rows]),
               'clip': np.asarray([r['clip_cosines'] for r in rows]),
               'fusion': np.asarray([fuse(r['ram_scores'], r['clip_cosines']) for r in rows])}
    report = {'scope': 'Exploratory class AP, not the preregistered primary F1 endpoint or full COCO-80 mAP',
              'images': len(rows), 'candidate_classes': len(names),
              'classes_with_positives': int((truth.sum(axis=0) > 0).sum()), 'methods': {},
              'definition': 'Sum of precision at each distinct score threshold weighted by recall increase; ties grouped',
              'mmr': 'Not reported: top-five MMR selections are not cross-image class confidence scores'}
    for method, matrix in vectors.items():
        per_class = {name: {'positives': int(truth[:, i].sum()),
                            'ap': average_precision(truth[:, i], matrix[:, i])}
                     for i, name in enumerate(names)}
        valid = [r['ap'] for r in per_class.values() if r['ap'] is not None]
        report['methods'][method] = {'mean_ap_present_classes': float(np.mean(valid)) if valid else None, 'per_class': per_class}
    return report

if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(summarize_class_ap(json.loads(args.run.read_text())), indent=2), encoding='utf-8')
