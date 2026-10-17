"""Build a human-readable report and CSV from the recorded real experiment."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(__doc__)
p.add_argument('--reports', type=Path, default=Path(__file__).resolve().parents[1] / 'reports')
args = p.parse_args()
root = args.reports
run = json.loads((root / 'validation-512.json').read_text())
summary = json.loads((root / 'validation-512-summary.json').read_text())
ap = json.loads((root / 'validation-512-class-ap.json').read_text())
evaluation = summary['partitions']['evaluation']
rows = [r for r in run['images'] if r['split'] == 'evaluation']
names = {'ram': 'RAM', 'clip': 'CLIP', 'fusion': 'Fusion', 'mmr': 'Fusion + MMR'}
lines = ['# COCO val2017: 512-image validation', '',
 'The protocol and deterministic image selection were recorded before model inference. '
 '128 development images and 384 evaluation images were processed with the same frozen checkpoints and parameters. '
 'The primary results below use only the 384 evaluation images. No tuning was performed.', '',
 '## Primary endpoint and supporting metrics', '',
 '| Method | Mean P@5 | Mean recall@5 | Mean F1@5 |', '|---|---:|---:|---:|']
for method, name in names.items():
    m = evaluation['mean_metrics'][method]
    lines.append(f"| {name} | {m['p_at_k']:.4f} | {m['recall_at_k']:.4f} | {m['f1_at_k']:.4f} |")
lines += ['', '## Paired differences from RAM', '',
 '| Method | F1 difference (percentage points) | 95% bootstrap interval | Improved / worse / tied images |',
 '|---|---:|---|---|']
for method, effect in evaluation['paired_f1_against_ram'].items():
    lo, hi = effect['bootstrap_95_percentile_interval']
    lines.append(f"| {names[method]} | {100*effect['mean_difference']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] | "
                 f"{effect['improved_images']} / {effect['worse_images']} / {effect['tied_images']} |")
lines += ['', 'Intervals use 10,000 paired image bootstrap resamples, seed 20261017. '
          'They are per-comparison percentile intervals, without a multiple-comparison adjustment. '
          'They do not measure training variation, annotation uncertainty or out-of-domain performance.', '']
for method in ('fusion', 'mmr'):
    effect = evaluation['paired_f1_against_ram'][method]
    lo, hi = effect['bootstrap_95_percentile_interval']
    interpretation = ('The interval includes zero; this comparison does not establish improvement.' if lo <= 0 <= hi
                      else 'The paired effect is negative on this cohort.' if hi < 0
                      else 'The paired effect is positive on this cohort; the interval is unadjusted and generalization is not established.')
    lines.append(f"- {names[method]}: {interpretation}")
total = evaluation['reference_positives']; excluded = evaluation['excluded_reference_positives']
lines += ['', '## Coverage and limits', '',
 f"The evaluation references contain {total} image/category positives; {excluded} fall outside the 69-name vocabulary. "
 f"The retained reference coverage is {(total-excluded)/total:.2%}. "
 f"{evaluation['images_without_in_vocabulary_positives']} evaluation images have no in-vocabulary positive reference. "
 'All images remain in the primary denominator. No image was replaced because of its predictions.', '',
 'The complete 512-image selection covers all 80 original COCO classes, but scoring is restricted to the 69 exact-name matches. '
 'Rare-category estimates are unstable. These are COCO validation images, not guaranteed absent from model pretraining. '
 'This is not full COCO-80 performance, a zero-shot claim or a reproduction of the thesis benchmark.', '',
 '## Exploratory class AP', '',
 f"This analysis was declared separately while inference was running. It averages non-interpolated AP over "
 f"{ap['classes_with_positives']} of the {ap['candidate_classes']} candidate classes with positive evaluation references. "
 'Every class support and AP are retained in the JSON artifact. No cross-image confidence is invented for MMR.', '',
 '| Method | Mean AP over present shared classes |', '|---|---:|']
for method, values in ap['methods'].items():
    lines.append(f"| {names[method]} | {values['mean_ap_present_classes']:.4f} |")
lines += ['', '## Measured CPU feature-stage latency', '',
 '| Stage | Median seconds/image | 95th percentile |', '|---|---:|---:|']
for stage in ('ram', 'clip'):
    seconds = [r[stage + '_seconds'] for r in rows]
    lines.append(f"| {names[stage]} | {np.median(seconds):.3f} | {np.quantile(seconds,.95):.3f} |")
lines += ['', 'One warm-up preceded each stage. Timings exclude model loading and image preprocessing; '
          'they do not include fusion/MMR postprocessing or constitute an end-to-end latency guarantee. '
          'RAM used independently computed candidate queries after native-output equivalence checks.', '',
          '## Error counts', '', 'The following counts use the unchanged source annotations; references are not revised after inspecting predictions.', '']
for method in names:
    fp = Counter(); fn = Counter()
    for row in rows:
        truth = set(row['truth']) & set(run['candidates']); prediction = set(row['predictions'][method])
        fp.update(prediction-truth); fn.update(truth-prediction)
    lines += [f"### {names[method]}", '',
              'Most frequent unmatched predictions: ' + ', '.join(f'{k} ({v})' for k,v in fp.most_common(10)) + '.', '',
              'Most frequent missed references: ' + ', '.join(f'{k} ({v})' for k,v in fn.most_common(10)) + '.', '']
lines += ['## Reproduction artifacts', '',
          '`validation-512.json` contains every image, raw scores, embeddings, predictions and timings. '
          '`validation-512-summary.json` contains both partitions and paired intervals. '
          '`validation-512-manifest.json` and `validation-registration.json` identify the source data, protocol and executed code. '
          'The CSV is an evaluation-only convenience export, not a replacement for the raw run.']
(root / 'VALIDATION_512.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
with (root / 'validation-512-evaluation.csv').open('w', newline='', encoding='utf-8') as out:
    fields = ['image_id', 'truth', 'excluded_truth'] + [m + '_f1_at_5' for m in names]
    writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
    for row in rows:
        writer.writerow({'image_id': row['image_id'], 'truth': ' | '.join(row['truth']),
                         'excluded_truth': ' | '.join(row['excluded_truth']),
                         **{m + '_f1_at_5': row['metrics'][m]['f1_at_k'] for m in names}})
