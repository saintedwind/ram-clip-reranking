"""Plot recorded evaluation metrics. Requires matplotlib; never generates observations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(__doc__)
p.add_argument('summary', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
report = json.loads(args.summary.read_text())['partitions']['evaluation']
methods = ['ram', 'clip', 'fusion', 'mmr']
names = ['RAM', 'CLIP', 'Fusion', 'Fusion + MMR']
plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
values = [report['mean_metrics'][m]['f1_at_k'] for m in methods]
left.barh(names, values, color=['#25485c', '#8d9ba5', '#497875', '#6f7c53'])
left.invert_yaxis()
left.set_xlim(0, 1)
left.set_xlabel('Mean per-image F1@5')
left.set_title(f"Fixed COCO evaluation subset (n={report['images']})")
for i, value in enumerate(values):
    left.text(value + .012, i, f'{value:.4f}', va='center')
for i, method in enumerate(methods[1:]):
    effect = report['paired_f1_against_ram'][method]
    mean = effect['mean_difference'] * 100
    low, high = [v * 100 for v in effect['bootstrap_95_percentile_interval']]
    right.errorbar(mean, i, xerr=[[mean - low], [high - mean]], fmt='o', color='#25485c', capsize=4)
right.set_yticks(range(3), names[1:])
right.invert_yaxis()
right.axvline(0, color='#929292', linestyle=':', linewidth=1)
right.set_xlabel('Difference from RAM (percentage points)')
right.set_title('Paired image bootstrap: 95% intervals')
fig.suptitle('RAM–CLIP ranking validation', fontsize=15)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output.with_suffix('.png'), dpi=180)
fig.savefig(args.output.with_suffix('.svg'))
