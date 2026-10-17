"""Validate the recorded real run; this does not rerun pretrained inference."""
import json
from pathlib import Path
import numpy as np
from tag_research.rerank import metrics

def test_recorded_pilot_is_internally_consistent():
    root = Path(__file__).resolve().parents[1]
    run = json.loads((root / 'reports/clip-pilot.json').read_text())
    source = json.loads((root / 'reports/data-manifest.json').read_text())
    assert len(run['images']) == len(source) == 8
    candidates = run['candidates']
    assert len(candidates) == len(set(candidates)) == 69
    for row, original in zip(run['images'], source):
        for key in ('image', 'sha256', 'annotation_sha256', 'truth'):
            assert row[key] == original[key]
        scores = np.asarray(row['cosines'])
        assert scores.shape == (len(candidates),)
        assert np.isfinite(scores).all() and np.all(np.abs(scores) <= 1.000001)
        ranking = [candidates[i] for i in np.argsort(-scores, kind='stable')]
        assert ranking[:5] == row['predictions']
        truth = set(row['truth']) & set(candidates)
        assert row['metrics'] == metrics(ranking, truth)
    for key, value in run['mean_metrics'].items():
        assert np.isclose(value, np.mean([row['metrics'][key] for row in run['images']]))
