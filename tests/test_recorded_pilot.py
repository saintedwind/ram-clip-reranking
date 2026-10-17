"""Validate the recorded real run; this does not rerun pretrained inference."""
import json
from pathlib import Path
import numpy as np
from tag_research.rerank import metrics, fuse

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

def test_full_recorded_comparison_recomputes_from_raw_scores():
    root = Path(__file__).resolve().parents[1]
    run = json.loads((root / 'reports/full-pilot.json').read_text())
    independent = json.loads((root / 'reports/clip-pilot.json').read_text())
    assert len(run['images']) == 8
    assert run['candidates'] == independent['candidates']
    names = run['candidates']
    for row, other in zip(run['images'], independent['images']):
        assert row['sha256'] == other['sha256']
        assert row['truth'] == other['truth']
        assert np.allclose(row['clip_cosines'], other['cosines'], atol=1e-6)
        vectors = {'ram': np.asarray(row['ram_scores']),
                   'clip': np.asarray(row['clip_cosines']),
                   'fusion': fuse(row['ram_scores'], row['clip_cosines'])}
        for method, scores in vectors.items():
            ranking = [names[i] for i in np.argsort(-scores, kind='stable')]
            assert ranking[:5] == row['predictions'][method]
        for method, prediction in row['predictions'].items():
            assert len(prediction) == len(set(prediction)) == 5
            truth = set(row['truth']) & set(names)
            assert row['metrics'][method] == metrics(prediction, truth)
    for method, values in run['mean_metrics'].items():
        for key, value in values.items():
            assert np.isclose(value, np.mean([row['metrics'][method][key] for row in run['images']]))
