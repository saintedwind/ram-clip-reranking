"""Consistency checks for the completed real run; no pretrained inference in CI."""
import json
import hashlib
from pathlib import Path
import numpy as np
from tag_research.rerank import fuse, metrics, select_mmr
from tag_research.validation import summarize
from tag_research.class_ap import summarize_class_ap

def test_recorded_512_run_and_paired_analysis():
    root = Path(__file__).resolve().parents[1] / 'reports'
    run = json.loads((root / 'validation-512.json').read_text())
    manifest = json.loads((root / 'validation-512-manifest.json').read_text())
    registration = json.loads((root / 'validation-registration.json').read_text())
    assert run['manifest_sha256'] == registration['manifest_sha256']
    assert len(run['images']) == len(manifest['images']) == 512
    assert run['parameters'] == {'alpha': .5, 'mmr_relevance_weight': .8, 'k': 5}
    assert run['query_subset'] is True
    assert run['native_subset_max_abs_error'] <= 1.1e-5
    candidates = run['candidates']
    text = np.asarray(run['text_embeddings'])
    assert len(candidates) == 69 and text.shape == (69, 512)
    for row, source in zip(run['images'], manifest['images']):
        for key in ('image_id', 'image', 'split', 'sha256', 'truth'):
            assert row[key] == source[key]
        assert row['excluded_truth'] == sorted(set(row['truth']) - set(candidates))
        ram = np.asarray(row['ram_scores']); clip = np.asarray(row['clip_cosines'])
        assert ram.shape == clip.shape == (69,)
        assert np.isfinite(ram).all() and np.all((ram >= 0) & (ram <= 1))
        assert np.isfinite(clip).all() and np.all(abs(clip) <= 1.000001)
        fused = fuse(ram, clip)
        orders = {method: np.argsort(-vector, kind='stable')[:5] for method, vector in
                  [('ram', ram), ('clip', clip), ('fusion', fused)]}
        orders['mmr'] = select_mmr(fused, text, 5)
        for method, order in orders.items():
            prediction = [candidates[i] for i in order]
            assert prediction == row['predictions'][method]
            assert metrics(prediction, set(row['truth']) & set(candidates)) == row['metrics'][method]
    expected = json.loads((root / 'validation-512-summary.json').read_text())
    actual = summarize(run)
    for partition in ('development', 'evaluation'):
        a = actual['partitions'][partition]; b = expected['partitions'][partition]
        assert a['images'] == b['images']
        for method in ('ram', 'clip', 'fusion', 'mmr'):
            for key in ('p_at_k', 'recall_at_k', 'f1_at_k'):
                assert np.isclose(a['mean_metrics'][method][key], b['mean_metrics'][method][key], atol=1e-12)
        for method in ('clip', 'fusion', 'mmr'):
            x = a['paired_f1_against_ram'][method]; y = b['paired_f1_against_ram'][method]
            assert np.allclose(x['bootstrap_95_percentile_interval'], y['bootstrap_95_percentile_interval'], atol=1e-12)
            for key in ('improved_images', 'worse_images', 'tied_images'):
                assert x[key] == y[key]
    ap = summarize_class_ap(run)
    stored_ap = json.loads((root / 'validation-512-class-ap.json').read_text())
    for method in ('ram', 'clip', 'fusion'):
        assert np.isclose(ap['methods'][method]['mean_ap_present_classes'],
                          stored_ap['methods'][method]['mean_ap_present_classes'], atol=1e-12)

def test_recorded_subset_equivalence_against_native_pilot():
    root = Path(__file__).resolve().parents[1] / 'reports'
    native = json.loads((root / 'full-pilot.json').read_text())
    subset = json.loads((root / 'subset-query-pilot.json').read_text())
    evidence = json.loads((root / 'subset-query-equivalence.json').read_text())
    assert len(native['images']) == len(subset['images']) == 8
    assert hashlib.sha256((root / 'subset-query-pilot.json').read_bytes()).hexdigest() == evidence['subset_run_sha256']
    for a, b in zip(native['images'], subset['images']):
        assert a['sha256'] == b['sha256']
        assert np.allclose(a['ram_scores'], b['ram_scores'], atol=1e-6, rtol=1e-5)
        assert a['predictions'] == b['predictions']
