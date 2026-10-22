import json
from pathlib import Path
import numpy as np
from tag_research.rerank import metrics

def test_recorded_ontology_run():
    p=Path(__file__).resolve().parents[1]/'reports'
    run=json.loads((p/'ontology-smoke.json').read_text())
    assert len(run['candidates'])==76
    assert len(run['images'])==8
    assert run['native_subset_max_abs_error'] < 1e-6
    for row in run['images']:
        assert len(row['ram_scores'])==76
        assert len(row['clip_cosines'])==76
        assert np.isfinite(row['ram_scores']).all()
        truth=set(row['truth']) & set(run['candidates'])
        for method,prediction in row['predictions'].items():
            assert len(prediction)==len(set(prediction))==5
            actual=metrics(prediction,truth)
            for key in actual:
                assert abs(actual[key]-row['metrics'][method][key])<1e-12
