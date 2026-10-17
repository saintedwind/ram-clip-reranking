import hashlib
import json
from pathlib import Path

def test_frozen_validation_manifest_and_source_registration():
    root = Path(__file__).resolve().parents[1]
    path = root / 'reports/validation-512-manifest.json'
    cohort = json.loads(path.read_text())
    registration = json.loads((root / 'reports/validation-registration.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == registration['manifest_sha256']
    rows = cohort['images']
    assert len(rows) == len({r['image_id'] for r in rows}) == 512
    assert len({r['sha256'] for r in rows}) == 512
    assert all(r['split'] == 'development' for r in rows[:128])
    assert all(r['split'] == 'evaluation' for r in rows[128:])
    keys = [hashlib.sha256(('ram-clip-validation-v1:' + str(r['image_id'])).encode()).hexdigest() for r in rows]
    assert keys == sorted(keys)
    assert cohort['annotation_sha256'] == 'e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f'
    for filename, field in [('experiment.py', 'experiment_sha256'), ('validation.py', 'analysis_sha256')]:
        assert hashlib.sha256((root / 'tag_research' / filename).read_bytes()).hexdigest() == registration[field]
