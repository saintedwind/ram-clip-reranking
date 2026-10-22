from pathlib import Path
import pytest
from tag_research.ontology import align, ALIASES, UNRESOLVED
from tag_research.data import COCO_NAMES

def test_aliases_preserve_reference_names():
    vocab = [ALIASES.get(c,c) for c in COCO_NAMES if c not in UNRESOLVED]
    candidates, indices = align(vocab)
    assert len(candidates) == 76
    assert [vocab[i] for i in indices] == [ALIASES.get(c,c) for c in candidates]
    assert set(COCO_NAMES)-set(candidates) == set(UNRESOLVED)

def test_missing_registered_alias_fails():
    with pytest.raises(ValueError, match='missing'):
        align([])

def test_duplicate_vocab_fails():
    with pytest.raises(ValueError, match='duplicate'):
        align(['plane','plane'])

def test_exact_name_has_priority():
    vocab = [ALIASES.get(c,c) for c in COCO_NAMES if c not in UNRESOLVED]+['airplane']
    candidates, indices = align(vocab)
    assert vocab[indices[candidates.index('airplane')]] == 'airplane'
