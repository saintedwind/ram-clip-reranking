import pytest
from tag_research.data import COCO_NAMES, manifest

def test_coco_category_order():
    assert len(COCO_NAMES) == len(set(COCO_NAMES)) == 80
    assert COCO_NAMES[0] == 'person'
    assert COCO_NAMES[58] == 'potted plant'
    assert COCO_NAMES[-1] == 'toothbrush'

def test_incomplete_dataset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Expected COCO8'):
        manifest(tmp_path)
