"""Conservative, outcome-independent COCO to RAM terminology alignment."""
from .data import COCO_NAMES

ALIASES = {'airplane': 'plane', 'fire hydrant': 'hydrant', 'dining table': 'dinning table', 'refrigerator': 'fridge', 'skis': 'ski', 'teddy bear': 'teddy', 'tv': 'television'}
# These are deliberately unresolved: the available RAM concepts differ in scope.
UNRESOLVED = {
    'cell phone': 'smartphone is narrower; phone includes landlines',
    'potted plant': 'houseplant is not defined by the presence of a pot',
    'suitcase': 'luggage includes other bags',
    'toilet': 'toilet bowl is a component rather than the complete object',
}

def align(vocabulary):
    names = [str(value).strip().lower() for value in vocabulary]
    if len(names) != len(set(names)):
        raise ValueError('Ambiguous duplicate RAM vocabulary')
    candidates, indices = [], []
    for category in COCO_NAMES:
        query = category if category in names else ALIASES.get(category)
        if query is not None:
            if query not in names:
                raise ValueError(f'Registered RAM query missing: {query}')
            candidates.append(category)
            indices.append(names.index(query))
    if len(indices) != len(set(indices)):
        raise ValueError('Two COCO categories share a RAM query')
    return candidates, indices
