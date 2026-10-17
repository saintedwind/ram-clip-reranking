"""Read existing COCO8 annotations without inferring labels from model output."""
import hashlib
from pathlib import Path

# COCO's contiguous 80-class order used by the distributed YOLO annotations.
COCO_NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,'
 'fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,'
 'zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,'
 'kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,'
 'cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,'
 'donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,'
 'keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,'
 'teddy bear,hair drier,toothbrush').split(',')

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def manifest(root):
    root = Path(root)
    rows = []
    for image in sorted((root / 'images').glob('*/*.jpg')):
        label = root / 'labels' / image.parent.name / (image.stem + '.txt')
        ids = set()
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                raise ValueError(f'Invalid annotation: {label}')
            cls = int(values[0])
            if not 0 <= cls < len(COCO_NAMES):
                raise ValueError('Class out of bounds')
            coords = list(map(float, values[1:]))
            if not all(0 <= x <= 1 for x in coords):
                raise ValueError('Invalid normalized box')
            ids.add(cls)
        rows.append({'image': image.relative_to(root).as_posix(),
                     'sha256': sha256(image), 'annotation_sha256': sha256(label),
                     'truth': [COCO_NAMES[i] for i in sorted(ids)]})
    if len(rows) != 8:
        raise ValueError(f'Expected COCO8, found {len(rows)} images')
    return rows
