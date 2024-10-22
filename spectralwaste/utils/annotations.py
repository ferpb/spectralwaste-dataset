import numpy as np


def coco_rle_to_mask(coco_rle):
    list = []
    val = False
    for count in coco_rle['counts']:
        list.append(np.full(count, val))
        val = not val
    mask = np.concatenate(list).reshape(coco_rle['size'][0], coco_rle['size'][1], order='F')
    return mask

def coco_polygon_to_mask(coco_polygon):
    raise NotImplementedError()

def mask_to_coco_rle(mask):
    h, w = mask.shape
    flat = mask.flatten(order='F')

    # Compute indices changes
    diff = flat[1:] ^ flat[:-1]
    change_indices = np.flatnonzero(diff)
    change_indices = np.concatenate([[0], change_indices + 1, [h * w]])

    # Encode run length
    lengths = change_indices[1:] - change_indices[:-1]
    counts = [] if flat[0] == 0 else [0]
    counts.extend(lengths)

    return {"size": (h, w), "counts": counts}

def semantic_from_instance(instance_labels):
    shape = instance_labels['masks'][0].shape
    assert all(mask.shape == shape for mask in instance_labels['masks'])
    semantic_labels = np.zeros(shape, dtype=np.uint8)
    for mask, cat in zip(instance_labels['masks'], instance_labels['categories']):
        semantic_labels[mask] = cat
    return semantic_labels

def merge_instances(instance_labels):
    """Merge all the instances into a a single tensor, assuming the masks are not overlapping"""
    class_counter = dict()
    out = np.zeros(instance_labels['masks'][0].shape, dtype=np.uint16)
    for mask, cat in zip(instance_labels['masks'], instance_labels['categories']):
        count = class_counter.get(cat, 0)
        out[mask] = cat * 1024 + count
        class_counter[cat] = count + 1
    return out