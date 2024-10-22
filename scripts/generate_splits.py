from pathlib import Path
import itertools
import json
import numpy as np

from spectralwaste.dataset import SpectralWasteDataset

# Divide images in the dataset into train/test/val

splits_division = {
    'train': 0.60,
    'test': 0.20,
    'val': 0.20
}

def split_list(lista: list, percents: list[float]):
    assert sum(percents) == 1

    # Ver cuántos elementos corresponden a cada lista
    counts = [round(len(lista) * perc) for perc in percents]

    # Add remaining to the split with highest percentage
    if sum(counts) < len(lista):
        counts[np.array(percents).argmax()] += len(lista) - sum(counts)

    splits = []
    start = 0
    for count in counts:
        end = start + count
        splits.append(lista[start:end])
        start = end

    return splits

def main():
    np.random.seed(42)

    dataset = SpectralWasteDataset('/data/spectralwaste/dataset', '/data/spectralwaste/metadata/annotations_rgb_iros2024.json')

    splits_lists = {split: [] for split in splits_division.keys()}

    # Remaining images to distribute
    images = np.random.permutation(dataset.get_images_with_label())

    # Count number of images that have each category
    cat_counts = {cat: sum([cat in img.categories for img in images]) for cat in dataset.categories}
    # Sort categories from fewer to more images
    sorted_cats = sorted(cat_counts, key=cat_counts.get)
    print(sorted_cats)

    remaining_images = images.copy()
    for cat in sorted_cats:
        selected_ids = [img.id for img in remaining_images if cat in img.categories]
        remaining_images = [img for img in remaining_images if cat not in img.categories]

        division = split_list(selected_ids, splits_division.values())
        for split, ids in zip(splits_division.keys(), division):
            splits_lists[split].extend(ids)

    # Split empty images
    for img in remaining_images:
        assert img.annotations == []
    selected_ids = [img.id for img in remaining_images]
    division = split_list(selected_ids, splits_division.values())
    for split, ids in zip(splits_division.keys(), division):
        splits_lists[split].extend(ids)

    # Sanity check
    distributed_ids = list(itertools.chain(*splits_lists.values()))
    assert sorted(distributed_ids) == sorted([img.id for img in images])

    # Check how many categories each split has
    for split in splits_lists.keys():
        images = dataset.get_images_with_label(lambda image: image.id in splits_lists[split])
        c = dict()
        for image in images:
            for cat in image.categories:
                c[cat] = c.get(cat, 0) + 1
        print(split, len(images), c)

    json.dump(splits_lists, open('splits.json', 'w'), indent=2)

if __name__ == "__main__":
    main()
