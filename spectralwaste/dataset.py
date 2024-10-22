from typing import Callable, Optional

import json
import numpy as np
from pathlib import Path
import imageio.v3 as imageio
import matplotlib.pyplot as plt

from spectralwaste import utils


# SpectralWaste data is stored with the following format:
#
# data
#  +-- 20220928
#       +-- 01
#            |-- rgb
#            |    +-- 20220928_01_105147.jpg
#            +-- hyper
#                 +-- 20220928_01_105147.tiff

# Each image has an id with the format format 'YYYYMMDD_xx_hhmmss'


class SpectralWasteImage:
    def __init__(self, id: str, paths: dict[str, Path], meta: dict, annotations: list, split: str, dataset_categories: list):
        self.id = id
        self.paths = paths
        self.meta = meta
        self.annotations = annotations
        self.split = split
        self.dataset_categories = dataset_categories
        self.image_categories = list(set([ann['category'] for ann in self.annotations]))

        self.has_annotation = self.meta is not None
        self.has_split = self.split is not None

    def read_rgb(self, scale=False):
        rgb =  imageio.imread(self.paths['rgb'])
        return rgb

    def read_hyper(self, scale=False):
        hyper = imageio.imread(self.paths['hyper']).transpose(1, 2, 0)
        return hyper

    def get_instance_labels(self):
        # Return inviduals boolean masks for each object
        shape = (self.meta['height'], self.meta['width'])

        masks = []
        masks_categories = []

        for ann in self.annotations:
            if ann['category'] in self.dataset_categories:
                category = self.dataset_categories.index(ann['category'])
            else:
                category = 0

            if type(ann['segmentation']) == dict:
                # segmentation is in RLE format
                assert shape == tuple(ann['segmentation']['size'])
                mask = utils.annotations.coco_rle_to_mask(ann['segmentation'])
            elif type(ann['segmentation']) == list:
                # segmentation is in polygon format
                mask = utils.annotations.coco_polygon_to_mask(ann['segmentation'])

            masks.append(mask)
            masks_categories.append(category)

        return dict(masks=masks, categories=masks_categories)

    def get_semantic_labels(self):
        # Return a single segmentation image with all the objects
        shape = (self.meta['height'], self.meta['width'])

        instance_labels = self.get_instance_labels()
        if instance_labels:
            return utils.annotations.semantic_from_instance(instance_labels)
        else:
            return np.zeros(shape, dtype=np.uint8)


class SpectralWasteDataset:
    def __init__(self, dataset_path: str, annotations_path: str = None, splits_path: str = None):
        self.dataset_path = Path(dataset_path)

        if annotations_path is not None:
            self.categories, self.meta, self.annotations, self.palette = self._load_json_annotations(annotations_path)
            self.num_categories = len(self.categories)
            self.has_annotations = True
        else:
            self.has_annotations = False

        if splits_path is not None:
            self.splits_names, self.splits = self._load_splits(splits_path)
            self.has_splits = True
        else:
            self.has_splits = False

    def _load_json_annotations(self, annotations_path):
        """
        Load annotations for a JSON file stored in the COCO format
        https://opencv.org/introduction-to-the-coco-dataset/
        """

        coco = json.load(open(annotations_path, 'r'))
        categories = {cat['id']: cat['name'] for cat in coco['categories']}
        palette = [cat['color'] for cat in coco['categories']]

        meta = {}
        annotations = {}
        for image in coco['images']:
            id = image['id']
            assert len(id) == 18

            # Get list of annotations for image
            image_anns = list(filter(lambda a: a['image_id'] == id, coco['annotations']))

            # replace category_id for category name in the annotations
            for ann in image_anns:
                ann['category'] = categories[ann['category_id']]

            meta[id] = image
            annotations[id] = image_anns

        return list(categories.values()), meta, annotations, palette

    def _load_splits(self, splits_path):
        """Load a dict that maps each image id to a split"""
        splits = json.load(open(splits_path, 'r'))
        names = list(splits.keys())

        # Expand lists into dictionary
        splits_dict = dict()
        for split, id_list in splits.items():
            for key in id_list:
                splits_dict[key] = split

        return names, splits_dict

    def get_image(self, id: str) -> SpectralWasteImage:
        """Retrieve a single image by its id"""

        if self.has_annotations:
            meta = self.meta.get(id, None)
            annotations = self.annotations.get(id, [])
        else:
            meta = None
            annotations = None

        if self.has_splits:
            split = self.splits.get(id, None)
        else:
            split = None

        # Get image paths from id
        day, seq, _ = id.split('_')
        paths = {
            'rgb': self.dataset_path / day / seq / 'rgb' / f'{id}.jpg',
            'hyper': self.dataset_path / day / seq / 'hyper' / f'{id}.tiff'
        }

        return SpectralWasteImage(id, paths, meta, annotations, split, self.categories)

    def get_images(self, filter: Optional[Callable[[SpectralWasteImage], bool]] = None) -> list[SpectralWasteImage]:
        """Retrieve all images in the dataset"""
        ids = (map(lambda path: path.stem, self.dataset_path.glob('*/*/rgb/*')))
        images = [self.get_image(id) for id in ids]
        if filter:
            return [image for image in images if filter(image)]
        else:
            return images

    # Annotated images

    def get_labeled_images(self, filter: Optional[Callable[[SpectralWasteImage], bool]] = None) -> list[SpectralWasteImage]:
        """Retrieve only annotated images"""
        if not self.has_annotations:
            raise ValueError('The dataset does not have annotations')

        labeled_images = [self.get_image(id) for id in self.meta.keys()]
        if filter:
            return [image for image in labeled_images if filter(image)]
        else:
            return labeled_images

    # Splits

    def get_images_in_split(self, split: str, filter: Optional[Callable[[SpectralWasteImage], bool]] = None) -> list[SpectralWasteImage]:
        """Retrieve images in a split"""
        if not self.has_splits:
            raise ValueError('The dataset does not have splits')

        ids = [k for k, s in self.splits.items() if s == split]
        images_in_split = [self.get_image(id) for id in ids]
        if filter:
            return [image for image in images_in_split if filter(image)]
        else:
            return images_in_split

    # Categories color palette

    def get_palette(self, categories = None):
        if not self.has_annotations:
            raise ValueError('The dataset does not have annotations')
        return [utils.plotting.hex_to_rgb(color) for color in self.palette]