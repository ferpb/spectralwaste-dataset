import numpy as np
import json
from pathlib import Path
import imageio.v3 as imageio

from spectralwaste.dataset import SpectralWasteDataset, SpectralWasteImage
from spectralwaste import alignment, utils


def preprocess_labeled_image(image: SpectralWasteImage, transfer_model: alignment.LabelTransferModel, output_path: Path):
    # Save:
    #   * downsampled rgb
    #   * downsampled hyper
    #   * downsampled rgb labels
    #   * downsampled rgb labels transfered to hyper
    # Images are stored in their original format (8 bits for rgb and 16 bits for hyper)

    rgb = image.read_rgb()
    hyper = image.read_hyper()
    rgb_instance = image.get_instance_labels()

    # Manual alignment
    rgb_align, hyper_align, rgb_instance_align, _ = alignment.multimodal_manual_align(rgb, hyper, rgb_instance, None)

    # Downsampling
    rgb_down = utils.image_processing.resize(rgb_align, (256, 256))
    hyper_down = utils.image_processing.resize(hyper_align, (256, 256))

    # Process labels
    if len(rgb_instance['masks']) != 0:
        # Label transfer
        hyper_align_color = utils.image_processing.false_color(hyper_align)
        hyper_instance_lt = transfer_model.transfer_instance_labels(rgb_align, hyper_align_color, rgb_instance_align)

        # Convert instance
        rgb_semantic_img = utils.annotations.semantic_from_instance(rgb_instance_align)
        rgb_instance_img = utils.annotations.merge_instances(rgb_instance_align)
        hyper_semantic_img = utils.annotations.semantic_from_instance(hyper_instance_lt)
        hyper_instance_img = utils.annotations.merge_instances(hyper_instance_lt)

        # Downsampling
        rgb_semantic_img_down = utils.image_processing.resize(rgb_semantic_img, (256, 256), anti_aliasing=False)
        rgb_instance_img_down = utils.image_processing.resize(rgb_instance_img, (256, 256), anti_aliasing=False)
        hyper_semantic_img_down = utils.image_processing.resize(hyper_semantic_img, (256, 256), anti_aliasing=False)
        hyper_instance_img_down = utils.image_processing.resize(hyper_instance_img, (256, 256), anti_aliasing=False)
    else:
        rgb_semantic_img_down = np.zeros((256, 256), dtype=np.uint8)
        rgb_instance_img_down = np.zeros((256, 256), dtype=np.uint16)
        hyper_semantic_img_down = np.zeros((256, 256), dtype=np.uint8)
        hyper_instance_img_down = np.zeros((256, 256), dtype=np.uint16)

    # Store images
    rgb_dir = output_path / 'rgb' / image.split
    hyper_dir = output_path / 'hyper' / image.split
    labels_rgb_dir = output_path / 'labels_rgb' / image.split
    labels_hyper_lt_dir = output_path / 'labels_hyper_lt' / image.split

    rgb_dir.mkdir(parents=True, exist_ok=True)
    hyper_dir.mkdir(parents=True, exist_ok=True)
    labels_rgb_dir.mkdir(parents=True, exist_ok=True)
    labels_hyper_lt_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(rgb_dir / f'{image.id}.png', rgb_down)
    imageio.imwrite(hyper_dir / f'{image.id}.tiff', hyper_down.transpose(2, 0, 1))
    imageio.imwrite(labels_rgb_dir / f'{image.id}_semantic.png', rgb_semantic_img_down)
    imageio.imwrite(labels_rgb_dir / f'{image.id}_instance.png', rgb_instance_img_down)
    imageio.imwrite(labels_hyper_lt_dir / f'{image.id}_semantic.png', hyper_semantic_img_down)
    imageio.imwrite(labels_hyper_lt_dir / f'{image.id}_instance.png', hyper_instance_img_down)


def preprocess_unlabeled_image(image: SpectralWasteImage, output_path: Path):
    # Save:
    #   * downsampled rgb
    #   * downsampled hyper
    # Images are stored in their original format (8 bits for RGB and 16 bits for hyper)

    rgb = image.read_rgb()
    hyper = image.read_hyper()

    # Manual alignment
    rgb_align, hyper_align, _, _ = alignment.multimodal_manual_align(rgb, hyper, None, None)

    # Downsampling
    rgb_down = utils.image_processing.resize(rgb_align, (256, 256))
    hyper_down = utils.image_processing.resize(hyper_align, (256, 256))

    rgb_path = output_path / 'rgb' / 'unlabeled' / f'{image.id}.png'
    hyper_path = output_path / 'hyper' / 'unlabeled' / f'{image.id}.tiff'

    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    hyper_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(rgb_path, rgb_down)
    imageio.imwrite(hyper_path, hyper_down.transpose(2, 0, 1))

def main():
    dataset = SpectralWasteDataset(
        '/data/spectralwaste/dataset',
        '/data/spectralwaste/metadata/annotations_rgb_iros2024.json',
        '/data/spectralwaste/metadata/splits_iros2024.json'
    )

    output_path = Path('out/spectralwaste_segmentation')

    transfer_model = alignment.LabelTransferModel(verbose=False)

    for image in dataset.get_images():
        if image.has_annotation:
            print(image.id, 'labeled')
            preprocess_labeled_image(image, transfer_model, output_path)
        else:
            print(image.id, 'unlabeled')
            preprocess_unlabeled_image(image, output_path)

    # Store metadata
    meta = {
        'categories': dataset.categories,
        'palette': dataset.get_palette(),
    }
    json.dump(meta, open(output_path / 'meta.json', 'w'))

if __name__ == "__main__":
    main()
