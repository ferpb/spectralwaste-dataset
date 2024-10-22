from spectralwaste.dataset import SpectralWasteDataset
from spectralwaste import alignment, utils

import torch
import torchmetrics

dataset_rgb = SpectralWasteDataset('../data/dataset', '../data/metadata/annotations_rgb_iros2024.json')
dataset_hyper = SpectralWasteDataset('../data/dataset', '../data/metadata/annotations_hyper_iros2024.json')

transfer_model = alignment.LabelTransferModel(verbose=False)

iou_ma_list = []
iou_lt_list = []

for hyper_image in dataset_hyper.get_labeled_images():
    rgb_image = dataset_rgb.get_image(hyper_image.id)

    rgb = hyper_image.read_rgb()
    hyper = hyper_image.read_hyper()

    rgb_instance = rgb_image.get_instance_labels()
    hyper_instance = hyper_image.get_instance_labels()

    rgb_semantic = rgb_image.get_semantic_labels()
    hyper_semantic = hyper_image.get_semantic_labels()

    rgb_align, hyper_align, rgb_instance_align, hyper_instance_align = alignment.multimodal_manual_align(rgb, hyper, rgb_instance, hyper_instance)
    _, _, rgb_semantic_align, hyper_semantic_align = alignment.multimodal_manual_align(rgb, hyper, rgb_semantic, hyper_semantic)

    hyper_align_color = utils.image_processing.false_color(hyper_align)
    transferred_instance = transfer_model.transfer_instance_labels(rgb_align, hyper_align_color, rgb_instance_align)
    transferred_semantic = utils.annotations.semantic_from_instance(transferred_instance)

    # compare semantic IoU
    iou_ma = torchmetrics.functional.jaccard_index(torch.tensor(rgb_semantic_align), torch.tensor(hyper_semantic_align), num_classes=dataset_rgb.num_categories, task='multiclass', average='none')
    iou_lt = torchmetrics.functional.jaccard_index(torch.tensor(transferred_semantic), torch.tensor(hyper_semantic_align), num_classes=dataset_rgb.num_categories, task='multiclass', average='none')

    print('iou_ma', iou_ma.numpy())
    print('iou_lt', iou_lt.numpy())
    print()

    iou_ma_list.append(iou_ma)
    iou_lt_list.append(iou_lt)

iou_ma = torch.vstack(iou_ma_list)
iou_ma = iou_ma.sum(dim=0) / (iou_ma != 0).sum(dim=0)
print(iou_ma)
print(iou_ma[1:].mean())

iou_lt = torch.vstack(iou_lt_list)
iou_lt = iou_lt.sum(dim=0) / (iou_lt != 0).sum(dim=0)
print(iou_lt)
print(iou_lt[1:].mean())