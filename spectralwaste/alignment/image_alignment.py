from spectralwaste import utils

def multimodal_manual_align(rgb, hyper, rgb_labels, hyper_labels, resize_size=(600, 600)):
    """
    Manually align the rgb and labels image to the hyper
    """

    assert rgb.shape == (1200, 1184, 3)
    assert hyper.shape == (600, 640, 224)

    rgb_align = rgb[:1190, :]
    hyper_align = hyper[:, :605]

    rgb_align = utils.image_processing.resize(rgb_align, resize_size)
    hyper_align = utils.image_processing.resize(hyper_align, resize_size)

    if rgb_labels is not None:
        if type(rgb_labels) is dict:
            # instance labels
            assert all([mask.shape == (1200, 1184) for mask in rgb_labels['masks']])
            masks_align = [mask[:1190, :] for mask in rgb_labels['masks']]
            masks_align = [utils.image_processing.resize(mask, resize_size, anti_aliasing=False) for mask in masks_align]
            rgb_labels_align = dict(masks=masks_align, categories=rgb_labels['categories'])
        else:
            # semantic labels
            assert rgb_labels.shape == (1200, 1184)
            rgb_labels_align = rgb_labels[:1190, :]
            rgb_labels_align = utils.image_processing.resize(rgb_labels_align, resize_size, anti_aliasing=False)
    else:
        rgb_labels_align = None

    if hyper_labels is not None:
        if type(hyper_labels) is dict:
            # instance labels
            assert all([mask.shape == (600, 640) for mask in hyper_labels['masks']])
            masks_align = [mask[:, :605] for mask in hyper_labels['masks']]
            masks_align = [utils.image_processing.resize(mask, resize_size, anti_aliasing=False) for mask in masks_align]
            hyper_labels_align = dict(masks=masks_align, categories=hyper_labels['categories'])
        else:
            # semantic labels
            assert hyper_labels.shape == (600, 640)
            hyper_labels_align = hyper_labels[:, :605]
            hyper_labels_align = utils.image_processing.resize(hyper_labels_align, resize_size, anti_aliasing=False)
    else:
        hyper_labels_align = None

    return rgb_align, hyper_align, rgb_labels_align, hyper_labels_align