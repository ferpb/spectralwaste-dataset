import numpy as np
import skimage

# We assume that both hyperspectral and RGB images have channels last

# /!\ When normalizing per channel, the scale of the radiance measurements will be different
# in each channel

def resize(input: np.ndarray, size, anti_aliasing=True, axis=-1):
    # move the axis (channels) to the last position
    if axis is not None and axis != -1:
        input = np.moveaxis(input, axis, -1)
    # Resize maintaining the data type
    result = skimage.transform.resize(input, size, preserve_range=True, order=0, anti_aliasing=anti_aliasing)
    result = result.astype(input.dtype)
    # restore axis position
    if axis is not None and axis != -1:
        result = np.moveaxis(result, -1, axis)
    return result

def max_value(dtype: np.dtype) -> int:
    if issubclass(dtype.type, np.integer):
        return np.iinfo(dtype).max
    else:
        return 1

def normalize(method, image: np.array, **kw_args):
    if method == 'max':
        return normalize_max(image, **kw_args)
    elif method == 'l1':
        return normalize_l1(image, **kw_args)
    elif method == 'minmax':
        return normalize_min_max(image, **kw_args)
    elif method == 'percentile':
        return normalize_percentile(image, **kw_args)
    else:
        raise KeyError(f'Unknown normalization method: {method}')

def normalize_max(image: np.array):
    max = max_value(image.dtype)
    return image.astype(np.float32) / max

def normalize_l1(image: np.array):
    with np.errstate(invalid='ignore'):
        hyper = hyper / np.linalg.norm(image, ord=1, axis=-1, keepdims=True)
        return np.nan_to_num(image, copy=False)

def normalize_min_max(image: np.array, per_channel=True):
    if per_channel:
        min_v = np.amin(image, axis=(0,1), keepdims=True)
        max_v = np.amax(image, axis=(0,1), keepdims=True)
    else:
        min_v = image.min()
        max_v = image.max()
    return (image - min_v) / (max_v - min_v)

def normalize_percentile(image, q_min=5, q_max=95, per_channel=True, clip=True):
    if per_channel:
        min_v = np.percentile(image, q_min, axis=(0,1), keepdims=True)
        max_v = np.percentile(image, q_max, axis=(0,1), keepdims=True)
    else:
        min_v = np.percentile(image, q_min)
        max_v = np.percentile(image, q_max)

    image = (image - min_v) / (max_v - min_v)
    if clip: image = np.clip(image, 0, 1)
    return image

def false_color(hyper):
    if hyper.shape[-1] <= 3:
        return hyper
    bands = hyper[:, :, [134, 170, 200]]
    bands_norm = normalize_percentile(bands, per_channel=True)
    return (bands_norm * 255).astype(np.uint8)