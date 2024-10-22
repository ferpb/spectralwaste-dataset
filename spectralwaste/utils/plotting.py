import matplotlib.pyplot as plt
import numpy as np


def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))

def get_color_labels(labels, palette):
    palette_array = np.array(palette, dtype=np.uint8)
    color_labels = palette_array[labels]
    return color_labels

def plot_labels(image, labels, palette, ax=None):
    if not ax:
        ax = plt.gca()
    color_labels_alpha = np.dstack([get_color_labels(labels, palette), np.zeros_like(labels)])
    color_labels_alpha[labels > 0, 3] = 180
    ax.imshow(image)
    ax.imshow(color_labels_alpha, interpolation='none')
    ax.axis('off')