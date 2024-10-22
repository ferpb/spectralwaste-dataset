# SpectralWaste Dataset

The SpectralWaste dataset contains sequences of multimodal RGB and hyperspectral images captured in a recycling facility. Each image is identified with the date, number of sequence and time it was captured, with the format `yyyymmdd-xx-HHMMSS`, and stored in directory structure like the following:

```
dataset
 +-- 20220928
      +-- 01
           |-- rgb
           |    +-- 20220928_01_105147.jpg
           +-- hyper
                +-- 20220928_01_105147.tiff
```

The dataset contains 7655 multimodal images, and 852 of those are annotated with object segmentation masks. You can download the dataset in the following links:

* Labeled dataset
     * RGB and HSI images (105 GB): [OneDrive][labeled-dataset]
* Complete dataset
     * RGB images (3.8 GB): [OneDrive][complete-rgb]
     * RGB and HSI images (1.3 TB): Please contact the [authors](mailto:fpena@unizar.es,acm@unizar.es).

The annotations are stored following the [COCO format](https://cocodataset.org/#format-data) for object detection. We provide two JSON files, one containing annotations for the RGB images and another with a reduced set of annotations for the hyperspectral images.

[labeled-dataset]: https://unizares-my.sharepoint.com/:f:/g/personal/756012_unizar_es/EkzyB3aciG1GncKAHUdh_sEB2Ch8kGwxLTyvLBHdKTiM_Q?e=feRBh6
[complete-rgb]: https://unizares-my.sharepoint.com/:u:/g/personal/756012_unizar_es/EWFDxj3rv1JFnPFVLCd0ePwBV3hsps2nwX84nq3orIgHzw?e=wtXoXC


## Installation
The repository requires Python 3.9+ and compatibility with PyTorch 2.0+ and `torchvision` 0.16+.

You can install the code in a virtual environment running:

```
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

In order to use the `spectralwaste.alignment` module it is required to install [COTR](https://github.com/ubc-vision/COTR) (which is included in this respository) and download its pretrained weights:

```
pip install -e third_party/COTR
curl https://www.cs.ubc.ca/research/kmyi_data/files/2021/cotr/default.zip -O
unzip default.zip third_party/COTR/out
```