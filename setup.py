import setuptools

setuptools.setup(
    name='spectralwaste_dataset',
    version=0.1,
    author="",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy<2.0',
        'matplotlib',
        'imageio',
        'easydict',
        'scikit-image',
        'opencv-python',
    ],
)