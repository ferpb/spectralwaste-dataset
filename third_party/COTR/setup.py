import setuptools

setuptools.setup(
    name='COTR',
    python_requires=">=3.8",
    packages=setuptools.find_packages(),
    install_requires=[
        'tables',
        'tqdm',
        'torch==2.2.2',
        'torchvision==0.17.2'
    ]
)