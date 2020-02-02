import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepscribe",
    version="0.1",
    python_requires='>=3',
    author="Edward Williams",
    author_email="eddiecwilliams@gmail.com",
    description="Character recognition on OCHRE cuneiform data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edwardclem/deepscribe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scikit-learn',
        'numpy',
        'matplotlib',
        'tqdm',
        'luigi',
        'h5py', #NOTE: opencv should be here, but conda and pip are fighting.
        'pillow',
        'tensorflow'
    ]
)
