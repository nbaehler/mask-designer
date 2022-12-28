import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mask-designer",
    version="0.0.1",
    author="Nicolas Bähler",
    author_email="nicolas.bahler@epfl.ch",
    description="Package to perform mask design for SLMs",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/nbaehler/mask-designer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "requests",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "scikit-image",
        "aotools",
        "opencv-python",
        "tensorboardx",
    ],
    extra_requires={
        "dev": [
            "click",
            "black",
            "pytest",
            "tensorboard",
            "torch_tb_profiler",
            "sphinx-rtd-theme",
            "docutils==0.16",
        ],
    },
)
