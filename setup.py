from setuptools import setup, find_packages

setup(
    name             = "aria-cl",
    version          = "1.0.0",
    author           = "Darshan Poudel",
    author_email     = "susanpdl77@gmail.com",
    description      = "ARIA — Adaptive Recurrent Intelligence Architecture for continual learning",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url              = "https://github.com/rsd-darshan/ARIA",
    packages         = find_packages(exclude=["tests*", "scripts*", "examples*"]),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "matplotlib>=3.6",
    ],
    extras_require   = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ]
    },
    classifiers      = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
