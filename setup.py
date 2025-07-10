from setuptools import find_packages, setup

setup(
    name="TARLoco",
    version="0.1.0",
    author="Amr Mousa",
    author_email="amrmousa.m@gmail.com",
    description="TARLoco - Extensions and RL implementations for quadruped robots based on Isaacsim.",
    packages=find_packages(include=["exts.tarloco", "exts.tarloco.*"]),
    install_requires=[
        "torch>=2.5.1",
        "gymnasium",
        "numpy",
        "torchvision",
        "omegaconf",
        "hydra-core",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: Linux",
    ],
)
