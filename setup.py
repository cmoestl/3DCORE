# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mfr3dcore",
    version="0.0",
    author="Andreas J. Weiss",
    author_email="andreas.weiss@oeaw.ac.at",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IWF-helio/3DCORE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
