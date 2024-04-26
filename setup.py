"""
Erik R Holmgren
2023-07-24
simtrk python - setup.py
"""

from setuptools import setup, find_packages

setup(
    name="tda",
    version="0.0.1",
    packages=find_packages(),
    license="Copyright 2024 Erik R Holmgren All Rights Reserved",
    description="Tracking and Data Association",
    long_description="Rotines for target simulation, measurement creation, tracking, and state estimation."
)