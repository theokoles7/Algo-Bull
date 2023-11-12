import os
from setuptools import find_packages, setup

setup(
    name =              "algo-bull",
    version =           "1.0.0",
    author =            "Gabriel C. Trahan",
    author_email =      "gabriel.trahan1@louisiana.edu",
    description =       (
        "Algorithmic trading automation."
    ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/Algo-Bull",
    long_description =  open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    packages =          find_packages()
)