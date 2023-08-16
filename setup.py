import os
from setuptools import find_packages, setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name =              "trading-bot",
    version =           "1.0.0",
    author =            "Gabriel C. Trahan",
    author_email =      "gabriel.trahan1@louisiana.edu",
    description =       (
        "Algorithmic trading automation."
    ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/Trading-Bot",
    long_description =  read('README.md'),
    packages =          find_packages()
)