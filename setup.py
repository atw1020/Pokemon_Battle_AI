'''

built out of IBM example program

'''

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Pokemon Battling Deep Learning AI',
    version='1.0.0',
    description='A Pokemon Battleing AI',
    long_description=long_description,
    url='https://github.com/ArthurTGW/Pokemon_Battle_AI',
    license='Apache-2.0'
)
