"""A setuptools based setup module."""

from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pytorch-ialgebra',
    version=open('VERSION').readline(),
    description='interactive interpretation for deep learning models', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='',
    author_email='',  # Optional
    keywords='interactive interpretation for deep learning models',
    packages=[
        'ialgebra',
        'ialgebra.interpreters',
        'ialgebra.models',
        'ialgebra.operations',
        'ialgebra.utils',
    ],
    python_requires='>=3.0.*',
    install_requires=[
        'argparse',
        'argparse',
        'httpserver',
        'threding',
        'webbrowser',
        'os',
        'sys',
        'pyyaml',
        'numpy',
        'tqdm',
        'collections',
        'cv2',
        'torch',
        'torchvision',
        'matplotlib',
        'visdom',
    ],
    project_urls='https://pypi.org/project/pytorch-ialgebra/',
)
