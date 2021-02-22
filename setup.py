import os.path
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name="datashift",
    version="0.0.2",
    description="Lightweight and generic data processor that allows quickly filtering, balancing and processing a data set from one form to another.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/MarcinStachowiak/datashift",
    author="Marcin Stachowiak",
    author_email="marcin@predictforce.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    keywords=['Pipeline', 'Data Processing', 'Preprocessing'],
    packages=find_packages(),
    install_requires=[
        'certifi==2020.12.5',
        'dill==0.3.3',
        'multiprocess==0.70.11.1',
        'numpy==1.20.1',
        'pandas==1.2.2',
        'python-dateutil==2.8.1',
        'pytz==2021.1',
        'PyYAML==5.4.1',
        'six==1.15.0'
    ]
)
