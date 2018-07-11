from setuptools import setup

with open('requirements.txt') as fle:
    dependencies = fle.readlines()

setup(name='Fuzzy K-modes Fuzzy Centroids',
      version='0.1',
      description='A clustering algorithm for categorical data',
      author='Jim Lemmers',
      author_email='shout@jimlemmers.com',
      licenses='BSD',
      packages=['fuzzykmodesfuzzycentroids'],
      install_requires=dependencies,
      zip_safe=False)
