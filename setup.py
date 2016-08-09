from setuptools import setup
import vcversioner
from os import path

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
long_description = f.read()

setup(
     name='knpackage',
      version=vcversioner.find_version().version,
      description='KnowEng toolbox',
      long_description=long_description,
      url='https://github.com/KnowEnG-Research/KnowEnG_Pipelines_Library',
      author='KnowEng Dev',
      author_email='knowengdev@gmail.com',
      license='MIT',
      packages=['knpackage'],
      zip_safe=False)
