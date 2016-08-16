from setuptools import setup
import vcversioner
from os import path

# Get the current directory
cur_dir = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(cur_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='knpackage',
    version='0.1.2',
    #version=vcversioner.find_version().version,
    #setup_requires=['vcversioner'],
    #vcversioner={
    #    'vcs_args': ['git', 'describe', '--long'],
    #},
    include_dev_version=True,
    description='KnowEng toolbox',
    long_description=long_description,
    url='https://github.com/KnowEnG-Research/KnowEnG_Pipelines_Library',
    author='KnowEng Dev',
    author_email='knowengdev@gmail.com',
    license='MIT',
    packages=['knpackage'],
    zip_safe=False)
