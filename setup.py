from setuptools import setup
import vcversioner

setup(name='knpackage',
      version=vcversioner.find_version().version,
      description='KnowEng toolbox',
      url='https://github.com/KnowEnG-Research/KnowEnG_Pipelines_Library',
      author='KnowEng Dev',
      author_email='knowengdev@gmail.com',
      license='MIT',
      packages=['knpackage'],
      zip_safe=False)
