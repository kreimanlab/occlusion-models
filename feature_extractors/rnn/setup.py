from setuptools import setup

setup(name='occlusion-models',
      version='0.1',
      description='Models for acquiring good features of occluded images',
      url='http://github.com/mschrimpf/occlusion-models',
      author='Martin Schrimpf',
      author_email='martin.schrimpf@outlook.com',
      license='MIT',
      install_requires=[
          'keras',
          'numpy',
          'scipy',
          'sklearn',
          'hdf5storage'
      ])
