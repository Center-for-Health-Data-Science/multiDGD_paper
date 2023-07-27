from setuptools import setup

setup(name='omicsdgd',
      version="0.0",
      description='omics implementation of the encoder-less representation learning using MAP estimation from https://arxiv.org/abs/2110.06672',
      author='Viktoria Schuster',
      licence='MIT',
      #url='https://github.com/Center-for-Health-Data-Science/DGD_universe',
      packages=['omicsdgd','omicsdgd.dataset', 'omicsdgd.latent', 'omicsdgd.nn', 'omicsdgd.functions'],
      #package_dir={'dgd': 'src/dgd'}
      #install_requires=['torch>=1.10']
     )