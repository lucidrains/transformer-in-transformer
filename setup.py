from setuptools import setup, find_packages

setup(
  name = 'transformer-in-transformer',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Transformer in Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/transformer-in-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformer',
    'image classification'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
