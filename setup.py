from setuptools import setup, find_packages

setup(
    name='pyfoil',
    version='0.0',
    description='GPflow/Tensorflow implementation of mixture of Gaussian process experts - uses sparse GPs and stochastic variational inference',
    project_urls={"repository": "https://github.com/ss1870/pyfoil"},
    author='Samuel Scott',
    author_email='ss1870@my.bristol.ac.uk',
    license='Apache-2.0',
    # packages=find_packages()
    packages=['pyfoil'],
)