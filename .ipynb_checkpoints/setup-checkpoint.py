from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='data_mining_ml',
    author='sueyama',
    license='MIT',
    install_requires=_requires_from_file('requirements.txt')
)
