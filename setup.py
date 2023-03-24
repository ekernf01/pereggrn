from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description = fh.read()

setup(
    name='perturbation_benchmarking_package',
    version='0.0.1',
    description='Systematic benchmarking of in silico genetic perturbations',
    long_description=long_description,
	long_description_content_type='text/markdown',
    #url
    author='Eric Kernfeld',
    author_email='eric.kern13@gmail.com',
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "anndata",
        "scanpy",
    ],
    python_requires=">=3.7", 
    url='https://github.com/ekernf01/perturbation_benchmarking_package',
)