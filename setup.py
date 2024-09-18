from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description = fh.read()

setup(
    name='pereggrn',
    version='0.0.1',
    description='Systematic benchmarking of in silico genetic perturbations',
    long_description=long_description,
	long_description_content_type='text/markdown',
	py_modules=["pereggrn.evaluator.py", "pereggrn.experimenter.py"],
    scripts=['do_one_experiment.py'],
    author='Eric Kernfeld',
    author_email='eric.kern13@gmail.com',
    install_requires=[
        "pandas",
        "numpy",
        "anndata",
        "scanpy",
		"memray",
        "pereggrn_perturbations",
        "pereggrn_networks",
        "joblib",
        "scipy",
        "altair",
		"vl-convert-python",
        "scikit-learn",
        "pyyaml",
        "ggrn",
		"ray[tune]",
    ],
    python_requires=">=3.7", 
    url='https://github.com/ekernf01/pereggrn',
    entry_points={
        'console_scripts': [
            'pereggrn = do_one_experiment:main'
        ]
    }
)
