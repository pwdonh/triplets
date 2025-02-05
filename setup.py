from setuptools import find_packages, setup

setup(
    name='triplets',
    version='0.1.0',
    packages=find_packages(),
    scripts=[
        'scripts/fit_triplets.py',
        'scripts/fit_triplets_crossval.py',
        'scripts/recon_embedding.py'
    ],
)
