from setuptools import setup, find_packages

setup(
    name="finding_fraudsters",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train-model=source.train_model:main',
        ],
    },
)
