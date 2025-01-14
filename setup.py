from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="PANNA",
    version="2.1.0",
    description='Properties from Artificial Neural Network Architectures',
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=['numpy', 'mendeleev', 'scipy'],
    extras_require={
        "TF": ['tensorflow>=2'],
        "JAX": ['tensorflow-cpu', 'jax[cuda12]', 'dm-haiku', 'optax'],
    },
    entry_points={
        'console_scripts': [
            'panna_train=panna.train:main', 'panna_evaluate=panna.evaluate:main',
            'panna_gvect_calculator=panna.gvect_calculator:main',
            'panna_tfr_packer=panna.tfr_packer:main',
            'panna_extract_weights=panna.extract_weights:main',
            'panna_jax=panna.train_jax:main'
        ],
    },
    author="",
    author_email="",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
