# setup.py
from setuptools import setup, find_packages

setup(
    name='tensorgraph',
    version='0.1',
    description='Your custom neural network framework',
    author='Pierre Mulder',
    packages=find_packages(include=['tensorgraph', 'tensorgraph.*']),  # Automatically finds tensorgraph and subpackages
    install_requires=[
        'numpy',
        'mlx',  # or mlx-core if you’re using it
        'pyvis',
        # 'tensorflow', only if you want to compare directly
    ],
    python_requires='>=3.8',
)
