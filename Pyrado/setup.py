from setuptools import setup, find_packages

setup(
    name='pyrado',
    version='0.1',
    description='A framework for reinforcement learning with domain randomization',
    author='Fabio Muratore & Felix Treede & Robin Menzenbach',
    author_email='muratore@ias.tu-darmstadt.de',
    # Specify empty py_modules to exclude pkgConfig.py
    py_modules=[],
    packages=find_packages(include=['pyrado', 'pyrado.*']),
    # Specify external packages as dependencies
    # Just added to shut down the warnings (either fulfilled by the anaconda environment of setup_deps.py)
    # install_requires=['numpy', 'torch', 'gym', 'pytest', 'matplotlib', 'pandas', 'joblib', 'optuna', 'tabulate', 'scipy', 'pyro', 'botorch', 'gpytorch'],
)
