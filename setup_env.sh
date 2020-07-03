#! /bin/bash
conda init

source ~/.bashrc

conda create -n pyrado python=3.7 blas cmake colorama coverage cython joblib lapack libgcc-ng mkl matplotlib-base numpy pandas patchelf pip pycairo pytest pytest-cov pytest-xdist pyyaml scipy seaborn setuptools sphinx sphinx-math-dollar sphinx_rtd_theme tabulate tqdm -c conda-forge

conda activate pyrado

conda run -n pyrado /bin/bash -c "pip install git+https://github.com/Xfel/init-args-serializer.git@master argparse box2d glfw gym pprint pytest-lazy-fixture vpython"
