conda activate pyrado

# Add the path to the RcsPySim libary in the build directory
export PYTHONPATH=PATH_TO/RcsPySim/build/lib:$PYTHONPATH

# Add the path to the rcsenv module in oder to find it form other projects outside of RcsPySim
export PYTHONPATH=PATH_TO/RcsPySim/src/python:$PYTHONPATH

# Add the path to your Pyrado folder
export PYTHONPATH=PATH_TO/Pyrado/:$PYTHONPATH
