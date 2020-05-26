#! /usr/bin/env python

import os
import os.path as osp
import shutil
import subprocess as sp
from urllib.request import urlretrieve
import tarfile
import zipfile
import tempfile
import errno
import sys
import argparse
import yaml

# Get the project's root directory
project_dir = osp.dirname(osp.abspath(__file__))

# Make sure the git submodules are up to date, otherwise this script might break them
sp.check_call(["git", "submodule", "update", "--init"], cwd=project_dir)

# Check if we are in HRI by looking for the SIT envionment variable
IN_HRI = 'SIT' in os.environ

# ================== #
# PARSE ARGS EAGERLY #
# ================== #
# Allows to use them in the configuration

parser = argparse.ArgumentParser(description='Setup RcsPySim dev env')
parser.add_argument('tasks', metavar='task', type=str, nargs='*', help='Subtasks to execute. Suggested tasks are `all` (includes every feature) or `no_rcs` (excludes Rcs and RcsPysim). To get a list of all availibe tasks, run `python setup_deps.py`.')
parser.add_argument('--vortex', action='store_true', default=False, help='Use vortex physics engine')
parser.add_argument('--use-cuda', dest='usecuda', action='store_true', default=False, help='Use CUDA for PyTorch')
parser.add_argument('--headless', action='store_true', help='Build in headless mode')
parser.add_argument('--local-torch', dest='uselibtorch', action='store_true', default=True, help='Use the local libtorch from the thirdParty directory')
parser.add_argument('--no-local-torch', dest='uselibtorch', action='store_false', help='Use the local libtorch from the thirdParty directory')
parser.add_argument('-j', default=1, type=int, help='Number of make threads')

args = parser.parse_args()
# Check for help print later, when the tasks are defined

# ====== #
# CONFIG #
# ====== #

# Common directories
dependency_dir = osp.join(project_dir, "thirdParty")
resources_dir = osp.join(dependency_dir, "resources")

# Global cmake prefix path
cmake_prefix_path = [
    # Anaconda env root directory
    os.environ['CONDA_PREFIX']
]

# Required packages
required_packages = [
    "g++-4.8",
    "qt5-default",
    "libqwt-qt5-dev",
    "libbullet-dev",
    "libfreetype6-dev",
    "libxml2-dev",
    "libglu1-mesa-dev",
    "freeglut3-dev",
    "mesa-common-dev",
    "libopenscenegraph-dev",
    "openscenegraph",
    "liblapack-dev"
]

# Environment for build processes
env_vars = {
    # Global cmake prefix path
    "CMAKE_PREFIX_PATH": ":".join(cmake_prefix_path)
}

# Number of threads for make
make_parallelity = args.j

# WM5
wm5_download_url = 'https://www.geometrictools.com/Downloads/WildMagic5p17.zip'
wm5_src_dir = osp.join(dependency_dir, "WildMagic5")

wm5_config = "ReleaseDynamic"
wm5_modules = ["LibCore", "LibMathematics"]

wm5_include_dir = osp.join(wm5_src_dir, "SDK/Include")
wm5_library_dir = osp.join(wm5_src_dir, "SDK/Library", wm5_config)

# Rcs
rcs_src_dir = osp.join(project_dir, "Rcs")
rcs_build_dir = osp.join(rcs_src_dir, "build")
rcs_cmake_vars = {
    "USE_BULLET": "2.83_float",
    "ENABLE_C++11": "ON",
    # Eigen is off for now until the dependency issues are fixed in Rcs.
    # Must specify include dir for Eigen 3.2
    #"EIGEN3_INCLUDE_DIR": eigen_include_dir,
    #"USE_EIGEN": "ON",
    "USE_WM5": "ON",  # for advanced collision models
}
# Optional headless mode
if args.headless:
    rcs_cmake_vars["HEADLESS_BUILD"] = "ON"

# Bullet
if IN_HRI:
    # Use bullet double from SIT
    rcs_cmake_vars["USE_BULLET"] = "2.83_double"
else:
    # Bullet from package is in float mode
    rcs_cmake_vars["USE_BULLET"] = "2.83_float"

# Vortex
if args.vortex:
    # Add to rcs dependencies
    rcs_cmake_vars["USE_VORTEX"] = "ESSENTIALS"

# WM5 collision library
if not IN_HRI:
    rcs_cmake_vars["WM5_INCLUDE_DIR"] = wm5_include_dir
    rcs_cmake_vars["WM5_LIBRARY_DIR"] = wm5_library_dir

# Kuka iiwa meshes
iiwa_repo_version = "1.2.5"
iiwa_url = f"https://github.com/IFL-CAMP/iiwa_stack/archive/{iiwa_repo_version}.tar.gz"

# Schunk SDH meshes
sdh_repo_version = "0.6.14"
sdh_url = f"https://github.com/ipa320/schunk_modular_robotics/archive/{sdh_repo_version}.tar.gz"

# Barrett WAM meshes (Pyrado)
wam_repo_version = "354c6e9"
wam_url = f"https://github.com/psclklnk/self-paced-rl/archive/{wam_repo_version}.tar.gz"

# PyTorch
# NOTE: Assumes that the current environment does NOT already contain PyTorch!
pytorch_version = "1.3.1"
pytorch_git_repo = "https://github.com/pytorch/pytorch.git"
pytorch_src_dir = osp.join(dependency_dir, "pytorch")

# RcsPySim
rcspysim_src_dir = osp.join(project_dir, "RcsPySim")
rcspysim_build_dir = osp.join(rcspysim_src_dir, "build")
uselibtorch = "ON" if args.uselibtorch else "OFF"
rcspysim_cmake_vars = {
    "PYBIND11_PYTHON_VERSION": "3.6",
    "SETUP_PYTHON_DEVEL": "ON",
    "Rcs_DIR": rcs_build_dir,
    "USE_LIBTORCH": uselibtorch,  # use the manually build PyTorch from thirdParty/pytorch 
    # Do NOT set CMAKE_PREFIX_PATH here, it will get overridden later on.
}

# Pyrado
pyrado_dir = osp.join(project_dir, "Pyrado")


# ======= #
# HELPERS #
# ======= #


def mkdir_p(path):
    """ Create directory and parents if it doesn't exist. """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def downloadAndExtract(url, destdir, archiveContentPath=None):
    """ Download an archive and extract it to the given destination. """
    # Select archive format
    if url.endswith(".tar.gz"):
        suffix = ".tar.gz"
        path_attr = "path"
    elif url.endswith(".zip"):
        suffix = ".zip"
        path_attr = "filename"
    else:
        raise ValueError("Unsupported archive file: {}".format(url))
    
    if osp.exists(destdir) and len(os.listdir(destdir)) != 0:
        # Exists, skip
        return
    with tempfile.NamedTemporaryFile(suffix=suffix) as tf:
        print("Downloading {}...".format(url))
        urlretrieve(url, tf.name)

        print("Extracting {}...".format(url))

        # Ensure destdir exists
        mkdir_p(destdir)

        # Filter
        if archiveContentPath is not None:
            # We only want to extract one subdirectory
            # Taken from https://stackoverflow.com/a/43094365
            def members(ml):
                subfolder = osp.normpath(archiveContentPath)
                l = len(subfolder)
                for member in ml:
                    # Skip directories in zip
                    isdir = getattr(member, 'is_dir', None)
                    if isdir and isdir():
                        continue

                    # Modify output path
                    path = getattr(member, path_attr)
                    rp = osp.relpath(path, subfolder)
                    if not rp.startswith(".."):
                        setattr(member, path_attr, rp)
                        yield member
        else:
            def members(ml):
                return ml

        if suffix == ".tar.gz":
            with tarfile.open(tf.name) as tar:
                tar.extractall(members=members(tar.getmembers()), path=destdir)
        else:
            with zipfile.ZipFile(tf.name) as zip:
                zip.extractall(members=members(zip.infolist()), path=destdir)


def buildCMakeProject(srcDir, buildDir, cmakeVars=None, env=env_vars, install_dir=None):
    """
    cd buildDir
    cmake srcDir -D...
    make
    (make install)
    """
    # Ensure build dir exists
    mkdir_p(buildDir)

    if env is not None:
        fullenv = dict(os.environ)
        fullenv.update(env)
        env = fullenv

    # Execute CMake command
    cmake_cmd = ["cmake", osp.relpath(srcDir, buildDir)]
    if cmakeVars is not None:
        for k, v in cmakeVars.items():
            if v is True:
                vstr = "ON"
            elif v is False:
                vstr = "OFF"
            else:
                vstr = v
            cmake_cmd.append("-D{}={}".format(k, vstr))
    if install_dir is not None:
        cmake_cmd.append("-DCMAKE_INSTALL_PREFIX={}".format(install_dir))
    sp.check_call(cmake_cmd, cwd=buildDir, env=env)

    # Execute make (build) command
    make_cmd = ["make", "-j{}".format(make_parallelity)]
    sp.check_call(make_cmd, cwd=buildDir, env=env)

    # Execute install command if desired
    if install_dir is not None:
        mkdir_p(install_dir)
        sp.check_call(["make", "install"], cwd=buildDir)

# =========== #
# SETUP TASKS #
# =========== #


def setup_dep_libraries():
    # Update
    sp.check_call(["sudo", "apt-get", "update", "-y"])
    # Install dependencies
    sp.check_call(["sudo", "apt-get", "install", "-y"] + required_packages)


def setup_wm5():
    # Download the sources
    downloadAndExtract(wm5_download_url, wm5_src_dir, "GeometricTools/WildMagic5")
    
    # Build relevant modules
    for module in wm5_modules:
        sp.check_call([
            "make", 
            "-f", 
            "makefile.wm5", 
            "build", 
            "CFG={}".format(wm5_config),
            "-j{}".format(make_parallelity),
        ], cwd=osp.join(wm5_src_dir, module))


def setup_rcs():
    # Build Rcs. We already have it in the submodule    
    buildCMakeProject(rcs_src_dir, rcs_build_dir, cmakeVars=rcs_cmake_vars)


def setup_pytorch():
    # Get PyTorch from git
    if not osp.exists(pytorch_src_dir):
        mkdir_p(pytorch_src_dir)
        sp.check_call(["git", "clone", "--recursive", "--branch", "v{}".format(pytorch_version), pytorch_git_repo, pytorch_src_dir])
    # Let it's setup do the magic
    env = os.environ.copy()
    env.update(env_vars)
    env["USE_CUDA"]="1" if args.usecuda else "0"  # CUDA is disabled by default
    env["USE_MKLDNN"]="0" # disable MKLDNN; mkl/blas deprecated error https://github.com/pytorch/pytorch/issues/17874
    env["_GLIBCXX_USE_CXX11_ABI"]="1"
    sp.check_call([sys.executable, "setup.py", "install"], cwd=pytorch_src_dir, env=env)


def setup_rcspysim():
    # Take care of RcsPySim
    buildCMakeProject(rcspysim_src_dir, rcspysim_build_dir, cmakeVars=rcspysim_cmake_vars)


def setup_iiwa():
    # The Kuka iiwa meshes
    downloadAndExtract(iiwa_url, osp.join(resources_dir, "iiwa_description"), f"iiwa_stack-{iiwa_repo_version}/iiwa_description")
    
    # Copy the relevant mesh files into RcsPySim's config folder
    # We already have the .tri meshes in there, just gives them company.
    src_dir = osp.join(resources_dir, "iiwa_description/meshes/iiwa14")
    dst_dir = osp.join(rcspysim_src_dir, "config/iiwa_description/meshes/iiwa14")
    
    # Collision and visual for links 0 - 7
    print("Copying the KUKA iiwa meshes to the RcsPySim config dir ...")
    for catdir in ['collision', 'visual']:
        for lnum in range(8):
            fname = osp.join(catdir, f"link_{lnum}.stl")
            
            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the KUKA iiwa meshes is done.")


def setup_schunk():
    # The Schunk SDH meshes
    downloadAndExtract(sdh_url, osp.join(resources_dir, "schunk_description"), f"schunk_modular_robotics-{sdh_repo_version}/schunk_description")
    
    # Copy the relevant mesh files into RcsPySim's config folder
    # We already have the .tri meshes in there, just gives them company.
    src_dir = osp.join(resources_dir, "schunk_description/meshes/sdh")
    dst_dir = osp.join(rcspysim_src_dir, "config/schunk_description/meshes/sdh")
    
    # Get all .stl files in the sdh subdir
    print("Copying the Schunk SDH meshes to the RcsPySim config dir ...")
    for fname in os.listdir(src_dir):
        if fname.endswith(".stl"):
            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the Schunk SDH meshes is done.")


def setup_wam():
    # Barrett WAM meshes (Pyrado)
    downloadAndExtract(wam_url, osp.join(resources_dir, "wam_description"), f"self-paced-rl-{wam_repo_version}/sprl/envs/xml/")
    
    # Copy the relevant mesh files into Pyrados's MuJoCo environments folder
    src_dir = osp.join(resources_dir, "wam_description/meshes")
    dst_dir = osp.join(pyrado_dir, "pyrado/environments/mujoco/assets/meshes/barrett_wam")

    # Get all .stl files in the wam subdir
    print("Copying the Barrett WAM meshes to the Pyrado assets dir ...")
    for fname in os.listdir(src_dir):
        if fname.endswith(".stl"):
            mkdir_p(osp.dirname(osp.join(dst_dir, fname)))
            shutil.copyfile(osp.join(src_dir, fname), osp.join(dst_dir, fname))
    print("Setting up the Barrett WAM meshes is done.")


def setup_meshes():
    # Set up all external meshes
    setup_iiwa()
    setup_schunk()
    setup_wam()


def setup_pyrado():
    # Set up Pyrado in development mode
    sp.check_call([sys.executable, "setup.py", "develop"], cwd=osp.join(project_dir, 'Pyrado'))


def setup_mujoco_py():
    # Set up mujoco-py (doing it via pip caused problems on some machines)
    sp.check_call([sys.executable, "setup.py", "install"], cwd=osp.join(project_dir, 'thirdParty', 'mujoco-py'))


def setup_pytorch_based():
    # Set up GPyTorch, BoTorch, Pyro without touching the PyTorch installation
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "gpytorch"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "botorch"])
    sp.check_call([sys.executable, "-m", "pip", "install", "-U", "--no-deps", "pyro-ppl"])


def setup_separate_pytorch():
    # We could do setup_dep_libraries() here, but it requires sudo rights
    if not IN_HRI:
        setup_wm5()
    setup_rcs()
    rcspysim_cmake_vars.update({"USE_LIBTORCH": "OFF"})  # don't use the local PyTorch but the one from anaconda/pip
    setup_rcspysim()
    setup_meshes()
    setup_mujoco_py()
    setup_pyrado()
    print("\nWM5, Rcs, RcsPySim, iiwa & Schunk & WAM meshes, mujoco-py, and Pyrado are set up!\n")


def setup_no_rcs():
    # Rcs will still be downloaded since it is a submodule
    setup_pytorch()
    setup_wam()  # ignoring the meshes used in RcsPySim
    setup_pyrado()
    setup_mujoco_py()
    setup_pytorch_based()
    print("\nPyTorch, RcsPySim, WAM meshes, mujoco-py, Pyrado (with GPyTorch, BoTorch, and Pyro) are set up!\n")


def setup_all():
    # We could do setup_dep_libraries() here, but it requires sudo rights
    if not IN_HRI:
        setup_wm5()
    setup_rcs()
    setup_pytorch()
    setup_rcspysim()
    setup_meshes()
    setup_pyrado()
    setup_mujoco_py()
    setup_pytorch_based()
    print("\nWM5, Rcs, PyTorch, RcsPySim, iiwa & Schunk & WAM meshes, mujoco-py, Pyrado (with GPyTorch, BoTorch, and Pyro) are set up!\n")
    

# All tasks list
tasks_by_name = {
    name[6:] : v  # cut the "setup_"
    for name, v in globals().items() if name.startswith('setup_')
}

# ==== #
# MAIN #
# ==== #
    
# Print help if none
if len(args.tasks) == 0:
    print("Available tasks:")
    for n in tasks_by_name.keys():
        print("  {}".format(n))

# Execute selected tasks
for task in args.tasks:
    tasks_by_name[task]()
