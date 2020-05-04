#!/bin/sh

# Assumptions:
# Equal user names and paths on both machines

# Usage:
# (Assuming you are in PROJECT_DIR/remotelaunch)
# bash remotelaunch_TEMPLATE.sh python PROJECT_DIR/Pyrado/scripts/training/SCRIPT_NAME.py

CMD="$@"

DSTHOST="..." # ADD NAME OF THE COMPUTER
PROOT="..." # ADD PATH TO PROJECT ROOT DIR

RCS_SRC_DIR="$PROOT/SimuRLacra/Rcs" # path to Rcs source dir
RCS_BUILD_DIR="$PROOT/SimuRLacra/Rcs/build" # path to Rcs build dir

MD_SRC_DIR="$PROOT/SimuRLacra"
RLAUNCH_DIR="$MD_SRC_DIR/remotelaunch"

RCSPYSIM_SRC_DIR="$MD_SRC_DIR/RcsPySim"
RCSPYSIM_BUILD_DIR="$RCSPYSIM_SRC_DIR/build"

# Synchronize code
$RLAUNCH_DIR/sync_to_host.sh $DSTHOST "$RCS_SRC_DIR" 
$RLAUNCH_DIR/sync_to_host.sh $DSTHOST "$MD_SRC_DIR"

# Specify the activation script
ACTIVATION_SCRIPT="activate_pyrado.sh"

# Now, run all this on the remote host
ssh -t -t $DSTHOST << EOF
shopt -s expand_aliases

mkdir -p "$RCS_BUILD_DIR"
cd "$RCS_BUILD_DIR"
cmake "$RCS_SRC_DIR"
make -j8

cd "$MD_SRC_DIR"
source "$ACTIVATION_SCRIPT"

mkdir -p "$RCSPYSIM_BUILD_DIR"
cd "$RCSPYSIM_BUILD_DIR"
cmake "$RCSPYSIM_SRC_DIR/build"
make -j8

$CMD

exit
EOF
