#!/bin/sh

# Run this script on the local machine.
# Two argument: 1) the target host; 2) the directory to sync

if [ "$#" -ne 2 ]
  then
    echo "Missing host or directory argument"
    exit
fi
# Store arguments
DSTHOST="$1"
SYNCDIR="$2"

SRC="$SYNCDIR/"
DST="$DSTHOST:$SYNCDIR/"

echo "$SRC"

# Use rsync
# Archive, compress, progress, ssh algo, delete removed
# Exclude git and SVN files
rsync -azPe ssh --delete \
    --exclude "Pyrado/data/evaluation" \
    --exclude "Pyrado/data/time_series" \
    --exclude "Pyrado/data/training/temp" \
    --exclude "thirdParty/" \
    --exclude "build/" \
    --exclude ".git/" \
    --exclude ".svn/" \
    --exclude "__pycache__" \
    --exclude-from="$(git -C "$SRC" ls-files --exclude-standard -oi --directory > /tmp/excludes; echo /tmp/excludes)" \
    "$SRC" "$DST"


