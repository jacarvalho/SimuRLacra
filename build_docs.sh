#! /bin/sh
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Build the Pyrado documentation (Sphinx)
cd $ROOT_DIR/Pyrado/doc
#rm -rf build
sphinx-apidoc -o source -t source/templates ..
cd source
for file in pyrado.*.rst; do
  if [ -e "$file" ]; then
    newname=`echo "$file" | sed 's/^pyrado.\(.*\)\.rst$/\1.rst/'`
    mv "$file" "$newname"
  fi
done
cd ..
sphinx-build -b html source build

# Build the RcsPySim documentation (Doxygen)
cd $ROOT_DIR/RcsPySim/build
make doc
