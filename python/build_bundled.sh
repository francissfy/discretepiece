#!/bin/sh

mkdir -p build

BUILD_DIR=./build
INSTALL_DIR=./build/root

if [ -f ./discretepiece/src/CMakeLists.txt ]; then
  SRC_DIR=./discretepiece
elif [ -f ../src/CMakeLists.txt ]; then
  SRC_DIR=..
else
  echo "unexpected error"
  exit 1
fi

cmake ${SRC_DIR} -B ${BUILD_DIR} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}
cmake --build ${BUILD_DIR} --config Release --target install --parallel 4

# for later python wrapper:
# cd src/discretepiece
# swig -c++ -python spm_client.i
# mv discretepiece.py __init__.py