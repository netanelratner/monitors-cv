#! /bin/bash

set -eax

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ `python --version | grep 3.7` ]]
then
OPENVINO_VERSION='/opt/intel/openvino/python/python3.7/'
fi

if [[ `python --version | grep 3.6` ]]
then
OPENVINO_VERSION='/opt/intel/openvino/python/python3.6/'
fi

if [[ `python --version | grep 3.5` ]]
then
OPENVINO_VERSION='/opt/intel/openvino/python/python3.5/'
fi

tmp_dir=$(mktemp -d -t openvino-XXXXXXXXXX)
mkdir $tmp_dir/openvino
cp -R $OPENVINO_VERSION/* $tmp_dir/
cp -R $DIR/setup/* $tmp_dir
cd $tmp_dir
pip wheel .
pip install openvino*.whl
cp openvino*.whl $DIR/..
ls $tmp_dir