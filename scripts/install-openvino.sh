#! /bin/bash

if [[ `lsb_release -a 2> /dev/null  | grep 16` ]]
then
    package=intel-openvino-dev-ubuntu16
else
    package=intel-openvino-dev-ubuntu18
fi

if ! apt show $package 2> /dev/null
then
    wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020  -O gpg-openvino-key 
    apt-key add gpg-openvino-key 
    echo "deb https://apt.repos.intel.com/openvino/2020 all main" > /etc/apt/sources.list.d/intel-openvino-2020.list
    apt update
    apt-get install -y $package
fi

folders=$(find /opt/intel/openvino/deployment_tools | grep \.so$ | xargs -I {} dirname {} | sort | uniq)
echo "$folders" > /etc/ld.so.conf.d/openvino.conf
ldconfig

