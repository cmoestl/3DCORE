#!/usr/bin/env bash

install_folder=$(pwd)

for kernel in $(find mfr3dcore/spice/kernels/* -maxdepth 1 -type d); do
    if [ -f $kernel/kernellist_url.txt ]; then
        while IFS='' read -r line || [ "$line" ]; do
            echo -ne "Downloading $line"
            cd $kernel
            curl $line -O -s
            cd $install_folder
            echo " ($(echo $(wc -c <"$kernel/$(basename $line)") | sed 's/^[ \t]*//;s/[ \t]*$//') bytes)"
        done < $kernel/kernellist_url.txt
    fi
done