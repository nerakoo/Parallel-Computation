# (WSL) Ubuntu Guide

## Install

```sh
sudo apt update
sudo apt install libopenmpi-dev openmpi-bin openmpi-common cmake libnetcdf-dev ncview g++
```

## Set Up CMake

```sh
cd build_wsl
cmake ..
```

## Compile

```sh
cmake --build .
# or
make
```