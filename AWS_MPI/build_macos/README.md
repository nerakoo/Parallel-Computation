# MacOS Guide

## Lib Install

Use brew to install openmpi; Use ports to install netcdf (don't use brew as it won't link with wave260)

```sh
brew install openmpi
brew install llvm    (to get clang with openmp support)
port install netcdf
```

## GDB

```sh
brew install gdb
```

Then follow [this guide](https://dev.to/jasonelwood/setup-gdb-on-macos-in-2020-489k) to give gdb a certificate so we can do remote process debug.
