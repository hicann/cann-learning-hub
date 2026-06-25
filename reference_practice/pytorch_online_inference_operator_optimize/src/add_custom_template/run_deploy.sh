#!/bin/bash
rm -rf build
mkdir -p build && cd build
cmake .. && make -j binary package
./custom_opp_*.run --install-path=${HOME}/
