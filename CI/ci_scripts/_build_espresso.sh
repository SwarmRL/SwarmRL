#!/usr/bin/bash

build_espresso () {
  cd espresso
  mkdir build
  cd build
  cmake ../
  cmake --build .
}

rm -r espresso/*
git submodule init
git submodule update
build_espresso
