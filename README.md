# KatoML

C++ machine learning library built from scratch that does not depends on any other external library.
(unless you use GPU, then we can't avoid using CUDA)

It contains three core modules:
* mltensor: dynamically typed tensor math library with numpy-like interface that supports multiple backends (e.g. CPU and GPU) 
* mlcompiler: automatic differentiation library that supports various graph optimization passes in intermediate representations (IR) level
* mlapp: high-level machine learning library built on top of mlcompiler and mltensor 

# CI

[![Build Status](https://github.com/sunho/KatoML/actions/workflows/test.yml/badge.svg)](https://github.com/sunho/KatoML/actions/workflows/test.yml)