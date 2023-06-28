# KatoML

C++ machine learning library built from scratch that does not depends on any other external library.
(unless you use GPU, then we can't avoid using CUDA)

It contains three core modules:
* mltensor: dynamically typed tensor math library with numpy-like interface that supports multiple backends (e.g. CPU and GPU) 
* mlcompiler: automatic differentiation library that supports various graph optimization passes in intermediate representations (IR) level
* mlapp: high-level machine learning library built on top of mlcompiler and mltensor 

I wrote this project for my own educational purpose, but they are powerful and robust enough to run (some cool model). (in fact, it's around 4000 lines of c++ code...)
