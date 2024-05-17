# KatoML

C++ machine learning library built from scratch that does not depends on any other external ML related library.
(in fact, only libraries it uses are STL, pybind11, and CUDA)

It contains two core C++ modules:
* mltensor: dynamically typed tensor math library with numpy-like interface that supports multiple backends (e.g. CPU and GPU) 
* mlcompiler: automatic differentiation library that supports various graph optimization passes in intermediate representations (IR) level

KatoML provides python binding as well so that the power of KatoML can be leveraged through python.

## Yuzu

This is a high-level python library that actually implements various machine learning techniques. There is an option to specify which ML framework it's going to use: either KatoML or PyTorch.

### Gallery

- [ ] 
- [ ] PuyoPuyo AI
- [ ] Chess AI
- [ ] Tetris AI
- [x] Cliff Walking
- [x] Taxi
- [x] Moon Lander
- [x] Atari Breakout
- [x] MNIST Letters
- [ ] MNIST Cloths

### Reinforcement Learning

- [x] Proximal Policy Optimization (PPO)
 - Orthogonal Initialization
 - Continuous Rollout Buffer
 - Entropy Loss
- [x] Deep Q Network (DQN)
 - Dual Deep Q Network (DDQN)
- [x] Vanilla Policy Gradient
 - "Reward to go"
 - Discounted Factor
- [x] Actor and Critic
- [x] Value Iteration

[![Build Status](https://github.com/sunho/KatoML/actions/workflows/test.yml/badge.svg)](https://github.com/sunho/KatoML/actions/workflows/test.yml)
