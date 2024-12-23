## Introduction

This repo contains the reimplementation of the paper "mean field reinforcement learning" in PyTorch, the original repo is [here](https://github.com/mlii/mfrl). 

The original repo uses MAgent library and tensorflow 1.0, which is outdated. This repo uses PyTorch and MAgent2 library(which uses pettingzoo api) to reimplement the paper.

## Installation
This repo based on MAgent2 library, pytorch and python 3.10(maybe also work in later python version), you can use ```pip``` to install it:
```bash
pip install magent2
```
However, there is a bug when you install magent2, magent2 will install a wrong version of pettingzoo library, you need to install the correct version of pettingzoo library manually:
```bash
pip install pettingzoo==1.22.3
```
After that, this code may work.(This code should work well in macos,linux and windows)