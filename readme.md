## Introduction

This repository provides a PyTorch-based reimplementation of the paper **"Mean Field Reinforcement Learning"**. The original implementation, available [here](https://github.com/mlii/mfrl), relies on the outdated MAgent library and TensorFlow 1.0. This project modernizes the approach using **PyTorch** and the **MAgent2** library, which is compatible with the PettingZoo API.

## Installation

This project requires **Python 3.10** or newer (other versions may work but are untested). To set up the environment, follow these steps:

1. Install the **MAgent2** library via `pip`:
    ```bash
    pip install magent2
    ```

2. Fix the known compatibility issue with PettingZoo:
    ```bash
    pip install pettingzoo==1.22.3
    ```

After completing these steps, the code should function correctly on **macOS**, **Linux**, and **Windows** platforms.

## Compatibility

This implementation has been verified to run successfully across multiple operating systems. If you encounter issues, ensure all dependencies are correctly installed and meet the required versions.
