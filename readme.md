## Introduction

This repository offers a PyTorch-based reimplementation of the paper **"Mean Field Multi-Agent Reinforcement Learning"**. The original implementation, available [here](https://github.com/mlii/mfrl), relied on the outdated MAgent library and TensorFlow 1.0. This project modernizes the approach by utilizing **PyTorch** and the **MAgent2** library, which is fully compatible with the PettingZoo API.

## Installation

This project requires **Python 3.10** or newer (other versions may work but are untested). Follow these steps to set up the environment:

1. Install the **MAgent2** library using `pip`:
    ```bash
    pip install magent2
    ```

2. Resolve a known compatibility issue with PettingZoo (you may see a version conflict warning during installation; it can be safely ignored):
    ```bash
    pip install pettingzoo==1.22.3
    ```

After completing these steps, the code should work seamlessly on **macOS**, **Linux**, and **Windows** platforms.

## Environment Support

The project currently supports the following environments:

1. **Battle**: Blue agents represent the "self_model," while red agents represent the "oppo_model."
2. **Combined Arms**: Green and black agents represent "self_melee" and "self_ranged," respectively, while red and blue agents represent "oppo_melee" and "oppo_ranged."

For detailed information about these environments, please refer to the [MAgent2 documentation](https://magent2.farama.org).

## Usage
The training script are "train_battle.py" and "train_combined_arms.py". You can run them with the following command:
```bash
python train_battle.py
```
```bash
python train_combined_arms.py
```
Note that my code will train the "self_model" and "oppo_model" in the same time. If you want to train them separately, you can modify the code in "train_battle.py" and "train_combined_arms.py".