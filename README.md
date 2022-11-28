# Policy Extraction from Q-Values

This is the repository for the code of the paper TODO.

**Authors**:
[Théo Matricon](https://theomat.github.io/), [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/)

<!-- toc -->
Table of contents:

- [Abstract](#abstract)
- [Usage](#usage)
  - [Installation](#installation)
  - [Reproducing the experiments](#reproducing-the-experiments)
- [Citing](#citing)

<!-- tocstop -->

TODO SOME FIGURE HERE

## Abstract

ABSTRACT HERE

## Usage

### Installation

```bash
# clone this repository

# First install polext
pip install .
# Or if you want dev install
pip install -e .

# Now to reproduce most experiments, you can either install the dependencies individually depending on what you are missing
# or just use
pip install -r requirements.txt
```

### Reproducing the experiments

Please see each ``README`` file in the corresponding folders:

| Environment  | State Space | Policy trained with rl-zoo | Folder |
|--------------|:-----:|:-----------:|:---------|
| [Acrobot-v1](https://www.gymlibrary.dev/environments/classic_control/acrobot/)    | Infinite  |        Yes | `infinite/acrobot` |
| [Cart-Pole-v1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)    | Finite    |        No  | `finite/cart-pole` |
| [Lunar-Lander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)    | Infinite  |        Yes | `infinite/lunar-lander` |
| [Mountain-Car-v0](https://www.gymlibrary.dev/environments/classic_control/mountain_car/) | Finite    |        Yes | `finite/mountain-car` |
| [Pong](https://www.gymlibrary.dev/environments/atari/pong/)            | Infinite  |        Yes | `infinite/pong` |

## Citing

TODO
