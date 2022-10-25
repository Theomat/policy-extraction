# Lunar Lander experiment

Once the setup is done, the experiment can be done by executing the ``experiment.sh`` script.

<!-- toc -->
Table of contents:

- [Setup](#setup)
- [Generating a Policy](#generating-a-policy)
- [Extracting a Decision Tree from a Policy](#extracting-a-decision-tree-from-a-policy)

<!-- tocstop -->

## Setup

```bash
# install swig (dependency for Box2D gym environments)
sudo apt install swig

# install dependencies
pip install -r requirements.txt
```

## Generating a Policy

Executing:

```bash
python generate_policy.py my_dqn.pt --env "LunarLander-v2"
```

will output a `my_dqn.pt` model file.

## Extracting a Decision Tree from a Policy

You can extract the policy using:

```bash
python -m polext extract.py my_dqn.pt
```

TODO test

