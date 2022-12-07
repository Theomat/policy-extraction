# [Lunar Lander](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)

The environement used is `LunarLander-v2` with the state space discretized where each dimension is split into linear bins.
This can be changed in the ``env.py`` file.

## Generating a policy

We recommend to either directly download a model from ``rl-zoo`` or to train it yourself, the latter having the advantage of beign compatible with your current versions of dependencies.

To download the model:
```
python -m rl_zoo3.load_from_hub --algo dqn --env LunarLander-v2 -f logs/ -orga sb3
```

To train a model from scratch:

```
python -m rl_zoo3.train --algo dqn --env LunarLander-v2 -f logs/
```

The model file that will be used in the next step can be found at ``logs/dqn/<env>/<env>.zip``

## Converting the DQN model obtained into a decision tree

This will evaluate each tree built on 100 episodes and this will build a tree for every method available for the infinite state space case:

```
python -m polext infinite/lunar-lander/env.py logs/dqn/LunarLander-v2_1/best_model.zip all 100 --depth 12 --iterations 2
```

You can also change the maximum allowed depth with ``--depth new_max``.
