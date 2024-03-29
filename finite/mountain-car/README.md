# [Mountain Car](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)

The environment used is `MountainCar-v0` with the state space discretized where each dimension is split into linear bins.
This can be changed in the ``env.py`` file.

## Generating a policy

We recommend to either directly download a model from ``rl-zoo`` or to train it yourself, the latter having the advantage of beign compatible with your current versions of dependencies.

To download the model: 
```
python -m rl_zoo3.load_from_hub --algo dqn --env MountainCar-v0 -f logs/ -orga sb3
```

To train a model from scratch:

```
python -m rl_zoo3.train --algo dqn --env MountainCar-v0 -f logs/
```

The model file that will be used in the next step can be found at ``logs/dqn/<env>/<env>.zip``

## Converting the DQN model obtained into a decision tree

This will evaluate each tree built on 100 episodes and this will build a tree for every method available for the finite state space case:

```
python -m polext finite/mountain-car/env.py logs/dqn/MountainCar-v0_1/best_model.zip all 100 --finite --depth 7
```

You can also change the maximum allowed depth with ``--depth new_max``.