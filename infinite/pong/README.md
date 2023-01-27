# [Pong](https://www.gymlibrary.dev/environments/atari/pong/)

The environment used is `Pong-ram-v4` with predicates that were hand designed.
This can be changed in the ``env.py`` file.

## Generating a policy

We recommend to train it yourself, the latter having the advantage of beign compatible with your current versions of dependencies.


To train a model from scratch:

```
python -m rl_zoo3.train --algo dqn --env Pong-ram-v4 -f logs/ --yaml ./infinite/pong/config.yml
```

The model file that will be used in the next step can be found at ``logs/dqn/<env>/<env>.zip``

## Converting the DQN model obtained into a decision tree

This will evaluate each tree built on 200 episodes and this will build a tree for every method available for the infinite state space case:

```
python -m polext infinite/pong/env.py logs/dqn/Pong-ram-v4_1/best_model.zip all 100 --depth 10 --iterations 2
```

You can also change the maximum allowed depth with ``--depth new_max``.
