# [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

The environement used is `CartPole-v1` with the state space discretized where each dimension is split into linear bins.
This can be changed in the ``env.py`` file.

## Generating a policy

```
python finite/cart-pole/generate_policy.py cart_pole.pt
```

## Converting the DQN model obtained into a decision tree

This will evaluate each tree built on 100 episodes and this will build a tree for every method available for the finite state space case:

```
python -m polext finite/cart-pole/env.py cart_pole.pt all 100 --finite
```

You can also change the maximum allowed depth with ``--depth new_max``.
