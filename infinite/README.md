# Comparisons

<!-- toc -->
Table of contents:

- [Comparing with dtControl](#comparing-with-dtcontrol)
- [Comparison with RL on predicate space MDP](#comparison-with-rl-on-predicate-space-mdp)

<!-- tocstop -->

## Comparing with dtControl

[dtControl](https://dtcontrol.readthedocs.io/en/latest/index.html) can be installed with:

```bash
pip install dtcontrol
```

To run [dtControl](https://dtcontrol.readthedocs.io/en/latest/index.html) you first need to produce a CSV model file to convert the neural network to input that can be digested using:

```bash
python infinite/dt_control.py infinite/<env>/env.py <model_file> -o my_dt_control_input.csv
```

You can then simply run [dtControl](https://dtcontrol.readthedocs.io/en/latest/index.html) with the input obtained:

```bash
dtcontrol --input my_dt_control_input.csv
```

## Comparison with RL on predicate space MDP

You should run the following command:

```bash
python infinite/discrete_rl.py infinite/<env>/env.py
```
