# Policy Extraction from Q-Values

This is the repository for the code of the paper TODO.

**Authors**:
[Théo Matricon](https://theomat.github.io/), [Nathanaël Fijalkow](https://nathanael-fijalkow.github.io/)

TODO SOME FIGURE EHRE

## Abstract

ABSTRACT HERE

## Usage

### Installation

```bash
# clone this repository

# First install polext
cd polext
poetry install
cd ..
```

### Reproducing the experiments

1. First, move into the experiment folder and install the requirements with:

```bash
pip install -r requirements.txt
```

2. You need to generate the policies, if there is no ``.pt`` file with:

```bash
python generate_policy.py <file_name>.pt
```

3. You can now extract the policy with:

```bash
python -m polext extract.py <file_name>.pt
```

TODO

## Citing

TODO
