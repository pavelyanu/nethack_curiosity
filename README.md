# Nethack Curiosity

## Installation

- clone the repository with the submodules:
```
git clone --recursive https://github.com/pavelyanu/nethack_curiosity.git
```
- install nle dependencies
```
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit
conda install -c conda-forge pybind11
```
- Install NLE and sample-factory as editable
```
cd externals/nle
pip install -e .
cd externals/sample-factory
pip install -e .[nethack]
pip install -e sf_examples/nethack/nethack_render_utils
```
- install nethack-curiosity
```
pip install -e .
```

## Running experiments

Experiments can be run through `experiments/nethack/run_nethack.py` python module.
