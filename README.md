# atflagger

A simple flagger for continuum UWL data. Flag persistent RFI first, then run this auto-flagger.

## Installation

Installing requires `pip` and `python3` (3.8 and up).

Stable version:
```
pip install atflagger
```

Latest version:
```
pip install git+https://github.com/AlecThomson/atflagger
```

## Usage
```
❯ atflagger -h
usage: atflagger [-h] [--beam BEAM] [--sigma SIGMA] [--n_windows N_WINDOWS] filenames [filenames ...]

Flag SDHDF data

positional arguments:
  filenames             Input SDHDF file(s)

optional arguments:
  -h, --help            show this help message and exit
  --beam BEAM           Beam label
  --sigma SIGMA         Sigma clipping threshold
  --n_windows N_WINDOWS
                        Number of windows to use in box filter
```
