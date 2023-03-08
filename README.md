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
usage: atflagger [-h] [-i] [-b BEAM] [-s SIGMA] [-n N_WINDOWS] [-w] [-r REPORT] [-c CORES] [-t THREADS_PER_WORKER] filenames [filenames ...]

atflagger - Automatic flagging of UWL data. This flagger divides each subband into a number of windows, and then uses sigma clipping to remove outliers. The number of windows is set by the 'n-windows' argument, and the number of sigma is set by the 'sigma' argument. Parallelism is handled by dask.distributed. The 'cores' argument sets the number of
Dask workers, and 'threads-per-worker' sets the number of threads. See https://docs.dask.org/en/stable/deploying-python.html#reference for more information.

positional arguments:
  filenames             Input SDHDF file(s)

optional arguments:
  -h, --help            show this help message and exit
  -i, --inplace         Update flags in-place (default: create new file)
  -b BEAM, --beam BEAM  Beam label
  -s SIGMA, --sigma SIGMA
                        Sigma clipping threshold
  -n N_WINDOWS, --n-windows N_WINDOWS
                        Number of windows to use in box filter
  -w, --use-weights     Use weights table instead of flag table
  -r REPORT, --report REPORT
                        Optionally save the Dask (html) report to a file
  -c CORES, --cores CORES
                        Number of workers to use (default: Dask automatic configuration)
  -t THREADS_PER_WORKER, --threads-per-worker THREADS_PER_WORKER
                        Number of threads per worker (default: Dask automatic configuration)
```
