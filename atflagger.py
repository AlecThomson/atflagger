#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
atflagger - Automatic flagging of UWL data.

This flagger divides each subband into a number of windows, and then uses
sigma clipping to remove outliers. The number of windows is set by the
'n-windows' argument, and the number of sigma is set by the 'sigma' argument.

Parallelism is handled by dask.distributed. The 'cores' argument sets the
number of Dask workers, and 'threads-per-worker' sets the number of threads.
See https://docs.dask.org/en/stable/deploying-python.html#reference for more
information.

"""
import datetime
import logging
import shutil
import socket
import warnings
from contextlib import nullcontext

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import xarray as xr
from astropy.stats import mad_std, sigma_clip
from astropy.table import QTable, Table
from dask import compute, delayed
from dask.distributed import Client, LocalCluster, performance_report
from IPython import embed
from tqdm.auto import tqdm, trange

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.setLevel(logging.INFO)


class AutoFlagError(Exception):
    def __init__(self, msg):
        self.msg = msg


@delayed()
def copy_file(filename, ext="hdf"):
    new_filename = filename[::-1].replace(
        f".{ext}"[::-1],
        f".atflagged.{ext}"[::-1],
        1,
    )[
        ::-1
    ]  # Reverse to replace .hdf with .atflagged.hdf
    logger.info(f"Copying file {filename}...")
    shutil.copy(filename, new_filename)
    logger.info(f"Created new file: {new_filename}")
    return new_filename


def box_filter(spectrum, sigma=3, n_windows=100):
    """
    Filter a spectrum using a box filter.
    """
    # Divide spectrum into windows
    window_size = len(spectrum) // n_windows
    dat_filt = np.zeros_like(spectrum).astype(bool)
    # Iterate through windows
    for i in range(n_windows):
        _dat = spectrum[i * window_size : window_size + i * window_size]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use sigma clipping to remove outliers
            _dat_filt = sigma_clip(
                _dat, sigma=sigma, maxiters=None, stdfunc=mad_std, masked=True
            )
        dat_filt[i * window_size : window_size + i * window_size] = _dat_filt.mask
    return dat_filt


@delayed()
def get_subbands(filename, beam_label="beam_0"):
    with h5py.File(filename, "r") as h5:
        # Read header info
        sb_avail = QTable.read(h5, path=beam_label + "/metadata/band_params")
    return list(sb_avail["LABEL"])


@delayed()
def update_history(filename, args, flagged=True):
    if not flagged:
        raise AutoFlagError("No flagging performed - not updating history")
    logger.info(f"Updating history in {filename}...")
    # Open HDF5 file
    with h5py.File(filename, mode="r+") as f:
        # Read header info
        dset = f["metadata"]["history"]
        history = Table(dset[:])
        history.add_row(
            {
                "DATE": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                "PROC": "atflagger",
                "PROC_DESCR": f"Automatic flagging of UWL data (version {pkg_resources.get_distribution('atflagger').version})",
                "PROC_ARGS": str(args),
                "PROC_HOST": socket.getfqdn(),
            }
        )
        # Datasets cannot be extended, so delete and recreate
        del f["metadata"]["history"]
        _ = f.create_dataset("metadata/history", data=history.as_array())

    logger.info(f"History updated in {filename}")


@delayed()
def flag(
    filename, sb_labels, beam_label="beam_0", sigma=3, n_windows=100, use_weights=False
) -> bool:
    for sb_label in sb_labels:
        logger.info(f"Starting flagging on {filename}, {sb_label}...")
        # Open HDF5 file
        with h5py.File(filename, "r+") as h5:
            # Read header info
            logger.info(f"Processing subband {sb_label} - {filename}")

            # We've decided on 'flags' and 'weights' as the schema
            # This will be enforced here

            # Look for old naming schemes
            old_flag = f"{beam_label}/{sb_label}/astronomy_data/flag" in h5
            new_flag = f"{beam_label}/{sb_label}/astronomy_data/flags" in h5
            old_weights = f"{beam_label}/{sb_label}/astronomy_data/data_weights" in h5
            new_weights = f"{beam_label}/{sb_label}/astronomy_data/weights" in h5

            if old_flag:
                logger.warning(
                    f"Old flagging scheme detected - {filename}:{beam_label}/{sb_label}/astronomy_data/flag."
                )
                if not new_flag:
                    logger.warning(
                        f"Renaming to {beam_label}/{sb_label}/astronomy_data/flags (new scheme)."
                    )
                    h5.copy(
                        f"{beam_label}/{sb_label}/astronomy_data/flag",
                        f"{beam_label}/{sb_label}/astronomy_data/flags",
                    )
                    del h5[f"{beam_label}/{sb_label}/astronomy_data/flag"]
                else:
                    logger.warning(
                        f"Also found {beam_label}/{sb_label}/astronomy_data/flags (new scheme)."
                    )
                    logger.warning(
                        f"Deleting {beam_label}/{sb_label}/astronomy_data/flag (old scheme) and using new scheme."
                    )
                    del h5[f"{beam_label}/{sb_label}/astronomy_data/flag"]

            elif not new_flag:
                raise AutoFlagError(
                    f"No flagging information found for {filename}:{beam_label}/{sb_label} - run persistent flagging first"
                )

            if old_weights:
                logger.warning(
                    f"Old weights scheme detected - {filename}:{beam_label}/{sb_label}/astronomy_data/data_weights."
                )
                if not new_weights:
                    logger.warning(
                        f"Renaming to {beam_label}/{sb_label}/astronomy_data/weights (new scheme)."
                    )
                    h5.copy(
                        f"{beam_label}/{sb_label}/astronomy_data/data_weights",
                        f"{beam_label}/{sb_label}/astronomy_data/weights",
                    )
                    del h5[f"{beam_label}/{sb_label}/astronomy_data/data_weights"]
                else:
                    logger.warning(
                        f"Also found {beam_label}/{sb_label}/astronomy_data/weights (new scheme)."
                    )
                    logger.warning(
                        f"Deleting {beam_label}/{sb_label}/astronomy_data/data_weights (old scheme) and using new scheme."
                    )
                    del h5[f"{beam_label}/{sb_label}/astronomy_data/data_weights"]

            elif not new_weights and use_weights:
                raise AutoFlagError(
                    f"No weights information found for {filename}:{beam_label}/{sb_label} - run persistent flagging first"
                )

            sb_data = f"{beam_label}/{sb_label}/astronomy_data/data"
            sb_flag = (
                f"{beam_label}/{sb_label}/astronomy_data/flags"
                if not use_weights
                else f"{beam_label}/{sb_label}/astronomy_data/weights"
            )
            sb_freq = f"{beam_label}/{sb_label}/astronomy_data/frequency"
            data = h5[sb_data]
            freq = np.squeeze(h5[sb_freq]) # Ensure 1D
            flag = np.array(h5[sb_flag])

            f_per = np.sum(flag) / np.sum(np.ones_like(flag)) * 100
            logger.info(f"Subband {sb_label} has {f_per:.2f}% flagged - {filename}")

            data_xr = xr.DataArray(
                data,
                dims=[
                    d.decode() if isinstance(d, bytes) else d
                    for d in h5[sb_data].attrs["DIMENSION_LABELS"]
                ],
                coords={
                    "frequency": freq,
                },
            )

            # Set flags
            # Ensure flag has same shape as data
            flag_reshape = flag.copy()
            for i, s in enumerate(data.shape):
                if i > len(flag_reshape.shape) - 1:
                    flag_reshape = np.expand_dims(flag_reshape, axis=-1)
                else:
                    if flag_reshape.shape[i] == s:
                        continue
                    else:
                        flag_reshape = np.expand_dims(flag_reshape, axis=i)
            data_xr_flg = data_xr.where(~flag_reshape.astype(bool))
            # Set chunks for parallel processing
            chunks = {d: 1 for d in data_xr_flg.dims}
            chunks["frequency"] = len(data_xr.frequency)
            data_xr_flg = data_xr_flg.chunk(chunks)
            mask = xr.apply_ufunc(
                box_filter,
                data_xr_flg,
                input_core_dims=[["frequency"]],
                output_core_dims=[["frequency"]],
                kwargs={"sigma": sigma, "n_windows": n_windows},
                dask="parallelized",
                vectorize=True,
                output_dtypes=(bool),
            )

            # Reduce mask
            dims = list(data_xr_flg.dims)
            dims.remove("frequency")
            dims.remove("time")
            mask_red = mask.sum(dim=dims) > 0
            logger.info(f"Flagging {sb_label} and writing to file {filename}...")
            # Write flags back to file
            h5[sb_flag][:] = mask_red.values.astype(int)

            f_per = np.sum(h5[sb_flag]) / np.sum(np.ones_like(h5[sb_flag])) * 100
            logger.info(f"Subband {sb_label} now has {f_per:.2f}% flagged - {filename}")
    logger.info(f"Finished processing file {filename}")
    return True


def main(
    filenames,
    inplace=False,
    beam_label="beam_0",
    sigma=3,
    n_windows=100,
    use_weights=False,
    report=None,
    cores=None,
    threads_per_worker=None,
):
    args = locals()
    _ = args.pop("filenames")
    if inplace:
        logger.warning("Running in-place - this will overwrite the previous flag data!")
    # Initialise dask
    with LocalCluster(
        n_workers=cores, threads_per_worker=threads_per_worker
    ) as cluster, Client(cluster) as client, performance_report(
        filename=report
    ) if report else nullcontext():
        logger.info(f"Dask running at {client.dashboard_link}")
        if report:
            logger.info(f"Writting report to {report}")

        todos = {}
        for filename in filenames:
            logger.info(f"Processing file {filename}")
            # Copy hdf5 file
            exts = ("hdf", "hdf5", "sdhdf", "h5")
            if not any(filename.endswith(f".{ext}") for ext in exts):
                raise ValueError(
                    f"I don't recognose the file extension of '{filename}' (must be one of {exts})"
                )
            for ext in exts:
                if filename.endswith(f".{ext}"):
                    break

            new_filename = copy_file(filename, ext=ext) if not inplace else filename

            sb_avail = get_subbands(new_filename, beam_label=beam_label)

            todos[new_filename] = sb_avail

        # Compute to concrete values
        todos = compute(todos)[0]  # First elemment of single-element tuple

        hists = []
        for new_filename in todos.keys():
            # Iterate through subbands inside flag function
            flagged = flag(
                new_filename,
                todos[new_filename],
                beam_label=beam_label,
                sigma=sigma,
                n_windows=n_windows,
                use_weights=use_weights,
            )
            hist = update_history(new_filename, args, flagged)
            hists.append(hist)

        _ = compute(hists)

    logger.info("Done!")


def cli():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="+", type=str, help="Input SDHDF file(s)")
    parser.add_argument(
        "-i",
        "--inplace",
        action="store_true",
        help="Update flags in-place (default: create new file)",
    )
    parser.add_argument("-b", "--beam", type=str, default="beam_0", help="Beam label")
    parser.add_argument(
        "-s", "--sigma", type=float, default=3, help="Sigma clipping threshold"
    )
    parser.add_argument(
        "-n",
        "--n-windows",
        type=int,
        default=100,
        help="Number of windows to use in box filter",
    )
    parser.add_argument(
        "-w",
        "--use-weights",
        action="store_true",
        help="Use weights table instead of flag table",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Optionally save the Dask (html) report to a file",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=None,
        help="Number of workers to use (default: Dask automatic configuration)",
    )
    parser.add_argument(
        "-t",
        "--threads-per-worker",
        type=int,
        default=None,
        help="Number of threads per worker (default: Dask automatic configuration)",
    )

    args = parser.parse_args()
    main(
        filenames=args.filenames,
        inplace=args.inplace,
        beam_label=args.beam,
        sigma=args.sigma,
        n_windows=args.n_windows,
        use_weights=args.use_weights,
        report=args.report,
        cores=args.cores,
        threads_per_worker=args.threads_per_worker,
    )


if __name__ == "__main__":
    cli()
