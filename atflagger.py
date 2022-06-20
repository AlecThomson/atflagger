import h5py
from astropy.table import Table, QTable
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, mad_std
from dask import delayed
from tqdm.auto import tqdm, trange
import xarray as xr
import warnings
import shutil
import logging as log
from dask.distributed import LocalCluster, Client


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


def get_subbands(filename, beam_label="beam_0"):
    with h5py.File(filename, "r") as h5:
        # Read header info
        sb_avail = QTable.read(h5, path=beam_label + "/metadata/band_params")
    return sb_avail["LABEL"]


def flag(filename, sb_label, beam_label="beam_0", sigma=3, n_windows=100):
    # Open HDF5 file
    with h5py.File(filename, "r") as h5:
        # Read header info
        log.info(f"Processing subband {sb_label} - {filename}")
        sb_data = f"{beam_label}/{sb_label}/astronomy_data/data"
        sb_flag = f"{beam_label}/{sb_label}/astronomy_data/flag"
        sb_freq = f"{beam_label}/{sb_label}/astronomy_data/frequency"
        data = h5[sb_data]
        freq = np.array(h5[sb_freq])
        try:
            flag = np.array(h5[sb_flag])
        except KeyError:
            raise KeyError(
                f"No flagging information found for subband {sb_label} - {filename} - run autoflagger first"
            )

        f_per = np.sum(flag) / np.sum(np.ones_like(flag)) * 100
        log.info(f"Subband {sb_label} has {f_per:.2f}% flagged - {filename}")

        data_xr = xr.DataArray(
            data,
            dims=h5[sb_data].attrs["DIMENSION_LABELS"],
            coords={"frequency": freq,},
        )

        # Set flags
        # Ensure flag has same shape as data
        flag_reshape = flag.copy()
        for i, s in enumerate(data.shape):
            if i > len(flag_reshape.shape) -1:
                flag_reshape = np.expand_dims(flag_reshape, axis=-1)
            else:
                if flag_reshape.shape[i] == s:
                    continue
                else:
                    flag_reshape = np.expand_dims(flag_reshape, axis=i)
        data_xr_flg = data_xr.where(
            ~flag_reshape.astype(bool)
        )
        # Set chunks for parallel processing
        chunks = {d:1 for d in data_xr_flg.dims}
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
        mask_red = mask.sum(dim=dims) > 0
        log.info(f"Flagging {sb_label} and writing to file...")
        # Write flags back to file
        h5[sb_flag][:] = mask_red.values.astype(int)

        f_per = (
            np.sum(h5[sb_flag].astype(bool)) / np.sum(np.ones_like(h5[sb_flag])) * 100
        )
        log.info(f"Subband {sb_label} now has {f_per:.2f}% flagged - {filename}")


def main(filenames, beam_label="beam_0", sigma=3, n_windows=100):
    # Initialise dask
    cluster = LocalCluster()
    client = Client(cluster)
    log.info(f"Dask running at {client.dashboard_link}")

    for filename in filenames:
        log.info(f"Processing file {filename}")
        # Copy hdf5 file
        new_filename = filename.replace(".hdf", ".atflagged.hdf")
        shutil.copy(filename, new_filename)

        log.info(f"Create new file: {new_filename}")

        sb_avail = get_subbands(new_filename, beam_label=beam_label)

        # Iterate through subbands
        for sb_label in sb_avail:
            flag(
                new_filename,
                sb_label,
                beam_label=beam_label,
                sigma=sigma,
                n_windows=n_windows,
            )

        log.info(f"Finished processing file {filename}")

    # Close dask
    client.close()
    cluster.close()
    log.info("Done!")


def cli():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Flag SDHDF data")
    parser.add_argument("filenames", nargs="+", type=str, help="Input SDHDF file(s)")
    parser.add_argument("--beam", type=str, default="beam_0", help="Beam label")
    parser.add_argument(
        "--sigma", type=float, default=3, help="Sigma clipping threshold"
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        default=100,
        help="Number of windows to use in box filter",
    )
    args = parser.parse_args()
    log.basicConfig(
        level=log.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args.filenames, args.beam, args.sigma, args.n_windows)


if __name__ == "__main__":
    cli()
