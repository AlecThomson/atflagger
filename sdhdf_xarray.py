from typing import Dict, List, Tuple, Union
import h5py
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip, mad_std
from dask import delayed
from tqdm.auto import tqdm, trange
import xarray as xr
import warnings
import shutil
import logging as log
import pandas as pd
from dask.distributed import LocalCluster, Client


def get_subbands(filename, beam_label="beam_0"):
    """Find available subbands for a givent beam.

    Args:
        filename (str): SDHDF file to read.
        beam_label (str, optional): Beam to look at. Defaults to "beam_0".

    Returns:
        Table: Available subbands
    """
    with h5py.File(filename, "r") as h5:
        # Read header info
        sb_avail = Table.read(h5, path=beam_label + "/metadata/band_params")
    return sb_avail["LABEL"]


def create(filename: str, in_memory: bool = True) -> Dict[str, xr.DataArray]:
    """Create a 'dataset' from SDHDF data
    Note - Creating an Xarray DataSet would be preferable, but seems slow
    at the moment. Using a Dict of Arrays instead.

    Args:
        filename (str): SDHDF file to open.
        in_memory (bool, optional): Load all data into memory. Defaults to True.

    Returns:
        Dict[str,xr.DataArray]: Data arrays for sub-band.
    """
    with h5py.File(filename, "r") as h5:
        keys = list(h5.keys())
        beam_labels = [key for key in keys if "beam_" in key]

        sb_avail = {}
        for beam_label in beam_labels:
            sb_avail[beam_label] = get_subbands(filename, beam_label=beam_label)

        data_arrs = {}
        for beam_label in tqdm(beam_labels, desc="Reading beams"):
            sb_labels = sb_avail[beam_label]
            for sb_label in tqdm(sb_labels, desc="Reading subbands"):
                sb_data = f"{beam_label}/{sb_label}/astronomy_data/data"
                sb_freq = f"{beam_label}/{sb_label}/astronomy_data/frequency"
                sb_para = f"{beam_label}/{sb_label}/metadata/obs_params"
                has_flags = (
                    "flag" in h5[f"{beam_label}/{sb_label}/astronomy_data"].keys()
                )
                data = h5[sb_data]
                if has_flags:
                    flag = h5[f"{beam_label}/{sb_label}/astronomy_data/flag"]
                    # Ensure flag has same shape as data
                    flag_reshape = flag[:].copy()
                    for i, s in enumerate(data.shape):
                        if i > len(flag_reshape.shape) - 1:
                            flag_reshape = np.expand_dims(flag_reshape, axis=-1)
                        else:
                            if flag_reshape.shape[i] == s:
                                continue
                            else:
                                flag_reshape = np.expand_dims(flag_reshape, axis=i)
                    flag = flag_reshape
                else:
                    flag = np.zeros_like(data)
                freq = h5[sb_freq]
                meta = h5[sb_para]
                if in_memory:
                    data = np.array(data)
                    freq = np.array(freq)
                    flag = np.array(flag)
                    meta = np.array(meta)
                names = meta.dtype.names
                coords = {name: ("time", meta[name]) for name in names}
                coords["frequency"] = freq
                coords["beam"] = beam_label
                data_xr = xr.DataArray(
                    data,
                    dims=h5[sb_data].attrs["DIMENSION_LABELS"],
                    coords=coords,
                    name=f"{sb_label}_data",
                )
                data_arrs[f"{sb_label}_data"] = data_xr
                flag_xr = xr.DataArray(
                    flag,
                    dims=h5[sb_data].attrs["DIMENSION_LABELS"],
                    coords=coords,
                    name=f"{sb_label}_flag",
                )
                data_arrs[f"{sb_label}_flag"] = flag_xr
            # dataset = xr.combine_by_coords(data_arrs)
        return data_arrs


def box_filter(
    spectrum: np.ndarray, sigma: float = 3, n_windows: int = 100
) -> np.ndarray:
    """Filter a spectrum using a box filter.

    Args:
        spectrum (np.ndarray): 1D spectrum array.
        sigma (int, optional): Stddev level to sigma clip to . Defaults to 3.
        n_windows (int, optional): Number of windows to divide the band into. Defaults to 100.

    Returns:
        np.ndarray: Flagged array
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


def flag_auto(
    dataset: Dict[str, xr.DataArray], sigma: float = 3, n_windows: int = 100
) -> Dict[str, xr.DataArray]:
    """Run autoflagging on dataset.

    Args:
        dataset (Dict[str, xr.DataArray]): SDHDF dataset.
        sigma (float, optional): Stddev level to clip to. Defaults to 3.
        n_windows (int, optional): Number of windows to divide the band. Defaults to 100.

    Returns:
        Dict[str, xr.DataArray]: Flagged dataset.
    """    """
    Flag data based on a box filter.
    """
    # Create a new dataset
    dataset_filt = dataset.copy()
    # Iterate through each subband
    data_sbs = [sb for sb in dataset.keys() if "data" in sb]
    for sb in tqdm(data_sbs, desc="Flagging subbands"):
        # Get the data
        dat = dataset[sb]
        flag = dataset[sb.replace("data", "flag")]
        dat_flg = dat.where(
            ~flag.astype(bool),
        )
        # Set chunks for parallel processing
        chunks = {d: 1 for d in dat_flg.dims}
        chunks["frequency"] = len(dat_flg.frequency)
        dat_flg = dat_flg.chunk(chunks)
        mask = xr.apply_ufunc(
            box_filter,
            dat_flg,
            input_core_dims=[["frequency"]],
            output_core_dims=[["frequency"]],
            kwargs={"sigma": sigma, "n_windows": n_windows},
            dask="parallelized",
            vectorize=True,
            output_dtypes=(bool),
        )
        dataset_filt[sb.replace("data", "flag")] = mask
    return dataset_filt


def get_persistent_rfi() -> pd.DataFrame:
    """Read persistent RFI file

    Returns:
        pd.Dataframe: Persistent RFI data.
    """
    rfi_df = pd.read_csv(
        "persistentRFI.dat",
        sep=",",
        # skip_blank_lines=True,
        comment="#",
        names=[
            "type",
            "observatory label",
            "receiver label",
            "freq0 MHz",
            "freq1 MHz",
            "MJD0",
            "MJD1",
            "text string for label",
        ],
    )
    return rfi_df


def flag_persistent(dataset: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray:
    """Flag persistent RFI

    Args:
        dataset (Dict[str, xr.DataArray]): SDHDF dataset.

    Returns:
        Dict[str, xr.DataArray: Flagged ata.
    """    
    rfi_df = get_persistent_rfi()
    flag_sbs = [sb for sb in dataset.keys() if "flag" in sb]
    for i, row in tqdm(
        rfi_df.iterrows(), desc="Flagging persistent RFI", total=len(rfi_df)
    ):
        for sb in tqdm(flag_sbs, desc="Flagging subbands", leave=False):
            # Get the data
            flag = dataset[sb]  # .astype(bool)
            if (
                row["freq0 MHz"] < flag.frequency.min()
                or row["freq1 MHz"] > flag.frequency.max()
            ):
                continue
            mask = (
                (flag.frequency > row["freq0 MHz"])
                & (flag.frequency < row["freq1 MHz"])
                & (flag.MJD > row["MJD0"])
                & (flag.MJD < row["MJD1"])
            )
            dataset[sb] = xr.where(mask, 1, flag)
    return dataset


def decimate(
    dataset: Dict[str, xr.DataArray], nchan: int = None, target_bw: float = None
) -> Dict[str, xr.DataArray]:
    """EXPERIMENTAL
    Decimate the data.

    Args:
        dataset (Dict[str, xr.DataArray]): SDHDF dataset.
        nchan (int, optional): Target channels per subband. Defaults to None.
        target_bw (float, optional): Target bandwidth per channel. Defaults to None.

    Raises:
        ValueError: If both nchan and target_bw are specified.
        ValueError: If neither nchan or target_bw are specified.

    Returns:
        Dict[str, xr.DataArray]: Flagged dataset.
    """
    if nchan is None and target_bw is None:
        raise ValueError("Must specify either nchan or target_bw")
    if nchan is not None and target_bw is not None:
        raise ValueError("Must specify either nchan or target_bw")
    # Assume that the bandwidth is the same for all subbands
    sb = list(dataset.keys())[0]
    if target_bw is not None:
        nchan = int(target_bw / dataset[sb].frequency.diff(dim="frequency").median())
    # Create a new dataset
    dataset_dec = dataset.copy()
    # Iterate through each subband
    data_sbs = [sb for sb in dataset.keys() if "data" in sb]
    for sb in tqdm(data_sbs, desc="Decimating subbands"):
        # Get the data
        dat = dataset[sb]
        # Set chunks for parallel processing
        chunks = {d: 1 for d in dat.dims}
        chunks["frequency"] = len(dat.frequency)
        dat = dat.chunk(chunks)
        flag = dataset[sb.replace("data", "flag")]
        flag = flag.chunk(chunks)

        # Decimate the data
        dat_dec, flag_dec = xr.apply_ufunc(
            _decimate_data,
            dat,
            flag,
            input_core_dims=[["frequency"], ["frequency"]],
            output_core_dims=[["frequency"], ["frequency"]],
            kwargs={"nchan": 1024},
            dask="parallelized",
            # vectorize=True,
            output_dtypes=(float, bool),
        )
        # Set the new data
        dataset_dec[sb] = dat_dec
        dataset_dec[sb.replace("data", "flag")] = flag_dec
    return dataset_dec


def _decimate_data(
    dat: np.ndarray, flag: np.ndarray, nchan: int = 1024
) -> Tuple[np.array, np.array]:
    """Decimate array

    Args:
        dat (np.ndarray): 1D spectrum to decimate.
        flag (np.ndarray): 1D flag spectrum to decimate.
        nchan (int, optional): Taget number of channels per subband. Defaults to 1024.

    Returns:
        Tuple[np.array, np.array]: Decimated data and flags
    """    """
    Decimate a dataset.
    """
    # Determine the number of channels average at time
    current_nchan = len(dat)
    nchan_avg = int(current_nchan / nchan)
    # Average the masked data
    dat = np.ma.array(dat, mask=flag)
    dat_avg = np.ma.mean(dat.reshape((nchan, nchan_avg)), axis=1)
    return dat_avg.data, dat_avg.mask
