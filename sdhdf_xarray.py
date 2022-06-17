
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

def get_subbands(filename, beam_label="beam_0"):
    with h5py.File(filename, "r") as h5:
        # Read header info
        sb_avail = Table.read(h5, path=beam_label + "/metadata/band_params")
    return sb_avail["LABEL"]

def create(filename:str) -> xr.Dataset:
    with h5py.File(filename, "r") as h5:
        keys = list(h5.keys())
        beam_labels = [key for key in keys if "beam_" in key]

        sb_avail = {}
        for beam_label in beam_labels:
            sb_avail[beam_label] = get_subbands(
                filename, beam_label=beam_label
            )

        data_arrs = []
        for beam_label in beam_labels:
            sb_labels = sb_avail[beam_label]
            for sb_label in sb_labels:
                sb_data = f"{beam_label}/{sb_label}/astronomy_data/data"
                sb_freq = f"{beam_label}/{sb_label}/astronomy_data/frequency"
                sb_para = f"{beam_label}/{sb_label}/metadata/obs_params"
                has_flags = "flag" in h5[f"{beam_label}/{sb_label}/astronomy_data"].keys()
                data = h5[sb_data]
                if has_flags:
                    flag = h5[f"{beam_label}/{sb_label}/astronomy_data/flag"]
                    # Ensure flag has same shape as data
                    flag_reshape = flag.astype(bool).copy()
                    for i, s in enumerate(data.shape):
                        if i > len(flag_reshape.shape) -1:
                            flag_reshape = np.expand_dims(flag_reshape, axis=-1)
                        else:
                            if flag_reshape.shape[i] == s:
                                continue
                            else:
                                flag_reshape = np.expand_dims(flag_reshape, axis=i)
                    flag = flag_reshape
                else:
                    flag = np.zeros_like(data).astype(bool)
                freq = h5[sb_freq]
                meta = h5[sb_para]
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
                data_arrs.append(data_xr)
                flag_xr = xr.DataArray(
                    flag,
                    dims=h5[sb_data].attrs["DIMENSION_LABELS"],
                    coords=coords,
                    name=f"{sb_label}_flag",
                )
                data_arrs.append(flag_xr)
            dataset = xr.combine_by_coords(data_arrs)
        return dataset