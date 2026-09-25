"""
Append new story-nudging vsw NetCDFs (produced by story-nudging.py:
3 climates x 3 levels x 5 realizations, same layout/schema as the ones
consolidated by story_nudging_zarr.py) from NETCDF_PATH directly onto the
existing IFS_FESOM_story_nudging.zarr store, along "time".

Store layout (unchanged):
  IFS_FESOM_story_nudging.zarr/
    cont/1, cont/2, cont/3
    hist/1, hist/2, hist/3
    Tplus2.0K/1, Tplus2.0K/2, Tplus2.0K/3

Each leaf group holds the 5 realizations combined along the existing
"realization" dimension, with the new NetCDFs' time steps appended after
whatever time range is already in the store.
"""
import io

import s3fs
import xarray as xr

S3_KEY = ""
S3_SECRET = ""

NETCDF_PATH = "destine-climate-dt/vsw/netcdf/SN"
ZARR_PATH = "destine-climate-dt/vsw/zarr/IFS_FESOM_story_nudging.zarr"

CLIMATES = ["cont", "hist", "Tplus2.0K"]
LEVELS = [1, 2, 3]
REALIZATIONS = [1, 2, 3, 4, 5]


def read_netcdf(eodc_s3, path):
    with eodc_s3.open(path, "rb") as f:
        return xr.open_dataset(io.BytesIO(f.read())).load()


def main():
    eodc_s3 = s3fs.S3FileSystem(
        key=S3_KEY,
        secret=S3_SECRET,
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"},
    )

    for climate in CLIMATES:
        for level in LEVELS:
            realizations = []
            for realization in REALIZATIONS:
                path = (
                    f"{NETCDF_PATH}/vsw_story-nudging_{climate}_level{level}"
                    f"_r{realization}.nc"
                )
                ds = read_netcdf(eodc_s3, path)
                ds = ds.expand_dims(realization=[realization])
                realizations.append(ds)

            combined = xr.concat(realizations, dim="realization")
            # A single Dask chunk along "time" for the write: the append
            # offset in the existing store won't generally land on a
            # multiple of the on-disk chunk size (100), so splitting the
            # new data into several time chunks risks two different Dask
            # chunks writing into the same physical zarr chunk. This only
            # affects how the write is split into tasks - the on-disk zarr
            # chunk grid (inherited from the existing store) is unchanged.
            combined = combined.chunk({"time": -1, "realization": -1, "points": -1})
            group = f"{climate}/{level}"
            # zarr3's FSMap-backed stores don't support to_zarr(..., group=...)
            # on top of an already-rooted store - build the group's path
            # directly into its own mapper instead.
            store = eodc_s3.get_mapper(f"{ZARR_PATH}/{group}")
            combined.to_zarr(store, mode="a", append_dim="time", zarr_format=3)
            print(f"appended to group {group}: {dict(combined.sizes)}")


if __name__ == "__main__":
    main()
