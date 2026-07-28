"""
Consolidate the 45 story-nudging vsw NetCDFs (produced by story-nudging.py:
3 climates x 3 levels x 5 realizations) into a single zarr3 store on S3,
called IFS_FESOM_story_nudging.zarr.

Store layout:
  IFS_FESOM_story_nudging.zarr/
    cont/1, cont/2, cont/3
    hist/1, hist/2, hist/3
    Tplus2.0K/1, Tplus2.0K/2, Tplus2.0K/3

Each leaf group holds the 5 realizations combined along a new
"realization" dimension.
"""
import io
import os

import s3fs
import xarray as xr

S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")

NETCDF_PATH = "destine-climate-dt/vsw/netcdf"
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
            combined = combined.chunk({"time": 100, "realization": -1, "points": -1})
            group = f"{climate}/{level}"
            # zarr3's FSMap-backed stores don't support to_zarr(..., group=...)
            # on top of an already-rooted store - build the group's path
            # directly into its own mapper instead.
            store = eodc_s3.get_mapper(f"{ZARR_PATH}/{group}")
            combined.to_zarr(store, mode="w", zarr_format=3)
            print(f"wrote group {group}: {dict(combined.sizes)}")


if __name__ == "__main__":
    main()
