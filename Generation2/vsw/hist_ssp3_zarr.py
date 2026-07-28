"""
Consolidate the 180 IFS-FESOM vsw NetCDFs (produced by models_hist_ssp3.py:
2 experiments x 3 levels x yearly chunks - 25 years for "hist", 35 years
for "SSP3-7.0") into a single zarr3 store on S3, called IFS-FESOM.zarr.

Store layout:
  IFS-FESOM.zarr/
    hist/1, hist/2, hist/3
    SSP3-7.0/1, SSP3-7.0/2, SSP3-7.0/3

Each leaf group holds the full experiment timerange, built by
concatenating the per-year NetCDFs along "time".
"""
import io
import os

import s3fs
import xarray as xr

S3_KEY = os.environ.get("S3_KEY")
S3_SECRET = os.environ.get("S3_SECRET")

NETCDF_PATH = "destine-climate-dt/vsw/netcdf/NEMO"
ZARR_PATH = "destine-climate-dt/vsw/zarr/IFS-NEMO.zarr"

MODEL = "IFS-NEMO"
LEVELS = [1, 2, 3]

EXPERIMENTS = {
    "hist": range(1990, 2015),
    "SSP3-7.0": range(2015, 2050),
}


def read_netcdf(eodc_s3, path):
    with eodc_s3.open(path, "rb") as f:
        return xr.open_dataset(io.BytesIO(f.read())).load()


def main():
    eodc_s3 = s3fs.S3FileSystem(
        key=S3_KEY,
        secret=S3_SECRET,
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"},
    )

    for experiment, years in EXPERIMENTS.items():
        for level in LEVELS:
            yearly = []
            for year in years:
                path = (
                    f"{NETCDF_PATH}/vsw_{MODEL}_{experiment}_level{level}"
                    f"_{year}-{year}.nc"
                )
                yearly.append(read_netcdf(eodc_s3, path))

            combined = xr.concat(yearly, dim="time")
            combined = combined.chunk({"time": 100, "points": -1})
            group = f"{experiment}/{level}"
            # zarr3's FSMap-backed stores don't support to_zarr(..., group=...)
            # on top of an already-rooted store - build the group's path
            # directly into its own mapper instead.
            store = eodc_s3.get_mapper(f"{ZARR_PATH}/{group}")
            combined.to_zarr(store, mode="w", zarr_format=3)
            print(f"wrote group {group}: {dict(combined.sizes)}")


if __name__ == "__main__":
    main()
