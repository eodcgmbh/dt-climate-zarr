"""
Consolidate the existing local lsm zarr stores in this folder into a
single zarr3 store on S3, land-sea-mask/lsm.zarr, with each source
store as its own group.
"""
import os

import s3fs
import xarray as xr

LSM_DIR = os.path.dirname(os.path.abspath(__file__))

S3_KEY = ""
S3_SECRET = ""

OUT_PATH = "destine-climate-dt/land-sea-mask/lsm.zarr"

SOURCE_STORES = [
    # "IFS_NEMO_ScenarioMIP_SSP3-7.zarr",
    "IFS_FESOM_sn_tplus2.zarr",
    "IFS_FESOM_sn_hist.zarr",
    "IFS_FESOM_sn_cont.zarr",
    # "IFS_NEMO_HighResMIP_cont.zarr",
]


def main():
    eodc_s3 = s3fs.S3FileSystem(
        key=S3_KEY,
        secret=S3_SECRET,
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"},
    )

    for store_name in SOURCE_STORES:
        group = store_name.removesuffix(".zarr")
        src_path = os.path.join(LSM_DIR, store_name)
        ds = xr.open_zarr(src_path)

        # zarr3's FSMap-backed stores don't support to_zarr(..., group=...)
        # on top of an already-rooted store - build the group's path
        # directly into its own mapper instead.
        store = eodc_s3.get_mapper(f"{OUT_PATH}/{group}")
        ds.to_zarr(store, mode="w", zarr_format=3)
        print(f"wrote group {group}: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
