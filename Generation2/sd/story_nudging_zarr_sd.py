"""
Consolidate the monthly story-nudging sd NetCDFs (produced by
story-nudging-sd.py: one file per calendar month per climate/realization)
into a single zarr3 store on S3, called IFS_FESOM_story_nudging.zarr.

Store layout:
  IFS_FESOM_story_nudging.zarr/
    hist

Only "hist" exists right now, but climate is still kept as a group (as
with the 2t store) so cont/Tplus2.0K can be added the same way later.
Each group holds the single realization as a size-1 "realization"
dimension, with all months concatenated along "time" (there's no level
axis for this variable).

story-nudging-sd.py fetches on the native HEALPix `cell` grid instead of
`polygon=shapes` (which reliably fails for this variable/levtype), so
these files don't have `points`/`latitude`/`longitude` like the 2t/vsw
ones. Since `cell` holds real HEALPix indices (nside=512, nested - the
same grid story-nudging-sd.py used to clip to Austria), their (lat, lon)
can be computed locally with astropy-healpix, so we rename `cell` to
`points` and attach latitude/longitude here, matching the 2t/vsw schema
without needing polygon= to work.
"""
import io

import astropy.units as u
import numpy as np
import s3fs
import xarray as xr
from astropy_healpix import HEALPix

S3_KEY = ""
S3_SECRET = ""

NETCDF_PATH = "destine-climate-dt/sd/netcdf"
ZARR_PATH = "destine-climate-dt/sd/zarr/IFS_FESOM_story_nudging.zarr"

CLIMATES = ["hist"]
REALIZATION = 1
NSIDE = 512  # story-nudging "high" resolution (9 km), matches story-nudging-sd.py


def read_netcdf(eodc_s3, path):
    with eodc_s3.open(path, "rb") as f:
        return xr.open_dataset(io.BytesIO(f.read())).load()


def attach_latlon(ds):
    """Rename the native HEALPix `cell` dim to `points` and attach
    latitude/longitude coordinates computed from the same nside=512
    nested HEALPix grid story-nudging-sd.py clipped to Austria with -
    matching the points/lat/lon schema of the polygon-fetched stores
    (2t, vsw) without needing polygon= (which fails for this variable)."""
    hp = HEALPix(nside=NSIDE, order="nested")
    lon, lat = hp.healpix_to_lonlat(ds["cell"].values)
    lon_deg = ((lon.to_value(u.deg) + 180) % 360) - 180
    lat_deg = lat.to_value(u.deg)

    ds = ds.rename({"cell": "points"})
    n_points = ds.sizes["points"]
    return ds.assign_coords(
        points=("points", np.arange(n_points, dtype="int64")),
        latitude=("points", lat_deg),
        longitude=("points", lon_deg),
    )


def main():
    eodc_s3 = s3fs.S3FileSystem(
        key=S3_KEY,
        secret=S3_SECRET,
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"},
    )

    for climate in CLIMATES:
        pattern = f"{NETCDF_PATH}/sd_story-nudging_{climate}_r{REALIZATION}_*.nc"
        paths = sorted(eodc_s3.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no files matched {pattern}")

        months = [read_netcdf(eodc_s3, path) for path in paths]
        ds = xr.concat(months, dim="time")
        ds = attach_latlon(ds)
        ds = ds.expand_dims(realization=[REALIZATION])
        ds = ds.chunk({"time": 100, "realization": -1, "points": -1})

        # zarr3's FSMap-backed stores don't support to_zarr(..., group=...)
        # on top of an already-rooted store - build the group's path
        # directly into its own mapper instead.
        store = eodc_s3.get_mapper(f"{ZARR_PATH}/{climate}")
        ds.to_zarr(store, mode="w", zarr_format=3)
        print(f"wrote group {climate} from {len(paths)} months: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
