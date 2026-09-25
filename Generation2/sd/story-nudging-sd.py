"""
Fetch daily (12:00) snow depth water equivalent (sd) for Austria from the
Climate DT story-nudging storyline (IFS-FESOM, sfc levtype, "hist"
experiment, realization 1), and upload the result as NetCDF to S3.

Unlike the vsw fetch (story-nudging.py), this keeps the native HEALPix
cell grid instead of converting to lat/lon points: the plain (non-feature)
.sel() used here always downloads the full global grid per timestep -
there's no server-side spatial filter on this path - so the Austria clip
happens locally via astropy-healpix + shapely after download, using the
store's own nside and DestinE's documented "nested" HEALPix ordering for
the climate-dt native grid.

Processed one month at a time: PolytopeZarrStore caches every fetched
chunk in a plain dict with no eviction (store._cache), so a whole-year
run keeps every downloaded full-global hourly grid (~12.6MB each) in
memory simultaneously - for a full year that's tens of GB. Building a
fresh store per month, uploading that month's NetCDF, then discarding the
store before the next month bounds peak memory to roughly one month's
worth of cached chunks instead of the whole range.

Credentials are passed as CLI arguments (see parse_args), so an Argo
Workflow can template them straight into the container's args: list.
"""
import argparse
import gc
import logging
import subprocess
import warnings

import astropy.units as u
import earthkit.data
import earthkit.geo.cartography
import numpy as np
import pandas as pd
import s3fs
from astropy_healpix import HEALPix
from shapely.geometry import MultiPolygon, Polygon
from shapely.vectorized import contains

from polytope_zarr import PolytopeZarrStore

REALIZATION = "1"
CLIMATE = "hist"
EXPERIMENTS = ["cont", "hist", "Tplus2.0K"]
COUNTRY = "Austria"
START_DATE = "2023-04-01T00:00:00"
END_DATE = "2026-07-31T23:00:00"
S3_PATH = "destine-climate-dt/sd/netcdf"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desp-user", required=True)
    parser.add_argument("--desp-password", required=True)
    parser.add_argument("--s3-key", required=True)
    parser.add_argument("--s3-secret", required=True)
    return parser.parse_args()


def authenticate(user, password):
    subprocess.run(
        [
            "python", "src/desp-authentication.py",
            "-u", user,
            "-p", password,
        ],
        check=True,
    )


def configure_logging():
    earthkit.data.config.set("cache-policy", "off")
    for name in ("polytope", "polytope.api", "earthkit.data", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def clip_to_shape(da, nside, shapes):
    """Reduce a HEALPix-nested `cell`-indexed DataArray to the cells whose
    center falls inside `shapes`, keeping the native cell index (no
    conversion to lat/lon points)."""
    hp = HEALPix(nside=nside, order="nested")
    n_cells = da.sizes["cell"]
    lon, lat = hp.healpix_to_lonlat(np.arange(n_cells))
    lon_deg = ((lon.to_value(u.deg) + 180) % 360) - 180
    lat_deg = lat.to_value(u.deg)

    polygons = [Polygon([(lo, la) for la, lo in ring]) for ring in shapes]
    country_shape = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
    mask = contains(country_shape, lon_deg, lat_deg)
    cell_idx = np.nonzero(mask)[0]

    clipped = da.isel(cell=cell_idx)
    clipped.attrs.pop("_polytope_store", None)  # not netCDF-serializable
    # String coordinates (e.g. "climate") come from the store with a zarr
    # fill_value of "", which xarray carries into encoding["_FillValue"].
    # to_netcdf() with no path can only use the scipy/netCDF3 engine, and
    # netCDF3 doesn't support _FillValue on string variables at all -
    # strip it here rather than fail at write time.
    for coord in clipped.coords.values():
        if coord.dtype.kind in ("U", "O"):
            coord.encoding.pop("_FillValue", None)
    return clipped


def process_month(eodc_s3, shapes, month_start, month_end):
    """Fetch, clip and upload one calendar month, using its own
    PolytopeZarrStore so nothing from this month's fetch cache survives
    into the next call."""
    store = PolytopeZarrStore.from_climate_dt(
        models=["IFS-FESOM"],
        experiment=EXPERIMENTS,
        activity="story-nudging",
        levtype="sfc",
        frequency="hourly",
        start_date=month_start.strftime("%Y-%m-%dT00:00:00"),
        end_date=month_end.strftime("%Y-%m-%dT23:00:00"),
        resolution="high",
        realization=REALIZATION,
    )
    store._filter_hours = [12]
    ds = store.open()

    # Fetching uses the store's default day-batching (all 24 hours of a
    # day in one request) rather than one request per timestamp: Polytope
    # is an async submit+poll API, so batching trades ~24x more downloaded
    # bytes for ~24x fewer requests, which is faster in practice since
    # per-request submit/poll latency dominates over transfer time here.
    time_slice = slice(month_start.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d"))
    sd = ds["sd"].polytope.sel(climate=CLIMATE, time=time_slice)
    sd_clipped = clip_to_shape(sd, store.nside, shapes)
    # Keep only the noon timestamp per day in the final NetCDF - the other
    # 23 hours were downloaded as part of each day's batch but are
    # discarded here rather than written out.
    sd_clipped = sd_clipped.sel(time=sd_clipped["time"].dt.hour == 12)
    # The store's "time" coordinate carries a nanoseconds-since-1970/int64
    # encoding, which overflows netCDF3's 32-bit integer limit (the only
    # format to_netcdf() can use for an in-memory, no-path write). Clearing
    # it lets xarray pick a netCDF3-safe unit instead - the dates still
    # read back as normal datetimes either way, only the invisible
    # internal storage unit changes.
    sd_clipped["time"].encoding = {}

    data_bytes = sd_clipped.to_netcdf()
    month_tag = month_start.strftime("%Y-%m")
    out_path = f"{S3_PATH}/sd_story-nudging_{CLIMATE}_r{REALIZATION}_{month_tag}.nc"
    with eodc_s3.open(out_path, "wb") as f:
        f.write(data_bytes)
    print(out_path)


def main():
    args = parse_args()
    authenticate(args.desp_user, args.desp_password)
    configure_logging()

    eodc_s3 = s3fs.S3FileSystem(
        key=args.s3_key,
        secret=args.s3_secret,
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"},
    )

    shapes = earthkit.geo.cartography.country_polygons([COUNTRY], resolution=50e6)

    for month_start in pd.date_range(START_DATE, END_DATE, freq="MS"):
        month_end = min(month_start + pd.offsets.MonthEnd(0), pd.Timestamp(END_DATE))
        process_month(eodc_s3, shapes, month_start, month_end)
        # Release this month's store (and its unbounded fetch cache) and
        # any lazy arrays referencing it before the next month starts.
        gc.collect()


if __name__ == "__main__":
    main()
