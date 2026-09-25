"""
Fetch volumetric soil water (vsw) for Austria from Climate DT
(ICON, IFS-FESOM, IFS-NEMO models, sol levtype), across the complete
timerange for two experiments - "hist" (1990-2014) and "SSP3-7.0"
(2015-2049) - and soil levels 1-3, uploading each result as NetCDF to S3.

Each experiment's timerange is split into 5-year chunks - one NetCDF
per (model, experiment, level, chunk) - since fetching the full range
in a single request doesn't work reliably.

Credentials are passed as CLI arguments (see parse_args), so an Argo
Workflow can template them straight into the container's args: list.
"""
import argparse
import io
import logging
import subprocess
import time
import warnings

import earthkit.data
import earthkit.geo.cartography
import rioxarray  # noqa: F401 -- registers the .rio accessor on xarray objects
import s3fs
from shapely.geometry import Polygon

from polytope_zarr import PolytopeZarrStore

# MODELS = ["ICON", "IFS-FESOM", "IFS-NEMO"]
MODELS = ["ICON"]
LEVELS = [1, 2, 3]
COUNTRY = "Austria"
S3_PATH = "destine-climate-dt/vsw/netcdf/ICON"
CHUNK_YEARS = 1

# North, West, South, East - MARS area order. Used with the "area" keyword
# (server-side regridding to a regular lat/lon grid) instead of the
# "feature: polygon" path, which hits the grid-hash-mismatch bug for ICON.
AREA = [49.0758, 9.4979, 46.308, 17.2416]

# EXPERIMENTS = {
#     "hist": {
#         "start_year": 1990,
#         "end_year": 2014,
#     },
#     "SSP3-7.0": {
#         "start_year": 2015,
#         "end_year": 2049,
#     },
# }

EXPERIMENTS = {
    "hist": {
        "start_year": 1992,
        "end_year": 2014,
    },
    "SSP3-7.0": {
        "start_year": 2015,
        "end_year": 2049,
    },
}

def year_chunks(start_year, end_year, chunk_size):
    """Split [start_year, end_year] into consecutive chunk_size-year ranges."""
    chunks = []
    year = start_year
    while year <= end_year:
        chunk_end = min(year + chunk_size - 1, end_year)
        chunks.append((year, chunk_end))
        year = chunk_end + 1
    return chunks


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
    # Give the freshly-issued token a moment to propagate before using it -
    # a container run authenticates and queries back-to-back with none of
    # the natural delay a human running notebook cells one at a time has.
    time.sleep(10)


def configure_logging():
    earthkit.data.config.set("cache-policy", "off")
    for name in ("polytope", "polytope.api", "earthkit.data", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    austria_polygon = Polygon([(lon, lat) for lat, lon in shapes[0]])

    failures = []

    for experiment, cfg in EXPERIMENTS.items():
        start_year, end_year = cfg["start_year"], cfg["end_year"]
        store = PolytopeZarrStore.from_climate_dt(
            models=MODELS,
            experiment=experiment,
            resolution="high",
            levtype="sol",
            frequency="hourly",
            start_date=f"{start_year}-01-01T00:00:00",
            end_date=f"{end_year}-12-31T23:00:00",
        )
        store._filter_hours = [12]
        ds = store.open()

        chunks = year_chunks(start_year, end_year, CHUNK_YEARS)

        for model in MODELS:
            for level in LEVELS:
                for chunk_start, chunk_end in chunks:
                    time_slice = slice(f"{chunk_start}-01-01", f"{chunk_end}-12-31")
                    try:
                        vsw = ds["vsw"].polytope.sel(
                            model=model,
                            time=time_slice,
                            level=level,
                            area=AREA,
                        )
                        vsw = (
                            vsw.rio.write_crs("EPSG:4326")
                            .rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
                            .rio.clip([austria_polygon], crs="EPSG:4326")
                        )
                        data_bytes = vsw.to_netcdf()
                        out_path = (
                            f"{S3_PATH}/vsw_{model}_{experiment}_level{level}"
                            f"_{chunk_start}-{chunk_end}.nc"
                        )
                        with eodc_s3.open(out_path, "wb") as f:
                            f.write(data_bytes)
                        print(out_path)
                    except Exception as e:
                        # polytope-client's own __str__ can crash (mixed str/int
                        # entries in e.messages), masking the real error - print
                        # the raw parts instead of str(e)/repr(e).
                        print(
                            f"FAILED: model={model} experiment={experiment} "
                            f"level={level} chunk={chunk_start}-{chunk_end}"
                        )
                        print("exception type:", type(e).__name__)
                        print("exception messages:", getattr(e, "messages", None))
                        print("exception args:", e.args)
                        failures.append((model, experiment, level, chunk_start, chunk_end))

    if failures:
        print(f"\n{len(failures)} combination(s) failed:")
        for model, experiment, level, chunk_start, chunk_end in failures:
            print(f"  model={model} experiment={experiment} level={level} chunk={chunk_start}-{chunk_end}")


if __name__ == "__main__":
    main()
