"""
Fetch daily (12:00) volumetric soil water (vsw) for Austria from the
Climate DT story-nudging storyline (IFS-FESOM, sol levtype), across all
realizations, experiments and soil levels, and upload each result as
NetCDF to S3.

Credentials are passed as CLI arguments (see parse_args), so an Argo
Workflow can template them straight into the container's args: list.
"""
import argparse
import io
import logging
import subprocess
import warnings

import earthkit.data
import earthkit.geo.cartography
import s3fs

from polytope_zarr import PolytopeZarrStore

REALIZATIONS = [1, 2, 3, 4, 5]
CLIMATES = ["cont", "hist", "Tplus2.0K"]
LEVELS = [1, 2, 3]
COUNTRY = "Austria"
START_DATE = "2026-01-01T00:00:00"
END_DATE = "2026-07-31T23:00:00"
TIME_SLICE = slice("2026-01-01", "2026-07-31")
S3_PATH = "destine-climate-dt/vsw/netcdf/SN"


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

    for realization in REALIZATIONS:
        realization = str(realization)
        store = PolytopeZarrStore.from_climate_dt(
            models=["IFS-FESOM"],
            experiment=CLIMATES,
            activity="story-nudging",
            levtype="sol",
            frequency="hourly",
            start_date=START_DATE,
            end_date=END_DATE,
            resolution="high",
            realization=realization,
        )
        store._filter_hours = [12]
        ds = store.open()

        for climate in CLIMATES:
            for level in LEVELS:
                vsw = ds["vsw"].polytope.sel(
                    climate=climate,
                    time=TIME_SLICE,
                    level=level,
                    polygon=shapes,
                )
                data_bytes = vsw.to_netcdf()
                out_path = (
                    f"{S3_PATH}/vsw_story-nudging_{climate}_level{level}_r{realization}.nc"
                )
                with eodc_s3.open(out_path, "wb") as f:
                    f.write(data_bytes)
                print(out_path)


if __name__ == "__main__":
    main()
