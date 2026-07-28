"""
fetch_climate_dt.py

Fetches Climate DT timeseries data from the Destination Earth PolytopE API
for a set of station points and stores results incrementally to a Zarr store.
"""

import os
import subprocess
import concurrent.futures
from typing import Literal

import pandas as pd
import xarray as xr
import s3fs




# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT = "SSP3-7.0"
ACTIVITY = "ScenarioMIP"
LEVEL_TYPE = "sfc"
DATESTRING = "2020-2024"
MODEL = "IFS-NEMO"
PARAMETER = "141"

POINTS_CSV = "destine-climate-dt/points_IFS_NEMO_ScenarioMIP_141.csv"
AUTH_SCRIPT = "./src/desp-authentication.py"
POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"

STORE_PATH = "destine-climate-dt/IFS-NEMO_sfc_ScenarioMIP_SSP3-7.0_snowDepth.zarr"

MAX_WORKERS = 25
STATION_START = 11834 
STATION_LIMIT = None  # set to None to process all stations
# ---------------------------------------------------------------------------
# S3 Configuration
# ---------------------------------------------------------------------------

# eodc_s3 = s3fs.S3FileSystem(
#     key=os.environ.get("S3_KEY"),
#     secret=os.environ.get("S3_SECRET"),
#     client_kwargs={"endpoint_url": "https://objects.eu"}
# )

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_access_token() -> str:
    """Run the DESP authentication script and return the access token."""
    result = subprocess.run(
        [
            "python", AUTH_SCRIPT,
            "-u", os.environ.get("DESP_USER"),
            "-p", os.environ.get("DESP_PASS")
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    token = result.stdout.strip().split("}\n")[-1]
    return token

import logging
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def get_datestring(temp_extent: str) -> str:
    """
    Convert a "YYYY-YYYY" range string to the API date format.

    Example: "2020-2024" -> "20200101/to/20241231"
    """
    years = temp_extent.split("-")
    return f"{years[0]}0101/to/{years[1]}1231"


def extract_cdt_ts(
    experiment: str,
    activity: str,
    level_type: str,
    datestring: str,
    model: str,
    parameter: str,
    location: list,
    feature: Literal["timeseries", "polygon"] = "timeseries",
    time_resolution: str = "0000/to/2300",
    resolution: str = "high",
) -> xr.Dataset | None:
    """
    Fetch a Climate DT timeseries (or polygon) from the Polytope API.

    Parameters
    ----------
    experiment      : e.g. "SSP3-7.0"
    activity        : e.g. "ScenarioMIP"
    level_type      : e.g. "sfc" or "sol"
    datestring      : API-formatted date range, e.g. "20200101/to/20241231"
    model           : e.g. "IFS-NEMO"
    parameter       : GRIB parameter ID as string, e.g. "141"
    location        : list of [lat, lon] pairs for timeseries, or polygon coords
    feature         : "timeseries" or "polygon"
    time_resolution : hourly range string, default covers full day
    resolution      : "high" or "low"

    Returns
    -------
    xr.Dataset or None on failure
    """
    import earthkit.data  # imported here to keep top-level imports lightweight

    if feature == "timeseries":
        feature_dict = {
            "type": "timeseries",
            "points": location,
            "time_axis": "date",
        }
    elif feature == "polygon":
        feature_dict = {
            "type": "polygon",
            "shape": location,
        }
    else:
        raise TypeError(f"Unsupported feature type: {feature!r}")

    request = {
        # Static Climate DT identifiers
        "class": "d1",
        "dataset": "climate-dt",
        "generation": "1",
        "expver": "0001",
        "stream": "clte",
        "type": "fc",
        # Variable parameters
        "activity": activity,
        "experiment": experiment,
        "levtype": level_type,
        "date": datestring,
        "model": model,
        "param": parameter,
        "realization": "1",
        "resolution": resolution,
        "time": time_resolution,
        "feature": feature_dict,
    }

    if level_type == "sol":
        request["levelist"] = "1"

    try:
        ds = earthkit.data.from_source(
            "polytope",
            "destination-earth",
            request,
            stream=False,
            address=POLYTOPE_ADDRESS,
        )
        return ds.to_xarray()
    except Exception as exc:
        print(f"[ERROR] earthkit request failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-station worker
# ---------------------------------------------------------------------------

def get_station_data(geosphere_station: pd.Series) -> xr.Dataset | None:
    """Fetch data for a single station and reshape for Zarr storage."""
    latlon = [[float(geosphere_station["latitude"]), float(geosphere_station["longitude"])]]

    ts = extract_cdt_ts(
        experiment=EXPERIMENT,
        activity=ACTIVITY,
        level_type=LEVEL_TYPE,
        datestring=get_datestring(DATESTRING),
        model=MODEL,
        parameter=PARAMETER,
        location=latlon,
    )

    if ts is None:
        return None

    point_id = int(geosphere_station["points"])

    return (
        ts.stack(pointid=("latitude", "longitude"))
        .reset_index("pointid")
        # .assign_coords({"stationid": ("pointid", np.array([str(geosphere_station["points"])], dtype="U10"))})
        .assign_coords({"pointid": [point_id]})
        .transpose("pointid", ...)
    )


# ---------------------------------------------------------------------------
# Zarr write helper
# ---------------------------------------------------------------------------

CHUNK_SPEC = {
    "pointid": 1,
    "levelist": 1,
    "number": 1,
    "datetime": 1,
    "t": "auto",
}


def write_to_zarr(dataset: xr.Dataset, store_path: str) -> None:
    zarr_store = s3fs.S3Map(root=store_path, s3=eodc_s3)
    chunked = dataset.chunk(chunks=CHUNK_SPEC)
    if not eodc_s3.exists(store_path):
        chunked.to_zarr(store=zarr_store, mode="w")
    else:
        chunked.to_zarr(store=zarr_store, append_dim="pointid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Script started", flush=True)

    # 1. Create S3 connection first
    global eodc_s3
    eodc_s3 = s3fs.S3FileSystem(
        key=os.environ.get("S3_KEY"),
        secret=os.environ.get("S3_SECRET"),
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"}
    )

    print(f"S3_KEY set: {bool(os.environ.get('S3_KEY'))}", flush=True)
    print(f"DESP_USER set: {bool(os.environ.get('DESP_USER'))}", flush=True)

    # 2. Then authenticate
    _ = get_access_token()

    # 3. Then read CSV
    with eodc_s3.open(POINTS_CSV, "r") as f:
        geosphere_stations = pd.read_csv(f)

    subset = geosphere_stations.iloc[STATION_START:STATION_LIMIT]

    print(f"Data will be stored in: {STORE_PATH}")

    failed_stations = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(get_station_data, station): station
            for idx, station in subset.iterrows()
        }

        for future in concurrent.futures.as_completed(futures):
            station = futures[future]
            result = future.result()

            if result is None:
                print(f"Skipping failed fetch for station {station['points']}")
                failed_stations.append(station)
                continue

            try:
                write_to_zarr(result, STORE_PATH)
            except Exception as e:
                print(f"Failed to write station {station['points']}: {e}")
                failed_stations.append(station)

    # Save failed stations to CSV if any
    if failed_stations:
        failed_df = pd.DataFrame(failed_stations)
        failed_csv_path = POINTS_CSV.replace(".csv", "_failed.csv")
        with eodc_s3.open(failed_csv_path, "w") as f:
            failed_df.to_csv(f, index=False)
        print(f"Failed stations saved to: {failed_csv_path}")
    else:
        print("All stations written successfully!")


if __name__ == "__main__":
    main()