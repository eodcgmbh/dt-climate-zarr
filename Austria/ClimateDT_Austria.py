"""
Fetches Climate DT timeseries data from the Destination Earth Polytope API
for multiple scenarios and station points, storing results incrementally
to Zarr stores on S3.
"""

import os
import sys
import logging
import subprocess
import concurrent.futures
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import s3fs

logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUTH_SCRIPT = "./src/desp-authentication.py"
POLYTOPE_ADDRESS = "polytope.lumi.apps.dte.destination-earth.eu"

SCENARIOS_PATH = "destine-climate-dt/climate_dt_scenarios_sorted.csv"
POINTS_FOLDER = "destine-climate-dt/points/"

MAX_WORKERS = 25
STATION_LIMIT = None  # set to an integer to process only a subset of stations


# ---------------------------------------------------------------------------
# Chunk specification for Zarr writing
# ---------------------------------------------------------------------------

CHUNK_SPEC = {
    "pointid": 1,
    "levelist": 1,
    "number": 1,
    "datetime": 1,
    "t": "auto",
}

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def get_access_token() -> str:
    """Run the DESP authentication script and return the access token."""
    result = subprocess.run(
        [
            "python", AUTH_SCRIPT,
            "-u", os.environ.get("DESP_USER"),
            "-p", os.environ.get("DESP_PASS"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    token = result.stdout.strip().split("}\n")[-1]
    return token


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
    resolution: str = "high",
    feature: Literal["timeseries", "polygon"] = "timeseries",
    time_resolution: str = "0000/to/2300",
) -> xr.Dataset | None:
    """
    Fetch a Climate DT timeseries from the Polytope API.

    Returns xr.Dataset or None on failure.
    """
    import earthkit.data

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
        "class": "d1",
        "dataset": "climate-dt",
        "generation": "1",
        "expver": "0001",
        "stream": "clte",
        "type": "fc",
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
        print(f"[ERROR] earthkit request failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Per-station worker
# ---------------------------------------------------------------------------

def get_station_data(
    geosphere_station: pd.Series,
    experiment: str,
    activity: str,
    level_type: str,
    datestring: str,
    model: str,
    parameter: str,
    resolution: str,
) -> xr.Dataset | None:
    """Fetch data for a single station and reshape for Zarr storage."""
    latlon = [[float(geosphere_station["latitude"]), float(geosphere_station["longitude"])]]

    ts = extract_cdt_ts(
        experiment=experiment,
        activity=activity,
        level_type=level_type,
        datestring=datestring,
        model=model,
        parameter=parameter,
        location=latlon,
        resolution=resolution,
    )

    if ts is None:
        return None

    point_id = int(geosphere_station["points"])

    return (
        ts.stack(pointid=("latitude", "longitude"))
        .reset_index("pointid")
        .assign_coords({"pointid": [point_id]})
        .transpose("pointid", ...)
    )


# ---------------------------------------------------------------------------
# Zarr write helper
# ---------------------------------------------------------------------------

def write_to_zarr(dataset: xr.Dataset, store_path: str, eodc_s3: s3fs.S3FileSystem) -> None:
    """Append a station dataset to the Zarr store (create if missing)."""
    zarr_store = s3fs.S3Map(root=store_path, s3=eodc_s3)
    chunked = dataset.chunk(chunks=CHUNK_SPEC)
    if not eodc_s3.exists(store_path):
        chunked.to_zarr(store=zarr_store, mode="w")
    else:
        chunked.to_zarr(store=zarr_store, append_dim="pointid")


# ---------------------------------------------------------------------------
# Per-scenario processor
# ---------------------------------------------------------------------------

def process_scenario(scenario: pd.Series, eodc_s3: s3fs.S3FileSystem) -> None:
    """Fetch and store data for all stations in a single scenario."""

    model = scenario["model"]
    parameter = scenario["params"]
    level_type = scenario["level type"]
    experiment = scenario["experiment"]
    resolution = scenario["resolution"]
    temp_extent = scenario["temporal extent"]
    activity = scenario["activity"]

    # Determine correct points file based on model
    if model in ["IFS-NEMO", "ICON"]:
        points_path = f"{POINTS_FOLDER}points_{model}_{activity}.csv"
    else:  # IFS-FESOM
        points_path = f"{POINTS_FOLDER}points_{model}_{experiment}.csv"

    print(f"\n--- Processing scenario ---", flush=True)
    print(f"Model:      {model}", flush=True)
    print(f"Experiment: {experiment}", flush=True)
    print(f"Activity:   {activity}", flush=True)
    print(f"Points:     {points_path}", flush=True)

    # Load points
    try:
        with eodc_s3.open(points_path, "r") as f:
            points = pd.read_csv(f)
    except Exception as e:
        print(f"[ERROR] Could not load points file {points_path}: {e}", flush=True)
        return

    subset = points.iloc[:STATION_LIMIT] if STATION_LIMIT is not None else points

    store_path = (
        f"destine-climate-dt/Austria/"
        f"{model}_{level_type}_{activity}_{experiment}.zarr"
    )
    print(f"Data will be stored in: {store_path}", flush=True)

    datestring = get_datestring(temp_extent)
    failed_stations = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                get_station_data,
                station,
                experiment,
                activity,
                level_type,
                datestring,
                model,
                parameter,
                resolution,
            ): station
            for _, station in subset.iterrows()
        }

        for future in concurrent.futures.as_completed(futures):
            station = futures[future]
            result = future.result()

            if result is None:
                print(f"Skipping failed fetch for station {station['points']}", flush=True)
                failed_stations.append(station)
                continue

            try:
                write_to_zarr(result, store_path, eodc_s3)
            except Exception as e:
                print(f"Failed to write station {station['points']}: {e}", flush=True)
                failed_stations.append(station)

    # Save failed stations to CSV if any
    if failed_stations:
        failed_df = pd.DataFrame(failed_stations)
        failed_csv_path = points_path.replace(".csv", "_failed.csv")
        try:
            with eodc_s3.open(failed_csv_path, "w") as f:
                failed_df.to_csv(f, index=False)
            print(f"Failed stations saved to: {failed_csv_path}", flush=True)
        except Exception as e:
            print(f"[ERROR] Could not save failed stations CSV: {e}", flush=True)
    else:
        print(f"All stations written successfully for {model} {experiment}!", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Script started", flush=True)

    # 1. Create S3 connection
    eodc_s3 = s3fs.S3FileSystem(
        key=os.environ.get("S3_KEY"),
        secret=os.environ.get("S3_SECRET"),
        client_kwargs={"endpoint_url": "https://objects.eodc.eu"}
    )

    print(f"S3_KEY set:    {bool(os.environ.get('S3_KEY'))}", flush=True)
    print(f"DESP_USER set: {bool(os.environ.get('DESP_USER'))}", flush=True)

    # 2. Authenticate
    _ = get_access_token()

    # 3. Load scenarios
    with eodc_s3.open(SCENARIOS_PATH, "r") as f:
        scenarios = pd.read_csv(f, sep=";")

    print(f"Found {len(scenarios)} scenario(s) to process", flush=True)
    print(scenarios, flush=True)

    # 4. Process each scenario
    for _, scenario in scenarios.iloc[:1].iterrows():
        process_scenario(scenario, eodc_s3)

    print("\nAll scenarios completed!", flush=True)


if __name__ == "__main__":
    main()