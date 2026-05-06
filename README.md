# dt-climate-zarr



This repository contains various notebooks and Python files for saving DestinE ClimateDT data as ZARR files.



SnowDepthWF contains the Workflow for the snow depth data of the IFS-NEMO model over the alps. Austria contains the workflow for different models for austria.



To process single models use dt-climate-data-extraction-austria-getPointCsv.ipynb to get a point.csv for you region, then use dt-climate-data-extraction-austria-getPointCsv.ipynb for the zarr processing.



If you want to rechunk or add multiple zarr stores into one use dt-climate-data-extraction-austria-getPointCsv.ipynb.

