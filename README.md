# dt-climate-zarr



This repository contains various notebooks and Python files for saving DestinE ClimateDT data as ZARR files.



The files are split into Generation1 and Generation2 depending on which data used from destinE.



\## Generation 1:



\### Austria 

Austria contains the workflow for different models for Austria.



To process single models use dt-climate-data-extraction-austria-getPointCsv.ipynb to get a point.csv for you region, then use ClimateDT\_Austria.py for downloading the data. You have to specify your desired scenarios in climate\_dt\_scenarios\_sorted.csv



\### SnowDepth

SnowDepthWF contains the Workflow for the snow depth data of the IFS-NEMO model over the alps.





\## Generation 2:



\### Volumetric soil moisture (vsm)

This folder contains the script for downloading the vsw data (as netcdf) and the writing them into a zarr3 store for different models





\## STAC

With dt2stac you can create the stac collection and python-stac\_ClimateDT\_Austria is an example notebook on how to access the data via stac

