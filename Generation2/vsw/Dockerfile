FROM python:3.10-slim

# Runtime libs cartopy/PROJ/GEOS need for country_polygons() etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/desp-authentication.py src/desp-authentication.py
COPY polytope_zarr.py .
COPY destine_portfolio.py .
COPY models_hist_ssp3.py .

ENTRYPOINT ["python", "models_hist_ssp3.py"]
