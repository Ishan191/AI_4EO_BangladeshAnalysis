# AI_4EO_BangladeshAnalysis

## Project Overview
This repository contains Python scripts for geospatial data processing and machine learning-based UHI prediction over Bangladesh using Random Forest Regressor and LSTM models.

## Files in This Repository
1. **RFC_Bangladesh.py** - Contains functions for preprocessing geospatial and environmental data and implementing a random forest regressor model.
2. **LSTM_Bangladesh.py** - Implements a LSTM model for predicting UHI using environmental variables.
3. **TifftoShp_ClipFilesAI4EO.py** - Extracts and processes raster (.tif) files, clipping them using shapefiles of the specified country/region.
4. **Scaling1to9.py** – Used for scaling down high-resolution (.tif) files from 1 km to 9 km.
5. **DataDownloadScript.js** – Used for downloading relevant feature .Tiff files from Google Earth Engine

## Dependencies
The scripts require the following Python libraries:
```sh
numpy
pandas
torch
matplotlib
sklearn
rasterio
geopandas
glob
```

## Usage
### 1. Downloading Data
- **DataDownloadScript.jsy** is used for downloading .tif raster files of the relevant features eg. NDVI,Albedo,Precipitation on Google Earth Engine.
  
### 2. Data Preprocessing
- **Scaling1to9.py** is used for features whose .tif raster files have a higher resolution and need to be scaled down to match the resolution of other datasets.
- **TifftoShp_ClipFilesAI4EO.py** processes .tif raster files and clips them using shapefiles.
- The processed data is then exported to a shapefile.

### 3. UHI Prediction Model
- **RFC_Bangladesh.py** processes the shapefiles and extracts data into dataframes. These dataframes are first saved and then used in the Random Forest Regressor model. A separate correlation matrix is also generated.
- **ProjectWeeks_LSTM.py** trains an LSTM model using geospatial and climate-related variables obtained from the datasets saved using RFC_Bangladesh.py.
- While The models are trained on district-wise environmental features and predict UHI values they only consider NDVI,Albedo,Wind,Wind Direction,Precipitation,GEOCODE,LST,DISTCODE and UHI as features.

### 4. Running the Scripts
For raster data download,use:
```sh
javascript DataDownloadScript.js  
```
- Ensure that a different folder name is set to store the raster files of different fetures under variable - driveFolder
- The dataset is also required to be changed for varying features under - var dataset = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
- The band can be set under the feature name e.g var uWind = dataset.select('u_component_of_wind_10m').mean() where uWind is the variable and 'u_component_of_wind_10m' is the band being downloaded from the dataset.
- Repeat process for downloading all the relevant datasets.

For raster data processing, use:
```sh
python Scaling1to9.py  # If necessary
python TifftoShp_ClipFilesAI4EO.py
```
To execute the Random Forest Regressor model training and evaluation, run:
```sh
python RFC_Bangladesh.py
```
To execute the LSTM model training and evaluation, run:
```sh
python RFC_Bangladesh.py  # Data preparation
python ProjectWeeks_LSTM.py
```
Ensure the target year and month are set within the duration of 2000-2020.

## Output
- Save and download raster data for different features.
- Clipped raster data is exported as a shapefile, based on whether scaling is required.
- The trained RFC model produces evaluation metrics (MSE, RMSE, MAE, R² Score) along with a correlation heatmap.
- The trained LSTM model produces evaluation metrics (MSE, RMSE, MAE, R² Score) and a model comparison of Actual Vs. Predicted UHI values for a selected area and month.


