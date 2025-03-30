# AI_4EO_BangladeshAnalysis

Project Overview
This repository contains Python scripts for geospatial data processing and machine learning-based UHI prediction over Bangladesh using Random Forest Regressor and LSTM models.
Files in This Repository
1.	RFC_Bangladesh.py - Contains functions for preprocessing geospatial and environmental data and implementing a random forest regressor model.
2.	LSTM_Bangladesh.py - Implements an LSTM model for predicting UHI using environmental variables.
3.	TifftoShp_ClipFilesAI4EO.py - Extracts and processes raster (.tif) files, clipping them using shapefiles of the set country/particular part of the country.
4.	Scaling1to9.py – Used for scaling down a set of high resolution(.tif) files from 1 to 9 km.
Dependencies
The scripts require the following Python libraries:
•	numpy
•	pandas
•	torch
•	matplotlib
•	sklearn
•	rasterio
•	geopandas
•	glob
Usage
1. Data Preprocessing
•	Scaling1to9.py script is used over features whose .tif raster files have a higher resolution and need to be scaled down to match the resolution of other datasets.
•	The TifftoShp_ClipFiles.py script processes .tif raster files and clips them using shapefiles.
•	The processed data is then exported to a shapefile.
2. UHI Prediction Model
•	RFC_Bangladesh.py processes the shape files and extracts data into a set of dataframes. These dataframes are first saved and then implemented as a part of the Random Forest Regressor model. A separate correlation matrix is also generated. 
•	ProjectWeeks_LSTM.py trains an LSTM model using geospatial and climate-related variables which are obtained from the datasets saved using RFC_Bangladesh.py.
•	The models are trained on district-wise environmental features and predicts UHI values.
3. Running the Scripts
For raster data processing, use:
Scaling1to9.py (if necessary) followed by TifftoShp_ClipFiles.py
To execute the Random Forest Regressor model training and evaluation, run:
RFC_Bangladesh.py
To execute the LSTM model training and evaluation, run:
RFC_Bangladesh.py followed by ProjectWeeks_LSTM.py
The target year and month is also required to be set in the script which should fall within the duration of 2000-2020.
Output
•	Clipped raster data based on whether scaling is required or not is exported as a shapefile.
•	The trained RFC model produces evaluation metrics (MSE, RMSE, MAE, R² Score) along with a correlation heatmap.
•	The trained LSTM model produces evaluation metrics (MSE, RMSE, MAE, R² Score) along with a model comparison of Actual Vs Predicted UHI values of the set area for a particular month.

