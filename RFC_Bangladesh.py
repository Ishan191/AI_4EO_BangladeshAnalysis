import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# Define paths to shapefiles
shapefile_paths = {
    "area": "D:/Python Test Codes/DISTCODE/Area4_RetainedData.shp",
    "ndvi": "D:/Python Test Codes/AI4EO/NDVIDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    "albedo": "D:/Python Test Codes/AI4EO/AlbedoDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    "lst": "D:/Python Test Codes/AI4EO/LSTDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    "UHI4": "D:/Python Test Codes/AI4EO/UHIDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    "prec": "D:/Python Test Codes/AI4EO/PrecpDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    
    "area3": "D:/Python Test Codes/DISTCODE/Area3_RetainedData.shp",
    "ndvi3": "D:/Python Test Codes/AI4EO/NDVIDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "albedo3": "D:/Python Test Codes/AI4EO/AlbedoDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "lst3": "D:/Python Test Codes/AI4EO/LSTDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "prec3": "D:/Python Test Codes/AI4EO/PrecpDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "UHI3": "D:/Python Test Codes/AI4EO/UHIDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    
    "area1": "D:/Python Test Codes/DISTCODE/Area1_RetainedData.shp",
    "ndvi1": "D:/Python Test Codes/AI4EO/NDVIDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "albedo1": "D:/Python Test Codes/AI4EO/AlbedoDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "prec1": "D:/Python Test Codes/AI4EO/PrecpDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "UHI1": "D:/Python Test Codes/AI4EO/UHIDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "lst1": "D:/Python Test Codes/AI4EO/LSTDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    
    "area2": "D:/Python Test Codes/DISTCODE/Area2_RetainedData.shp",
    "ndvi2": "D:/Python Test Codes/AI4EO/NDVIDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "albedo2": "D:/Python Test Codes/AI4EO/AlbedoDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "prec2": "D:/Python Test Codes/AI4EO/PrecpDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "UHI2": "D:/Python Test Codes/AI4EO/UHIDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "lst2": "D:/Python Test Codes/AI4EO/LSTDistrict_dissolved_band_data_with_tiffs_Area2.shp"
}

shapefile_pathsWind = {
    "area1u": "D:/Python Test Codes/AI4EO/WindUDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "area2u": "D:/Python Test Codes/AI4EO/WindUDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "area3u": "D:/Python Test Codes/AI4EO/WindUDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "area4u": "D:/Python Test Codes/AI4EO/WindUDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    
    "area1v": "D:/Python Test Codes/AI4EO/WindVDistrict_dissolved_band_data_with_tiffs_Area1.shp",
    "area2v": "D:/Python Test Codes/AI4EO/WindVDistrict_dissolved_band_data_with_tiffs_Area2.shp",
    "area3v": "D:/Python Test Codes/AI4EO/WindVDistrict_dissolved_band_data_with_tiffs_Area3.shp",
    "area4v": "D:/Python Test Codes/AI4EO/WindVDistrict_dissolved_band_data_with_tiffs_Area4.shp",
    
}

# Load shapefiles into GeoDataFrames
gdf_area1U = gpd.read_file(shapefile_pathsWind["area1u"])
gdf_area2U = gpd.read_file(shapefile_pathsWind["area2u"])
gdf_area3U = gpd.read_file(shapefile_pathsWind["area3u"])
gdf_area4U = gpd.read_file(shapefile_pathsWind["area4u"])

gdf_area1V = gpd.read_file(shapefile_pathsWind["area1v"])
gdf_area2V = gpd.read_file(shapefile_pathsWind["area2v"])
gdf_area3V = gpd.read_file(shapefile_pathsWind["area3v"])
gdf_area4V = gpd.read_file(shapefile_pathsWind["area4v"])

# Load shapefiles into GeoDataFrames
gdf_area4 = gpd.read_file(shapefile_paths["area"])
gdf_ndvi4 = gpd.read_file(shapefile_paths["ndvi"])
gdf_albedo4 = gpd.read_file(shapefile_paths["albedo"])
gdf_lst4 = gpd.read_file(shapefile_paths["lst"])
gdf_prec4 = gpd.read_file(shapefile_paths["prec"])
gdf_uhi4 = gpd.read_file(shapefile_paths["UHI4"])

gdf_area3 = gpd.read_file(shapefile_paths["area3"])
gdf_ndvi3 = gpd.read_file(shapefile_paths["ndvi3"])
gdf_albedo3 = gpd.read_file(shapefile_paths["albedo3"])
gdf_lst3 = gpd.read_file(shapefile_paths["lst3"])
gdf_prec3 = gpd.read_file(shapefile_paths["prec3"])
gdf_uhi3 = gpd.read_file(shapefile_paths["UHI3"])

gdf_area1 = gpd.read_file(shapefile_paths["area1"])
gdf_ndvi1 = gpd.read_file(shapefile_paths["ndvi1"])
gdf_albedo1 = gpd.read_file(shapefile_paths["albedo1"])
gdf_lst1 = gpd.read_file(shapefile_paths["lst1"])
gdf_prec1 = gpd.read_file(shapefile_paths["prec1"])
gdf_uhi1 = gpd.read_file(shapefile_paths["UHI1"])

gdf_area2 = gpd.read_file(shapefile_paths["area2"])
gdf_ndvi2 = gpd.read_file(shapefile_paths["ndvi2"])
gdf_albedo2 = gpd.read_file(shapefile_paths["albedo2"])
gdf_lst2 = gpd.read_file(shapefile_paths["lst2"])
gdf_prec2 = gpd.read_file(shapefile_paths["prec2"])
gdf_uhi2 = gpd.read_file(shapefile_paths["UHI2"])


# Convert to DataFrames and drop geometry columns
df_area4 = gdf_area4.drop(columns=['geometry'])
df_ndvi4 = gdf_ndvi4.drop(columns=['geometry'])
df_albedo4 = gdf_albedo4.drop(columns=['geometry'])
df_lst4 = gdf_lst4.drop(columns=['geometry'])
df_prec4 = gdf_prec4.drop(columns=['geometry'])
df_uhi4 = gdf_uhi4.drop(columns=['geometry'])
df_wind4u = gdf_area4U.drop(columns=['geometry'])
df_wind4v = gdf_area4V.drop(columns=['geometry'])

df_area3 = gdf_area3.drop(columns=['geometry'])
df_ndvi3 = gdf_ndvi3.drop(columns=['geometry'])
df_albedo3 = gdf_albedo3.drop(columns=['geometry'])
df_lst3 = gdf_lst3.drop(columns=['geometry'])
df_prec3 = gdf_prec3.drop(columns=['geometry'])
df_uhi3 = gdf_uhi3.drop(columns=['geometry'])
df_wind3u = gdf_area3U.drop(columns=['geometry'])
df_wind3v = gdf_area3V.drop(columns=['geometry'])

df_area1 = gdf_area1.drop(columns=['geometry'])
df_ndvi1 = gdf_ndvi1.drop(columns=['geometry'])
df_albedo1 = gdf_albedo1.drop(columns=['geometry'])
df_lst1 = gdf_lst1.drop(columns=['geometry'])
df_prec1 = gdf_prec1.drop(columns=['geometry'])
df_uhi1 = gdf_uhi1.drop(columns=['geometry'])
df_wind1u = gdf_area1U.drop(columns=['geometry'])
df_wind1v = gdf_area1V.drop(columns=['geometry'])

df_area2 = gdf_area2.drop(columns=['geometry'])
#df_areageo1 = gdf_area1.copy()
df_ndvi2 = gdf_ndvi2.drop(columns=['geometry'])
df_albedo2 = gdf_albedo2.drop(columns=['geometry'])
df_lst2 = gdf_lst2.drop(columns=['geometry'])
df_prec2 = gdf_prec2.drop(columns=['geometry'])
df_uhi2 = gdf_uhi2.drop(columns=['geometry'])
df_wind2u = gdf_area2U.drop(columns=['geometry'])
df_wind2v = gdf_area2V.drop(columns=['geometry'])

# Identify columns for NDVI, Albedo, Precipitaion ,UHI ,Wind and LST
ndvi_columns4 = [col for col in df_ndvi4.columns if col.startswith("Bd")]
albedo_columns4 = [col for col in df_albedo4.columns if col.startswith("Bd")]
lst_columns4 = [col for col in df_lst4.columns if col.startswith("Bd")]
prec_columns4 = [col for col in df_prec4.columns if col.startswith("Bd")]
uhi_columns4 = [col for col in df_uhi4.columns if col.startswith("Bd")]
windu_columns4 = [col for col in df_wind4u.columns if col.startswith("Bd")]
windv_columns4 = [col for col in df_wind4v.columns if col.startswith("Bd")]


ndvi_columns3 = [col for col in df_ndvi3.columns if col.startswith("Bd")]
albedo_columns3 = [col for col in df_albedo3.columns if col.startswith("Bd")]
lst_columns3 = [col for col in df_lst3.columns if col.startswith("Bd")]
prec_columns3 = [col for col in df_prec3.columns if col.startswith("Bd")]
uhi_columns3 = [col for col in df_uhi3.columns if col.startswith("Bd")]
windu_columns3 = [col for col in df_wind3u.columns if col.startswith("Bd")]
windv_columns3 = [col for col in df_wind3v.columns if col.startswith("Bd")]


ndvi_columns1 = [col for col in df_ndvi1.columns if col.startswith("Bd")]
albedo_columns1 = [col for col in df_albedo1.columns if col.startswith("Bd")]
lst_columns1 = [col for col in df_lst1.columns if col.startswith("Bd")]
prec_columns1 = [col for col in df_prec1.columns if col.startswith("Bd")]
uhi_columns1 = [col for col in df_uhi1.columns if col.startswith("Bd")]
windu_columns1 = [col for col in df_wind1u.columns if col.startswith("Bd")]
windv_columns1 = [col for col in df_wind1v.columns if col.startswith("Bd")]

ndvi_columns2 = [col for col in df_ndvi2.columns if col.startswith("Bd")]
albedo_columns2 = [col for col in df_albedo2.columns if col.startswith("Bd")]
lst_columns2 = [col for col in df_lst2.columns if col.startswith("Bd")]
prec_columns2 = [col for col in df_prec2.columns if col.startswith("Bd")]
uhi_columns2 = [col for col in df_uhi2.columns if col.startswith("Bd")]
windu_columns2 = [col for col in df_wind2u.columns if col.startswith("Bd")]
windv_columns2 = [col for col in df_wind2v.columns if col.startswith("Bd")]


# Melt Albedo data 
df_albedo_long4 = df_albedo4.melt(id_vars=['DISTNAME'], value_vars=albedo_columns4,
                                 var_name='Band', value_name='Albedo_Value')
df_albedo_long3 = df_albedo3.melt(id_vars=['DISTNAME'], value_vars=albedo_columns3,
                                  var_name='Band', value_name='Albedo_Value')
df_albedo_long1 = df_albedo1.melt(id_vars=['DISTNAME'], value_vars=albedo_columns1,
                                 var_name='Band', value_name='Albedo_Value')
df_albedo_long2 = df_albedo2.melt(id_vars=['DISTNAME'], value_vars=albedo_columns2,
                                 var_name='Band', value_name='Albedo_Value')

# Melt NDVI data 
df_ndvi_long4 = df_ndvi4.melt(id_vars=['DISTNAME'], value_vars=ndvi_columns4,
                             var_name='Band', value_name='NDVI_Value')
df_ndvi_long3 = df_ndvi3.melt(id_vars=['DISTNAME'], value_vars=ndvi_columns3,
                             var_name='Band', value_name='NDVI_Value')
df_ndvi_long1 = df_ndvi1.melt(id_vars=['DISTNAME'], value_vars=ndvi_columns1,
                             var_name='Band', value_name='NDVI_Value')
df_ndvi_long2 = df_ndvi2.melt(id_vars=['DISTNAME'], value_vars=ndvi_columns2,
                             var_name='Band', value_name='NDVI_Value')


# Melt LST data 
df_lst_long4 = df_lst4.melt(id_vars=['DISTNAME'], value_vars=lst_columns4,
                          var_name='Band', value_name='lst_Value')
df_lst_long3 = df_lst3.melt(id_vars=['DISTNAME'], value_vars=lst_columns3,
                          var_name='Band', value_name='lst_Value')
df_lst_long1 = df_lst1.melt(id_vars=['DISTNAME'], value_vars=lst_columns1,
                          var_name='Band', value_name='lst_Value')
df_lst_long2 = df_lst2.melt(id_vars=['DISTNAME'], value_vars=lst_columns2,
                          var_name='Band', value_name='lst_Value')


# Melt Precipitation data 
df_prec_long4 = df_prec4.melt(id_vars=['DISTNAME'], value_vars=prec_columns4,
                          var_name='Band', value_name='precipitation_Value')
df_prec_long3 = df_prec3.melt(id_vars=['DISTNAME'], value_vars=prec_columns3,
                          var_name='Band', value_name='precipitation_Value')
df_prec_long1 = df_prec1.melt(id_vars=['DISTNAME'], value_vars=prec_columns1,
                          var_name='Band', value_name='precipitation_Value')
df_prec_long2 = df_prec2.melt(id_vars=['DISTNAME'], value_vars=prec_columns2,
                          var_name='Band', value_name='precipitation_Value')


# Melt UHI data 
df_uhi_long4 = df_uhi4.melt(id_vars=['DISTNAME'], value_vars=uhi_columns4,
                          var_name='Band', value_name='uhi_Value')
df_uhi_long3 = df_uhi3.melt(id_vars=['DISTNAME'], value_vars=uhi_columns3,
                          var_name='Band', value_name='uhi_Value')
df_uhi_long1 = df_uhi1.melt(id_vars=['DISTNAME'], value_vars=uhi_columns1,
                          var_name='Band', value_name='uhi_Value')
df_uhi_long2 = df_uhi2.melt(id_vars=['DISTNAME'], value_vars=uhi_columns2,
                          var_name='Band', value_name='uhi_Value')


# Melt Wind U data
df_windu_long4 = df_wind4u.melt(id_vars=['DISTNAME'], value_vars=windu_columns4,
                          var_name='Band', value_name='windu_Value')
df_windu_long3 = df_wind3u.melt(id_vars=['DISTNAME'], value_vars=windu_columns3,
                          var_name='Band', value_name='windu_Value')
df_windu_long1 = df_wind1u.melt(id_vars=['DISTNAME'], value_vars=windu_columns1,
                          var_name='Band', value_name='windu_Value')
df_windu_long2 = df_wind2u.melt(id_vars=['DISTNAME'], value_vars=windu_columns2,
                          var_name='Band', value_name='windu_Value')

# Melt Wind V data (long format)
df_windv_long4 = df_wind4v.melt(id_vars=['DISTNAME'], value_vars=windv_columns4,
                          var_name='Band', value_name='windv_Value')
df_windv_long3 = df_wind3v.melt(id_vars=['DISTNAME'], value_vars=windv_columns3,
                          var_name='Band', value_name='windv_Value')
df_windv_long1 = df_wind1v.melt(id_vars=['DISTNAME'], value_vars=windv_columns1,
                          var_name='Band', value_name='windv_Value')
df_windv_long2 = df_wind2v.melt(id_vars=['DISTNAME'], value_vars=windv_columns2,
                          var_name='Band', value_name='windv_Value')

# Merge all features of Area 4 into a single using DISTNAME and Band
df_merged4 = pd.merge(df_albedo_long4,df_ndvi_long4, on=['DISTNAME', 'Band'], how='left')
df_merged4 = pd.merge(df_merged4, df_lst_long4, on=['DISTNAME', 'Band'], how='left')
df_merged4 = pd.merge(df_merged4, df_prec_long4, on=['DISTNAME', 'Band'], how='left')
df_merged4 = pd.merge(df_merged4, df_uhi_long4, on=['DISTNAME', 'Band'], how='left')
df_merged4 = pd.merge(df_merged4, df_windu_long4, on=['DISTNAME', 'Band'], how='left')
df_merged4 = pd.merge(df_merged4, df_windv_long4, on=['DISTNAME', 'Band'], how='left')

# Separate Month and Year 
df_final_int4 = df_area4.merge(df_merged4, on='DISTNAME', how='left')
df_final_int4['Year'] = [int(i[4:8]) for i in df_final_int4['Band']]  
df_final_int4['Month'] = [int(i[8:].replace('_', '')) + 1 for i in df_final_int4['Band']] 

# Calculate Wind and Wind Direction
df_final4 = df_final_int4.interpolate(method='polynomial', order=2)
df_final4['wind_Value'] = np.sqrt(df_final4['windu_Value']**2 + df_final4['windv_Value']**2)
df_final4['wind_Direction'] = (270 - np.degrees(np.arctan2(df_final4['windv_Value'], df_final4['windu_Value']))) % 360
df_final4['uhi_Value'] = df_final4['uhi_Value'].fillna(0)
df_final4 = df_final4[(df_final4['Year'] < 2021)]
df_final4 = df_final4[df_final4['DISTCODE'] != 0]

# Merge all features of Area 3 into a single using DISTNAME and Band
df_merged3 = pd.merge(df_albedo_long3, df_ndvi_long3,  on=['DISTNAME', 'Band'], how='left')
df_merged3 = pd.merge(df_merged3, df_lst_long3, on=['DISTNAME', 'Band'], how='left')
df_merged3 = pd.merge(df_merged3, df_prec_long3, on=['DISTNAME', 'Band'], how='left')
df_merged3 = pd.merge(df_merged3, df_uhi_long3, on=['DISTNAME', 'Band'], how='left')
df_merged3 = pd.merge(df_merged3, df_windu_long3, on=['DISTNAME', 'Band'], how='left')
df_merged3 = pd.merge(df_merged3, df_windv_long3, on=['DISTNAME', 'Band'], how='left')

# Separate Month and Year 
df_final3_int = df_area3.merge(df_merged3, on='DISTNAME', how='left')
df_final3_int['Year'] = [int(i[4:8]) for i in df_final3_int['Band']]  # Extracting the first 4 characters for the year
df_final3_int['Month'] = [int(i[8:].replace('_', '')) + 1 for i in df_final3_int['Band']]

# Calculate Wind and Wind Direction
df_final3 = df_final3_int.interpolate(method='polynomial', order=2)
df_final3['wind_Value'] = np.sqrt(df_final3['windu_Value']**2 + df_final3['windv_Value']**2)
df_final3['wind_Direction'] = (270 - np.degrees(np.arctan2(df_final3['windv_Value'], df_final3['windu_Value']))) % 360
df_final3['uhi_Value'] = df_final3['uhi_Value'].fillna(0)
df_final3 = df_final3[(df_final3['Year'] < 2021)]
df_final3 = df_final3[df_final3['DISTCODE'] != 0]

# Merge all features of Area 1 into a single using DISTNAME and Band
df_merged1 = pd.merge(df_albedo_long1, df_ndvi_long1,  on=['DISTNAME', 'Band'], how='left')
df_merged1 = pd.merge(df_merged1, df_lst_long1, on=['DISTNAME', 'Band'], how='left')
df_merged1 = pd.merge(df_merged1, df_prec_long1, on=['DISTNAME', 'Band'], how='left')
df_merged1 = pd.merge(df_merged1, df_uhi_long1, on=['DISTNAME', 'Band'], how='left')
df_merged1 = pd.merge(df_merged1, df_windu_long1, on=['DISTNAME', 'Band'], how='left')
df_merged1 = pd.merge(df_merged1, df_windv_long1, on=['DISTNAME', 'Band'], how='left')

# Separate Month and Year 
df_final1_int = df_area1.merge(df_merged1, on='DISTNAME', how='left')
df_final1_int['Band'] = df_final1_int['Band'].astype(str)
df_final1_int['Year'] = [int(i[4:8]) if i != 'nan' else None for i in df_final1_int['Band']]  # Extracting the first 4 characters for the year
df_final1_int['Month'] = [int(i[8:].replace('_', '')) + 1 if i != 'nan' else None for i in df_final1_int['Band']]  # Extracting the remaining part for the month

# Calculate Wind and Wind Direction
df_final1 = df_final1_int.interpolate(method='polynomial', order=2)
df_final1['wind_Value'] = np.sqrt(df_final1['windu_Value']**2 + df_final1['windv_Value']**2)
df_final1['wind_Direction'] = (270 - np.degrees(np.arctan2(df_final1['windv_Value'], df_final1['windu_Value']))) % 360
df_final1['uhi_Value'] = df_final1['uhi_Value'].fillna(0)
df_final1 = df_final1[(df_final1['Year'] < 2021)]
df_final1 = df_final1[df_final1['DISTCODE'] != 0]
#df_final1 = pd.merge(df_final1, df_geo1, on=['DISTNAME', 'DISTCODE'], how='left')

# Merge all features of Area 2 into a single using DISTNAME and Band
df_merged2 = pd.merge(df_albedo_long2, df_ndvi_long2,  on=['DISTNAME', 'Band'], how='left')
df_merged2 = pd.merge(df_merged2, df_lst_long2, on=['DISTNAME', 'Band'], how='left')
df_merged2 = pd.merge(df_merged2, df_prec_long2, on=['DISTNAME', 'Band'], how='left')
df_merged2 = pd.merge(df_merged2, df_uhi_long2, on=['DISTNAME', 'Band'], how='left')
df_merged2 = pd.merge(df_merged2, df_windu_long2, on=['DISTNAME', 'Band'], how='left')
df_merged2 = pd.merge(df_merged2, df_windv_long2, on=['DISTNAME', 'Band'], how='left')

# Separate Month and Year 
df_final2_int = df_area2.merge(df_merged2, on='DISTNAME', how='left')
df_final2_int['Band'] = df_final2_int['Band'].astype(str)
df_final2_int['Year'] = [int(i[4:8]) if i != 'nan' else None for i in df_final2_int['Band']]  # Extracting the first 4 characters for the year
df_final2_int['Month'] = [int(i[8:].replace('_', '')) + 1 if i != 'nan' else None for i in df_final2_int['Band']]  # Extracting the remaining part for the month

# Calculate Wind and Wind Direction
df_final2 = df_final2_int.interpolate(method='polynomial', order=2)
df_final2['wind_Value'] = np.sqrt(df_final2['windu_Value']**2 + df_final2['windv_Value']**2)
df_final2['wind_Direction'] = (270 - np.degrees(np.arctan2(df_final2['windv_Value'], df_final2['windu_Value']))) % 360
df_final2['uhi_Value'] = df_final2['uhi_Value'].fillna(0)
df_final2['lst_Value'] = df_final2['lst_Value'].fillna(0)
df_final2['precipitation_Value'] = df_final2['precipitation_Value'].fillna(0)
df_final2 = df_final2[(df_final2['Year'] < 2021)]
df_final2 = df_final2[df_final2['DISTCODE'] != 0]

df_final4 = df_final4[df_final4["Band"] != "nan"]
df_final1 = df_final1[df_final1["Band"] != "nan"]
df_final2 = df_final2[df_final2["Band"] != "nan"]
df_final3 = df_final3[df_final3["Band"] != "nan"]

df_final4.to_feather('df_finalArea4.feather')
df_final3.to_feather('df_finalArea3.feather')
df_final1.to_feather('df_finalArea1.feather')
df_final2.to_feather('df_finalArea2.feather')


features = ['NDVI_Value','Albedo_Value','GEOCODE', 'DISTCODE'
            ,'precipitation_Value','wind_Value','wind_Direction']
target = ['uhi_Value']

df_combinedFinal = pd.concat([df_final4,df_final2])
X_train = df_combinedFinal[features]
y_train = df_combinedFinal[target]
X_val = df_final3[features]
y_val = df_final3[target]
X_test = df_final1[features]
y_test = df_final1[target]

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=10
                              , min_samples_split=10, random_state=42)
model.fit(X_train, y_train)


# Evaluate on the validation set
# Predictions for test and validation sets

y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Calculate regression metrics for the validation set
val_mae = mean_absolute_error(y_val, y_pred_val)
val_mse = mean_squared_error(y_val, y_pred_val)
val_rmse = np.sqrt(val_mse)  # Root Mean Squared Error
val_r2 = r2_score(y_val, y_pred_val)
print(val_mae,val_mse,val_rmse,val_r2)

# Calculate regression metrics for the test set
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)  # Root Mean Squared Error
test_r2 = r2_score(y_test, y_pred_test)
print(test_mae,test_mse,test_rmse,test_r2)

# Specify the variables you want to calculate the correlation for
selected_columns = [
    'NDVI_Value', 'Albedo_Value', 'lst_Value','precipitation_Value'
    ,'wind_Value','wind_Direction','uhi_Value'  
]

# Select only the specified columns
df_combinedFinal = pd.concat([df_final1,df_final2,df_final3,df_final4])
df_selected = df_combinedFinal[selected_columns]

# Calculate the correlation matrix
corr_matrix = df_selected.corr()

# Plot the correlation matrix using a heatmap
plt.figure()
sns.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1)
plt.title('Correlation Heatmap - All Datasets')
plt.show()


