// Define the region of Bangladesh using coordinates.
var bangladesh = ee.Geometry.Polygon([
    [[88.0, 20.5],
     [92.5, 20.5],
     [92.5, 26.5],
     [88.0, 26.5]]
]);

// Define the start and end years.
var startYear = 2000;
var endYear = 2024;

var driveFolder = 'Bangladesh_Wind_Exports'; 

for (var year = startYear; year <= endYear; year++) {
  for (var month = 1; month <= 12; month++) {
    
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate = startDate.advance(1, 'month');
    
    // Filter the ERA5-LAND Monthly dataset for the current month and region
    var dataset = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
                   .filter(ee.Filter.date(startDate, endDate))
                   .filterBounds(bangladesh);  

    // Select the U and V wind component bands and compute the monthly mean
    var uWind = dataset.select('u_component_of_wind_10m').mean();  
    var vWind = dataset.select('v_component_of_wind_10m').mean();  

    // Export U component of wind
    Export.image.toDrive({
      image: uWind,
      description: 'Bangladesh_U_Wind_' + year + '_' + (month < 10 ? '0' + month : month),
      folder: driveFolder,  
      scale: 1000,  
      region: bangladesh,  
      fileFormat: 'GeoTIFF',
      maxPixels: 1e8  
    });

    // Export V component of wind
    Export.image.toDrive({
      image: vWind,
      description: 'Bangladesh_V_Wind_' + year + '_' + (month < 10 ? '0' + month : month),  
      folder: driveFolder,  
      scale: 1000, 
      region: bangladesh,  
      fileFormat: 'GeoTIFF',
      maxPixels: 1e8  
    });
  }
}
