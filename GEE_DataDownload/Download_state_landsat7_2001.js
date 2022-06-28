var cloud_masks = require('users/fitoprincipe/geetools:cloud_masks');
var maskClouds = cloud_masks.landsatSR(); // mask function

/**
 * Function to mask clouds based on the pixel_qa band of Landsat SR data.
 * @param {ee.Image} image Input Landsat SR image
 * @return {ee.Image} Cloudmasked Landsat image
 */
var india = ee.FeatureCollection("users/chahatresearch/India_Boundary");

var cloudMaskL457 = function(image) {
  var qa = image.select('pixel_qa');
  // If the cloud bit (5) is set and the cloud confidence (7) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var cloud = qa.bitwiseAnd(1 << 5)
                  .and(qa.bitwiseAnd(1 << 7))
                  .or(qa.bitwiseAnd(1 << 3));
  // Remove edge pixels that don't occur in all bands
  var mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud.not()).updateMask(mask2);
};


var year_list = ['2001'];

// var state_list =['Madhya Pradesh','Maharashtra','Haryana','Punjab','Karnataka','Jharkhand','Andhra Pradesh','Telangana','Chhattisgarh','Uttar Pradesh'];
// var state_list = ['Assam', 'Bihar', 'Chhattisgarh', 'Jharkhand', 'Manipur']
//var state_list = ['Uttarakhand','Tripura','Tamil Nadu','Sikkim','Kerala','Gujarat','Bihar','Odisha','Rajasthan','Manipur','Mizoram','Assam']
//var state_list = ['Andhra Pradesh', 'Bihar', 'Chhattisgarh', 'Gujarat', 'Jharkhand', 'Karnataka', 'Maharashtra', 
//                    'Madhya Pradesh', 'Odisha', 'Punjab', 'Rajasthan', 'Tamil Nadu', 'Uttar Pradesh']
// var state_list = ['Haryana']
var state_list = ['West Bengal']

for (var i in state_list) {
  
  var state_name = state_list[i];
  
  var state = ee.FeatureCollection('users/hariomahlawat/India_States')
    .filter(ee.Filter.eq('ST_NM',state_name));
    
  for (var j in year_list)
  {
     var year = year_list[j];
     var str = state_name.replace(/\s/g,'');
   
    var state_image = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
    .filterBounds(state)
    .filterDate(year+'-01-01',year+'-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 5))
    .sort('CLOUD_COVER')
    .map(maskClouds);  
    //.map(cloudMaskL457);
  
    var visParams = {
    bands: ['B3', 'B2', 'B1'],
    min: 0,
    max: 3000,
    gamma: 1.4,
    };
  
    // Map.addLayer(state_image.median().clip(state), visParams);

    var bands = ['B3','B2','B1'];    // RGB bands for Landsat 7
    state_image = state_image.select(bands).median();
    Map.addLayer(state_image.clip(state), visParams);

  
    print(state_image);  
   
   
   
     
    Export.image.toDrive({
      image: state_image.clip(state),
      description: 'Landsat7_SR_'+str + '_' + year + 'new',
      maxPixels: 499295920080,
      scale: 30,
      folder: 'Village_level_new',
      region: state.geometry().bounds()
      });
  }
 
}