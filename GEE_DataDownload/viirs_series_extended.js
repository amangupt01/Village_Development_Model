
var viirs = ee.Image("users/gaurav-chauhan/LongNTL_2019")

var dist = ee.FeatureCollection("users/chahatresearch/india_district_boundaries");
var distlist = dist.toList(dist.size())

var year = 2019
//var viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG").filterDate(ee.Date.fromYMD(year, 1, 1), ee.Date.fromYMD(year, 12, 31)).select('avg_rad').mean()

// var t1 = ee.Number(3);
// var t2 = ee.Number(4);

/*
// print(t2.max(t1));
var t1 = ee.Number(3);
print(t1.lt(ee.Number(3.7)))
print(t1.lt(ee.Number(3.7)).eq(ee.Number(1)))
var result = ee.Algorithms.If(t1.lt(ee.Number(3.7)).eq(ee.Number(1)), "True", "False")
print(result)
if(result === "True")
  print("Works")
*/

// Return the DN that maximizes interclass variance in B5 (in the region).
var otsu = function(histogram) {
  var counts = ee.Array(ee.Dictionary(histogram).get('histogram'));
  var means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'));
  var size = means.length().get([0]);
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = sum.divide(total);
  
  var indices = ee.List.sequence(1, size);
  
  // Compute between sum of squares, where each mean partitions the data.
  var bss = indices.map(function(i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans = means.slice(0, 0, i);
    var aMean = aMeans.multiply(aCounts)
        .reduce(ee.Reducer.sum(), [0]).get([0])
        .divide(aCount);
    var bCount = total.subtract(aCount);
    var bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2)).add(
           bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  
  // print(ui.Chart.array.values(ee.Array(bss), 0, means));
  
  // Return the mean value corresponding to the maximum BSS.
  return means.sort(bss).get([-1]);
};

var getVectors = function(t1, nightimg, dist1){
  var zones = nightimg.gt(t1);
  zones = zones.updateMask(zones.neq(0));
  
  var vectors = zones.addBands(nightimg).reduceToVectors({
    reducer: ee.Reducer.mean(),
    geometry: dist1.geometry,
    scale: 1000,
    geometryType: 'polygon',
    eightConnected: false,
    labelProperty: 'nighlight_intensity',
  });
  
  return vectors.toList(vectors.size());
};

var getNoVectors = function(){
  var null_feat = ee.Feature(ee.Geometry.Point(0.0,0.0), {'mean':null})
  return ee.List([null_feat])
}

var blobs = ee.List([])
var len = distlist.size()
var index = 0

var calcVectors = function(dist1){
  var nightimg = viirs.clip(dist1);

  // create histogram using histogram reducer
  var hist1a = nightimg.reduceRegion({
    reducer: ee.Reducer.histogram(null, null)
        .combine('mean', null, true)
        .combine('variance', null, true), 
   
    scale: 1000,
    bestEffort: true
  });
  
  //var t1 = otsu(hist1a.get('avg_rad_histogram'))
  var t1 = otsu(hist1a.get('b1_histogram'))
  return ee.Algorithms.If(t1.lt(ee.Number(1.0)), getNoVectors(), getVectors(t1, nightimg, dist1))
  return t1
}

//2336 (2272)

var blobs = distlist.map(calcVectors)
//print(blobs.size())
//print(blobs)

blobs = blobs.flatten()
//print(blobs.size())
//print(blobs)

var nullFilter = ee.Filter.notNull(['mean'])
blobs = blobs.filter(nullFilter)

var numblobs = blobs.size()

//print(numblobs)
//print(blobs)

var blobsCollection = ee.FeatureCollection(blobs)

//var max_intensity = blobsCollection.aggregate_max('mean')
//print(max_intensity)


Map.setCenter(77.21,28.64, 10);
//Map.addLayer(blobsCollection, {color: 'FF0000'}, 'colored');


 Export.table.toDrive({
   collection: blobsCollection,
   description:'NightLight_District_Blobs_extended_time_series_viirs_2019',
   fileFormat: 'GeoJSON'
 });
 