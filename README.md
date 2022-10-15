# Village Development Model

The trained CNN models for arch1 of all indicators can be found [here](https://drive.google.com/drive/folders/1eTUKyMq1z0dGoJaJ-BS80Q80T5Ucsrgq?usp=sharing)
  
## Directories and their functions : 
  * ```GEE_DataDownload\``` - These scripts are used to download the scripts from google earth Engine. They download RGB bands of respectives States of India.
  * ```PreProcessing_Images\``` - These scripts are just to cut the state tiff images into village images in png format. These crops are restricted to be fixed size of 150x150 pixels.
  * ```PreProcessing_Data\Generate_VillageLevels\``` - These scripts are used to classify villages into levels of rudimentary, intermediate & advanced.

## Generating Nightlight Data
  * Download the year wise tif file from the [site](https://doi.org/10.7910/DVN/YGIVCD)
  * To get an idea of what blobs contain and how do they look check the [map image](PreProcessing_Data/Nightlight%20Generation/Visualisation_Blob.png) 
  * Upload the LongNTL_{year}.tif as assets in the GEE.
  * Update the path and run the [script](PreProcessing_Data/Nightlight%20Generation/viirs_series_extended.js) to obtain the blob collection
  *  Run the VIIRS extended section of the [notebook](PreProcessing_Data/Nightlight%20Generation/Get_blob_details(Generate%20NTL%20Features).ipynb) to generate the selected feature computation scheme of nightlight. Ensure that you select the correct year
  *  Use the [script](PreProcessing_Data/Nightlight%20Generation/nightlight_scoring_schemes.ipynb) to score the features of nightlight data

## Generating Population Data
  Use the [script](PreProcessing_Data/make_population.ipynb) to generate the data about population and number of households in a village. Wrapper functions like logarithm and square root was also applied to them. Note, that this script takes the original census data as input to generate these population features. 

## Steps to Reproduce the Pipeline
  * Download shapefiles of states using the [script](GEE_DataDownload/Download_state_landsat7_2001.js), and make sure to select the correct year and states list.
  * The next step is to cut the shapes of villages from the shapefile of the state. Use the [script](PreProcessing_Images/cutVillage.sh) [You need to set the correct path to i) Where the state fill files are contained, ii) Where the state geojsons are present iii) Where you want the split files to be placed]
  * Now we will run the [Jupyter Notebook](PreProcessing_Data/Final_generate_village_centroids.ipynb), which takes the shape files of all villages and calculates the center.
  * Using the village centroid data obtained in the previous step, we will compute the nearest neighboring villages for each village: Follow the [script](PreProcessing_Data/find_out_nearest_neighbours_logic.ipynb)
  * Generate the nightlight and population features as explained above.