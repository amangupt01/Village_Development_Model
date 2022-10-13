# Village Development Model

The trained CNN models for arch1 of all indicators can be found [here](https://drive.google.com/drive/folders/1eTUKyMq1z0dGoJaJ-BS80Q80T5Ucsrgq?usp=sharing)
  
## Directories and their functions : 
  * ```GEE_DataDownload\``` - These scripts are used to download the scripts from google earth Engine. They download RGB bands of respectives States of India.
  * ```PreProcessing_Images\``` - These scripts are just to cut the state tiff images into village images in png format. These crops are restricted to be fixed size of 150x150 pixels.
  * ```PreProcessing_Data\Generate_VillageLevels\``` - These scripts are used to classify villages into levels of rudimentary, intermediate & advanced.

## Steps to Reproduce the pipeline
  * Download shapefiles of states using the [script](GEE_DataDownload/Download_state_landsat7_2001.js), and make sure to select the correct year and states list.
  * The next step is to cut the shapes of villages from the shapefile of the state. Use the [script](PreProcessing_Images/cutVillage.sh) [You need to set the correct path to i) Where the state fill files are contained, ii) Where the state geojsons are present iii) Where you want the split files to be placed]
  * Now we will run the [Jupyter Notebook](PreProcessing_Data/Final_generate_village_centroids.ipynb), which takes the shape files of all villages and calculates the center.
  * Using the village centroid data obtained in the previous step, we will compute the nearest neighboring villages for each village: Follow the [script](PreProcessing_Data/find_out_nearest_neighbours_logic.ipynb)