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
  Post this follow the pipeline of individual model that you want to train and test.

## Arch-1
1. Generating Features
- We first run clustering on the raw census data to generate levels of each census indicator like BF, FC, ASSET etc, using the [script](Arch1/Get_unoutliered_labels_indicators_for_district.ipynb)
- Use [scripts](Arch1/dataset_maker.py) to create an 80:20 split in datasets for generating the training and testing data sets. This creates a train and test set of each cluster(level) for each indicator. The directory structure for each indicator should look like [this](Arch1/directory_structure.png)
- In the generated dataset, there was Class Imbalance with cluster-1 having fewer villages, so data augmentation was performed using the [script](Arch1/dataaugment_literacy.py) (Done only in BF and Literacy)
2. Training the CNN Model
- We also use penalty functions to handle class imbalance for indicators literacy and BF. {Both use different loss functions} Check the [script](Arch1/train_model_weight_balance_new_literacy.py). Not for the other three indicators i.e. MSW, FC & Assets which don't have class imbalance we don't use augmentation or penalty functions.
- Trained models obatained can be found [here](https://drive.google.com/drive/folders/1eTUKyMq1z0dGoJaJ-BS80Q80T5Ucsrgq?usp=sharing)

## Arch-2