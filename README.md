# Village Development Model
The complete dataset can be found [here](https://drive.google.com/drive/folders/1xtaTGiaPJxDLr2t4RRqHYyJBkS3NcXTm?usp=sharing) 
## Directories and their functions : 
  * ```GEE_DataDownload\``` - These scripts are used to download the scripts from google earth Engine. They download RGB bands of respectives States of India.
  * ```PreProcessing_Images\``` - These scripts are just to cut the state tiff images into village images in png format. These crops are restricted to be fixed size of 150x150 pixels.
  * ```PreProcessing_Data\``` - These scripts are used to classify villages into levels of rudimentary, intermediate & advanced. They also contain scripts to generate population and nightlight features.
  * ```Arch1, Arch2 & Arch3_Scripts``` - These scripts are used to generate the features for each model, training models and also the evaluation scripts.
  * ```Hypothesis_Testing``` - Performs the hypothesis validation based on the predicted outputs.
  * ```Visualisations & Error_Analysis``` - These scripts generate development/nightlight encoded geojsons to visualise the development of villages. Also contains the scripts for occlusion studies, error analysis & statistics.

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
Note that the CNN models are the same for Arch-1 & Arch-2
- Use the [script](Arch2/Combining_nightlight_and_pop.ipynb) to combine all the features including nightlight, population, nearest neighbours to generate the final input for training regression models for Arch-2. The combined data for training can be found [here](https://drive.google.com/drive/folders/1LrTGcCuCWEnaKl4f9wII0Cobb-DxeSr4)
- To perform temporal transferability of the trained model we use the [script](Arch2/Common_Indicators_Classification_2011.ipynb) to identify features that provide maximum temporal transferability.
- Use the [script](Arch2/Copy_arch3_regression_grid_search.ipynb) to train regression models on the selected feature for each indicator on each level. The trained regression models can be found [here](https://drive.google.com/drive/folders/1Wf_L2ZgYdpBnvazuvz5Au5Zk4ITtzI6X?usp=sharing)
- After obtaining the regression outputs we run the [script](Arch2/Household_Indicators_Predictions.ipynb) to get the regression output on the test set
- Finally to check the performance of the models we use clusters trained on groundtruth to discretize the output of Arch-2 and calculate the RMSE using [this](Arch2/Copy_of_Classification_Performance_Arch_2.ipynb)
  
## Arch3
We retrain the CNN regression models for each level of each indicator. 
- Use the training [scripts](Arch3_Scripts/trainmodel.py) on the same train-val image split as done in Arch-1. 
- Use the [script](Arch3_Scripts/combine.py) to combine and make the outputs of all the models as well as the nearest neighbours data.
- Combine the obtained data with the nightlight and population features generated earlier.
- Use similar scipts like that of arch2 to find neccesary indicators that ensure temporal transferability.
- Train the regression model on the combined dataset using [script](Arch3_Scripts/Copy_arch3_regression_grid_search_new.ipynb)

## Hypothesis Testing
Use the scripts in the ```Hypothesis_Testing\``` directory to run and validate the hypothesis mentioned in the paper.

## Visualisation and Error Analysis
The following are the files and their purposes in the directory ```Visualisations & Error_Analysis```:-
- ```plot.ipynb``` & ```plot_ntl.ipynb``` - these create development/nightlight encoded list of villages within a district as a geojsons. These geojsons can be imported to [geojson plotter](https://geojson.io/#map=2/20.0/0.0) to plot and analyse development. The plotted images looks like [this](Visualisations%20&%20Error_Analysis/bokaro.png)
- ```Excluded_Villages.ipynb``` - this file collates the list of villages that were excluded from the analysis study either due to cloud cover or while recovering villages from the tiff tiles of states.
- ```Error_analysis.ipynb``` - this file analysis the trends in the villages whose ADI values are miscalculated by the model
- ```stats.ipynb``` - this file calculates the statewise mean and standard deviation of error in predicted and calculated ADIs. It also creates CDF plots between ADI_2001, ADI_2011 & ADI_2019
- ```corelation.ipynb``` - analysis whether the decrement in the ADI values has a correlation with nightlights
- ```occlusion.ipynb``` - helps validate if the CNN models learns important features on the map of villages. Overlaying the heatmaps on the actual village images lead to visualisations like [these](Visualisations%20&%20Error_Analysis/occlusion.png)
