# Village Development Model

The trained CNN models for arch1 of all indicators can be found [here](https://drive.google.com/drive/folders/1eTUKyMq1z0dGoJaJ-BS80Q80T5Ucsrgq?usp=sharing)
  
## Directories and their functions : 
  * ```GEE_DataDownload\``` - These scripts are used to download the scripts from google earth Engine. They download RGB bands of respectives States of India.
  * ```PreProcessing_Images\``` - These scripts are just to cut the state tiff images into village images in png format. These crops are restricted to be fixed size of 150x150 pixels.
  * ```PreProcessing_Data\Generate_VillageLevels\``` - These scripts are used to classify villages into levels of rudimentary, intermediate & advanced.
