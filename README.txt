# DGA Domain Detection - Binary Classifier

This project provides pre-trained models to classify domain names as either DGA-generated or legitimate.

## Files
- Model.h5: Main detection model.
- Model_hardtodetectdomains.h5: Model specialized for harder DGA domains.
- DGA_test.py: Script to predict and evaluate.
- requirements.txt: Required Python libraries.

## Dataset format
The input file must be a CSV containing two columns: 'domain' and 'label'.


# DGA Domain Detection - Binary Classifier

This project provides pre-trained models to classify domain names as either DGA-generated or legitimate.

## Files (Quick Testing)
- Model.h5: Main detection model.
- Model_hardtodetectdomains.h5: Model specialized for harder DGA domains.
- DGA_test.py: Script to predict and evaluate.
- requirements.txt: Required Python libraries.

## Dataset format
The input file must be a CSV containing two columns: 'domain' and 'label'.

## Additional Classification Methods

### Binary_Classification/
Complete binary classification folder with training:
- model.h5: Main detection model
- Training_testing.py: Script for both training and testing

**Dataset format:** CSV named `test.csv` with columns:
- `domain`: Domain names
- `label`: Numeric labels 


### Multiclass_Classification/

#### All_Families_Classification/
Classifies domains into individual DGA families 
- model.h5: Individual family classification model
- Test.py: Test script for individual families  
- Train.py: Training script
- Label_Map.csv: Maps family names to numeric labels

**Dataset format:** CSV named `test.csv` with columns:
- `domain`: Domain names
- `label`: Numeric family labels (check Label_Map.csv for mapping)


#### Embedding_Clustering_Classification/
Classifies domains into superfamilies 
- model.h5: Superfamily classification model
- test.py: Test script for superfamilies
- train.py: Training script  
- Label_Map.csv: Maps families to superfamily labels

**Dataset format:** CSV named `test.csv` with columns:
- `domain`: Domain names
- `superfamily_label`: Numeric superfamily labels (check Label_Map.csv for mapping)


#### PCA_Analysis_Classification/
Classifies domains using PCA-based family groupin into superfamilies
- model.h5: PCA-based classification model
- Test.py: Test script for PCA classification
- Train.py: Training script
- Label_Map.csv: Maps families to PCA cluster labels

**Dataset format:** CSV named `test.csv` with columns:
- `domain`: Domain names
- `superfamily_label`: Numeric PCA cluster labels (check Label_Map.csv for mapping)


