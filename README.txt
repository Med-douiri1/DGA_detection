# DGA Domain Detection - Binary Classifier

This project provides pre-trained models to classify domain names as either DGA-generated or legitimate.

## Files
- Model.h5: Main detection model.
- Model_hardtodetectdomains.h5: Model specialized for harder DGA domains.
- DGA_test.py: Script to predict and evaluate.
- requirements.txt: Required Python libraries.

## Dataset format
The input file must be a CSV containing two columns: 'domain' and 'label'.


