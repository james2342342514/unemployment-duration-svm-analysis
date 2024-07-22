# Unemployment Duration Analysis using Support Vector Machines (SVM)

## Overview
This project analyzes unemployment duration data using Support Vector Machines (SVM). It includes data preprocessing, model training with different kernels, and performance evaluation.

## Dataset
The dataset used is the 'UnempDur' dataset from the Ecdat package, which contains information about unemployment duration and related factors.
I'm attaching a .txt file with some notes about the dataset and its source 

## Project Structure
- Data preprocessing and splitting
- SVM model training with various kernels (linear, polynomial, radial, sigmoid)
- Model evaluation using query and test sets
- Performance metrics calculation (accuracy, sensitivity, specificity, etc.)

## Requirements
- R
- Libraries: tidyverse, kernlab, e1071, Ecdat

## Usage
1. Run the script to load and preprocess the data.
2. Train SVM models with different kernels.
3. Evaluate model performance on query and test sets.
4. View results and performance metrics.

## Results
The linear and polynomial kernel SVMs showed the best performance, with high accuracy and sensitivity. Detailed results are provided in the script output.

## Future Work
- Hyperparameter tuning for SVM models
- Feature importance analysis
- Comparison with other machine learning algorithms
