# Breast Cancer Diagnostic Prediction
# BME 3968 - Medical AI - Project 1 Group 4
This repository includes EDA notebooks, modular source code for preprocessing and model training, evaluation scripts, saved models, results visualizations, a demo notebook for inference, and full documentation with dependency specifications for reproducibility. CHANGE THIS

# Group Members & Roles 
1. Ava Frank: (title of role)
    * Role description: 

2. Caroline Horey: (title of role)
    * Role description: 

3. Ayushi Elhence: (title of role)
    * Role description: 

# Overview
This project develops machine learning (ML) models to assist clinicians in classifying breast tumors as benign or malignant based on nuclear morphology features extracted from fine needle aspiration (FNA) biopsies. 

# Clinical Context 
This will be added at the end based on paper (see intro)

# Problem Statement
We aim to evaluate classic ML algorithms for the binary classification of breast tumors and determine whether a reduced feature subset can maintain diagnostic performance. 

# Data Information FORMAT NEEDS FIXING 
This project uses the Breast Cancer Wisconsin (Diagnostic) dataset (OpenML ID: 1510), originally collected at the University of Wisconsin and distributed through the UCI Machine Learning Repository. 
* Dataset Summary 
    * Samples: 569 patients
    * Features: 30 numeric features 
    * Target Classes: 2 (Benign, Malignant)
* Feature Description   
Features are computed from digitalized images of FNA biopsies of breast masses.
They quantify nuclear morphology charateristics such as: 
    * Radius
    * Texture
    * Perimeter
    * Area 
    * Smoothness 
    * Compactness
    * Concavity
    * Concave points 
    * Symmetry 
    * Fractal dimension
Each of these variables is calculated as mean, standard error, and worst (largest) value, resulting in 30 total predictors 
* Target Variable    
The task is a binary classification problem: 
    * Benign 
    * Malignant

# Repository Structure
- **data/** - Raw dataset files
- **notebooks/** - EDA and modeling notebooks
- **src/** - Modular preprocessing and training scripts
- **models/** - Saved trained models (.pkl) 
- **reslts/** - ROC curves, confusion matrices, performance tables 

# Setup Instructions 

# Run Commands

# Results

# Summary 

# Folder Explanations

# Dependencies