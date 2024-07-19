# Evaluating the Impact of Oversampling Techniques on Imbalanced Healthcare Data: A Case Study on Stroke Prediction

## Introduction
In healthcare, predictive models can significantly improve patient outcomes by enabling early intervention and personalized treatment plans. However, many healthcare datasets, such as those predicting stroke occurrences, suffer from class imbalance. This study evaluates the effectiveness of various oversampling techniques on an imbalanced healthcare dataset related to stroke prediction.

## Dataset
The dataset used in this project contains information necessary to predict the occurrence of a stroke. Each row in the dataset represents a patient, and the dataset includes the following attributes:

- **id:** Unique identifier
- **gender:** "Male", "Female", or "Other"
- **age:** Age of the patient
- **hypertension:** 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- **heart_disease:** 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- **ever_married:** "No" or "Yes"
- **work_type:** "Children", "Govt_job", "Never_worked", "Private", or "Self-employed"
- **Residence_type:** "Rural" or "Urban"
- **avg_glucose_level:** Average glucose level in the blood
- **bmi:** Body mass index
- **smoking_status:** "Formerly smoked", "Never smoked", "Smokes", or "Unknown"
- **stroke:** 1 if the patient had a stroke, 0 if not

## Problem Definition
Many healthcare datasets are imbalanced, leading to biased models that perform poorly in identifying the minority class. This study aims to:

1. **Assess Model Performance on Imbalanced Data:** Understand model performance when trained on imbalanced data without adjustments.
2. **Implement Oversampling Techniques:** Apply various oversampling methods to balance the dataset.
3. **Comparative Analysis:** Compare model performance on original vs. oversampled datasets using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
4. **Insights and Recommendations:** Identify which oversampling techniques yield the most significant improvements and provide recommendations for handling class imbalance in similar datasets.

## Methods
1. **Data Preprocessing:** Cleaning and preparing the data for analysis.
2. **Baseline Model:** Training a baseline model on the original imbalanced dataset.
3. **Oversampling Techniques:** Implementing the following oversampling techniques:
   - **Random Over-Sampling:** Randomly replicating minority class examples to balance the dataset.
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Generating synthetic examples for the minority class by interpolating between existing examples.
   - **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE, but adaptively focuses on harder-to-learn examples by generating more synthetic data points in regions where the model struggles.
   - **Borderline-SMOTE:** An extension of SMOTE that focuses on generating synthetic data points near the borderline of the minority and majority classes.
   - **SVMSMOTE:** Uses support vector machines (SVM) to create synthetic instances that are close to the decision boundary.
   - **KMeansSMOTE:** Combines K-means clustering and SMOTE to generate synthetic samples based on cluster centroids.
   - **SMOTEENN:** A combination of SMOTE and Edited Nearest Neighbors (ENN) that first generates synthetic samples using SMOTE and then cleans the dataset using ENN.
   - **SMOTETomek:** Combines SMOTE and Tomek Links to generate synthetic samples and then remove noisy samples.

4. **Model Training:** Training models on both the original and oversampled datasets.
5. **Evaluation:** Comparing model performance using accuracy, precision, recall, F1-score, and AUC-ROC.

## Results
The results section will detail the performance of models on both the imbalanced and oversampled datasets, highlighting the improvements achieved through oversampling.

## Conclusion
This study provides insights into the effectiveness of various oversampling techniques in improving predictive model performance on imbalanced healthcare datasets. Recommendations for handling class imbalance in similar datasets will be provided.

