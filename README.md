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

## Models
The following machine learning models were used in this study:
1. **Logistic Regression:** A linear model for binary classification that estimates probabilities using the logistic function.
2. **Random Forest:** An ensemble learning method that uses multiple decision trees to improve model accuracy and control overfitting.
3. **Support Vector Machine (SVM):** A powerful classifier that finds the hyperplane that best separates the classes, with support for probability estimates.
4. **Gradient Boosting:** An ensemble technique that builds models sequentially to correct the errors of previous models.
5. **AdaBoost:** An ensemble method that combines multiple weak classifiers to form a strong classifier, focusing on harder-to-classify instances.
6. **k-Nearest Neighbors (k-NN):** A simple, instance-based learning algorithm that classifies instances based on the majority class of their k-nearest neighbors.
7. **Decision Tree:** A non-parametric model that splits data into subsets based on feature values, forming a tree-like structure.
8. **Naive Bayes:** A probabilistic classifier based on Bayes' theorem, assuming independence between features.
9. **Linear Discriminant Analysis:** A linear classifier that finds a linear combination of features that best separates the classes.
10. **Quadratic Discriminant Analysis:** An extension of LDA that models quadratic decision boundaries.
11. **Extra Trees:** An ensemble learning method similar to Random Forests but with more randomization during tree building.

## Results
The results section will detail the performance of models on both the imbalanced and oversampled datasets.
### Performance on Imbalanced Data
The table below summarizes the performance of various machine learning models when trained on the imbalanced stroke dataset:

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score | Confusion Matrix      |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|----------------|-----------------------|
| Logistic Regression          | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.851          | [[960, 0], [62, 0]]   |
| Random Forest                | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.797          | [[960, 0], [62, 0]]   |
| Support Vector Machine       | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.628          | [[960, 0], [62, 0]]   |
| Gradient Boosting            | 0.938    | 0.94          | 0.33          | 1.00       | 0.02       | 0.97         | 0.03         | 0.835          | [[958, 2], [61, 1]]   |
| AdaBoost                     | 0.937    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.793          | [[958, 2], [62, 0]]   |
| k-Nearest Neighbors          | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.647          | [[960, 0], [62, 0]]   |
| Decision Tree                | 0.920    | 0.95          | 0.27          | 0.97       | 0.19       | 0.96         | 0.23         | 0.580          | [[928, 32], [50, 12]] |
| Naive Bayes                  | 0.867    | 0.96          | 0.22          | 0.89       | 0.47       | 0.93         | 0.30         | 0.829          | [[857, 103], [33, 29]]|
| Linear Discriminant Analysis | 0.934    | 0.94          | 0.27          | 0.99       | 0.05       | 0.97         | 0.08         | 0.842          | [[952, 8], [59, 3]]   |
| Quadratic Discriminant Analysis | 0.880 | 0.96          | 0.24          | 0.91       | 0.45       | 0.93         | 0.31         | 0.830          | [[871, 89], [34, 28]] |
| Extra Trees                  | 0.935    | 0.94          | 0.17          | 0.99       | 0.02       | 0.97         | 0.03         | 0.771          | [[955, 5], [61, 1]]   |



## Conclusion
This study provides insights into the effectiveness of various oversampling techniques in improving predictive model performance on imbalanced healthcare datasets. Recommendations for handling class imbalance in similar datasets will be provided.
