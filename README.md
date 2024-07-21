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

## Evaluation Metrics
The following evaluation metrics were used in this study:
- **Accuracy:** The ratio of correctly predicted instances to the total instances. Measures the overall correctness of the model. However, it can be misleading in imbalanced datasets, as it might reflect high values even if the model fails to predict the minority class correctly.
- **Precision:** The ratio of correctly predicted positive instances to the total predicted positives. Indicates the accuracy of the positive predictions made by the model. High precision means that there are fewer false positives.
- **Recall (Sensitivity or True Positive Rate):** The ratio of correctly predicted positive instances to all actual positives. Measures the model's ability to identify all relevant instances. High recall means that there are fewer false negatives.
- **F1-Score:** The harmonic mean of precision and recall. Provides a single metric that balances both precision and recall. It is particularly useful when the class distribution is imbalanced.
- **ROC AUC Score:** The area under the Receiver Operating Characteristic (ROC) curve.
   - The ROC curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC score indicates how well the model distinguishes between the classes. A score of 1 indicates perfect discrimination, while a score of 0.5 indicates no discrimination (random guessing).

## Results
The results section will detail the performance of models on both the imbalanced and oversampled datasets.
### Performance on Imbalanced Data
The table below summarizes the performance of various machine learning models when trained on the imbalanced stroke dataset:

| Model                        | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|----------------|
| Logistic Regression          | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.851          |
| Random Forest                | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.797          |
| Support Vector Machine       | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.628          |
| Gradient Boosting            | 0.938    | 0.94          | 0.33          | 1.00       | 0.02       | 0.97         | 0.03         | 0.835          |
| AdaBoost                     | 0.937    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.793          |
| k-Nearest Neighbors          | 0.939    | 0.94          | 0.00          | 1.00       | 0.00       | 0.97         | 0.00         | 0.647          |
| Decision Tree                | 0.920    | 0.95          | 0.27          | 0.97       | 0.19       | 0.96         | 0.23         | 0.580          |
| Naive Bayes                  | 0.867    | 0.96          | 0.22          | 0.89       | 0.47       | 0.93         | 0.30         | 0.829          |
| Linear Discriminant Analysis | 0.934    | 0.94          | 0.27          | 0.99       | 0.05       | 0.97         | 0.08         | 0.842          |
| Quadratic Discriminant Analysis | 0.880 | 0.96          | 0.24          | 0.91       | 0.45       | 0.93         | 0.31         | 0.830          |

The evaluations demonstrate that most models struggled to detect stroke cases in an imbalanced dataset. High accuracy did not translate to effective stroke detection, as the minority class (stroke cases) was often missed. The ROC AUC scores indicated that while models could distinguish between classes overall, they were biased towards the majority class.
### Performce on oversampled datasets
Below are the results of each oversampling technique:
#### Random Over-Sampling
| Model                          | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|--------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression            | 0.749    | 0.98          | 0.16          | 0.75       | 0.76       | 0.85         | 0.27         | 0.851         |
| Random Forest                  | 0.938    | 0.94          | 0.40          | 1.00       | 0.03       | 0.97         | 0.06         | 0.814         |
| Support Vector Machine         | 0.771    | 0.96          | 0.14          | 0.79       | 0.53       | 0.87         | 0.22         | 0.778         |
| Gradient Boosting              | 0.793    | 0.97          | 0.18          | 0.80       | 0.66       | 0.88         | 0.28         | 0.815         |
| AdaBoost                       | 0.740    | 0.98          | 0.16          | 0.74       | 0.77       | 0.84         | 0.27         | 0.839         |
| k-Nearest Neighbors            | 0.868    | 0.95          | 0.18          | 0.90       | 0.32       | 0.93         | 0.23         | 0.649         |
| Decision Tree                  | 0.911    | 0.94          | 0.16          | 0.96       | 0.11       | 0.95         | 0.13         | 0.538         |
| Naive Bayes                    | 0.738    | 0.98          | 0.15          | 0.74       | 0.74       | 0.84         | 0.26         | 0.829         |
| Linear Discriminant Analysis   | 0.735    | 0.98          | 0.16          | 0.73       | 0.76       | 0.84         | 0.26         | 0.850         |
| Quadratic Discriminant Analysis| 0.750    | 0.98          | 0.16          | 0.75       | 0.74       | 0.85         | 0.26         | 0.831         |
| Extra Trees                    | 0.937    | 0.94          | 0.33          | 1.00       | 0.03       | 0.97         | 0.06         | 0.758         |

Logistic Regression shows improved recall for the minority class (stroke) after Random Over-Sampling. However, precision for the minority class remains low, indicating many false positives. The model correctly predicts 75% of non-stroke cases and 76% of stroke cases, resulting in an overall accuracy of 74.9%.

Random Forest maintains high accuracy but struggles to recall the minority class (stroke). The ROC AUC score indicates that the model has some discriminatory power but fails to detect stroke cases effectively.

Support Vector Machine shows moderate improvement in recall for the minority class (stroke) after Random Over-Sampling but still has low precision, indicating false positives. The overall accuracy is 77.1%. Gradient Boosting shows a balanced performance with improved recall for the minority class and a good ROC AUC score, indicating better discrimination between classes.

AdaBoost improves recall for the minority class significantly but at the cost of precision, resulting in many false positives. The overall accuracy is 74.0%.k-Nearest Neighbors shows moderate improvement in recall for the minority class, but precision is still low, indicating many false positives. Decision Tree maintains high precision and recall for the majority class but struggles significantly with the minority class, resulting in low performance metrics for stroke prediction.

Naive Bayes shows improved recall for the minority class with moderate precision. The overall accuracy is 73.8%. Linear Discriminant Analysis shows moderate improvement in recall for the minority class with low precision, resulting in many false positives. Quadratic Discriminant Analysis shows moderate improvement in recall for the minority class with low precision, resulting in many false positives. Extra Trees maintains high precision and recall for the majority class but struggles significantly with the minority class, resulting in low performance metrics for stroke prediction.

Therefore, Random Over-Sampling generally improves recall for the minority class across most models, but precision remains low, indicating a high number of false positives. The ROC AUC score improves for most models, indicating better discrimination between classes. Models like Gradient Boosting and Logistic Regression show balanced improvements with Random Over-Sampling.

#### SMOTE
| Model                            | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|----------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression              | 0.759    | 0.98          | 0.17          | 0.76       | 0.76       | 0.86         | 0.28         | 0.849         |
| Random Forest                    | 0.917    | 0.94          | 0.19          | 0.97       | 0.11       | 0.96         | 0.14         | 0.805         |
| Support Vector Machine           | 0.777    | 0.96          | 0.14          | 0.79       | 0.52       | 0.87         | 0.22         | 0.764         |
| Gradient Boosting                | 0.875    | 0.96          | 0.21          | 0.91       | 0.39       | 0.93         | 0.27         | 0.820         |
| AdaBoost                         | 0.822    | 0.97          | 0.18          | 0.84       | 0.55       | 0.90         | 0.27         | 0.812         |
| k-Nearest Neighbors              | 0.829    | 0.96          | 0.16          | 0.86       | 0.42       | 0.90         | 0.23         | 0.686         |
| Decision Tree                    | 0.869    | 0.95          | 0.15          | 0.91       | 0.26       | 0.93         | 0.19         | 0.583         |
| Naive Bayes                      | 0.727    | 0.98          | 0.15          | 0.73       | 0.74       | 0.83         | 0.25         | 0.830         |
| Linear Discriminant Analysis     | 0.743    | 0.98          | 0.16          | 0.74       | 0.76       | 0.84         | 0.26         | 0.848         |
| Quadratic Discriminant Analysis  | 0.747    | 0.98          | 0.17          | 0.74       | 0.82       | 0.85         | 0.28         | 0.839         |
| Extra Trees                      | 0.912    | 0.94          | 0.15          | 0.96       | 0.10       | 0.95         | 0.12         | 0.784         |

Logistic Regression shows a significant improvement in recall for the minority class (stroke), indicating that SMOTE helps in identifying more stroke cases. However, precision is still low, suggesting many false positives.

Random Forest maintains high accuracy but struggles with recall and precision for the minority class, indicating that it is still not very effective at identifying stroke cases despite oversampling.

The SVM model benefits from SMOTE by significantly improving recall for the minority class. However, like Logistic Regression, it also suffers from low precision, leading to false positives.

Gradient Boosting shows a balanced improvement in both recall and precision, making it more effective at identifying stroke cases with fewer false positives compared to Logistic Regression and SVM. AdaBoost shows similar trends to Gradient Boosting, with improved recall and moderate precision, indicating a balanced identification of stroke cases.

k-NN shows a reasonable improvement in recall but has a relatively low precision, suggesting it is more prone to false positives. Decision Tree shows moderate improvements in recall and precision but still struggles to balance between false positives and true positives effectively.

Naive Bayes benefits significantly from SMOTE, showing high recall for the minority class, although precision remains low, resulting in many false positives. LDA shows improved recall similar to Logistic Regression but suffers from low precision, indicating many false positives.

QDA benefits from SMOTE with the highest recall among the models, but its precision is still low, resulting in a significant number of false positives. Extra Trees maintain high accuracy but have poor recall and precision for the minority class, indicating limited effectiveness in identifying stroke cases despite oversampling.

Overall, applying SMOTE improves the recall of most models for the minority class (stroke), meaning that the models are better at identifying stroke cases. However, precision generally remains low across the models, indicating a high number of false positives. This trade-off is common in dealing with imbalanced datasets. Gradient Boosting, AdaBoost, and Logistic Regression seem to provide a more balanced improvement in both recall and precision, making them potentially better choices for this particular problem. 

#### ADASYN
| Model                            | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|----------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression              | 0.756    | 0.98          | 0.17          | 0.76       | 0.76       | 0.85         | 0.27         | 0.849         |
| Random Forest                    | 0.916    | 0.94          | 0.17          | 0.97       | 0.10       | 0.96         | 0.12         | 0.804         |
| Support Vector Machine           | 0.776    | 0.96          | 0.14          | 0.79       | 0.52       | 0.87         | 0.22         | 0.763         |
| Gradient Boosting                | 0.870    | 0.96          | 0.19          | 0.90       | 0.35       | 0.93         | 0.25         | 0.818         |
| AdaBoost                         | 0.827    | 0.97          | 0.19          | 0.84       | 0.56       | 0.90         | 0.28         | 0.816         |
| k-Nearest Neighbors              | 0.817    | 0.96          | 0.14          | 0.84       | 0.40       | 0.90         | 0.21         | 0.678         |
| Decision Tree                    | 0.853    | 0.95          | 0.11          | 0.89       | 0.21       | 0.92         | 0.15         | 0.552         |
| Naive Bayes                      | 0.711    | 0.98          | 0.14          | 0.71       | 0.76       | 0.82         | 0.24         | 0.828         |
| Linear Discriminant Analysis     | 0.738    | 0.98          | 0.16          | 0.74       | 0.76       | 0.84         | 0.26         | 0.847         |
| Quadratic Discriminant Analysis  | 0.735    | 0.98          | 0.16          | 0.73       | 0.82       | 0.84         | 0.27         | 0.838         |
| Extra Trees                      | 0.914    | 0.94          | 0.14          | 0.97       | 0.08       | 0.95         | 0.10         | 0.787         |

Logistic Regression shows a significant improvement in recall for the minority class (stroke), indicating that ADASYN helps in identifying more stroke cases. However, precision is still low, suggesting many false positives.

Random Forest maintains high accuracy but struggles with recall and precision for the minority class, indicating that it is still not very effective at identifying stroke cases despite oversampling. The SVM model benefits from ADASYN by significantly improving recall for the minority class. However, like Logistic Regression, it also suffers from low precision, leading to false positives.

Gradient Boosting shows a balanced improvement in both recall and precision, making it more effective at identifying stroke cases with fewer false positives compared to Logistic Regression and SVM. AdaBoost shows similar trends to Gradient Boosting, with improved recall and moderate precision, indicating a balanced identification of stroke cases.

k-NN shows a reasonable improvement in recall but has a relatively low precision, suggesting it is more prone to false positives. Decision Tree shows moderate improvements in recall and precision but still struggles to balance between false positives and true positives effectively.

Naive Bayes benefits significantly from ADASYN, showing high recall for the minority class, although precision remains low, resulting in many false positives.LDA shows improved recall similar to Logistic Regression but suffers from low precision, indicating many false positives.

QDA benefits from ADASYN with the highest recall among the models, but its precision is still low, resulting in a significant number of false positives. Extra Trees maintain high accuracy but have poor recall and precision for the minority class, indicating limited effectiveness in identifying stroke cases despite oversampling.

Overall, applying ADASYN improves the recall of most models for the minority class (stroke), meaning that the models are better at identifying stroke cases. However, precision generally remains low across the models, indicating a high number of false positives. This trade-off is common in dealing with imbalanced datasets. Gradient Boosting, AdaBoost, and Logistic Regression seem to provide a more balanced improvement in both recall and precision, making them potentially better choices for this particular problem. 

#### Borderline-SMOTE
| Model                            | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|----------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression              | 0.798    | 0.98          | 0.19          | 0.80       | 0.73       | 0.88         | 0.30         | 0.854         |
| Random Forest                    | 0.920    | 0.95          | 0.22          | 0.97       | 0.13       | 0.96         | 0.16         | 0.811         |
| Support Vector Machine           | 0.835    | 0.96          | 0.18          | 0.86       | 0.50       | 0.91         | 0.27         | 0.794         |
| Gradient Boosting                | 0.860    | 0.95          | 0.17          | 0.89       | 0.32       | 0.92         | 0.22         | 0.830         |
| AdaBoost                         | 0.841    | 0.97          | 0.20          | 0.86       | 0.55       | 0.91         | 0.29         | 0.829         |
| k-Nearest Neighbors              | 0.857    | 0.96          | 0.18          | 0.89       | 0.37       | 0.92         | 0.24         | 0.690         |
| Decision Tree                    | 0.886    | 0.95          | 0.21          | 0.92       | 0.32       | 0.94         | 0.26         | 0.623         |
| Naive Bayes                      | 0.767    | 0.98          | 0.17          | 0.77       | 0.71       | 0.86         | 0.27         | 0.836         |
| Linear Discriminant Analysis     | 0.767    | 0.98          | 0.17          | 0.77       | 0.74       | 0.86         | 0.28         | 0.851         |
| Quadratic Discriminant Analysis  | 0.784    | 0.98          | 0.18          | 0.79       | 0.73       | 0.87         | 0.29         | 0.845         |
| Extra Trees                      | 0.919    | 0.94          | 0.18          | 0.97       | 0.10       | 0.96         | 0.13         | 0.785         |

Logistic Regression shows a balanced improvement in recall and precision for the minority class, indicating that Borderline-SMOTE helps in identifying more stroke cases while maintaining a reasonable number of false positives.

Random Forest maintains high accuracy but has limited effectiveness in improving recall for the minority class, indicating that it still misses many stroke cases. SVM benefits from Borderline-SMOTE by significantly improving recall for the minority class, although precision remains moderate, indicating a balanced identification of stroke cases with some false positives.

Gradient Boosting shows moderate improvements in both recall and precision, making it effective at identifying stroke cases with a reasonable number of false positives. AdaBoost shows a balanced improvement in recall and precision, indicating effective identification of stroke cases with fewer false positives compared to some other models.

k-NN shows a reasonable improvement in recall, but precision is still moderate, suggesting a balanced identification of stroke cases with some false positives. Decision Tree shows moderate improvements in recall and precision, indicating a balanced identification of stroke cases with a reasonable number of false positives.

Naive Bayes benefits significantly from Borderline-SMOTE, showing high recall for the minority class, although precision remains moderate, resulting in some false positives. LDA shows improved recall similar to Logistic Regression but maintains moderate precision, indicating some false positives.

QDA benefits from Borderline-SMOTE with high recall and moderate precision, resulting in balanced identification of stroke cases. Extra Trees maintain high accuracy but have limited effectiveness in improving recall for the minority class, indicating it still misses many stroke cases.

Overall, applying Borderline-SMOTE generally improves the recall for most models for the minority class (stroke), meaning that the models are better at identifying stroke cases. However, precision often remains moderate, indicating a balanced number of false positives. Gradient Boosting, AdaBoost, and Logistic Regression provide a more balanced improvement in both recall and precision, making them potentially better choices for this particular problem.

#### SVMSMOTE
| Model                           | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-Score (0) | F1-Score (1) | ROC AUC Score |
|---------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| **Logistic Regression**         | 0.864    | 0.97          | 0.24          | 0.88       | 0.56       | 0.92         | 0.33         | 0.854         |
| **Random Forest**               | 0.928    | 0.94          | 0.23          | 0.98       | 0.08       | 0.96         | 0.12         | 0.816         |
| **Support Vector Machine**      | 0.875    | 0.95          | 0.18          | 0.91       | 0.31       | 0.93         | 0.23         | 0.787         |
| **Gradient Boosting**           | 0.918    | 0.95          | 0.28          | 0.96       | 0.23       | 0.96         | 0.25         | 0.836         |
| **AdaBoost**                    | 0.888    | 0.95          | 0.20          | 0.93       | 0.29       | 0.94         | 0.24         | 0.832         |
| **k-Nearest Neighbors**         | 0.871    | 0.95          | 0.18          | 0.91       | 0.31       | 0.93         | 0.22         | 0.693         |
| **Decision Tree**               | 0.876    | 0.95          | 0.14          | 0.92       | 0.21       | 0.93         | 0.17         | 0.564         |
| **Naive Bayes**                 | 0.816    | 0.97          | 0.19          | 0.83       | 0.61       | 0.89         | 0.29         | 0.835         |
| **Linear Discriminant Analysis**| 0.861    | 0.97          | 0.23          | 0.88       | 0.55       | 0.92         | 0.32         | 0.848         |
| **Quadratic Discriminant Analysis** | 0.819 | 0.98          | 0.21          | 0.83       | 0.69       | 0.90         | 0.32         | 0.844         |
| **Extra Trees**                 | 0.923    | 0.94          | 0.16          | 0.98       | 0.06       | 0.96         | 0.09         | 0.784         |

Logistic Regression shows a balanced improvement in recall and precision for the minority class, indicating that SVMSMOTE helps in identifying more stroke cases while maintaining a reasonable number of false positives.

Random Forest maintains high accuracy but has limited effectiveness in improving recall for the minority class, indicating that it still misses many stroke cases. SVM benefits from SVMSMOTE by significantly improving recall for the minority class, although precision remains moderate, indicating a balanced identification of stroke cases with some false positives.

Gradient Boosting shows moderate improvements in both recall and precision, making it effective at identifying stroke cases with a reasonable number of false positives. AdaBoost shows a balanced improvement in recall and precision, indicating effective identification of stroke cases with fewer false positives compared to some other models.

k-NN shows a reasonable improvement in recall, but precision is still moderate, suggesting a balanced identification of stroke cases with some false positives. Decision Tree shows moderate improvements in recall and precision, indicating a balanced identification of stroke cases with a reasonable number of false positives.

Naive Bayes benefits significantly from SVMSMOTE, showing high recall for the minority class, although precision remains moderate, resulting in some false positives. LDA shows improved recall similar to Logistic Regression but maintains moderate precision, indicating some false positives.

QDA benefits from SVMSMOTE with high recall and moderate precision, resulting in balanced identification of stroke cases. Extra Trees maintain high accuracy but have limited effectiveness in improving recall for the minority class, indicating it still misses many stroke cases.

Overall, applying SVMSMOTE generally improves the recall for most models for the minority class (stroke), meaning that the models are better at identifying stroke cases. However, precision often remains moderate, indicating a balanced number of false positives. Gradient Boosting, AdaBoost, and Logistic Regression provide a more balanced improvement in both recall and precision, making them potentially better choices for this particular problem. 

#### SMOTEENN
| Model                              | Accuracy | Precision (0) | Precision (1) | Recall (0) | Recall (1) | F1-score (0) | F1-score (1) | ROC AUC Score |
|------------------------------------|----------|---------------|---------------|------------|------------|--------------|--------------|---------------|
| Logistic Regression                | 0.708    | 0.98          | 0.15          | 0.70       | 0.79       | 0.82         | 0.25         | 0.848         |
| Random Forest                      | 0.860    | 0.95          | 0.17          | 0.89       | 0.32       | 0.92         | 0.22         | 0.798         |
| Support Vector Machine             | 0.747    | 0.97          | 0.14          | 0.76       | 0.61       | 0.85         | 0.23         | 0.771         |
| Gradient Boosting                  | 0.822    | 0.97          | 0.19          | 0.84       | 0.58       | 0.90         | 0.28         | 0.824         |
| AdaBoost                           | 0.786    | 0.98          | 0.18          | 0.79       | 0.74       | 0.87         | 0.30         | 0.838         |
| k-Nearest Neighbors                | 0.764    | 0.96          | 0.14          | 0.78       | 0.55       | 0.86         | 0.22         | 0.714         |
| Decision Tree                      | 0.841    | 0.96          | 0.17          | 0.87       | 0.40       | 0.91         | 0.23         | 0.636         |
| Naive Bayes                        | 0.717    | 0.98          | 0.14          | 0.72       | 0.74       | 0.83         | 0.24         | 0.825         |
| Linear Discriminant Analysis       | 0.698    | 0.99          | 0.15          | 0.69       | 0.84       | 0.81         | 0.25         | 0.847         |
| Quadratic Discriminant Analysis    | 0.729    | 0.98          | 0.15          | 0.73       | 0.77       | 0.83         | 0.26         | 0.828         |
| Extra Trees                        | 0.851    | 0.95          | 0.16          | 0.88       | 0.34       | 0.92         | 0.22         | 0.774         |

Logistic Regression after SMOTEENN showed improved recall for the minority class (stroke cases) at the expense of precision. This means it identified more true stroke cases but also had more false positives. The overall accuracy dropped, indicating a trade-off between identifying stroke cases and maintaining precision.

Random Forest after SMOTEENN maintained high accuracy and improved recall for the minority class. It managed to identify more stroke cases while keeping false positives relatively low, making it a balanced model for both classes.

Support Vector Machine after SMOTEENN improved recall for the minority class, identifying more stroke cases but with reduced precision. The overall accuracy was lower, indicating more false positives were introduced.

Gradient Boosting after SMOTEENN achieved good recall and precision balance. It effectively identified stroke cases with moderate false positives, maintaining a good overall accuracy. AdaBoost after SMOTEENN improved recall significantly for the minority class, identifying more stroke cases with increased false positives. The overall accuracy decreased, indicating the model was more sensitive to stroke cases.

k-Nearest Neighbors after SMOTEENN showed improved recall for the minority class but at the cost of precision. The overall accuracy dropped, reflecting a trade-off between identifying stroke cases and maintaining precision. Decision Tree after SMOTEENN maintained a good balance between recall and precision for both classes. It identified more stroke cases while keeping false positives at a moderate level, resulting in a good overall accuracy.

Naive Bayes after SMOTEENN showed significant improvement in recall for the minority class. It identified more stroke cases but had increased false positives, reflecting a lower overall accuracy. Linear Discriminant Analysis after SMOTEENN had a significant increase in recall for the minority class. It identified more stroke cases but had more false positives, resulting in a lower overall accuracy.

Linear Discriminant Analysis after SMOTEENN had a significant increase in recall for the minority class. It identified more stroke cases but had more false positives, resulting in a lower overall accuracy. Quadratic Discriminant Analysis after SMOTEENN showed improved recall for the minority class with an increased number of false positives. The overall accuracy decreased, indicating the model was more sensitive to stroke cases. 

Extra Trees after SMOTEENN maintained high accuracy and showed moderate improvement in recall for the minority class. It identified more stroke cases with a balanced precision, making it a reliable model for predicting stroke cases.

Overall, applying SMOTEENN has improved the recall for the minority class across various models, allowing them to identify more stroke cases. However, this improvement often comes at the cost of precision and overall accuracy, reflecting a trade-off between sensitivity to stroke cases and maintaining precision. Models like Random Forest and Gradient Boosting achieved a good balance, making them effective for both classes.

#### SMOTETomek
| Model                        | Accuracy | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | ROC AUC Score |
|------------------------------|----------|---------------|------------|--------------|---------------|------------|--------------|---------------|
| Logistic Regression          | 0.757    | 0.98          | 0.76       | 0.85         | 0.17          | 0.74       | 0.27         | 0.849         |
| Random Forest                | 0.916    | 0.94          | 0.97       | 0.96         | 0.17          | 0.10       | 0.12         | 0.796         |
| Support Vector Machine       | 0.783    | 0.96          | 0.80       | 0.87         | 0.15          | 0.55       | 0.23         | 0.766         |
| Gradient Boosting            | 0.863    | 0.96          | 0.89       | 0.92         | 0.19          | 0.37       | 0.25         | 0.821         |
| AdaBoost                     | 0.832    | 0.97          | 0.84       | 0.90         | 0.21          | 0.63       | 0.31         | 0.818         |
| k-Nearest Neighbors          | 0.820    | 0.96          | 0.85       | 0.90         | 0.15          | 0.42       | 0.22         | 0.678         |
| Decision Tree                | 0.858    | 0.95          | 0.90       | 0.92         | 0.12          | 0.21       | 0.15         | 0.555         |
| Naive Bayes                  | 0.726    | 0.98          | 0.72       | 0.83         | 0.15          | 0.76       | 0.25         | 0.831         |
| Linear Discriminant Analysis | 0.741    | 0.98          | 0.74       | 0.84         | 0.16          | 0.74       | 0.26         | 0.847         |
| Quadratic Discriminant Analysis | 0.744 | 0.98          | 0.74       | 0.84         | 0.17          | 0.82       | 0.28         | 0.839         |
| Extra Trees                  | 0.913    | 0.94          | 0.97       | 0.95         | 0.15          | 0.10       | 0.12         | 0.776         |

Logistic Regression after SMOTETomek showed improved recall for the minority class (stroke cases) but had low precision, indicating a higher number of false positives. The overall accuracy is moderate, reflecting a balance between identifying stroke cases and maintaining precision. 

Random Forest after SMOTETomek maintained high accuracy and recall for the majority class but struggled with the minority class, showing low recall and precision for stroke cases. SVM after SMOTETomek showed improved recall for the minority class but at the cost of precision, leading to a moderate number of false positives.

Gradient Boosting achieved a good balance between recall and precision for both classes, making it effective at identifying stroke cases with a reasonable number of false positives. AdaBoost showed significant improvement in recall for the minority class, effectively identifying stroke cases with a moderate number of false positives.

k-NN showed improved recall for the minority class but at the cost of precision, leading to more false positives and a moderate overall accuracy. Decision Tree maintained high accuracy and recall for the majority class but struggled with the minority class, showing low recall and precision for stroke cases.

Naive Bayes showed significant improvement in recall for the minority class, effectively identifying stroke cases but at the cost of precision, leading to a moderate number of false positives. LDA showed balanced recall for both classes, making it effective at identifying both positives and negatives.

LDA showed balanced recall for both classes, making it effective at identifying both positives and negatives. Extra Trees model maintained high accuracy and recall for the majority class but struggled with recall for the minority class, similar to the random forest model.

Overall, applying SMOTETomek improved recall for the minority class across various models, allowing them to identify more stroke cases. However, this improvement often comes at the cost of precision and overall accuracy, reflecting a trade-off between sensitivity to stroke cases and maintaining precision. Models like Gradient Boosting and AdaBoost achieved a good balance, making them effective for both classes.

## Conclusion
This study provides insights into the effectiveness of various oversampling techniques in improving predictive model performance on imbalanced healthcare datasets. Recommendations for handling class imbalance in similar datasets will be provided.
