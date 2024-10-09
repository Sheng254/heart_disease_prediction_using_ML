# Heart Disease Prediction Using Machine Learning


## Table of Contents

1. [Abstract](#abstract)
2. [Dataset Description](#dataset-Description)
3. [Machine Learning Algorithms Used](#machine-learning-algorithms-used)
4. [Machine Learning Pipeline From Data Collection to Model Evaluation](#machine-learning-pipeline-from-data-collection-to-model-evaluation)
5. [Results](#results)
6. [Future improvements](#future-improvements)
7. [Remarks](#remarks)

<br>

## Abstract
This project leverages data analysis, data science, and machine learning techniques to predict heart disease using the UCI Cleveland dataset. Among the models tested, k-Nearest Neighbors (kNN) (accuracy: 87.5%, precision: 90.48%, recall: 79.17%, F1: 84.44%) and Random Forest (accuracy: 89.29%, precision: 95.00%, recall: 79.17%, F1: 86.36%) demonstrated the strongest performance. In this context, recall is a crucial metric as it minimizes false negatives, ensuring that more cases of heart disease are accurately identified. While both models performed well, Random Forest proved to be the overall best model, offering superior accuracy and a balanced performance across all metrics. These models are well-suited for healthcare applications where early and accurate detection of heart disease is vital.

<br>

## Dataset Description
The dataset contains 14 attributes related to heart disease diagnoses, including patient demographics, heart-related metrics, and the target variable indicating heart disease presence. This project focused on the Cleveland subset of the UCI Heart Disease dataset.

<br>

## Machine Learning Algorithms Used
- **K-Nearest Neighbors (kNN)**: Effective for non-linear data and instance-based reasoning.  
- **Logistic Regression (LR)**: Suitable for binary classification tasks.  
- **Random Forests (RF)**: Reduces overfitting through ensemble learning.  
- **Multi-layer Perceptron (MLP)**: A simple neural network used to capture complex relationships.

<br>

## Machine Learning Pipeline From Data Collection to Model Evaluation

The code followed a structured approach to build and evaluate machine learning models for heart disease prediction.

1. **Data Collection and Exploration**:
   - The dataset was retrieved from the UCI Machine Learning Repository, specifically focusing on the Cleveland heart disease dataset, and explored to understand its structure and identify important variables.

2. **Data Cleaning**:
   - Missing values were checked and handled by removing rows with incomplete data, as they made up a small portion of the dataset.
   - Duplicate records were identified and removed to maintain the dataset's integrity.
   - Outliers in continuous variables (e.g., blood pressure, cholesterol levels) were identified using boxplots and removed to improve model robustness.

3. **Feature Selection and Transformation**:
   - Mutual information and chi-squared tests were performed to identify the most significant features related to heart disease.
   - The top categorical features included number of major vessels (ca), thalassemia (thal), and chest pain type (cp).
   - The selected continuous features included oldpeak (ST depression) and thalach (maximum heart rate).
   - One-Hot Encoding was applied to categorical variables.
   - Continuous features were normalized using Min-Max Scaling to prepare the data for ML models.
     
4. **Model Selection**:
   - Implemented those 4 ML algorithms as mentioned previously.
   - Each model was selected for its strengths: kNN for handling non-linear relationships, Logistic Regression for interpretability, Random Forest for reducing overfitting, and MLP for capturing complex patterns.

5. **Hyperparameter Tuning**:
   - Hyperparameter tuning was conducted using GridSearchCV to optimize the models' performance.
   - Parameters such as the number of neighbors for kNN, regularization strength for Logistic Regression, number of trees for Random Forest, and layer configuration for MLP were tuned.
     
6. **Model Evaluation**:
   - The models were evaluated based on key metrics: accuracy, precision, recall, and F1 score.

<br>

## Results

| Model                | Accuracy | Precision | Recall  | F1      |
|----------------------|----------|-----------|---------|---------|
| K-Nearest Neighbors   | 0.875000 | 0.904762  | 0.791667| 0.844444|
| Logistic Regression   | 0.821429 | 0.937500  | 0.625000| 0.750000|
| Random Forest         | 0.892857 | 0.950000  | 0.791667| 0.863636|
| Multi-layer Perceptron| 0.803571 | 0.882353  | 0.625000| 0.731707|

**Table 1: Model Performance Metrics after Hyperparameter Tuning**

After hyperparameter tuning, Random Forest emerged as the best-performing model with the highest accuracy (89.29%), precision (95.00%), and F1 score (86.36%), along with a recall of 79.17%, indicating a strong balance between minimizing false positives and correctly identifying heart disease cases. k-Nearest Neighbors (kNN) also performed well, achieving an accuracy of 87.5%, precision of 90.48%, recall of 79.17%, and an F1 score of 84.44%, making it another reliable model for heart disease prediction. Both models demonstrated a good balance of precision and recall, which is crucial in healthcare for ensuring early detection and minimizing false negatives. Logistic Regression and Multi-layer Perceptron (MLP) had relatively high precision (93.75% and 88.24%, respectively) but lower recall (62.5%), increasing the risk of missed heart disease cases.

Overall, Random Forest and kNN are the more suitable models for this dataset, with a better trade-off between identifying true positives and minimizing false negatives. But Random Forest stands out as the most well-rounded model since it provides a balanced performance across all 4 metrics.

<br>

## Future improvements
Future work could focus on incorporating additional metrics like ROC-AUC and experimenting with ensemble methods to improve accuracy and recall further.

<br>

## Remarks
This project was done using Jupyter Notebook, where code is executed one by one to simulate the data analysis process. In industry, Jupyter Notebook is typically suitable for code exploration, and ML engineering code should be developed in a proper IDE. All code should be structured in files, ensuring that every line of code can be run together efficiently, which is effective in production environments.

Most fancy data visualizations graphs won’t bring much insight into the next step of data analysis. They are cool to generate but they don’t add much value when it comes to understanding the data or making actionable decisions. Effective data analysis relies more on simple visualizations that help identify patterns, trends, or outliers that inform the next analytical steps.

This project has not been tested on real-life inference data, which is a critical aspect of machine learning engineering. Many ML projects are successful in development environments but perform poorly in real-world scenarios due to issues such as insufficient handling of edge cases, changes in data distribution, and the need for further hyperparameter tuning.
