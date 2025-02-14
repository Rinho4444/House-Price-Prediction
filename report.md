# House Price Prediction Report

## Abstract
This project aims to accurately predict house prices using Machine Learning. We utilize **AutoML, LightGBM, and Optuna** for model optimization. Our approach improves on baseline models, achieving a lower RMSE and better generalization.

## Introduction
This project was initially developed as an assignment for the AI4B class, where we were tasked with building a predictive model for house prices. While fulfilling the course requirements, our team recognized the broader significance of this problem in the real estate industry. Accurate price predictions can aid buyers, sellers, and investors in making informed decisions, optimizing market strategies, and assessing property values more effectively.

To ensure our model has practical applications beyond the classroom, we extend this project with advanced machine learning techniques, such as **LightGBM**, **Optuna**, and **AutoML**, to enhance prediction accuracy. This approach allows us to both meet the academic expectations of our coursework and contribute meaningfully to real-world applications of AI in real estate. 

To create the best project, the team members were encouraged to optimise their teamwork and communication to help each other. The tasks were divided as follow:
- Exploring/understanding the data: Le Xuan Trong
- Data cleaning and : Pham Ha Khanh Chi
- Building the models: Le Ky Nam
- Optimising the models: Pham Dang Hung
- Writing the report: Pham Ha Khanh Chi 
Each task was completed successfully by not just the assigned member but also by the help of the whole team during the process.
The AIM for this project was for us to successfully build a model by the information given, applying all the knowledge we had been taught in AI4B class. Plus, we also learned teamwork and communication skills to be able to solve any bugs, obstacles or, problems passing by during the process.

## Dataset & Data Processing
- **Raw data**: The data set that we used was taken from Kaggle in [King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). But, our teachers had already split the data set into three: X_train, y_train and X_test, in which X_train and y_train were used to train the model, and X_test would be used as the data set to produce the predicted house sales for scoring and ranking teams in AI4B class.
-  **Processed Data**: Cleaned and feature-engineered dataset ready for model training.

## Exploratory Data Analysis (EDA)
- Missing values handled using **!?**.
- Outliers removed for better model stability.
- Feature correlations analyzed to select relevant variables.
- Visualizations created using **Matplotlib, Seaborn, and WandB**.

## Model Training & Optimization
- **Baseline Model**: The best model choosing between K-Nearest Neighbors, Linear Regression, Decision Tree, Random Forest, and XGBoost with hyperparameter tuning using GridSearchCV.
- **Optimized Model**: The best model choosing from **AutoGluon** (LightGBM) with hyperparameter tuning using **Optuna**.

### Baseline Model Performance
| Model | RMSE | 
|--------|------|----------|
| K-Nearest Neighbors | 150,600 |
| Linear Regression | 172,200 |
| Decision Tree | 161,400 |
| Random Forest | 127,300 |
| XGBoost| 121,500 |
-> XGBoost (tuning using GridSearchCV) is the best among these models, so it will be our Baseline Model.

### Final Model Performance
| Model | RMSE Validation | RMSE Test |
| XGBoost | | |
| LightGBM | | | 

