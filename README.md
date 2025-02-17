# House Price Prediction - Machine Learning Project
Author: HVAInternational (Pham Dang Hung, Pham Ha Khanh Chi, Le Xuan Trong, and Le Ky Nam)

## 1. Abstract
This project aims to accurately predict house prices using Machine Learning. We utilize **AutoML, LightGBM, and Optuna** for model optimization. Our approach improves on baseline models, achieving a lower RMSE and better generalization.

## 2. Introduction
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

## 3. Data Analysis 

### 3.1. Dataset Overview:
The data set that we used was [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) from Kaggle. But, our teachers had already split the data set into three: X_train, y_train and X_test, in which X_train and y_train were used to train the model, and X_test would be used as the data set to produce the predicted house sales for scoring and ranking teams in AI4B class.

### 3.2. Exploratory Data Analysis (EDA)
To understand our dataset better and prepare it for modeling, we conducted several visual analyses:

#### **Missing value**
We didn't find any missing values in this data.

##### **Histogram Analysis**
📌 Purpose: Check if the house price distribution is Gaussian, detect skewness and outliers.
Here is the example of the price distribution:
![Price Distribution](images/price_distribution_before.png)
###### Feature Distribution Insights
- Most values fall within **$0 - $2M**, peaking around **$400,000**. Although there are a lot of outliers shown in this histogram, most are **natural outliers**, still following the overall distribution of data. This means the histogram displays very few **extreme outliers**.
- The distributions exhibit a roughly **Gaussian shape**, suggesting that the features follow an **approximately normal distribution**. 

###### Impact on Data Processing
- Since the distribution is **close to normal** and has **few extreme outliers**, applying **StandardScaler** is the optimal choice for feature scaling.
- Applying **StandardScaler** ensures the data is **centered around zero with unit variance**, improving model stability and performance in algorithms sensitive to feature magnitudes (e.g., Linear Regression).


##### **Heatmap (Correlation Matrix)**
📌 Purpose: Determine the relationship between variables, find important attributes that affect house prices.
![Heatmap](images/heatmap_before.png)

###### Feature Distribution Insights
- The features that have the highest correlation with price in decreasing order are **sqft_living, grade, sqft_above, sqft_living15, bathrooms, and view**. They have the most significant impact on the price.
- The features **sqft_basement, bedrooms, lat, waterfront, and floors** also influence the price but to a lesser extent.
- The features with little to no correlation with price, which can be removed, are **zipcode, yr_built, yr_renovated, sqft_lot, long, and sqft_lot15**.

###### Impact on Data Processing
To simplify the dataset and reduce the risk of **overfitting**, we will consider removing the low-impact features: **zipcode, yr_built, yr_renovated, sqft_lot, long, and sqft_lot15**. This helps improve model efficiency and generalization.

##### **Bar Chart Analysis**
📌 Purpose: Check data balance across categorical features.
Here is the example of the price by bulit decade (grouping years to analysis easier):
![Built Year Distribution](images/built_year_distribution_before.png)

###### Feature Distribution Insights
- With the exception of two periods—1920-1940 (The Great Depression) and 2010-2020 (The COVID-19 Pandemic)—the number of houses built per decade has **generally followed an upward trend**.
- The number of houses built is expected to **grow unless a major economic or global event disrupts the trend**.

###### Impact on Data Processing
- Since economic crises like The Great Depression and the COVID-19 Pandemic are the primary causes of these downward trends, rather than removing or imputing data from these periods, we introduce a new binary feature: ```economic_crisis_year``` (```1``` if the house was built in these periods, otherwise ```0```).
- Given that the overall trend remains upward, we decide to retain the built_year feature instead of removing it, as initially considered after analyzing the Correlation Matrix.

##### **Box Plot Analysis**
📌 Purpose: Detect outliers and understand the spread of continuous variables.
![Price vs Grade](images/price_vs_grade_before.png)

###### Feature Distribution Insights
- **There is a clear upward trend in price as the grade increases**. Higher-grade houses tend to have significantly higher median prices, confirming that grade is an important factor in predicting house prices.
- **Outliers are mostly observed in grades 6 to 10**. These could represent exceptionally expensive or underpriced houses within their respective grades, possibly due to location, renovations, or other hidden factors.

###### Impact on Data Processing
- We will retain ```grade``` because it is strongly correlated with house prices.
- Outlier Handling:
  - We will consider location as a possible cause of outliers in grades 6 to 10. To do this, we will draw a box plot of ```price``` vs ```zipcode```. If the **median and range of the box plots for each zipcode are non-overlapping**, it suggests that location is a **significant factor** causing the outliers.
  - If this is the case, we will create a new feature called ```high_value_location``` (set to 1 if the house is in a high-value location, and 0 otherwise). This will help capture the influence of location on house prices and better handle outliers.
  - If there is no clear separation in the box plots across zipcodes, we will remove zipcode and explore other potential features that might be contributing to outliers.

      
##### **Scatter Plot Analysis**
📌 Purpose: Examine relationships between numerical features and house prices.
![Price vs Living Area](images/price_vs_living_area_before.png)

###### Feature Distribution Insights
- The price has an **upward trend** as the living area increases from 0 to 4000 sq ft. This suggests that larger houses generally tend to have higher prices.
- After reaching 4000 sq ft, the price continues to increase, but at a much **slower and inconsistent rate**. This indicates that for houses with **very large living areas**, the price no longer grows linearly and may be influenced by other factors, such as location, condition, or amenities.
- There are houses with **very large living areas** that have prices **lower than expected**. These outliers could be due to several factors, including location, condition, or amenities.

###### Impact on Data Processing
- Since the price growth becomes inconsistent after 4000 sq ft, we may apply a **logarithmic transformation or binning** for large living areas. This will help us better capture the relationship between living area and price, without allowing extreme values to skew the results.
- We will handle outliers in the **scatter plot (price vs living area)** in the same way as we did with the box plot, by investigating potential causes and considering appropriate data transformations.

#### Remind:
You can visit the ```/images/``` folder in this repository to see more visualizations that we used to explore and analyze the data.

### 3.3. Data Preprocessing

## 4. Modelling
### 4.1. Baseline Models
These are the baseline models that we used:
- K-Nearest Neighbors
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
### 4.2. Hyperparameter tuning for Baseline models
To optimize the performance of the baseline models, we performed **hyperparameter tuning**. The hyperparameters evaluated for each model are as follows:
- KNN (KNeighborsRegressor):
  - n_neighbors: [3, 5, 7]
  - weights: ['uniform', 'distance']
- Linear Regression (LR):
  - fit_intercept: [True, False]
- Decision Tree (DT):
  - max_depth: [5, 10, None]
  - min_samples_split: [2, 5]
- Random Forest (RF):
  - n_estimators: [50, 100]
  - max_depth: [10, None]
- XGBoost (XGB):
  - n_estimators: [50, 100]
  - learning_rate: [0.01, 0.1]

To perform the tuning, we used GridSearchCV to explore the best combination of hyperparameters for each model, thereby improving their performance.
### 4.3. Advanced Model
Since our data is tabular, we used **AutoGluon** help us find the best model.  
![AutoGluon leaderboard](images/autogluon_leaderboard.png)   
According to AutoGluon Leaderboard, from the models that are easy to code and easy to use, LightGBM has the best score, so we will continue by optimizing **LightGBM**.
### 4.4. Hyperparameter tuning for Advanced model
To optimize the performance of the advanced model, we also performed **hyperparameter tuning**. The hyperparameters evaluated for the model are as follows:
```
param_grid = {
    #         "device_type": trial.suggest_categorical("device_type", ['gpu']),
    "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
    "max_depth": trial.suggest_int("max_depth", 3, 12),
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
    "max_bin": trial.suggest_int("max_bin", 200, 300),
    "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
    "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
    "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    "bagging_fraction": trial.suggest_float(
        "bagging_fraction", 0.2, 0.95, step=0.1
    ),
    "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
    "feature_fraction": trial.suggest_float(
        "feature_fraction", 0.2, 0.95, step=0.1
    ),
    "random_state": 42
}
```
Since there are a lot of hyperparameters we want to calculate, we will use **Optuna** to help us efficiently search for the best set of hyperparameters.

## 5. Data Processing & Model Training Pipeline
![Model Pipeline](images/model_pipeline.png)

## 6. Model Results

| Model | RMSE | 
|---------|------|
| K-Nearest Neighbors | 188239 | 
| Linear Regression | 64453485530 | 
| Decision Tree | 195260 | 
| Random Forest | 178764 | 
| XGBoost | 188172 | 
| **LightGBM** | **169731** | 

## 7. Conclusion
The LightGBM model, combined with Optuna, successfully optimized performance, achieving higher accuracy than traditional models. This project can be expanded by collecting more data or applying deep learning techniques.



