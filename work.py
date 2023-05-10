import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Data Cleaning

# Load the sales data
sales_df = pd.read_csv('sales.csv')

# Examine the sales data
print("Sales data info:")
print(sales_df.info())
print("\nSales data top 10 rows:")
print(sales_df.head(10))
print("\nSales data basic statistics:")
print(sales_df.describe())

# Show the missing and incorrect values for each column
print("\nMissing and incorrect values in sales data:")
print(sales_df.isnull().sum())
print(sales_df[sales_df['Weekly_Sales'] < 0])

# Handle missing and incorrect values
# For missing values, we can either drop the rows or fill them with appropriate values
sales_df.dropna(inplace=True)
# For negative sales values, we can either drop the rows or replace them with appropriate values
sales_df = sales_df[sales_df['Weekly_Sales'] >= 0]

# Load the weather data
weather_df = pd.read_csv('weather.csv')

# Examine the weather data
print("\nWeather data info:")
print(weather_df.info())
print("\nWeather data top 10 rows:")
print(weather_df.head(10))
print("\nWeather data basic statistics:")
print(weather_df.describe())

# Show the missing and incorrect values for each column
print("\nMissing and incorrect values in weather data:")
print(weather_df.isnull().sum())
print(weather_df[weather_df['Temperature'] < -100])

# Handle missing and incorrect values
# For missing values, we can either drop the rows or fill them with appropriate values
weather_df.dropna(inplace=True)
# For incorrect values, we can either drop the rows or replace them with appropriate values
weather_df = weather_df[weather_df['Temperature'] >= -100]

# Load the fuel pricing data
fuel_df = pd.read_csv('fuel.csv')

# Examine the fuel pricing data
print("\nFuel pricing data info:")
print(fuel_df.info())
print("\nFuel pricing data top 10 rows:")
print(fuel_df.head(10))
print("\nFuel pricing data basic statistics:")
print(fuel_df.describe())

# Show the missing and incorrect values for each column
print("\nMissing and incorrect values in fuel pricing data:")
print(fuel_df.isnull().sum())
print(fuel_df[fuel_df['Fuel_Price'] < 0])

# Handle missing and incorrect values
# For missing values, we can either drop the rows or fill them with appropriate values
fuel_df.dropna(inplace=True)
# For negative fuel prices, we can either drop the rows or replace them with appropriate values
fuel_df = fuel_df[fuel_df['Fuel_Price'] >= 0]

# Merge all datasets into data frame based on the date and store
merged_df = pd.merge(sales_df, weather_df, on=['Date', 'Store'], how='left')
merged_df = pd.merge(merged_df, fuel_df, on=['Date', 'Store'], how='left')

# Examine the merged data
print("\nMerged data info:")
print(merged_df.info())
print("\nMerged data top 10 rows:")
print(merged_df.head(10))
print("\nMerged data basic statistics:")
print(merged_df.describe())


# Visualization

"""
Chart to illustrate if weekly sales are increasing or decreasing over time:
"""

# Group the sales data by week and calculate the total sales for each week
weekly_sales = sales_df.groupby('Date')['Weekly_Sales'].sum()

# Create a line chart to show the trend of weekly sales over time
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales.index, weekly_sales.values)
plt.xlabel('Week')
plt.ylabel('Total Sales ($)')
plt.title('Weekly Sales Trend')
plt.show()

"""
Chart to show how much each brand sells:
"""

# Group the sales data by brand and calculate the total sales for each brand
brand_sales = sales_df.groupby('Category')['Weekly_Sales'].sum()

# Create a bar chart to show the total sales for each brand
plt.figure(figsize=(10, 6))
plt.bar(brand_sales.index, brand_sales.values)
plt.xlabel('Category')
plt.ylabel('Total Sales ($)')
plt.title('Sales by Brand')
plt.show()


"""
Determining the top ten selling stores:
"""

# Group the sales data by store and calculate the total sales for each store
store_sales = sales_df.groupby('Store')['Weekly_Sales'].sum()

# Sort the stores by total sales and select the top 10
top_stores = store_sales.sort_values(ascending=False).head(10)

# Print the top 10 stores
print("Top 10 Selling Stores:")
print(top_stores)


"""
Histogram to show the top 10 stores sales:
"""

# Select the sales data for the top 10 stores
top_stores_sales = sales_df[sales_df['Store'].isin(top_stores.index)]

# Create a histogram to show the distribution of sales for the top 10 stores
plt.figure(figsize=(10, 6))
plt.hist(top_stores_sales['Weekly_Sales'], bins=20, alpha=0.5, label='Top 10 Stores')
plt.xlabel('Weekly Sales ($)')
plt.ylabel('Frequency')
plt.title('Sales Distribution for Top 10 Stores')
plt.legend()
plt.show()


"""
Chart to compare average weekly sales for the top ten selling stores during holidays and non-holidays:
"""

# Group the sales data by store and holiday status and calculate the average sales for each group
store_holiday_sales = sales_df.groupby(['Store', 'Holiday'])['Weekly_Sales'].mean().unstack()

# Select the sales data for the top 10 stores
top_stores_sales = store_holiday_sales.loc[top_stores.index]

# Create a bar chart to compare the average sales for the top 10 stores during holidays and non-holidays
plt.figure(figsize=(10, 6))
plt.bar(top_stores_sales.index, top_stores_sales[False], label='Non-Holiday')
plt.bar(top_stores_sales.index, top_stores_sales[True], bottom=top_stores_sales[False], label='Holiday')
plt.xlabel('Store')
plt.ylabel('Average Sales ($)')
plt.title('Average Sales for Top 10 Stores by Holiday Status')
plt.legend()
plt.show()


"""
Chart to display the average weekly sales for each brand department for the top 10 selling stores:
"""

# Group the sales data by store, brand, and department and calculate the average sales for each group
store_brand_dept_sales = sales_df.groupby(['Store', 'Category'])['Weekly_Sales'].mean().unstack()

# Select the sales data for the top 10 stores
top_stores_sales = store_brand_dept_sales.loc[top_stores.index]

# Create a stacked bar chart to show the average sales for each brand department for the top 10 stores
plt.figure(figsize=(10, 6))
top_stores_sales.plot(kind='bar', stacked=True)
plt.xlabel('Store')
plt.ylabel('Average Sales ($)')
plt.title('Average Sales by Brand Department for Top 10 Stores')
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.show()


"""
Line chart to show the relationship between weekly sales and weather Temperature:
"""

# Group the sales data by week and calculate the total sales for each week
weekly_sales = sales_df.groupby('Date')['Weekly_Sales'].sum()

# Group the weather data by week and calculate the average temperature for each week
weekly_temp = weather_df.groupby('Date')['Temperature'].mean()

# Create a line chart to show the relationship between weekly sales and temperature
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales.index, weekly_sales.values, label='Weekly Sales')
plt.plot(weekly_temp.index, weekly_temp.values, label='Temperature')
plt.xlabel('Week')
plt.ylabel('Total Sales ($) / Temperature (F)')
plt.title('Relationship between Weekly Sales and Temperature')
plt.legend()
plt.show()


"""
Line chart to show the relationship between the cost of fuel and weather weekly sales:
"""

# Group the sales data by week and calculate the total sales for each week
weekly_sales = sales_df.groupby('Date')['Weekly_Sales'].sum()

# Group the fuel data by week and calculate the average fuel price for each week
weekly_fuel = fuel_df.groupby('Date')['Fuel_Price'].mean()

# Group the weather data by week and calculate the average temperature for each week
weekly_temp = weather_df.groupby('Date')['Temperature'].mean()

# Create a line chart to show the relationship between fuel price, temperature, and weekly sales
plt.figure(figsize=(10, 6))
plt.plot(weekly_sales.index, weekly_sales.values, label='Weekly Sales')
plt.plot(weekly_fuel.index, weekly_fuel.values, label='Fuel Price')
plt.plot(weekly_temp.index, weekly_temp.values, label='Temperature')
plt.xlabel('Week')
plt.ylabel('Total Sales ($) / Fuel Price ($) / Temperature (F)')
plt.title('Relationship between Weekly Sales, Fuel Price, and Temperature')
plt.legend()
plt.show()


# """Modeling"""

# # Data Preprocessing:

# # Load the sales data
# sales_df = pd.read_csv('sales.csv')

# # Split the data into features and target
# X = sales_df.drop('Weekly_Sales', axis=1)
# y = sales_df['Weekly_Sales']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Impute missing values in numerical features
# num_imputer = SimpleImputer(strategy='mean')
# num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
# X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
# X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# # Scale the numerical features
# scaler = StandardScaler()
# X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
# X_test[num_cols] = scaler.transform(X_test[num_cols])

# # One-hot encode categorical features
# cat_cols = X_train.select_dtypes(include=['object']).columns
# ohe = OneHotEncoder(handle_unknown='ignore')
# X_train_ohe = ohe.fit_transform(X_train[cat_cols])
# X_test_ohe = ohe.transform(X_test[cat_cols])
# X_train_ohe_df = pd.DataFrame(X_train_ohe.toarray(), columns=ohe.get_feature_names_out(cat_cols))
# X_test_ohe_df = pd.DataFrame(X_test_ohe.toarray(), columns=ohe.get_feature_names_out(cat_cols))
# X_train = pd.concat([X_train[num_cols], X_train_ohe_df], axis=1)
# X_test = pd.concat([X_test[num_cols], X_test_ohe_df], axis=1)


# # Model Selection:

# # Create a list of models to try
# models = [
#     LinearRegression(),
#     DecisionTreeRegressor(),
#     RandomForestRegressor(),
#     GradientBoostingRegressor()
# ]

# # Impute missing values in numerical features
# # num_imputer = SimpleImputer(strategy='mean')
# # num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
# # X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
# # X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# # Train the linear regression model
# # model = LinearRegression()
# # model.fit(X_train, y_train)

# # Read data from a CSV file
# df = pd.read_csv('sales.csv')


# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(df.drop('Store', axis=1), df['Store'], test_size=0.2, random_state=42)


# # Impute missing values in numerical features
# num_imputer = SimpleImputer(strategy='mean')
# num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
# X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
# X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# # Impute missing values in numerical features

# num_imputer = SimpleImputer(strategy='mean')
# num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
# X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
# X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# # Generate synthetic data
# num_samples = 1000
# num_features = 10
# X = pd.DataFrame(np.random.randn(num_samples, num_features), columns=['feature{}'.format(i) for i in range(num_features)])
# y = pd.Series(np.random.randn(num_samples), name='target')

# # One-hot encode categorical features

# cat_cols = X_train.select_dtypes(include=['object']).columns
# ohe = OneHotEncoder(handle_unknown='ignore')
# X_train_ohe = ohe.fit_transform(X_train[cat_cols])
# X_test_ohe = ohe.transform(X_test[cat_cols])
# X_train_ohe_df = pd.DataFrame(X_train_ohe.toarray(), columns=ohe.get_feature_names_out(cat_cols))
# X_test_ohe_df = pd.DataFrame(X_test_ohe.toarray(), columns=ohe.get_feature_names_out(cat_cols))
# X_train = pd.concat([X_train[num_cols], X_train_ohe_df], axis=1)
# X_test = pd.concat([X_test[num_cols], X_test_ohe_df], axis=1)

# # Ensure that X_train and y_train have the same number of samples
# y_train = y_train.loc[X_train.index]

# # Train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)


# # Train and evaluate each model on the training and testing sets
# for model in models:
#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     print(f'{type(model).__name__} Train Score: {train_score:.3f}')
#     print(f'{type(model).__name__} Test Score: {test_score:.3f}\n')


# # Model Training and Evaluation:

# # Select the best model based on the testing set performance
# model = RandomForestRegressor()
# model.fit(X_train, y_train)

# # Make predictions on the testing set
# y_pred = model.predict(X_test)

# # Evaluate the performance of the model on the testing set
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse:.2f}')
# print(f'Mean Absolute Error: {mae:.2f}')
# print(f'R-squared Score: {r2:.2f}')


# # Model Comparison:

# # Train and evaluate each model on the training and testing sets
# train_scores = []
# test_scores = []
# for model in models:
#     model.fit(X_train, y_train)
#     train_score = model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     train_scores.append(train_score)
#     test_scores.append(test_score)

# # Plot the training and testing set performance for each model
# plt.figure(figsize=(10, 6))
# plt.bar(np.arange(len(models)), train_scores, label='Train Score')
# plt.bar(np.arange(len(models)), test_scores, label='Test Score')
# plt.xticks(np.arange(len(models)), [type(model).__name__ for model in models], rotation=45)
# plt.xlabel('Model')
# plt.ylabel('Score')
# plt.title('Performance Comparison of Different Models')
# plt.legend()
# plt.show()


# # Clustering:

# # Select the number of clusters
# k_range = range(2, 11)
# wss = []
# sil_scores = []
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_train)
#     wss.append(sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])
#     sil_scores.append(silhouette_score(X_train, kmeans.labels_))

# # Plot the within-cluster sum of squares (WSS) and silhouette score for different numbers of clusters
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.plot(k_range, wss, 'bx-')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WSS')
# plt.title('Elbow Method')

# plt.subplot(1, 2, 2)
# plt.plot(k_range, sil_scores, 'bx-')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Method')

# plt.show()

# # Select the number of clusters based on the elbow method or silhouette method
# num_clusters = 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans.fit(X_train)

# # Visualize the clusters using a scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=kmeans.labels_)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title(f'K-Means Clustering with {num_clusters} Clusters')
# plt.show()
