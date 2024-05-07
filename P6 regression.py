# Simple Linear Regression

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset
housing = fetch_california_housing()
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target variable 'PRICE' to the DataFrame
housing_df['PRICE'] = housing.target

# Simple Linear Regression
X = housing_df[['AveRooms']]
y = housing_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

mse = mean_squared_error(y_test, model.predict(X_test))
r2 = r2_score(y_test, model.predict(X_test))

print("Simple Linear Regression:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Intercept: ", model.intercept_)
print("Coefficient:", model.coef_)



# Multiple Linear Regression
X = housing_df.drop('PRICE', axis=1)
y = housing_df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMultiple Linear Regression:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Intercept: ", model.intercept_)
print("Coefficients:")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.coef_[i]}")
