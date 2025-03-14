
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('dataset/SriLanka_Weather_Dataset.csv')

# Convert 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Extract useful time-based features
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['day_of_week'] = data['time'].dt.dayofweek

# One-hot encoding for 'city' column
data = pd.get_dummies(data, columns=['city'], drop_first=True)

# Drop unnecessary columns
data.drop(columns=['time', 'country', 'sunrise', 'sunset', 'apparent_temperature_max', 
                   'apparent_temperature_min', 'snowfall_sum'], inplace=True)

# Define input features (X) and target variable (y) for temperature prediction
# You can optionally include 'latitude' and 'longitude' if you want to use geographical features
X = data.drop(columns=['temperature_2m_mean'])  # Features for prediction
y_temp = data['temperature_2m_mean']  # Target for temperature prediction

# Define input features (X) and target variable (y) for precipitation prediction
y_rain = data['precipitation_sum']  # Target for rain prediction

# Split data into training (80%) and testing (20%) for both targets
X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
X_train_rain, X_test_rain, y_rain_train, y_rain_test = train_test_split(X, y_rain, test_size=0.2, random_state=42)

# Initialize models
linear_reg_temp = LinearRegression()
decision_tree_rain = DecisionTreeRegressor(random_state=42)

# Train models
linear_reg_temp.fit(X_train, y_temp_train)
decision_tree_rain.fit(X_train_rain, y_rain_train)

# Make predictions
y_temp_pred = linear_reg_temp.predict(X_test)
y_rain_pred = decision_tree_rain.predict(X_test_rain)

# Evaluate Linear Regression Model (Temperature Prediction)
print("Linear Regression Model Evaluation (Temperature Prediction):")
print("Mean Absolute Error:", mean_absolute_error(y_temp_test, y_temp_pred))
print("Mean Squared Error:", mean_squared_error(y_temp_test, y_temp_pred))
print("R-squared Score:", r2_score(y_temp_test, y_temp_pred))

# Evaluate Decision Tree Model (Precipitation Prediction)
print("\nDecision Tree Model Evaluation (Precipitation Prediction):")
print("Mean Absolute Error:", mean_absolute_error(y_rain_test, y_rain_pred))
print("Mean Squared Error:", mean_squared_error(y_rain_test, y_rain_pred))
print("R-squared Score:", r2_score(y_rain_test, y_rain_pred))


### Debugging
# Debugging - Ensure new_data has all columns (including missing ones)
city_columns = [col for col in X.columns if col.startswith('city_')]

# Example input for prediction (You can replace the values as needed)
new_data = pd.DataFrame({
    'year': [2025],
    'month': [3],
    'day': [15],
    'day_of_week': [6],
    'windspeed_10m_max': [10],
    'shortwave_radiation_sum': [200],
    'precipitation_sum': [5],
    'latitude': [6.9271],  # Example latitude for Colombo
    'longitude': [79.8612], # Example longitude for Colombo
    # One-hot encoded city column (Assuming 'Colombo' is included in the one-hot encoding)
    'city_Colombo': [1],
    'city_Kandy': [0],  # Set other cities to 0
    'city_Galle': [0],
    # Add any other cities you have in your data, as needed.
})

# Ensure all city columns are in the new_data
for col in city_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Align new_data columns with the training data columns
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict temperature
predicted_temp = linear_reg_temp.predict(new_data)
print("Predicted Temperature:", predicted_temp)

# Predict precipitation
predicted_rain = decision_tree_rain.predict(new_data)
print("Predicted Precipitation:", predicted_rain)

