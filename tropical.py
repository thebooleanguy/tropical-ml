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

# Drop unnecessary columns
data.drop(columns=['time', 'country', 'latitude', 'longitude', 'elevation', 'sunrise', 'sunset',
                   'apparent_temperature_max', 'apparent_temperature_min', 'snowfall_sum'], inplace=True)

# Define input features (X) and target variable (y) for temperature prediction
X = data.drop(columns=['temperature_2m_mean', 'city'])  # Predicting temperature
y_temp = data['temperature_2m_mean']

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

