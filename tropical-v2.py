import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load weather data
weather = pd.read_csv('dataset/SriLanka_Weather_Dataset.csv')

# Convert the 'time' column to datetime
weather['time'] = pd.to_datetime(weather['time'])

# Preprocess the data for a specific city (e.g., Kandy)
def preprocess_data(city_name):
    # Filter for the chosen city
    weather_city = weather[weather['city'] == city_name]
    weather_city.set_index('time', inplace=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['country', 'latitude', 'longitude', 'elevation', 'sunrise', 'sunset', 'snowfall_sum', 'apparent_temperature_max', 'apparent_temperature_min']
    weather_city = weather_city.drop(columns=columns_to_drop)
    
    # Create year, month, day, weekday features from the date
    weather_city['year'] = weather_city.index.year
    weather_city['month'] = weather_city.index.month
    weather_city['day'] = weather_city.index.day
    weather_city['weekday'] = weather_city.index.weekday
    
    # Handle outliers (simple method: remove values that are too extreme, like temperatures above 40°C or below 0°C)
    weather_city = weather_city[(weather_city['temperature_2m_mean'] < 40) & (weather_city['temperature_2m_mean'] > 0)]
    
    # Handle lag features for weather forecasting (e.g., lag of 1 day for temperature and precipitation)
    weather_city['temp_lag_1'] = weather_city['temperature_2m_mean'].shift(1)
    weather_city['rain_lag_1'] = weather_city['rain_sum'].shift(1)
    
    # Drop rows with NaN values due to shifting
    weather_city.dropna(inplace=True)
    
    return weather_city

# Predict temperature using Linear Regression
def predict_temperature(weather_city):
    X = weather_city[['year', 'month', 'day', 'weekday', 'temp_lag_1', 'rain_lag_1']]
    y_temp = weather_city['temperature_2m_mean']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    
    # Initialize the model and train it
    model_temp = LinearRegression()
    model_temp.fit(X_train, y_train)
    
    # Predictions
    y_pred_temp = model_temp.predict(X_test)
    
    # Evaluation
    print('Temperature Prediction (Linear Regression):')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred_temp)}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_temp)}')
    print(f'R-squared: {r2_score(y_test, y_pred_temp)}')

    return model_temp

# Predict weather code using Decision Tree Classifier
def predict_weather_code(weather_city):
    X = weather_city[['year', 'month', 'day', 'weekday', 'temp_lag_1', 'rain_lag_1']]
    y_weather_code = weather_city['weathercode']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_weather_code, test_size=0.2, random_state=42)
    
    # Initialize and train Decision Tree model
    model_weather_code = DecisionTreeClassifier(random_state=42)
    model_weather_code.fit(X_train, y_train)
    
    # Predictions
    y_pred_weather_code = model_weather_code.predict(X_test)
    
    # Evaluation
    print('Weather Code Prediction (Decision Tree):')
    print(f'Accuracy: {model_weather_code.score(X_test, y_test)}')
    
    return model_weather_code

# Function to predict future values iteratively (for temperature and weather code)
def predict_future_weather(city_name, future_dates):
    # Preprocess data for the city
    weather_city = preprocess_data(city_name)
    
    # Get the most recent available data (the last row)
    last_row = weather_city.iloc[-1]

    # Initialize the models for temperature and weather code
    model_temp = predict_temperature(weather_city)
    model_weather_code = predict_weather_code(weather_city)

    # Create an empty list to store predictions for future dates
    future_predictions = []

    for future_date in future_dates:
        # Generate future features (assume the date is continuous, e.g., next day after the last)
        future_data = last_row.copy()
        future_data['year'] = future_date.year
        future_data['month'] = future_date.month
        future_data['day'] = future_date.day
        future_data['weekday'] = future_date.weekday()

        # Make predictions
        temp_pred = model_temp.predict([[
            future_data['year'], future_data['month'], future_data['day'], future_data['weekday'], 
            future_data['temp_lag_1'], future_data['rain_lag_1']
        ]])
        
        weather_code_pred = model_weather_code.predict([[
            future_data['year'], future_data['month'], future_data['day'], future_data['weekday'], 
            future_data['temp_lag_1'], future_data['rain_lag_1']
        ]])

        # Store the predictions
        future_predictions.append({
            'date': future_date,
            'predicted_temperature': temp_pred[0],
            'predicted_weather_code': weather_code_pred[0]
        })

        # Update last row with the predicted values for the next iteration
        last_row['temperature_2m_mean'] = temp_pred[0]
        last_row['weathercode'] = weather_code_pred[0]
        last_row['temp_lag_1'] = temp_pred[0]
        last_row['rain_lag_1'] = 0  # Assuming no rain for simplicity (you can adjust this)

    return future_predictions

# Example usage for future prediction
city_name = 'Kandy'
future_dates = pd.to_datetime(['2025-03-12', '2025-03-22', '2025-03-23'])  # Example future dates
future_predictions = predict_future_weather(city_name, future_dates)

# Display predictions
for prediction in future_predictions:
    print(f"Date: {prediction['date'].strftime('%Y-%m-%d')}, "
          f"Predicted Temperature: {prediction['predicted_temperature']}°C, "
          f"Predicted Weather Code: {prediction['predicted_weather_code']}")

