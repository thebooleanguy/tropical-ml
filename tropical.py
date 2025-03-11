import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('dataset/SriLanka_Weather_Dataset.csv')

# Display basic information about the dataset
print(data.head())
print(data.info())
print(data.describe())

# Convert 'time' column to datetime format
data['time'] = pd.to_datetime(data['time'])

# Extract useful date features
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month
data['day'] = data['time'].dt.day
data['day_of_year'] = data['time'].dt.dayofyear
data['week_of_year'] = data['time'].dt.isocalendar().week

# Convert 'sunrise' and 'sunset' to datetime and calculate daylight duration
data['sunrise'] = pd.to_datetime(data['sunrise'])
data['sunset'] = pd.to_datetime(data['sunset'])
data['daylight_hours'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 3600

# Drop unnecessary columns (raw time, text-based location info)
data = data.drop(columns=['time', 'sunrise', 'sunset', 'country'])

# Encode 'city' as numerical labels
encoder = LabelEncoder()
data['city'] = encoder.fit_transform(data['city'])

# Select target variable (change based on prediction goal)
target = 'temperature_2m_mean'  # Change to 'weathercode' for classification
X = data.drop(columns=[target])
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features to standardize value ranges
scaler = StandardScaler()
numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Print final dataset shape
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
