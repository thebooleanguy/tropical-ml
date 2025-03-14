# Import necessary library
import pandas as pd

# Load dataset
data = pd.read_csv('dataset/SriLanka_Weather_Dataset.csv')

# Extract unique city names
unique_cities = data['city'].unique()

# Print the unique city names
print("Unique city names:")
for city in unique_cities:
    print(city)

