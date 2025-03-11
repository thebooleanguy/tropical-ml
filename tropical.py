import pandas as pd
data = pd.read_csv('dataset/SriLanka_Weather_Dataset.csv')

print(data.head())
print(data.info())
print(data.describe())
