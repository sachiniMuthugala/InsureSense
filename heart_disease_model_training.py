import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Preprocessing
# Convert categorical variables to numerical
df['gender'] = df['gender'].map({'male': 0, 'female': 1})

# Split the data into input features and target variable
X = df.drop('risk', axis=1)
y = df['risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Save the model
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)
