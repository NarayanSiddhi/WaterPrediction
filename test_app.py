import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the trained model
model = joblib.load("svm.pkl")

# Load the validation dataset
validation_data = pd.read_csv("test_data.csv")

# Prepare input features and target labels
X_val = validation_data.drop(columns=["target_column"])
y_val = validation_data["target_column"]

# Make predictions on the validation data
y_pred = model.predict(X_val)

# Evaluate performance
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
