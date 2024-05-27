# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
data = pd.DataFrame(‘/content/InsFraudDataset6.csv’)

# Train Isolation Forest
model = IsolationForest(contamination=0.25, random_state=42)
model.fit(data)

# Predict anomaly scores
data['Anomaly_Score'] = model.decision_function(data)
data['Anomaly'] = model.predict(data)

# Display the DataFrame with anomaly scores
print(data)

# Explain the anomalies using SHAP library
explainer = shap.Explainer(model.predict, data)
shap_values = explainer(data)
# Visualizing SHAP Values for the First Data Point
shap.plots.waterfall(shap_values[0])
# Printing Anomalies and Their SHAP Values
for i in range(len(data)):
    if anomalies[i] == -1:
        print(f"Anomaly detected in data point {i}: {data.iloc[i]}")
        shap.plots.waterfall(shap_values[i])
