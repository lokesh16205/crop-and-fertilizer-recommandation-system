import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the crop recommendation dataset
data_crop = pd.read_csv('Crop_recommendation.csv')

# Step 2: Encode the target (crop label)
le_crop = LabelEncoder()
data_crop['label_enc'] = le_crop.fit_transform(data_crop['label'])

# Step 3: Select Features (N, P, K, temperature, humidity, pH, rainfall) and Target (label_enc)
X_crop = data_crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = data_crop['label_enc']

# Step 4: Train the RandomForest model
model_crop = RandomForestClassifier()
model_crop.fit(X_crop, y_crop)

# Step 5: Save the model and label encoder
joblib.dump(model_crop, 'model_crop.pkl')
joblib.dump(le_crop, 'label_encoder_crop.pkl')

print("Crop recommendation model and label encoder saved as 'model_crop.pkl' and 'label_encoder_crop.pkl'.")
