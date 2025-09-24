import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the fertilizer dataset
data_fertilizer = pd.read_csv('Fertilizer.csv')

# Step 2: Encode the target (fertilizer name)
le_fertilizer = LabelEncoder()
data_fertilizer['fertilizer_enc'] = le_fertilizer.fit_transform(data_fertilizer['Fertilizer Name'])

# Step 3: Select Features (Nitrogen, Potassium, Phosphorous) and Target (fertilizer_enc)
X_fertilizer = data_fertilizer[['Nitrogen', 'Potassium', 'Phosphorous']]
y_fertilizer = data_fertilizer['fertilizer_enc']

# Step 4: Train the RandomForest model
model_fertilizer = RandomForestClassifier()
model_fertilizer.fit(X_fertilizer, y_fertilizer)

# Step 5: Save the model and label encoder
joblib.dump(model_fertilizer, 'model_fertilizer.pkl')
joblib.dump(le_fertilizer, 'label_encoder_fertilizer.pkl')

print("Fertilizer recommendation model and label encoder saved as 'model_fertilizer.pkl' and 'label_encoder_fertilizer.pkl'.")
