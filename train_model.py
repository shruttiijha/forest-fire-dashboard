import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Create fake dataset
data = pd.DataFrame({
    'temperature': [20, 25, 30, 35, 40, 22, 33],
    'humidity': [30, 50, 70, 20, 10, 60, 25],
    'wind_speed': [10, 20, 15, 5, 25, 18, 12],
    'fire_risk': ['LOW', 'MEDIUM', 'HIGH', 'LOW', 'HIGH', 'MEDIUM', 'HIGH']
})

# 2. Train model
X = data[['temperature', 'humidity', 'wind_speed']]
y = data['fire_risk']

model = RandomForestClassifier()
model.fit(X, y)

# 3. Save model
joblib.dump(model, 'model.pkl')
print("✅ Model trained and saved as 'model.pkl'")
