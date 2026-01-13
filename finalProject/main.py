from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np

my_models = {}
my_models['MyLSTM_1'] = load_model('lstm_traffic_model.h5')
my_scaler = joblib.load('my_scaler.save')

train_df=pd.read_csv('train_df.csv')



# 1. Extract training feature names (replace X with your actual DataFrame used for training)
train_features = train_df.columns.tolist()
print(" Features used in training:", len(train_features))

# 2. Create some realistic dummy test data
# Example: 3 rows of data
np.random.seed(42)

# Example realistic ranges for features
test_data_dict = {
    "hour": [10, 14, 20],            # morning, afternoon, night
    "weekday": [5, 3, 6],           # Mon, Wed, Sat
    "month": [1, 6, 12],            # Jan, Jun, Dec
    "temp": [15, 30, 22],    # Celsius
    "humidity": [40, 70, 55],       # %
    "wind_speed": [15, 15, 10],       # km/h
    "weather_Clear": [1, 0, 0],
    "weather_Clouds": [0, 1, 0],
    "weather_Rain": [0, 0, 1],
}

# 3. For missing features (since your project had 79 features), fill with random values
for f in train_features:
    if f not in test_data_dict:
        test_data_dict[f] = np.random.randint(0, 100, size=3)

# 4. Build test DataFrame with correct feature order
test_data = pd.DataFrame(test_data_dict)[train_features]

print("\nTest Data (before scaling):")
print(test_data.head())

# 5. Scale using the same scaler
scaled_test_data = my_scaler.transform(test_data)

# 6. Reshape for LSTM [batch, time, features]
scaled_test_data = np.expand_dims(scaled_test_data, axis=1)

# 7. Predict using trained model
predictions = my_models['MyLSTM_1'].predict(scaled_test_data)
predictions = np.maximum(0, predictions) 

print("\n Predictions:")
print(predictions)




import numpy as np

# Denormalization formula: original_value = scaled_value * (max - min) + min
# According to the notebook, min and max for traffic_volume are 0 and 7280
min_traffic_volume = 0
max_traffic_volume = 7280

# The scaled predictions you provided
scaled_predictions = np.array(predictions)

# Apply the denormalization formula
denormalized_predictions = scaled_predictions * (max_traffic_volume - min_traffic_volume) + min_traffic_volume

print(" Denormalized Predictions:")
print(denormalized_predictions)
