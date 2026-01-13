# _main_.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from metro_interstate_clean import load_and_preprocess

def create_sequences(X, y, time_steps=24):
    """
    Create sequences of time_steps for LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape):
    """
    Build LSTM model.
    """
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    return model

if _name_ == "_main_":
    # Load dataset
    data_path = "metro_interstate_clean/Metro_Interstate_Traffic_Volume.csv"
    X_scaled, y_scaled, scaler_X, scaler_y, df = load_and_preprocess(data_path)

    # Create sequences
    TIME_STEPS = 24
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred)
    y_test_rescaled = scaler_y.inverse_transform(y_test)

    print("Sample Predictions:", y_pred_rescaled[:5].flatten())
    print("Actual Values:", y_test_rescaled[:5].flatten())