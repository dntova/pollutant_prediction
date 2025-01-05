#!/usr/bin/env python

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRAIN_CSV = 'data/daily_averages.csv'
TEST_CSV = 'data/predict.csv'

RNN_MODEL_FILE = 'all_pollutants_rnn_model.keras'
SCALER_FILE = 'all_pollutants_rnn_model_scaler.pkl'

POLLUTANT_COLS = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

WEATHER_COLS = ['tp', 't2m', 'u10', 'v10', 'sp']


def create_tf_dataset(X, y, sequence_length, batch_size):
    """
    Creates a tf.data.Dataset for RNN training/evaluation where each sample
    is a sequence of length 'sequence_length'.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.window(sequence_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(
        lambda x, y: tf.data.Dataset.zip((x.batch(sequence_length), y.batch(sequence_length)))
    )
    # For each sequence window, we feed the entire X window into the model
    # but only use the final row's y as the label.
    dataset = dataset.map(
        lambda x, y: (tf.reshape(x, (sequence_length, -1)), y[-1])
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_rnn_for_all_pollutants():
    """
    Train and save a single LSTM-based RNN model that predicts
    all pollutants simultaneously.
    """
    df = pd.read_csv(TRAIN_CSV, parse_dates=['time'])

    # For each pollutant, add lag and rolling average features
    for pollutant in POLLUTANT_COLS:
        for lag in [1, 2, 3]:
            df[f'{pollutant}_lag_{lag}'] = df[pollutant].shift(lag)
        df[f'{pollutant}_rolling_avg'] = df[pollutant].rolling(window=3).mean()

    # Drop rows with NaN from shifting/rolling
    df = df.dropna()

    # Construct feature columns:
    #  - Weather features
    #  - Lags and rolling averages for each pollutant
    feature_cols = []
    feature_cols.extend(WEATHER_COLS)
    for pollutant in POLLUTANT_COLS:
        feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]]
        feature_cols.append(f'{pollutant}_rolling_avg')

    # Inputs (X) and multi-output targets (Y) 
    X = df[feature_cols].values
    Y = df[POLLUTANT_COLS].values  # shape: (num_samples, 6)

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create tf.data.Dataset for sequences
    sequence_length = 10
    batch_size = 16
    train_dataset = create_tf_dataset(X_scaled, Y, sequence_length, batch_size)

    model = keras.Sequential([
        keras.layers.LSTM(128, activation='tanh', return_sequences=True,
                          input_shape=(sequence_length, X_scaled.shape[1])),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, activation='tanh'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dense(len(POLLUTANT_COLS))
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=20,
                                                   restore_best_weights=True)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=10)

    model.fit(train_dataset, 
              epochs=500, 
              validation_data=train_dataset, 
              callbacks=[early_stopping, lr_scheduler],
              verbose=1)

    model.save(RNN_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print(f"Trained multi-pollutant LSTM model saved to '{RNN_MODEL_FILE}'")
    print(f"Scaler saved to '{SCALER_FILE}'")


def evaluate_rnn_for_all_pollutants(plot_results=True):
    """
    Evaluate the saved LSTM model on test data for all pollutants simultaneously.
    Computes MAE, RMSE, and R^2 for each pollutant.
    """
    df_test = pd.read_csv(TEST_CSV, parse_dates=['time'])

    for pollutant in POLLUTANT_COLS:
        for lag in [1, 2, 3]:
            df_test[f'{pollutant}_lag_{lag}'] = df_test[pollutant].shift(lag)
        df_test[f'{pollutant}_rolling_avg'] = df_test[pollutant].rolling(window=3).mean()

    df_test = df_test.dropna()

    feature_cols = []
    feature_cols.extend(WEATHER_COLS)
    for pollutant in POLLUTANT_COLS:
        feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]]
        feature_cols.append(f'{pollutant}_rolling_avg')

    X_test = df_test[feature_cols].values
    Y_test = df_test[POLLUTANT_COLS].values

    scaler = joblib.load(SCALER_FILE)
    X_test_scaled = scaler.transform(X_test)

    sequence_length = 10
    batch_size = 1
    test_dataset = create_tf_dataset(X_test_scaled, Y_test, sequence_length, batch_size)

    model = keras.models.load_model(RNN_MODEL_FILE)
    predictions = model.predict(test_dataset)  # shape: (num_samples, 6)

    Y_test_aligned = Y_test[sequence_length - 1:]  # shape should match predictions

    if len(Y_test_aligned) != len(predictions):
        print(f"Length mismatch! Y_test_aligned: {len(Y_test_aligned)}, predictions: {len(predictions)}")
        return

    for i, pollutant in enumerate(POLLUTANT_COLS):
        y_true = Y_test_aligned[:, i]
        y_pred = predictions[:, i]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\nEvaluation for '{pollutant}':")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R^2:  {r2:.4f}")

        if plot_results:
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.xlabel(f"Actual {pollutant.upper()}")
            plt.ylabel(f"Predicted {pollutant.upper()}")
            plt.title(f"LSTM - Predicted vs. Actual {pollutant.upper()}")
            min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.show()


if __name__ == "__main__":
    print("\nAvailable modes for multi-pollutant RNN:")
    print("1) train")
    print("2) evaluate")

    try:
        mode_number = int(input("\nPlease select a mode (1 or 2): ").strip())
    except ValueError:
        print("Invalid input. Please enter 1 or 2.")
        exit()

    if mode_number == 1:
        train_rnn_for_all_pollutants()
    elif mode_number == 2:
        evaluate_rnn_for_all_pollutants(plot_results=True)
    else:
        print("Invalid selection. Exiting.")
