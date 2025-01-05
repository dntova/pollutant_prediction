#!/usr/bin/env python

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import RandomizedSearchCV

# ------------------------------------------------------------------------
TRAIN_CSV = 'data/daily_averages.csv'
TEST_CSV = 'data/predict.csv'


def train_and_save_model(pollutant):
    """
    Train and save a Random Forest model for a specific pollutant.
    """
    target_col = pollutant  # e.g., 'pm10', 'pm25', ...
    MODEL_FILE = f'{pollutant}_forecast_model.pkl'

    df = pd.read_csv(TRAIN_CSV, parse_dates=['time'])

    # Weather features
    feature_cols = ['tp', 't2m', 'u10', 'v10', 'sp']

    # Add lag and rolling average features for the chosen pollutant
    for lag in [1, 2, 3]:
        df[f'{pollutant}_lag_{lag}'] = df[target_col].shift(lag)
    df[f'{pollutant}_rolling_avg'] = df[target_col].rolling(window=3).mean()
    df = df.dropna()

    feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]] + [f'{pollutant}_rolling_avg']
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }
    rf = RandomForestRegressor(random_state=42)
    rf_random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=-1)
    rf_random_search.fit(X_scaled, y)

    best_model = rf_random_search.best_estimator_
    joblib.dump((best_model, scaler), MODEL_FILE)
    print(f"Tuned Random Forest model for '{pollutant}' saved to '{MODEL_FILE}'")


def load_model_and_evaluate(pollutant, plot_results=True):
    """
    Load a Random Forest model for a specific pollutant and evaluate on test data.
    """
    target_col = pollutant
    MODEL_FILE = f'{pollutant}_forecast_model.pkl'

    model, scaler = joblib.load(MODEL_FILE)
    df_test = pd.read_csv(TEST_CSV, parse_dates=['time'])

    # Weather features
    feature_cols = ['tp', 't2m', 'u10', 'v10', 'sp']

    # Add lag and rolling average features
    for lag in [1, 2, 3]:
        df_test[f'{pollutant}_lag_{lag}'] = df_test[target_col].shift(lag)
    df_test[f'{pollutant}_rolling_avg'] = df_test[target_col].rolling(window=3).mean()
    df_test = df_test.dropna()  # Drop rows with NaN due to lag and rolling mean

    feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]] + [f'{pollutant}_rolling_avg']
    X_test = df_test[feature_cols].copy()
    y_test = df_test[target_col].copy()

    # Scale the features using the same scaler from training
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Evaluation for '{pollutant}' on test dataset:\n"
          f"  MAE:  {mae:.4f}\n"
          f"  RMSE: {rmse:.4f}\n"
          f"  R^2:  {r2:.4f}")

    if plot_results:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.xlabel(f"Actual {pollutant.upper()}")
        plt.ylabel(f"Predicted {pollutant.upper()}")
        plt.title(f"Random Forest - Predicted vs. Actual {pollutant.upper()} (Test Data)")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()


def train_prophet_and_save_model(pollutant):
    """
    Train and save a Prophet model for a specific pollutant.
    """
    target_col = pollutant
    PROPHET_MODEL_FILE = f'{pollutant}_forecast_prophet.pkl'

    df = pd.read_csv(TRAIN_CSV, parse_dates=['time'])
    prophet_df = df[['time', target_col]].copy()
    prophet_df.columns = ['ds', 'y']

    model = Prophet()
    model.fit(prophet_df)

    joblib.dump(model, PROPHET_MODEL_FILE)
    print(f"Prophet model for '{pollutant}' saved to '{PROPHET_MODEL_FILE}'")



def prophet_evaluate(pollutant, plot_results=True):
    """
    Evaluate a previously trained Prophet model for a specific pollutant.
    """
    target_col = pollutant
    PROPHET_MODEL_FILE = f'{pollutant}_forecast_prophet.pkl'

    model = joblib.load(PROPHET_MODEL_FILE)
    df_test = pd.read_csv(TEST_CSV, parse_dates=['time'])
    prophet_test = df_test[['time']].copy()
    prophet_test.columns = ['ds']

    y_test = df_test[target_col].values
    forecast = model.predict(prophet_test)
    df_test[f'{pollutant}_predicted'] = forecast['yhat']
    predictions = df_test[f'{pollutant}_predicted'].values

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Prophet Evaluation for '{pollutant}' on test dataset:\n"
          f"  MAE:  {mae:.4f}\n"
          f"  RMSE: {rmse:.4f}\n"
          f"  R^2:  {r2:.4f}")

    if plot_results:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.xlabel(f"Actual {pollutant.upper()}")
        plt.ylabel(f"Predicted {pollutant.upper()}")
        plt.title(f"Prophet - Predicted vs. Actual {pollutant.upper()}")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()

    return df_test


def create_tf_dataset(X, y, sequence_length, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.window(sequence_length, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(sequence_length), y.batch(sequence_length))))
    dataset = dataset.map(lambda x, y: (tf.reshape(x, (sequence_length, -1)), y[-1]))  # Keep the last y value
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_rnn_and_save_model(pollutant):
    """
    Train and save an LSTM-based RNN model for a specific pollutant.
    """
    target_col = pollutant
    RNN_MODEL_FILE = f'{pollutant}_rnn_model.keras'
    SCALER_FILE = f'{pollutant}_rnn_model_scaler.pkl'

    df = pd.read_csv(TRAIN_CSV, parse_dates=['time'])
    feature_cols = ['tp', 't2m', 'u10', 'v10', 'sp']

    # Add lag and rolling average features
    for lag in [1, 2, 3]:
        df[f'{pollutant}_lag_{lag}'] = df[target_col].shift(lag)
    df[f'{pollutant}_rolling_avg'] = df[target_col].rolling(window=3).mean()
    df = df.dropna()

    feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]] + [f'{pollutant}_rolling_avg']
    X = df[feature_cols].values
    y = df[target_col].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create tf.data.Dataset for sequences
    sequence_length = 10
    batch_size = 16
    dataset = create_tf_dataset(X_scaled, y, sequence_length, batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True,
                             input_shape=(sequence_length, len(feature_cols))),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32, activation='tanh'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

    model.fit(dataset, epochs=500, validation_data=dataset, verbose=1,
              callbacks=[early_stopping, lr_scheduler])
    model.save(RNN_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"LSTM model for '{pollutant}' saved to '{RNN_MODEL_FILE}' and scaler saved to '{SCALER_FILE}'")


def evaluate_rnn_model(pollutant, plot_results=True):
    """
    Evaluate the saved RNN model for a specific pollutant.
    """
    target_col = pollutant
    RNN_MODEL_FILE = f'{pollutant}_rnn_model.keras'
    SCALER_FILE = f'{pollutant}_rnn_model_scaler.pkl'

    df_test = pd.read_csv(TEST_CSV, parse_dates=['time'])
    feature_cols = ['tp', 't2m', 'u10', 'v10', 'sp']

    # Add lag and rolling average features
    for lag in [1, 2, 3]:
        df_test[f'{pollutant}_lag_{lag}'] = df_test[target_col].shift(lag)
    df_test[f'{pollutant}_rolling_avg'] = df_test[target_col].rolling(window=3).mean()
    df_test = df_test.dropna()

    feature_cols += [f'{pollutant}_lag_{lag}' for lag in [1, 2, 3]] + [f'{pollutant}_rolling_avg']
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    scaler = joblib.load(SCALER_FILE)
    X_test_scaled = scaler.transform(X_test)

    # Create tf.data.Dataset for evaluation
    sequence_length = 10
    batch_size = 1
    test_dataset = create_tf_dataset(X_test_scaled, y_test, sequence_length, batch_size)

    model = tf.keras.models.load_model(RNN_MODEL_FILE)
    predictions = model.predict(test_dataset).flatten()

    # Align y_test and predictions
    y_test_aligned = y_test[sequence_length - 1:]

    if len(y_test_aligned) != len(predictions):
        print(f"Length mismatch! y_test: {len(y_test_aligned)}, predictions: {len(predictions)}")
        return

    mae = mean_absolute_error(y_test_aligned, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_aligned, predictions))
    r2 = r2_score(y_test_aligned, predictions)

    print(f"RNN Evaluation for '{pollutant}' on test dataset:\n"
          f"  MAE:  {mae:.4f}\n"
          f"  RMSE: {rmse:.4f}\n"
          f"  R^2:  {r2:.4f}")

    if plot_results:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test_aligned, predictions, alpha=0.5)
        plt.xlabel(f"Actual {pollutant.upper()}")
        plt.ylabel(f"Predicted {pollutant.upper()}")
        plt.title(f"LSTM - Predicted vs. Actual {pollutant.upper()}")
        plt.plot([y_test_aligned.min(), y_test_aligned.max()],
                 [y_test_aligned.min(), y_test_aligned.max()], 'r--')
        plt.show()



if __name__ == "__main__":
    # List available modes
    print("\nAvailable modes:")
    print("1) train")
    print("2) evaluate")
    print("3) prophet_train")
    print("4) prophet_evaluate")
    print("5) rnn_train")
    print("6) rnn_evaluate")
    
    try:
        mode_number = int(input("\nPlease select a mode (1-6): ").strip())
    except ValueError:
        print("Invalid input. Please enter a number from 1 to 6.")
        exit()

    # Ask user which pollutant they want to work with
    print("\nAvailable pollutants: pm25, pm10, o3, no2, so2, co")
    pollutant = input("Please select a pollutant: ").strip().lower()
    valid_pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    if pollutant not in valid_pollutants:
        print(f"Invalid pollutant '{pollutant}'. Exiting.")
        exit()

    # Toggle to True/False if you want plots
    plot_results = True

    if mode_number == 1:
        train_and_save_model(pollutant)
    elif mode_number == 2:
        load_model_and_evaluate(pollutant, plot_results=plot_results)
    elif mode_number == 3:
        train_prophet_and_save_model(pollutant)
    elif mode_number == 4:
        prophet_evaluate(pollutant, plot_results=plot_results)
    elif mode_number == 5:
        train_rnn_and_save_model(pollutant)
    elif mode_number == 6:
        evaluate_rnn_model(pollutant, plot_results=plot_results)
    else:
        print("Invalid number selected.")
