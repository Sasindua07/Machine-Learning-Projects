import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
dataset = pd.read_csv('/Users/sasindu/Desktop/Predicta/historical_weather.csv')

# Drop unnecessary columns
dataset_2 = dataset.drop(columns=[ 'min_temp_c', 'max_temp_c', 'precipitation_mm', 'snow_depth_mm', 'avg_wind_dir_deg', 'avg_wind_speed_kmh'])

#dataset_2 = dataset_2.loc[dataset_2.date < '2018-01-01']

# Apply label encoding to the 'city_id' column
label_encoder = LabelEncoder()
dataset_2['city_id'] = label_encoder.fit_transform(dataset_2['city_id'])

# Split the data into training and test sets
#Train = dataset_2.loc[(dataset_2.date < '2018-01-01') & (dataset_2.date > '2018-01-07') ]
#Test = dataset_2.loc[(dataset_2.date >= '2018-01-01') & (dataset_2.date <= '2018-01-07')]

#Test['date'] = pd.to_datetime(Test['date'])
#Test.set_index('date', inplace=True)

dataset_2['date'] = pd.to_datetime(dataset_2['date'])
dataset_2.set_index('date', inplace=True)

PREDICTIONS = []
for i in range(100):
    city1 = dataset_2[dataset_2['city_id'] == i].copy()  # Explicitly create a copy
    city1 = city1.drop(columns='city_id')

    avg_temp = city1['avg_temp_c'].mean()

    # Fill missing values with the mean
    city1['avg_temp_c'] = city1['avg_temp_c'].fillna(avg_temp)

    # Test size: 50 days
    w_test_size = 50
    w_test_index = len(city1) - w_test_size

    # Dataset is split into training and test data
    w_train = city1.iloc[:w_test_index]
    w_test = city1.iloc[w_test_index:]

    # Scaling the training and test data
    w_scaler = MinMaxScaler()
    w_scaler.fit(w_train)
    city1_scaled = w_scaler.transform(city1)
    w_scaled_train = w_scaler.transform(w_train)
    w_scaled_test = w_scaler.transform(w_test)

    # Parameters
    length1 = 16
    length2 = 16
    n_features = 1
    w_time_series_generator = TimeseriesGenerator(w_scaled_train, w_scaled_train, length=length1, batch_size=1)

    # Initialize the model
    initializer = tf.keras.initializers.he_uniform(seed=0)
    model = Sequential()

    # Add LSTM layer
    model.add(LSTM(16, activation='relu', input_shape=(length1, n_features), kernel_initializer=initializer, bias_initializer=initializers.Constant(0.01)))

    # Add Dense layer
    model.add(Dense(1, activation='linear', kernel_initializer=initializer, bias_initializer=initializers.Constant(0.01)))

    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=opt, loss='mse')

    # Create the validation generator
    w_time_series_val_generator = TimeseriesGenerator(w_scaled_test, w_scaled_test, length=length2, batch_size=1)

    # Define EarlyStopping callback
    Early_Stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)

    # Train the model
    model.fit(w_time_series_generator,epochs=100,
                    validation_data=w_time_series_val_generator ,
                callbacks=[Early_Stopping])
    # Generate predictions for the next 7 days
    test_outputs = []
    batch =  w_scaled_test[-length1:].reshape((1, length1, n_features))

    for j in range(7):
        test_out = model.predict(batch)[0]
        test_outputs.append(test_out)
        batch = np.append(batch[:, 1:, :], [[test_out]], axis=1)
    
    print('successfully done')
    # Apply the inverse_transform function to the test_outputs to get their Average WeeklyMinTemp values
    lstm_predictions = w_scaler.inverse_transform(test_outputs)

    # A new test feature is created to comprise the LSTM test predictions
    PREDICTIONS.extend(lstm_predictions.flatten())
    print(len(PREDICTIONS))
    print(PREDICTIONS)

