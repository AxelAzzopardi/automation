from keras.models import load_model
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from keras.callbacks import EarlyStopping

# Read in the training data
data = pd.read_csv('train_data.csv', usecols=['Year', 'Make', 'Model', 'Trim', 'Transmission', 'Displacement',
                                              'Configuration', 'Aspiration', 'Weight', 'Power', 'W/P Ratio',
                                              'DriveTrain', 'Price', 'Lap Time'])

data['Aspiration'].fillna('NA', inplace=True)

# Extract the target column (lap time)
lap_time = data['Lap Time']
data = data.drop(columns=['Lap Time'])

# Encode text inputs using LabelEncoder
le = LabelEncoder()
data = data.apply(le.fit_transform)

for i in range(len(lap_time)):
    minutes, seconds = str(lap_time[i]).split(':')
    lap_time[i] = int(minutes) * 60 + float(seconds)

# Scale the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(data, lap_time, test_size=0.2, random_state=1)

# Create the model
model = Sequential()
model.add(Dense(128, activation='sigmoid', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

# Compile the model
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error'])

y_train = np.asarray(y_train).astype(np.float64)
y_val = np.asarray(y_val).astype(np.float64)

# Create a callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=300, min_delta=0.0001, mode='min')

# Train the model
history = model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_val, y_val)
                    , callbacks=[early_stopping])

model.save("trained_model.h5")

# load the test data
test_data = pd.read_csv("test_data.csv", usecols=['Year', 'Make', 'Model', 'Trim', 'Transmission', 'Displacement',
                                                  'Configuration', 'Aspiration', 'Weight', 'Power', 'W/P Ratio',
                                                  'DriveTrain', 'Price', 'Lap Time'])

test_data['Aspiration'].fillna('NA', inplace=True)
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col])

# separate the input features and target column
X_test = test_data.drop("Lap Time", axis=1)
X_test = scaler.transform(X_test)
y_test = test_data["Lap Time"]

model = load_model("trained_model.h5")

# make predictions on the test data
y_prediction = model.predict(X_test)
y_prediction = np.array(y_prediction, dtype=str)

for i in range(len(y_prediction)):
    minutes, seconds = divmod(float(y_prediction[i]), 60)
    formatted_time = "{:02d}:{:06.3f}".format(int(minutes), seconds)
    y_prediction[i] = formatted_time

print("Predicted:", y_prediction)