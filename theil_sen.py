import os
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import timeit

## do not shuffle time series data
print("setting up parameters...")
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8   
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("loading training data...")
csv_path = './data/original_training/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv'
df = pd.read_csv(csv_path, parse_dates=['Timestamp'])
df = df.sort_values('Timestamp') # ensure the timeseries is right

print("preprocessing...")
# remove nans
df = df.dropna()

# scaler to scale to [0, 1]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# use close price as prediction variable (y)
y = df_scaled[:, 4]
X = np.delete(df_scaled, 4, axis=1)

# split data
print("splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size = 0.05)

# define the model 
# TODO: test results using random state vs not using it 
#model = TheilSenRegressor(random_state=RANDOM_SEED)
model = TheilSenRegressor(verbose=10)

# fit the model to training data
print("fitting model to training data...")
start_time = timeit.default_timer()
model.fit(X_train, y_train)
elapsed = timeit.default_timer() - start_time
print(f"time elapsed during training {elapsed} seconds")

# make prediction on test data
print("predicting on test data...")
ytest_ = model.predict(X_test)

#for i in range(len(ytest_)):
#    label = scaler.inverse_transform()