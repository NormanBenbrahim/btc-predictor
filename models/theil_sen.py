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
import joblib
import timeit

VERSION_NUMBER = 1

# based on this paper
# https://www.researchgate.net/publication/328989226_Machine_Learning_Models_Comparison_for_Bitcoin_Price_Prediction

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
#df_scaled = scaler.fit_transform(df)

# use close price as prediction variable (y)
df_subset = df.tail(100000) # 30s for training this subset
y = df_subset.drop(['Timestamp', 'Open', 'High', 'Low', 'Volume_(BTC)',
       'Volume_(Currency)', 'Weighted_Price'], axis=1)
X = df_subset[['Timestamp', 'Open', 'High', 'Low', 'Volume_(BTC)',
       'Volume_(Currency)', 'Weighted_Price']]

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# split data
print("splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, shuffle=False, test_size = 0.05)

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
y_hat = model.predict(X_test)

print(f'saving model...')
joblib.dump(model, f"./data/saved_models/theilsen-V{VERSION_NUMBER}.model")

# plot results
print("plotting result...")
y_test_inverse = scaler.inverse_transform(y_test.reshape(1,-1))
y_hat_inverse = scaler.inverse_transform(y_hat.reshape(1,-1))

plt.plot(y_test_inverse[0,:], label="Actual Price", color='green')
plt.plot(y_hat_inverse[0, :], label="Predicted Price", color='red') 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.show()
