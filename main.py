# Machine learning
from sklearn.svm import SVC

# For data manipulation
import pandas as pd
import numpy as np
import polygon
import PolygonImport

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

SPY = pd.read_csv('/Users/samfuller/Desktop/Trading/SPY.csv', index_col=0)
puts_concat = PolygonImport.put_dict
put_df = pd.json_normalize(puts_concat)
print(put_df.head())

# Convert index to datetime format
SPY.index = pd.to_datetime(SPY.index)

# Print the first five rows
SPY.head()

# Create predictor variables
SPY['Open-Close'] = SPY.Open - SPY.Close
SPY['High-Low'] = SPY.High - SPY.Low

# Store all predictor variables in a variable X
X = SPY[['Open-Close', 'High-Low']]
X.head()

# Target variables
y = np.where(SPY['Close'].shift(-1) > SPY['Close'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(SPY))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)

accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

# Predicted Signal
SPY['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
SPY['Return'] = SPY.Close.pct_change()

# Calculate strategy returns
SPY['Strategy_Return'] = SPY.Return * SPY.Predicted_Signal.shift(1)

# Calculate geometric returns
geometric_returns = (SPY.Strategy_Return.iloc[split:]+1).cumprod()

# Plot geometric returns
geometric_returns.plot(figsize=(10, 7),color='g')
plt.ylabel("Strategy Returns (%)")
plt.xlabel("Date")
plt.show()