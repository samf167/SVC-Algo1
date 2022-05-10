# S&P ETF May 20th 400 put SVC model based on vol, o-c, h-l (range)

# For data manip
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

puts_concat = PolygonImport.put_dict
put_df = pd.json_normalize(puts_concat)
print(put_df.head())

# Create predictor variables
put_df['Open-Close'] = put_df.o - put_df.c
put_df['High-Low'] = put_df.h - put_df.l
#put_df['Volume'] = put_df.v

# Store all predictor variables in a variable X
X = put_df[['Open-Close', 'High-Low']]
print(X.head())

# Target variables
y = np.where(put_df['c'].shift(-1) > put_df['c'], 1, 0)

split_percentage = 0.8
split = int(split_percentage*len(put_df))

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
put_df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
put_df['Return'] = put_df.c.pct_change()

# Calculate strategy returns
put_df['Strategy_Return'] = put_df.Return * put_df.Predicted_Signal.shift(1)

# Calculate geometric returns
geometric_returns = (put_df.Strategy_Return.iloc[split:]+1).cumprod()

# Plot geometric returns
geometric_returns.plot(figsize=(10, 7),color='g')
plt.ylabel("Strategy Returns (%)")
plt.xlabel("Date")
plt.show()
