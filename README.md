## DEVELOPED BY: YOHESH KUMAR R.M
## REGISTER NO: 21222224118
## DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
# # Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the Microsoft stock data from the CSV file
data = pd.read_csv('Microsoft_Stock.csv', index_col='Date', parse_dates=True)

# Display the first few rows (GIVEN DATA)
print("GIVEN DATA:")
print(data.head())

# Ensure 'Close' prices are numeric and handle NaN values
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna()  # Drop rows with NaN values if any

# Perform Augmented Dickey-Fuller test for stationarity
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit an AutoRegressive (AR) model with 13 lags
model = AutoReg(train['Close'], lags=13)
model_fit = model.fit()

# Make predictions using the AR model
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test['Close'], predictions)
print('Mean Squared Error:', mse)

# Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
plt.figure(figsize=(10, 6))
plt.subplot(211)
plot_pacf(train['Close'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['Close'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

# PREDICTION
print("PREDICTION:")
print(predictions)

# Plot the test data and predictions (FINAL PREDICTION)
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Close'], label='Actual Price')
plt.plot(test.index, predictions, color='red', label='Predicted Price')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()  
```
## OUTPUT:

### GIVEN DATA
![image](https://github.com/user-attachments/assets/4ef280a6-ec25-4b7c-a677-b213acfda9de)

### ADF-STATISTIC AND P-VALUE
![image](https://github.com/user-attachments/assets/ac1b9d40-e379-45fa-81ca-b60283603ae5)


### PACF - ACF
![image](https://github.com/user-attachments/assets/2846d9a4-90d3-446c-bcaf-286427f63916)


### MSE VALUE
![image](https://github.com/user-attachments/assets/a56ef7ed-a38e-441f-abb4-1a9f416dae47)


### PREDICTION
![image](https://github.com/user-attachments/assets/f1a69958-bf46-42fe-8440-4a6f6d415853)

### FINAL PREDICTION
![image](https://github.com/user-attachments/assets/a2241bb9-9e0c-4cb8-8272-a5ac0de328cf)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
