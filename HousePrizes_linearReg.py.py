import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('Housing.csv')

X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error: ",mae)
print("R-squared: ", r2)

new_data = pd.DataFrame({
    'sqft': [2000, 1500],
    'bedrooms': [3, 2],
    'bathrooms': [2, 1]
})

new_predictions = model.predict(new_data)
print("Predicted Prices:", new_predictions)
