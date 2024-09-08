from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

df = pd.read_csv('afldata.csv')
df = df.drop(columns=['homescore', 'awayscore'])
df.head()

y = df['winningmargin']
X = df.drop(columns=['winningmargin'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

input_data = pd.DataFrame([[2, 1, 58000]], columns=['hometeam', 'awayteam', 'attendance'])
print(model.predict(input_data))