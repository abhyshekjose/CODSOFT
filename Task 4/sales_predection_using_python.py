import pandas as pd

file_path = r'D:\codsoft\Task 4\advertising.csv'
data = pd.read_csv(file_path)
print(data.head())

print(data.info())
print(data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data)
plt.show()

data = data.dropna()


from sklearn.model_selection import train_test_split

X = data.drop('Sales', axis=1)  
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R^2 Score:', r2_score(y_test, y_pred))

import pandas as pd
import numpy as np

X_train_df = pd.DataFrame(X_train, columns=data.columns[:-1])  # Assuming last column is 'Sales'
X_test_df = pd.DataFrame(X_test, columns=data.columns[:-1])

new_data = pd.DataFrame(np.array([[150, 30, 70]]), columns=['TV', 'Radio', 'Newspaper'])

predictions = model.predict(new_data)
print(predictions)
