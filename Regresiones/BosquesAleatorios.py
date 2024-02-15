from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

california = datasets.fetch_california_housing()


X = california.data[:, np.newaxis,5]
Y = california.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, shuffle=False)

bar = RandomForestRegressor(n_estimators=300, max_depth=8)

bar.fit(x_train, y_train)

y_pred = bar.predict(x_test)

# Mala elección de modelo
print('Precisión del modelo: ', bar.score(x_test, y_test))
