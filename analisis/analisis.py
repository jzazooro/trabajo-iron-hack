import pandas as pd
import sklearn

df = pd.read_csv('regresion_data.csv')
df.head()

# analisis del csv
df.describe()

# graficar
import matplotlib.pyplot as plt
plt.scatter(df['x'], df['y'])
plt.show()

# regresion lineal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = df['x'].values.reshape(-1, 1)
y = df['y'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regresion = LinearRegression()
regresion.fit(x_train, y_train)

# graficar
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regresion.predict(x_train), color='blue')
plt.title('Regresion Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# predecir
y_pred = regresion.predict(x_test)

# graficar
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regresion.predict(x_train), color='blue')
plt.title('Regresion Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# regresion polinomial
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

regresion_poly = LinearRegression()
regresion_poly.fit(x_poly, y)

# graficar
plt.scatter(x, y, color='red')
plt.plot(x, regresion_poly.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Regresion Polinomial')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# predecir
y_pred = regresion_poly.predict(poly_reg.fit_transform(x_test))

# graficar
plt.scatter(x_test, y_test, color='red')
plt.plot(x, regresion_poly.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Regresion Polinomial')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# regresion logistica
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('clasificacion_data.csv')
df.head()

# media de una columna en especifico
df['y'].mean()

# graficar
plt.scatter(df['x1'], df['x2'], c=df['y'])
plt.show()

# varianza de una columna en especifico
df['x1'].var()

# desviacion tipica de una columna en especifico
df['x1'].std()

